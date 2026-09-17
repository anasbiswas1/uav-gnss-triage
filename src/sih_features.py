#!/usr/bin/env python3
"""
sih_features.py  --  ULog -> physics-consistency window features (SIH-generated and Whelan live logs).

Design rules (from the plan):
  * Simulator truth topics (*_groundtruth) are never read here. Truth lives only in the manifest / verify step.
  * Receiver self-report fields (fix_type, satellites_used, eph, jamming_indicator, noise_per_ms, s_variance)
    are extracted into a SEPARATE column family (rx_*) that the core detector excludes. The single exception is
    fix validity: GPS samples with fix_type < 3 are masked out of the position/velocity channels (an investigator
    discards no-fix samples too) and the no-fix condition is exposed as gps_nofix_frac. The detector is therefore
    receiver-blind except for validity masking.
  * The time grid spans the whole recorded log, not the GPS-valid span, so GPS loss at the start or end of a flight
    stays visible. Gap-aware interpolation never bridges an outage.
  * Windows are timestamp-indexed half-open intervals [t, t + 5 s) advanced by exactly 2.5 s.
  * Every feature row carries flight_id so all splits can be flight-grouped.
  * Window labels are onset-aware, with onset/restore read from the log itself (spoof parameter change,
    failure-injection messages, SIM_GPS_USED changes for no-fix reporting); the manifest is the fallback.

Usage:
  python sih_features.py --sih <run dir> [--sih <another run dir>] --whelan_zip <UAVAttackData.zip> [--real_dir <folder of .ulg>] --out <dir>
Outputs: <out>/windows.csv, <out>/flights.csv, and a printed inspection + per-label feature summary.
"""
import argparse
import json
import math
import pathlib
import warnings
import zipfile

import numpy as np
import pandas as pd

WIN_S, HOP_S, HZ = 5.0, 2.5, 5.0   # window length, hop, common sample rate
FEATURE_VERSION = 'v3_fullgrid_validity_vecimu_exacthop'
SKIP_TOPICS = ('groundtruth',)      # never read simulator truth


# ------------------------------------------------------------------ ULog helpers
def load_ulog(path):
    from pyulog import ULog
    u = ULog(str(path))
    topics = {}
    for d in u.data_list:
        if any(s in d.name for s in SKIP_TOPICS):
            continue
        topics.setdefault(d.name, {})[d.multi_id] = d
    return u, topics


def first_instance(topics, *names):
    for n in names:
        if n in topics:
            inst = sorted(topics[n].keys())[0]
            return n, topics[n][inst]
    return None, None


def col(d, *names, scale=1.0, default=None):
    """First present field among names, as float array (optionally scaled)."""
    for n in names:
        if n in d.data:
            return np.asarray(d.data[n], dtype=float) * scale
    return default


def tsec(d, t0):
    return (np.asarray(d.data['timestamp'], dtype=float) - t0) / 1e6


def interp_to(grid, t, x, max_gap=1.0):
    """Linear interpolation onto grid, NaN wherever the nearest valid raw sample is farther than max_gap seconds,
    so sensor outages are preserved instead of bridged."""
    if x is None or t is None or len(t) < 2:
        return np.full_like(grid, np.nan, dtype=float)
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return np.full_like(grid, np.nan, dtype=float)
    tt, xx = t[ok], x[ok]
    y = np.interp(grid, tt, xx, left=np.nan, right=np.nan)
    j = np.clip(np.searchsorted(tt, grid), 1, len(tt) - 1)
    nearest = np.minimum(np.abs(grid - tt[j - 1]), np.abs(tt[j] - grid))
    y[nearest > max_gap] = np.nan
    return y


def hold_to(grid, t, x, max_gap=1.0):
    """Nearest-sample hold for categorical or step-like fields; NaN beyond max_gap from any raw sample."""
    if x is None or t is None or len(t) == 0:
        return np.full_like(grid, np.nan, dtype=float)
    j = np.clip(np.searchsorted(t, grid), 1, len(t) - 1) if len(t) > 1 else np.zeros(len(grid), dtype=int)
    if len(t) > 1:
        left, right = j - 1, j
        use_right = np.abs(t[right] - grid) < np.abs(grid - t[left])
        k = np.where(use_right, right, left)
    else:
        k = np.zeros(len(grid), dtype=int)
    y = np.asarray(x, dtype=float)[k]
    y[np.abs(t[k] - grid) > max_gap] = np.nan
    return y


def presence(grid, t, max_gap):
    """1 where any raw message lies within max_gap seconds of the grid point, else 0."""
    if t is None or len(t) == 0:
        return np.zeros(len(grid))
    j = np.clip(np.searchsorted(t, grid), 1, len(t) - 1) if len(t) > 1 else np.zeros(len(grid), dtype=int)
    if len(t) > 1:
        nearest = np.minimum(np.abs(grid - t[j - 1]), np.abs(t[j] - grid))
    else:
        nearest = np.abs(grid - t[0])
    return (nearest <= max_gap).astype(float)


def align_quaternions(q):
    """Flip signs so consecutive quaternions sit on the same hemisphere (q and -q are the same rotation)."""
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def quat_to_rotm(q):
    """q = [w, x, y, z] -> 3x3 rotation matrix body->NED."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# ------------------------------------------------------------------ per-flight channel table
def build_channels(path, inspect=False):
    u, topics = load_ulog(path)
    t0 = u.start_timestamp
    duration = (u.last_timestamp - t0) / 1e6
    info = getattr(u, 'msg_info_dict', {}) or {}
    out = {'log': path.name, 'duration_s': duration,
           'ver_sw': str(info.get('ver_sw', ''))[:10], 'ver_sw_release': str(info.get('ver_sw_release', '')),
           'ver_hw': str(info.get('ver_hw', '')), 'sys_name': str(info.get('sys_name', ''))}

    # exact event times from the log
    ev = {'t_inject': np.nan, 't_clear': np.nan, 't_spoof': np.nan, 't_nofix': np.nan, 't_nofix_clear': np.nan}
    for m in getattr(u, 'logged_messages', []):
        txt = m.message if isinstance(m.message, str) else m.message.decode(errors='ignore')
        if 'Injected:' in txt and np.isnan(ev['t_inject']):
            ev['t_inject'] = (m.timestamp - t0) / 1e6
        elif 'Cleared:' in txt and np.isnan(ev['t_clear']):
            ev['t_clear'] = (m.timestamp - t0) / 1e6
    for (ts_, k, v) in getattr(u, 'changed_parameters', []):
        tsx = (ts_ - t0) / 1e6
        if k == 'SIM_GPS_SPF_EN' and v == 1 and np.isnan(ev['t_spoof']):
            ev['t_spoof'] = tsx
        if k == 'SIM_GPS_USED' and v < 4 and np.isnan(ev['t_nofix']):
            ev['t_nofix'] = tsx
        if k == 'SIM_GPS_USED' and v >= 4 and not np.isnan(ev['t_nofix']) and np.isnan(ev['t_nofix_clear']):
            ev['t_nofix_clear'] = tsx
    out['events'] = ev

    gps_name, gps = first_instance(topics, 'sensor_gps', 'vehicle_gps_position')
    if gps is None:
        raise RuntimeError(f'{path.name}: no GPS topic')
    # units by field identity: current PX4 publishes degrees, 2020-era PX4 publishes int32 1e7 degrees and mm
    if 'latitude_deg' in gps.data:
        lat, lon, alt = col(gps, 'latitude_deg'), col(gps, 'longitude_deg'), col(gps, 'altitude_msl_m')
    else:
        lat, lon, alt = col(gps, 'lat') / 1e7, col(gps, 'lon') / 1e7, col(gps, 'alt') / 1e3
    tg_all = tsec(gps, t0)
    fix = col(gps, 'fix_type')
    valid = np.ones(len(tg_all), dtype=bool) if fix is None else (fix >= 3)
    valid &= np.isfinite(lat) & np.isfinite(lon)
    if valid.sum() < 2:
        raise RuntimeError(f'{path.name}: fewer than 2 valid GPS fixes')
    tg = tg_all[valid]
    lat0, lon0 = lat[valid][0], lon[valid][0]
    gN = (lat[valid] - lat0) * 111320.0
    gE = (lon[valid] - lon0) * 111320.0 * math.cos(math.radians(lat0))
    gD = -(alt[valid] - alt[valid][0])
    gvN, gvE, gvD = col(gps, 'vel_n_m_s')[valid], col(gps, 'vel_e_m_s')[valid], col(gps, 'vel_d_m_s')[valid]

    # common grid over the whole recorded log
    grid = np.arange(0.0, max(duration, 1.0), 1.0 / HZ)
    ch = {'t': grid}
    for k, v in (('gN', gN), ('gE', gE), ('gD', gD), ('gvN', gvN), ('gvE', gvE), ('gvD', gvD)):
        ch[k] = interp_to(grid, tg, v, max_gap=0.6)
    ch['gps_msg'] = presence(grid, tg_all, 0.6)                       # any GPS message present
    ch['gps_valid'] = hold_to(grid, tg_all, valid.astype(float), 0.6)  # nearest message has a valid fix
    # receiver self-report channel (kept separate): continuous fields interpolated, categorical fields held
    for k, names in (('rx_eph', ('eph',)), ('rx_noise', ('noise_per_ms',)), ('rx_svar', ('s_variance_m_s',))):
        ch[k] = interp_to(grid, tg_all, col(gps, *names), max_gap=0.6)
    for k, names in (('rx_sats', ('satellites_used',)), ('rx_fix', ('fix_type',)), ('rx_jam', ('jamming_indicator',))):
        ch[k] = hold_to(grid, tg_all, col(gps, *names), 0.6)

    # barometric altitude (GNSS-independent pressure altitude)
    _, air = first_instance(topics, 'vehicle_air_data')
    if air is not None:
        ch['baroAlt'] = interp_to(grid, tsec(air, t0), col(air, 'baro_alt_meter'))
    else:
        _, baro = first_instance(topics, 'sensor_baro')
        p = col(baro, 'pressure') if baro is not None else None
        if p is not None:
            if np.nanmedian(p) < 2000:      # hPa/mbar in older logs
                p = p * 100.0
            ch['baroAlt'] = interp_to(grid, tsec(baro, t0), 44330.0 * (1.0 - (p / 101325.0) ** 0.1903))
        else:
            ch['baroAlt'] = np.full_like(grid, np.nan)

    # EKF local position and attitude (EKF-derived, GNSS-dependent; documented as such)
    _, lp = first_instance(topics, 'vehicle_local_position')
    if lp is not None:
        tl = tsec(lp, t0)
        for k, n in (('eN', 'x'), ('eE', 'y'), ('eD', 'z'), ('evN', 'vx'), ('evE', 'vy'), ('evD', 'vz')):
            ch[k] = interp_to(grid, tl, col(lp, n))
    _, att = first_instance(topics, 'vehicle_attitude')
    if att is not None:
        ta = tsec(att, t0)
        q = np.stack([col(att, 'q[0]'), col(att, 'q[1]'), col(att, 'q[2]'), col(att, 'q[3]')], 1)
        good = np.all(np.isfinite(q), 1) & (np.abs(np.linalg.norm(q, axis=1) - 1.0) < 0.05)
        q = align_quaternions(q[good]); ta = ta[good]
        qi = np.stack([interp_to(grid, ta, q[:, i]) for i in range(4)], 1)
        nrm = np.linalg.norm(qi, axis=1)
        qi = qi / np.where(nrm > 0, nrm, np.nan)[:, None]
        for i in range(4):
            ch[f'q{i}'] = qi[:, i]

    # raw IMU, block-averaged to the grid (a block with no samples stays NaN)
    _, imu = first_instance(topics, 'sensor_combined')
    if imu is not None:
        ti = tsec(imu, t0)
        acc = np.stack([col(imu, 'accelerometer_m_s2[0]'), col(imu, 'accelerometer_m_s2[1]'), col(imu, 'accelerometer_m_s2[2]')], 1)
        gyr = np.stack([col(imu, 'gyro_rad[0]'), col(imu, 'gyro_rad[1]'), col(imu, 'gyro_rad[2]')], 1)
        idx = np.clip(np.searchsorted(grid, ti) - 1, 0, len(grid) - 1)
        for j, k in enumerate(('ax', 'ay', 'az')):
            s = pd.Series(acc[:, j]).groupby(idx).mean()
            ch[k] = s.reindex(range(len(grid))).to_numpy()
        s = pd.Series(np.linalg.norm(gyr, axis=1)).groupby(idx).std()
        ch['gyro_std'] = s.reindex(range(len(grid))).to_numpy()

    # magnetometer norm (derived topic; stops publishing when the sensor is stuck)
    _, mag = first_instance(topics, 'vehicle_magnetometer', 'sensor_mag')
    if mag is not None:
        m = np.stack([col(mag, 'magnetometer_ga[0]', 'x'), col(mag, 'magnetometer_ga[1]', 'y'),
                      col(mag, 'magnetometer_ga[2]', 'z')], 1)
        ch['magNorm'] = interp_to(grid, tsec(mag, t0), np.linalg.norm(m, axis=1))

    # raw sensor topics (logged at low rate, keep publishing constant values when stuck)
    _, rmag = first_instance(topics, 'sensor_mag')
    if rmag is not None:
        m = np.stack([col(rmag, 'x'), col(rmag, 'y'), col(rmag, 'z')], 1)
        ch['magRaw'] = interp_to(grid, tsec(rmag, t0), np.linalg.norm(m, axis=1), max_gap=2.5)
    _, rbaro = first_instance(topics, 'sensor_baro')
    if rbaro is not None:
        pr = col(rbaro, 'pressure')
        if pr is not None and np.nanmedian(pr) < 2000:      # hPa in older logs -> Pa
            pr = pr * 100.0
        ch['baroRaw'] = interp_to(grid, tsec(rbaro, t0), pr, max_gap=2.5)

    # EKF innovations and test ratios
    for topic, prefix in (('estimator_innovations', 'inn_'), ('estimator_innovation_test_ratios', 'tr_')):
        _, e = first_instance(topics, topic)
        if e is None:
            continue
        te = tsec(e, t0)
        for k, n in (('gpsHpos0', 'gps_hpos[0]'), ('gpsHpos1', 'gps_hpos[1]'), ('gpsVpos', 'gps_vpos'),
                     ('gpsHvel0', 'gps_hvel[0]'), ('gpsHvel1', 'gps_hvel[1]'), ('gpsVvel', 'gps_vvel'),
                     ('baroVpos', 'baro_vpos'), ('heading', 'heading'),
                     ('mag0', 'mag_field[0]'), ('mag1', 'mag_field[1]'), ('mag2', 'mag_field[2]')):
            ch[prefix + k] = interp_to(grid, te, col(e, n))

    if inspect:
        print(f'\n[inspect] {path.name}: GPS topic = {gps_name}, valid fixes {int(valid.sum())}/{len(valid)}, '
              f'log duration {duration:.0f}s, fields = {sorted(gps.data.keys())[:40]}')
        for topic in ('vehicle_air_data', 'sensor_baro', 'vehicle_local_position', 'vehicle_attitude',
                      'sensor_combined', 'vehicle_magnetometer', 'sensor_mag', 'estimator_innovations',
                      'estimator_innovation_test_ratios'):
            n, d = first_instance(topics, topic)
            print(f'  {topic:34s}', 'present' if d is not None else 'MISSING',
                  '' if d is None else f'rows={len(d.data["timestamp"])}')
    out['ch'] = pd.DataFrame(ch)
    return out


# ------------------------------------------------------------------ window features
def window_features(df):
    t = df['t'].to_numpy()
    dt = 1.0 / HZ
    have = set(df.columns)
    feats = []
    warnings.simplefilter('ignore', category=RuntimeWarning)

    # rotated IMU acceleration in NED (gravity removed) where attitude and IMU are present
    aN = aE = None
    if {'q0', 'q1', 'q2', 'q3', 'ax', 'ay', 'az'} <= have:
        q = df[['q0', 'q1', 'q2', 'q3']].to_numpy()
        a = df[['ax', 'ay', 'az']].to_numpy()
        aN, aE = np.full(len(df), np.nan), np.full(len(df), np.nan)
        for i in range(len(df)):
            if np.all(np.isfinite(q[i])) and np.all(np.isfinite(a[i])):
                v = quat_to_rotm(q[i]) @ a[i]
                aN[i], aE[i] = v[0], v[1]
    integ = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

    n_windows = int(np.floor((t[-1] - t[0] - WIN_S) / HOP_S)) + 1 if t[-1] - t[0] >= WIN_S else 0
    for k in range(max(n_windows, 0)):
        t_s = t[0] + k * HOP_S
        m = (t >= t_s) & (t < t_s + WIN_S)
        if m.sum() < int(0.8 * WIN_S * HZ):
            continue
        w = df.loc[m]
        f = {'t_start': float(t_s), 't_end': float(t_s + WIN_S)}
        gN, gE, gD = w['gN'].to_numpy(), w['gE'].to_numpy(), w['gD'].to_numpy()
        gvN, gvE = w['gvN'].to_numpy(), w['gvE'].to_numpy()
        f['gps_gap_frac'] = float(np.isnan(gN).mean())                                 # no valid position
        f['gps_nofix_frac'] = float(((w['gps_msg'] > 0) & (w['gps_valid'] < 0.5)).mean())  # messages, no fix
        f['gps_silent_frac'] = float((w['gps_msg'] < 0.5).mean())                      # no messages at all
        gps_ok = np.isfinite(gN)
        if gps_ok.sum() >= 3:
            i0 = int(np.argmax(gps_ok)); vi = np.where(gps_ok)[0]
            # 1. GPS position-derivative vs GPS-reported velocity (incoherent spoof, jump)
            dN, dE = np.gradient(gN, dt), np.gradient(gE, dt)
            f['posvel_rms'] = float(np.sqrt(np.nanmean((dN - gvN) ** 2 + (dE - gvE) ** 2)))
            # 2. step speed implied by consecutive fixes vs reported speed (jump detector)
            step = np.hypot(np.diff(gN), np.diff(gE)) / dt
            f['step_speed_max'] = float(np.nanmax(step))
            f['step_minus_vel_max'] = float(np.nanmax(step - np.hypot(gvN, gvE)[1:]))
            f['gps_speed_mean'] = float(np.nanmean(np.hypot(gvN, gvE)))
            # 3. GPS vertical increment vs barometric increment (GNSS-independent altitude)
            if 'baroAlt' in have:
                b = w['baroAlt'].to_numpy()
                f['gps_baro_dz_rms'] = float(np.sqrt(np.nanmean(((-gD - (-gD[i0])) - (b - b[i0])) ** 2)))
            # 4. GPS velocity change vs integrated IMU acceleration, as vectors over the same interval;
            #    a missing acceleration sample inside the interval leaves the feature undefined (attitude is
            #    EKF-derived: documented dependence)
            if aN is not None:
                an, ae = aN[m], aE[m]
                lo, hi = vi[0], vi[-1]
                seg_n, seg_e = an[lo:hi + 1], ae[lo:hi + 1]
                if hi > lo and np.all(np.isfinite(seg_n)) and np.all(np.isfinite(seg_e)):
                    dv_gps = np.array([gvN[hi] - gvN[lo], gvE[hi] - gvE[lo]])
                    dv_imu = np.array([integ(seg_n, dx=dt), integ(seg_e, dx=dt)])
                    f['dv_gps_imu_vecdiff'] = float(np.linalg.norm(dv_gps - dv_imu))
                f['acc_horiz_rms'] = float(np.sqrt(np.nanmean(an ** 2 + ae ** 2)))
            # 5. course over ground vs EKF heading while moving
            if {'q0', 'q1', 'q2', 'q3'} <= have:
                q = w[['q0', 'q1', 'q2', 'q3']].to_numpy()
                yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
                spd = np.hypot(gvN, gvE)
                cog = np.arctan2(gvE, gvN)
                d = np.angle(np.exp(1j * (cog - yaw)))
                moving = np.isfinite(spd) & (spd > 1.0) & np.isfinite(d)
                f['cog_yaw_absdiff_deg'] = float(np.degrees(np.mean(np.abs(d[moving])))) if moving.sum() > 3 else np.nan
            # 6. GPS vs EKF local-position increments (EKF-dependent; large when EKF rejects GPS)
            if {'eN', 'eE'} <= have:
                eN, eE = w['eN'].to_numpy(), w['eE'].to_numpy()
                f['gps_ekf_dpos_rms'] = float(np.sqrt(np.nanmean(((gN - gN[i0]) - (eN - eN[i0])) ** 2 + ((gE - gE[i0]) - (eE - eE[i0])) ** 2)))
        else:
            if aN is not None:
                an, ae = aN[m], aE[m]
                f['acc_horiz_rms'] = float(np.sqrt(np.nanmean(an ** 2 + ae ** 2)))
        # 7. sensor-health statistics (baro/mag faults, vibration)
        if 'baroAlt' in have:
            f['baro_std'] = float(np.nanstd(w['baroAlt']))
            f['baro_gap_frac'] = float(np.isnan(w['baroAlt']).mean())
        if 'magNorm' in have:
            f['mag_norm_std'] = float(np.nanstd(w['magNorm']))
            f['mag_norm_mean'] = float(np.nanmean(w['magNorm']))
            f['mag_gap_frac'] = float(np.isnan(w['magNorm']).mean())
        if 'magRaw' in have:
            f['mag_raw_std'] = float(np.nanstd(w['magRaw']))
        if 'baroRaw' in have:
            f['baro_raw_std'] = float(np.nanstd(w['baroRaw']))
        if 'gyro_std' in have:
            f['gyro_std_mean'] = float(np.nanmean(w['gyro_std']))
        # 8. EKF innovation statistics
        for c in [c for c in df.columns if c.startswith('inn_') or c.startswith('tr_')]:
            x = w[c].to_numpy()
            f[c + '_absmean'] = float(np.nanmean(np.abs(x)))
            f[c + '_absmax'] = float(np.nanmax(np.abs(x)))
        # receiver self-report channel (separate family of columns)
        for c in ('rx_eph', 'rx_sats', 'rx_fix', 'rx_jam', 'rx_noise', 'rx_svar'):
            f[c + '_mean'] = float(np.nanmean(w[c]))
        feats.append(f)
    return pd.DataFrame(feats)


# ------------------------------------------------------------------ labels
def window_label(fam, t_start, t_end, onset, restore):
    if fam == 'unlabelled':
        return 'unlabelled'
    if fam == 'nominal' or onset is None or np.isnan(onset):
        return fam
    if t_end < onset:
        return 'nominal'
    if restore is not None and not np.isnan(restore) and t_start > restore:
        return 'nominal'
    return fam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sih', action='append', default=[], help='folder with manifest.jsonl and *.ulg (repeatable)')
    ap.add_argument('--whelan_zip', default=None, help='UAVAttackData.zip; the three live logs are extracted')
    ap.add_argument('--real_dir', action='append', default=[], help='folder of real PX4 .ulg files with no ground truth (repeatable)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    jobs = []   # (path, meta)
    for folder in args.sih:
        folder = pathlib.Path(folder)
        for line in (folder / 'manifest.jsonl').read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get('ok') or not r.get('ulog'):
                continue
            v = r.get('verify', {})
            onset = v.get('onset_sim_s')
            restore = None
            if r.get('failure') and r['failure'].get('restore_after_s') and onset is not None:
                restore = onset + r['failure']['restore_after_s']
            if r.get('rxmode') and r['rxmode'].get('restore_after_s') and onset is not None:
                restore = onset + r['rxmode']['restore_after_s']
            nz = r.get('nuisance', {})
            jobs.append((folder / r['ulog'], {'flight_id': r['flight_id'], 'source': 'sih', 'run': folder.name,
                                              'family': r['family'], 'subtype': r['subtype'],
                                              'onset_s': onset, 'restore_s': restore, 'home_lat': r['home'][0],
                                              'noise_k': nz.get('SIM_GPS_NOISE_K'), 'noise_t': nz.get('SIM_GPS_NOISE_T'),
                                              'noise_p': nz.get('SIM_GPS_NOISE_P')}))
    if args.whelan_zip:
        wdir = out / 'whelan_live'; wdir.mkdir(exist_ok=True)
        fam_map = {'benign': 'nominal', 'jamming': 'gps_degrade', 'spoofing': 'spoof'}
        with zipfile.ZipFile(args.whelan_zip) as z:
            for name in z.namelist():
                if name.lower().endswith('.ulg') and 'live gps' in name.lower():
                    target = wdir / pathlib.Path(name).name
                    if not target.exists():
                        target.write_bytes(z.read(name))
                    fname = pathlib.Path(name).name.lower()
                    fam = next((v for k, v in fam_map.items() if k in fname), 'unknown')
                    jobs.append((target, {'flight_id': 'whelan_' + target.stem, 'source': 'whelan', 'run': 'whelan',
                                          'family': fam, 'subtype': 'live_' + fam, 'onset_s': np.nan, 'restore_s': np.nan,
                                          'home_lat': np.nan, 'noise_k': np.nan, 'noise_t': np.nan, 'noise_p': np.nan}))

    for folder in args.real_dir:
        folder = pathlib.Path(folder)
        for path in sorted(folder.glob('*.ulg')):
            jobs.append((path, {'flight_id': 'real_' + path.stem, 'source': 'real', 'run': folder.name,
                                'family': 'unlabelled', 'subtype': 'unlabelled', 'onset_s': np.nan, 'restore_s': np.nan,
                                'home_lat': np.nan, 'noise_k': np.nan, 'noise_t': np.nan, 'noise_p': np.nan}))

    rows, flights = [], []
    seen_sources = set()
    for path, meta in jobs:
        try:
            chd = build_channels(path, inspect=(meta['source'] not in seen_sources))
        except Exception as e:
            print(f'SKIP {path.name}: {type(e).__name__}: {e}')
            continue
        seen_sources.add(meta['source'])
        wf = window_features(chd['ch'])
        ev = chd['events']
        onset, restore = meta['onset_s'], meta['restore_s']
        if meta['source'] == 'sih':
            if meta['family'] == 'spoof' and not np.isnan(ev['t_spoof']):
                onset = ev['t_spoof']
            elif meta['family'] in ('gps_degrade', 'sensor_fault'):
                if not np.isnan(ev['t_nofix']):
                    onset = ev['t_nofix']
                    restore = ev['t_nofix_clear'] if not np.isnan(ev['t_nofix_clear']) else restore
                else:
                    if not np.isnan(ev['t_inject']):
                        onset = ev['t_inject']
                    if not np.isnan(ev['t_clear']):
                        restore = ev['t_clear']
        meta = {**meta, 'onset_s': onset, 'restore_s': restore}
        for k, v in meta.items():
            wf[k] = v
        wf['window_label'] = [window_label(meta['family'], a, b, onset, restore)
                              for a, b in zip(wf['t_start'], wf['t_end'])]
        rows.append(wf)
        flights.append({**meta, 'log': path.name, 'duration_s': round(chd['duration_s'], 1), 'n_windows': len(wf),
                        'ver_sw': chd.get('ver_sw', ''), 'ver_sw_release': chd.get('ver_sw_release', ''),
                        'ver_hw': chd.get('ver_hw', ''), 'sys_name': chd.get('sys_name', '')})
        print(f'{meta["flight_id"]:40s} {meta["family"]:13s} windows={len(wf):4d} duration={chd["duration_s"]:.0f}s')

    if not rows:
        print('no flights processed (every log was skipped or no inputs were given)'); return
    W = pd.concat(rows, ignore_index=True)
    F = pd.DataFrame(flights)
    W.attrs['feature_version'] = FEATURE_VERSION
    W.to_csv(out / 'windows.csv', index=False)
    F.to_csv(out / 'flights.csv', index=False)
    (out / 'feature_version.txt').write_text(FEATURE_VERSION + '\n')
    print(f'\nwrote {len(W)} windows from {len(F)} flights ->', out, '| feature version', FEATURE_VERSION)

    key = ['gps_gap_frac', 'gps_nofix_frac', 'gps_silent_frac', 'posvel_rms', 'step_minus_vel_max', 'gps_baro_dz_rms',
           'dv_gps_imu_vecdiff', 'gps_ekf_dpos_rms', 'baro_gap_frac', 'baro_raw_std', 'mag_gap_frac', 'mag_raw_std',
           'tr_gpsHpos0_absmax', 'tr_baroVpos_absmax', 'tr_heading_absmax', 'rx_jam_mean']
    key = [k for k in key if k in W.columns]
    pd.set_option('display.width', 240); pd.set_option('display.max_columns', 40)
    print('\nMedian of key features by (source, window_label):')
    print(W.groupby(['source', 'window_label'])[key].median().round(3).to_string())
    print('\nWindows per (source, family, window_label):')
    print(W.groupby(['source', 'family', 'window_label']).size().to_string())


if __name__ == '__main__':
    main()
