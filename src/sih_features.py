#!/usr/bin/env python3
"""
sih_features.py  --  ULog -> physics-consistency window features (SIH-generated and Whelan live logs).

Design rules (from the plan):
  * Simulator truth topics (*_groundtruth) are never read here. Truth lives only in the manifest / verify step.
  * Receiver self-report fields (fix_type, satellites_used, eph, jamming_indicator, noise_per_ms, s_variance)
    are extracted into a SEPARATE column family (rx_*) so the core detector can exclude them.
  * Every feature row carries flight_id so all splits can be flight-grouped.
  * Window label is onset-aware: before onset a window in an attacked flight is 'nominal'; for gps_degrade and
    sensor_fault, windows after the restore time are 'nominal' again; spoof persists to the end of the flight.

Usage:
  python sih_features.py --sih /content/drive/MyDrive/datasets/sih_pilot4 \
                         --whelan_zip /content/drive/MyDrive/datasets/uav_attack/UAVAttackData.zip \
                         --out /content/drive/MyDrive/datasets/features_pilot4
Outputs: <out>/windows.csv, <out>/flights.csv, and a printed inspection + per-label feature summary.
"""
import argparse
import json
import math
import pathlib
import sys
import zipfile

import numpy as np
import pandas as pd

WIN_S, HOP_S, HZ = 5.0, 2.5, 5.0   # window length, hop, common sample rate

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
    """Interpolate x(t) onto grid, but leave NaN wherever the nearest raw sample is farther than max_gap seconds
    (so sensor outages are preserved instead of bridged)."""
    if x is None or len(t) < 2:
        return np.full_like(grid, np.nan)
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return np.full_like(grid, np.nan)
    tt, xx = t[ok], x[ok]
    y = np.interp(grid, tt, xx, left=np.nan, right=np.nan)
    j = np.clip(np.searchsorted(tt, grid), 1, len(tt) - 1)
    nearest = np.minimum(np.abs(grid - tt[j - 1]), np.abs(tt[j] - grid))
    y[nearest > max_gap] = np.nan
    return y


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
    out = {'log': path.name, 'duration_s': (u.last_timestamp - t0) / 1e6}

    # exact event times from the log: failure-injection messages and the spoof-enable parameter change
    ev = {'t_inject': np.nan, 't_clear': np.nan, 't_spoof': np.nan}
    for m in getattr(u, 'logged_messages', []):
        txt = m.message if isinstance(m.message, str) else m.message.decode(errors='ignore')
        if 'Injected:' in txt and np.isnan(ev['t_inject']):
            ev['t_inject'] = (m.timestamp - t0) / 1e6
        elif 'Cleared:' in txt and np.isnan(ev['t_clear']):
            ev['t_clear'] = (m.timestamp - t0) / 1e6
    for (ts_, k, v) in getattr(u, 'changed_parameters', []):
        if k == 'SIM_GPS_SPF_EN' and v == 1 and np.isnan(ev['t_spoof']):
            ev['t_spoof'] = (ts_ - t0) / 1e6
    out['events'] = ev

    gps_name, gps = first_instance(topics, 'sensor_gps', 'vehicle_gps_position')
    if gps is None:
        raise RuntimeError(f'{path.name}: no GPS topic')
    lat = col(gps, 'latitude_deg', 'lat')
    lon = col(gps, 'longitude_deg', 'lon')
    alt = col(gps, 'altitude_msl_m', 'alt')
    if np.nanmax(np.abs(lat)) > 1000:          # int32 1e7 degrees, alt in mm (older PX4)
        lat, lon, alt = lat / 1e7, lon / 1e7, alt / 1e3
    tg = tsec(gps, t0)
    lat0, lon0 = lat[np.isfinite(lat)][0], lon[np.isfinite(lon)][0]
    gN = (lat - lat0) * 111320.0
    gE = (lon - lon0) * 111320.0 * math.cos(math.radians(lat0))
    gD = -(alt - alt[np.isfinite(alt)][0])
    gvN, gvE, gvD = col(gps, 'vel_n_m_s'), col(gps, 'vel_e_m_s'), col(gps, 'vel_d_m_s')

    # common grid across the GPS-valid span
    grid = np.arange(tg[0], tg[-1], 1.0 / HZ)
    ch = {'t': grid}
    for k, v in (('gN', gN), ('gE', gE), ('gD', gD), ('gvN', gvN), ('gvE', gvE), ('gvD', gvD)):
        ch[k] = interp_to(grid, tg, v, max_gap=0.6)
    # receiver self-report channel (kept separate)
    for k, names in (('rx_eph', ('eph',)), ('rx_sats', ('satellites_used',)), ('rx_fix', ('fix_type',)),
                     ('rx_jam', ('jamming_indicator',)), ('rx_noise', ('noise_per_ms',)),
                     ('rx_svar', ('s_variance_m_s',))):
        ch[k] = interp_to(grid, tg, col(gps, *names), max_gap=0.6)

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
        for i in range(4):
            ch[f'q{i}'] = interp_to(grid, ta, q[:, i])

    # raw IMU, block-averaged to the grid
    _, imu = first_instance(topics, 'sensor_combined')
    if imu is not None:
        ti = tsec(imu, t0)
        acc = np.stack([col(imu, 'accelerometer_m_s2[0]'), col(imu, 'accelerometer_m_s2[1]'), col(imu, 'accelerometer_m_s2[2]')], 1)
        gyr = np.stack([col(imu, 'gyro_rad[0]'), col(imu, 'gyro_rad[1]'), col(imu, 'gyro_rad[2]')], 1)
        idx = np.clip(np.searchsorted(grid, ti) - 1, 0, len(grid) - 1)
        for j, k in enumerate(('ax', 'ay', 'az')):
            s = pd.Series(acc[:, j]).groupby(idx).mean()
            ch[k] = s.reindex(range(len(grid))).to_numpy()
        gnorm = np.linalg.norm(gyr, axis=1)
        s = pd.Series(gnorm).groupby(idx).std()
        ch['gyro_std'] = s.reindex(range(len(grid))).to_numpy()

    # magnetometer norm (derived topic; stops publishing when the sensor is stuck)
    _, mag = first_instance(topics, 'vehicle_magnetometer', 'sensor_mag')
    if mag is not None:
        m = np.stack([col(mag, 'magnetometer_ga[0]', 'x'), col(mag, 'magnetometer_ga[1]', 'y'),
                      col(mag, 'magnetometer_ga[2]', 'z')], 1)
        ch['magNorm'] = interp_to(grid, tsec(mag, t0), np.linalg.norm(m, axis=1))

    # raw sensor topics (logged at low rate, keep publishing constant values when stuck): nearest-sample hold
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
        print(f'\n[inspect] {path.name}: GPS topic = {gps_name}, fields = {sorted(gps.data.keys())[:40]}')
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
    feats = []
    n_win = int(WIN_S * HZ)
    hop = int(HOP_S * HZ)
    have = set(df.columns)

    # precompute rotated IMU acceleration in NED if attitude + IMU are present
    aN = aE = aD = None
    if {'q0', 'q1', 'q2', 'q3', 'ax', 'ay', 'az'} <= have:
        q = df[['q0', 'q1', 'q2', 'q3']].to_numpy()
        a = df[['ax', 'ay', 'az']].to_numpy()
        aN, aE, aD = np.full(len(df), np.nan), np.full(len(df), np.nan), np.full(len(df), np.nan)
        for i in range(len(df)):
            if np.all(np.isfinite(q[i])) and np.all(np.isfinite(a[i])):
                v = quat_to_rotm(q[i]) @ a[i]
                aN[i], aE[i], aD[i] = v[0], v[1], v[2] + 9.80665   # remove gravity (NED, D positive down)

    import warnings
    for s in range(0, len(df) - n_win + 1, hop):
        w = df.iloc[s:s + n_win]
        f = {'t_start': float(t[s]), 't_end': float(t[s + n_win - 1])}
        gN, gE, gD = w['gN'].to_numpy(), w['gE'].to_numpy(), w['gD'].to_numpy()
        gvN, gvE, gvD = w['gvN'].to_numpy(), w['gvE'].to_numpy(), w['gvD'].to_numpy()
        f['gps_gap_frac'] = float(np.isnan(gN).mean())      # GPS outage fraction: a feature, not a reason to drop
        warnings.simplefilter('ignore', category=RuntimeWarning)
        gps_ok = np.isfinite(gN)
        if gps_ok.sum() >= 3:
            # use the first valid fix in the window as the reference for increment features
            i0 = int(np.argmax(gps_ok))
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
            # 4. GPS velocity change vs integrated IMU acceleration (attitude is EKF-derived: documented dependence)
            if aN is not None:
                an, ae = aN[s:s + n_win], aE[s:s + n_win]
                vi = np.where(gps_ok)[0]
                dv_gps = np.hypot(gvN[vi[-1]] - gvN[vi[0]], gvE[vi[-1]] - gvE[vi[0]])
                dv_imu = np.hypot(np.nansum(an[vi[0]:vi[-1] + 1]) * dt, np.nansum(ae[vi[0]:vi[-1] + 1]) * dt)
                f['dv_gps_minus_imu'] = float(dv_gps - dv_imu)
                f['acc_horiz_rms'] = float(np.sqrt(np.nanmean(an ** 2 + ae ** 2)))
            # 5. course over ground vs EKF heading while moving
            if {'q0', 'q1', 'q2', 'q3'} <= have:
                q = w[['q0', 'q1', 'q2', 'q3']].to_numpy()
                yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
                spd = np.hypot(gvN, gvE)
                cog = np.arctan2(gvE, gvN)
                d = np.angle(np.exp(1j * (cog - yaw)))
                moving = np.isfinite(spd) & (spd > 1.0)
                f['cog_yaw_absdiff_deg'] = float(np.degrees(np.nanmean(np.abs(d[moving])))) if moving.sum() > 3 else np.nan
            # 6. GPS vs EKF local-position increments (EKF-dependent; large when EKF rejects GPS)
            if {'eN', 'eE'} <= have:
                eN, eE = w['eN'].to_numpy(), w['eE'].to_numpy()
                f['gps_ekf_dpos_rms'] = float(np.sqrt(np.nanmean(((gN - gN[i0]) - (eN - eN[i0])) ** 2 + ((gE - gE[i0]) - (eE - eE[i0])) ** 2)))
        else:
            # (near-)total GPS outage: physics features undefined; gap fraction carries the signal
            if aN is not None:
                an, ae = aN[s:s + n_win], aE[s:s + n_win]
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
    if fam == 'nominal' or onset is None or np.isnan(onset):
        return fam if fam != 'nominal' else 'nominal'
    if t_end < onset:
        return 'nominal'
    if restore is not None and not np.isnan(restore) and t_start > restore:
        return 'nominal'
    return fam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sih', action='append', default=[], help='folder with manifest.jsonl and *.ulg (repeatable)')
    ap.add_argument('--whelan_zip', default=None, help='UAVAttackData.zip; the three live logs are extracted')
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
            jobs.append((folder / r['ulog'], {'flight_id': r['flight_id'], 'source': 'sih', 'family': r['family'],
                                              'subtype': r['subtype'], 'onset_s': onset, 'restore_s': restore,
                                              'home_lat': r['home'][0], 'noise_k': r.get('nuisance', {}).get('SIM_GPS_NOISE_K'),
                                              'noise_t': r.get('nuisance', {}).get('SIM_GPS_NOISE_T'),
                                              'noise_p': r.get('nuisance', {}).get('SIM_GPS_NOISE_P')}))
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
                    jobs.append((target, {'flight_id': 'whelan_' + target.stem, 'source': 'whelan', 'family': fam,
                                          'subtype': 'live_' + fam, 'onset_s': np.nan, 'restore_s': np.nan,
                                          'home_lat': np.nan, 'noise_k': np.nan, 'noise_t': np.nan, 'noise_p': np.nan}))

    rows, flights = [], []
    for i, (path, meta) in enumerate(jobs):
        try:
            chd = build_channels(path, inspect=(i == 0 or meta['source'] == 'whelan' and not any(f['source'] == 'whelan' for f in flights)))
        except Exception as e:
            print(f'SKIP {path.name}: {type(e).__name__}: {e}')
            continue
        wf = window_features(chd['ch'])
        ev = chd['events']
        onset, restore = meta['onset_s'], meta['restore_s']
        if meta['source'] == 'sih':
            if meta['family'] == 'spoof' and not np.isnan(ev['t_spoof']):
                onset = ev['t_spoof']
            elif meta['family'] in ('gps_degrade', 'sensor_fault'):
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
        flights.append({**meta, 'log': path.name, 'duration_s': round(chd['duration_s'], 1), 'n_windows': len(wf)})
        print(f'{meta["flight_id"]:40s} {meta["family"]:13s} windows={len(wf):4d} duration={chd["duration_s"]:.0f}s')

    W = pd.concat(rows, ignore_index=True)
    F = pd.DataFrame(flights)
    W.to_csv(out / 'windows.csv', index=False)
    F.to_csv(out / 'flights.csv', index=False)
    print(f'\nwrote {len(W)} windows from {len(F)} flights ->', out)

    key = ['gps_gap_frac', 'posvel_rms', 'step_minus_vel_max', 'gps_baro_dz_rms', 'dv_gps_minus_imu',
           'gps_ekf_dpos_rms', 'baro_gap_frac', 'baro_raw_std', 'mag_gap_frac', 'mag_raw_std',
           'tr_gpsHpos0_absmax', 'tr_gpsHvel0_absmax', 'tr_baroVpos_absmax', 'tr_heading_absmax', 'rx_jam_mean']
    key = [k for k in key if k in W.columns]
    pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)
    print('\nMedian of key features by (source, window_label):')
    print(W.groupby(['source', 'window_label'])[key].median().round(3).to_string())
    print('\nWindows per (source, family, window_label):')
    print(W.groupby(['source', 'family', 'window_label']).size().to_string())


if __name__ == '__main__':
    main()
