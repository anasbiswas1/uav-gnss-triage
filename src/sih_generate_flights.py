#!/usr/bin/env python3
"""
sih_generate_flights.py  --  generate independent, labelled PX4 SIH flights for four families.

Families (one label per flight, injected mid-mission at a random onset):
  nominal       no injection
  spoof         GNSS manipulation via the SensorGpsSim patch (subtypes: jump / drift_incoherent / drift_coherent)
  gps_degrade   receiver degradation via PX4 failure injection (subtypes: off_then_ok / stuck)
  sensor_fault  non-GNSS sensor fault via PX4 failure injection (subtypes: baro_stuck / mag_stuck)

Each flight: fresh rootfs (no parameter carry-over), random home, random mission, random onset.
Ground truth for onset is the parameter-change timestamp inside the ULog (SIM_GPS_SPF_EN or SYS_FAILURE_EN).
Simulator truth topics (vehicle_*_position_groundtruth) stay inside the ULog; the analysis parser must
exclude them from features. Outputs: <out>/<flight_id>.ulg, <out>/<flight_id>.json, <out>/manifest.jsonl

Usage (Colab):
  python sih_generate_flights.py --out /content/drive/MyDrive/datasets/sih_flights --n_per_family 1 --speed 4
Resume-safe: existing flight_ids in manifest.jsonl are skipped.
"""
import argparse
import asyncio
import hashlib
import inspect
import json
import math
import os
import pathlib
import random
import shutil
import subprocess
import sys
import time

FAMILIES = ['nominal', 'spoof', 'gps_degrade', 'sensor_fault']
HOMES = [  # lat, lon, alt_m
    (25.3463, 55.4209, 5.0),    # Sharjah
    (24.4539, 54.3773, 5.0),    # Abu Dhabi
    (24.2075, 55.7447, 260.0),  # Al Ain
    (25.1288, 56.3265, 10.0),   # Fujairah
    (25.7895, 55.9432, 5.0),    # Ras Al Khaimah
    (50.7962, -1.0803, 20.0),   # Portsmouth
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def offset_latlon(lat, lon, dn, de):
    dlat = dn / 6371000.0 * 180.0 / math.pi
    dlon = de / 6371000.0 * 180.0 / math.pi / math.cos(math.radians(lat))
    return lat + dlat, lon + dlon


def make_mission_item(lat, lon, alt, speed):
    from mavsdk.mission import MissionItem
    params = inspect.signature(MissionItem.__init__).parameters
    kw = dict(latitude_deg=lat, longitude_deg=lon, relative_altitude_m=alt, speed_m_s=speed,
              is_fly_through=True, gimbal_pitch_deg=float('nan'), gimbal_yaw_deg=float('nan'),
              camera_action=MissionItem.CameraAction.NONE, loiter_time_s=float('nan'),
              camera_photo_interval_s=float('nan'), acceptance_radius_m=float('nan'),
              yaw_deg=float('nan'), camera_photo_distance_m=float('nan'))
    if 'vehicle_action' in params:
        kw['vehicle_action'] = MissionItem.VehicleAction.NONE
    return MissionItem(**{k: v for k, v in kw.items() if k in params})


def draw_config(rng, family):
    """Sample the per-flight injection configuration. Everything here is recorded in the manifest."""
    cfg = {'family': family, 'subtype': None, 'onset_s': None, 'params': {}, 'failure': None,
           # nuisance: receiver noise model randomised for every family. SIM_GPS_NOISE_T > 0 selects the
           # realism model: slow bias with correlation time T (s) and std P (m), white jitter J (m);
           # velocity noise about 0.04 to 0.10 m/s.
           'nuisance': {'SIM_GPS_NOISE_T': round(rng.uniform(20.0, 120.0), 1),
                        'SIM_GPS_NOISE_P': round(rng.uniform(0.8, 2.5), 2),
                        'SIM_GPS_NOISE_J': round(rng.uniform(0.02, 0.08), 3),
                        'SIM_GPS_NOISEV_K': round(rng.uniform(1.5, 4.0), 1),
                        'SIM_GPS_NOISE_K': 1.0}}
    if family == 'nominal':
        cfg['subtype'] = 'none'
        return cfg
    cfg['onset_s'] = round(rng.uniform(20.0, 70.0), 1)          # sim seconds after mission start
    if family == 'spoof':
        heading = rng.uniform(0, 2 * math.pi)
        sub = rng.choice(['jump', 'drift_incoherent', 'drift_coherent'])
        cfg['subtype'] = sub
        if sub == 'jump':
            jump = rng.uniform(20.0, 150.0)
            cfg['params'] = {'SIM_GPS_SPF_JN': jump * math.cos(heading), 'SIM_GPS_SPF_JE': jump * math.sin(heading),
                             'SIM_GPS_SPF_VN': 0.0, 'SIM_GPS_SPF_VE': 0.0, 'SIM_GPS_SPF_VD': 0.0,
                             'SIM_GPS_SPF_MAX': 0.0, 'SIM_GPS_SPF_VEL': 0}
        else:
            rate = rng.uniform(0.5, 5.0)
            cfg['params'] = {'SIM_GPS_SPF_JN': 0.0, 'SIM_GPS_SPF_JE': 0.0,
                             'SIM_GPS_SPF_VN': rate * math.cos(heading), 'SIM_GPS_SPF_VE': rate * math.sin(heading),
                             'SIM_GPS_SPF_VD': rng.choice([0.0, rng.uniform(-0.5, 0.5)]),
                             'SIM_GPS_SPF_MAX': rng.uniform(100.0, 400.0),
                             'SIM_GPS_SPF_VEL': 1 if sub == 'drift_coherent' else 0}
    elif family == 'gps_degrade':
        sub = rng.choice(['off_then_ok', 'stuck_then_ok'])
        cfg['subtype'] = sub
        cfg['failure'] = {'unit': 'SENSOR_GPS', 'type': 'OFF' if sub == 'off_then_ok' else 'STUCK',
                          'restore_after_s': round(rng.uniform(15.0, 60.0), 1)}
    elif family == 'sensor_fault':
        sub = rng.choice(['baro_stuck', 'mag_stuck'])
        cfg['subtype'] = sub
        cfg['failure'] = {'unit': 'SENSOR_BARO' if sub == 'baro_stuck' else 'SENSOR_MAG', 'type': 'STUCK',
                          'restore_after_s': round(rng.uniform(15.0, 60.0), 1)}
    return cfg


def draw_mission(rng, home):
    lat0, lon0, _ = home
    n = rng.randint(3, 5)
    alt = rng.uniform(15.0, 60.0)
    speed = rng.uniform(3.0, 10.0)
    items = []
    for _ in range(n):
        dn, de = rng.uniform(-150, 150), rng.uniform(-150, 150)
        lat, lon = offset_latlon(lat0, lon0, dn, de)
        items.append((lat, lon, alt + rng.uniform(-5, 5), speed))
    return items


async def fly_one(px4_bin, px4_etc, rootfs, home, speed_factor, cfg, mission, log_txt, server_port):
    from mavsdk import System
    from mavsdk.mission import MissionPlan
    from mavsdk.failure import FailureUnit, FailureType

    # fresh rootfs so parameters never carry over between flights
    if rootfs.exists():
        shutil.rmtree(rootfs)
    rootfs.mkdir(parents=True)

    env = dict(os.environ, PX4_SIM_MODEL='sihsim_quadx', PX4_SIMULATOR='sihsim',
               PX4_SIM_SPEED_FACTOR=str(speed_factor),
               PX4_HOME_LAT=str(home[0]), PX4_HOME_LON=str(home[1]), PX4_HOME_ALT=str(home[2]))
    subprocess.run(['pkill', '-f', 'bin/px4'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(['pkill', '-f', 'mavsdk_server'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    await asyncio.sleep(1.0)
    out = open(log_txt, 'w')
    proc = subprocess.Popen(['stdbuf', '-oL', str(px4_bin), str(px4_etc), '-s', 'etc/init.d-posix/rcS', '-d'],
                            cwd=rootfs, env=env, stdout=out, stderr=subprocess.STDOUT)
    result = {'ok': False, 'notes': []}
    sim = lambda wall: wall / speed_factor   # sim seconds -> wall seconds

    try:
        d = System(port=server_port)
        await d.connect(system_address='udpin://0.0.0.0:14540')

        async def wait_conn():
            async for s in d.core.connection_state():
                if s.is_connected:
                    return
        await asyncio.wait_for(wait_conn(), 120)

        async def wait_health():
            async for h in d.telemetry.health():
                if h.is_global_position_ok and h.is_home_position_ok and h.is_armable:
                    return
        await asyncio.wait_for(wait_health(), 120)

        # pre-flight parameters (injector armed but disabled; failure injection gate closed)
        for k, v in cfg['nuisance'].items():
            await d.param.set_param_float(k, float(v))
        for k, v in cfg['params'].items():
            if isinstance(v, int):
                await d.param.set_param_int(k, v)
            else:
                await d.param.set_param_float(k, float(v))
        await d.param.set_param_int('SIM_GPS_SPF_EN', 0)
        await d.param.set_param_int('SYS_FAILURE_EN', 0)

        items = [make_mission_item(*m) for m in mission]
        await d.mission.set_return_to_launch_after_mission(True)
        await d.mission.upload_mission(MissionPlan(items))
        for attempt in range(6):
            try:
                await d.action.arm()
                if attempt:
                    result['notes'].append(f'armed on attempt {attempt + 1}')
                break
            except Exception as e:
                if attempt == 5:
                    raise
                await asyncio.sleep(1.5)
        await d.mission.start_mission()
        t_start = time.time()

        async def wait_airborne():
            async for a in d.telemetry.in_air():
                if a:
                    return
        await asyncio.wait_for(wait_airborne(), sim(60))

        async def inject_retry(unit, ftype, label):
            for attempt in range(3):
                try:
                    await d.failure.inject(unit, ftype, 0)
                    if attempt:
                        result['notes'].append(f'{label} ok on attempt {attempt + 1}')
                    return True
                except Exception as e:
                    last = f'{type(e).__name__}: {e}'
                    await asyncio.sleep(0.5)
            result['notes'].append(f'{label} failed after 3 attempts: {last}')
            return False

        # injection at onset (sim seconds after mission start)
        if cfg['family'] != 'nominal':
            await asyncio.sleep(max(0.0, sim(cfg['onset_s']) - (time.time() - t_start)))
            if cfg['family'] == 'spoof':
                await d.param.set_param_int('SIM_GPS_SPF_EN', 1)     # onset marker in ULog
            else:
                await d.param.set_param_int('SYS_FAILURE_EN', 1)     # onset marker in ULog
                unit = getattr(FailureUnit, cfg['failure']['unit'])
                ftype = getattr(FailureType, cfg['failure']['type'])
                await inject_retry(unit, ftype, 'failure inject')
                if cfg['failure']['restore_after_s']:
                    await asyncio.sleep(sim(cfg['failure']['restore_after_s']))
                    await inject_retry(unit, FailureType.OK, 'failure restore')
            result['onset_wall_s'] = round(time.time() - t_start, 2)

        # wait for landing (mission end + RTL, or a failsafe landing), bounded
        async def wait_landed():
            async for a in d.telemetry.in_air():
                if not a:
                    return
        try:
            await asyncio.wait_for(wait_landed(), sim(480))
        except asyncio.TimeoutError:
            result['notes'].append('landing timeout; forcing RTL')
            landed = False
            for step, wait_s in (('rtl', 120), ('land', 60)):
                try:
                    if step == 'rtl':
                        await d.action.return_to_launch()
                    else:
                        await d.action.land()
                    await asyncio.wait_for(wait_landed(), sim(wait_s))
                    landed = True
                    break
                except Exception as e:
                    result['notes'].append(f'{step} failed: {type(e).__name__}: {e}')
            if not landed:
                result['notes'].append('killed in sim (still airborne at cap)')
                try:
                    await d.action.kill()
                    await asyncio.wait_for(wait_landed(), sim(30))
                except Exception as e:
                    result['notes'].append(f'kill failed: {type(e).__name__}: {e}')
        await asyncio.sleep(3)
        result['ok'] = True
    except Exception as e:
        result['notes'].append(f'flight error: {type(e).__name__}: {e}')
    finally:
        subprocess.run([str(px4_bin.parent / 'px4-shutdown')], cwd=rootfs,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            proc.terminate()
        out.close()
        subprocess.run(['pkill', '-f', 'mavsdk_server'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    logs = sorted((rootfs / 'fs/log').rglob('*.ulg'), key=os.path.getmtime)
    result['ulog'] = logs[-1] if logs else None
    return result


def verify(ulog_path):
    """Max horizontal divergence between reported GPS and simulator truth, plus onset from param changes."""
    from pyulog import ULog
    u = ULog(str(ulog_path))
    names = {d.name: d for d in u.data_list}
    info = {'duration_s': round((u.last_timestamp - u.start_timestamp) / 1e6, 1),
            'topics': len(names), 'has_truth': 'vehicle_global_position_groundtruth' in names}
    onset = [(t, k, v) for (t, k, v) in u.changed_parameters if k in ('SIM_GPS_SPF_EN', 'SYS_FAILURE_EN') and v == 1]
    info['onset_sim_s'] = round((onset[0][0] - u.start_timestamp) / 1e6, 2) if onset else None
    if info['has_truth'] and 'sensor_gps' in names:
        import numpy as np
        g, t = names['sensor_gps'].data, names['vehicle_global_position_groundtruth'].data
        lat_g = g.get('latitude_deg', g.get('lat'))
        lon_g = g.get('longitude_deg', g.get('lon'))
        if lat_g is not None and np.nanmax(np.abs(lat_g)) > 1000:  # int32 1e7 scaling in older logs
            lat_g, lon_g = lat_g / 1e7, lon_g / 1e7
        lat_t = np.interp(g['timestamp'], t['timestamp'], t['lat'])
        lon_t = np.interp(g['timestamp'], t['timestamp'], t['lon'])
        dn = (lat_g - lat_t) * 111320.0
        de = (lon_g - lon_t) * 111320.0 * np.cos(np.radians(lat_t))
        div = np.hypot(dn, de)
        ts = (g['timestamp'] - u.start_timestamp) / 1e6
        info['max_gps_vs_truth_m'] = round(float(np.nanmax(div)), 1)
        if info['onset_sim_s'] is not None:
            pre, post = div[ts < info['onset_sim_s']], div[ts >= info['onset_sim_s']]
            info['div_pre_onset_max_m'] = round(float(np.nanmax(pre)), 1) if len(pre) else None
            info['div_post_onset_max_m'] = round(float(np.nanmax(post)), 1) if len(post) else None
        last = div[ts >= ts[-1] - 30.0]
        info['div_last30s_max_m'] = round(float(np.nanmax(last)), 1) if len(last) else None
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--px4', default='/content/PX4-Autopilot')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_per_family', type=int, default=1)
    ap.add_argument('--speed', type=float, default=4.0)
    ap.add_argument('--seed', type=int, default=20260909)
    ap.add_argument('--families', default=','.join(FAMILIES))
    args = ap.parse_args()

    px4 = pathlib.Path(args.px4)
    bld = px4 / 'build/px4_sitl_default'
    px4_bin, px4_etc = bld / 'bin/px4', bld / 'etc'
    rootfs = bld / 'rootfs_gen'
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'manifest.jsonl'
    done = set()
    if manifest.exists():
        done = {json.loads(l)['flight_id'] for l in manifest.read_text().splitlines() if l.strip()}

    try:
        fw_commit = subprocess.run(['git', '-C', str(px4), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except Exception:
        fw_commit = None

    families = args.families.split(',')
    jobs = [(fam, i) for i in range(args.n_per_family) for fam in families]
    for fam, i in jobs:
        seed = args.seed * 1000 + FAMILIES.index(fam) * 100 + i
        rng = random.Random(seed)
        flight_id = f'f{seed}_{fam}'
        if flight_id in done:
            print('skip (done):', flight_id); continue
        home = rng.choice(HOMES)
        cfg = draw_config(rng, fam)
        mission = draw_mission(rng, home)
        print(f'\n=== {flight_id}  subtype={cfg["subtype"]}  onset={cfg["onset_s"]}  home={home[:2]} ===', flush=True)
        log_txt = out / f'{flight_id}.px4.txt'
        res = asyncio.run(fly_one(px4_bin, px4_etc, rootfs, home, args.speed, cfg, mission, log_txt, 50040 + (i % 50)))
        if not res['ok']:
            print('  flight failed, retrying once with the same config:', res['notes'], flush=True)
            first_notes = res['notes']
            res = asyncio.run(fly_one(px4_bin, px4_etc, rootfs, home, args.speed, cfg, mission, log_txt, 50040 + (i % 50)))
            res['notes'] = [f'first attempt failed: {first_notes}'] + res['notes']
        rec = {'flight_id': flight_id, 'seed': seed, 'home': home, 'speed_factor': args.speed,
               'firmware_commit': fw_commit, 'family': fam, 'subtype': cfg['subtype'],
               'onset_s_requested': cfg['onset_s'], 'params': cfg['params'], 'nuisance': cfg['nuisance'],
               'failure': cfg['failure'], 'mission': mission, 'ok': res['ok'], 'notes': res['notes'],
               'onset_wall_s': res.get('onset_wall_s')}
        if res['ulog'] is not None:
            dst = out / f'{flight_id}.ulg'
            shutil.copy(res['ulog'], dst)
            rec['ulog'] = dst.name; rec['sha256'] = sha256(dst); rec['bytes'] = dst.stat().st_size
            try:
                rec['verify'] = verify(dst)
            except Exception as e:
                rec['verify'] = {'error': f'{type(e).__name__}: {e}'}
        else:
            rec['ulog'] = None
        (out / f'{flight_id}.json').write_text(json.dumps(rec, indent=2))
        with open(manifest, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        print(json.dumps({k: rec[k] for k in ('ok', 'ulog', 'notes', 'verify') if k in rec}), flush=True)


if __name__ == '__main__':
    main()
