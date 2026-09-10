#!/usr/bin/env python3
"""
sih_spoof_patch.py  --  add a parameterised GNSS spoof injector to PX4's sensor_gps_sim.

Usage:  python sih_spoof_patch.py /content/PX4-Autopilot
Then:   cd /content/PX4-Autopilot && make px4_sitl_default

What it adds (all in src/modules/simulation/sensor_gps_sim):
  New parameters (group Simulator):
    SIM_GPS_SPF_EN   0/1   enable injector (set to 1 mid-flight = spoof onset; logged as a param change)
    SIM_GPS_SPF_VEL  0/1   0 = reported velocity stays truthful (incoherent), 1 = drift added to velocity (coherent)
    SIM_GPS_SPF_JN   m     initial offset north at onset (jump)
    SIM_GPS_SPF_JE   m     initial offset east at onset (jump)
    SIM_GPS_SPF_VN   m/s   drift rate north
    SIM_GPS_SPF_VE   m/s   drift rate east
    SIM_GPS_SPF_VD   m/s   drift rate down
    SIM_GPS_SPF_MAX  m     horizontal offset cap (0 = no cap)
  Injection happens before EKF2 consumes the fix, so the closed-loop vehicle response is real.
  IMU, baro and mag remain truthful. Receiver self-report fields (fix_type, satellites_used,
  jamming/spoofing_state) are untouched, i.e. a naive receiver.

The script is idempotent: running it twice is a no-op.
"""
import pathlib
import sys

PX4 = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else '/content/PX4-Autopilot')
MOD = PX4 / 'src/modules/simulation/sensor_gps_sim'
HPP, CPP, YML = MOD / 'SensorGpsSim.hpp', MOD / 'SensorGpsSim.cpp', MOD / 'parameters.yaml'
MARK = 'SIM_GPS_SPF_EN'


def must_replace(text, old, new, label):
    if old not in text:
        raise SystemExit(f'PATCH FAILED: anchor not found for {label}. The module source changed; '
                         f'send me the current {label} file.')
    if text.count(old) != 1:
        raise SystemExit(f'PATCH FAILED: anchor for {label} is not unique.')
    return text.replace(old, new)


# ---------------------------------------------------------------- header
hpp = HPP.read_text()
if MARK in hpp:
    print('header already patched')
else:
    hpp = must_replace(
        hpp,
        '\tDEFINE_PARAMETERS(\n'
        '\t\t(ParamInt<px4::params::SIM_GPS_USED>)      _sim_gps_used,\n'
        '\t\t(ParamFloat<px4::params::SENS_GPS1_OFFX>)  _param_gps1_offx,\n'
        '\t\t(ParamFloat<px4::params::SENS_GPS1_OFFY>)  _param_gps1_offy\n'
        '\t)\n',
        '\t// GNSS spoof injector state (research patch)\n'
        '\tbool _spf_active{false};\n'
        '\tuint64_t _spf_t0{0};\n'
        '\n'
        '\tDEFINE_PARAMETERS(\n'
        '\t\t(ParamInt<px4::params::SIM_GPS_USED>)      _sim_gps_used,\n'
        '\t\t(ParamFloat<px4::params::SENS_GPS1_OFFX>)  _param_gps1_offx,\n'
        '\t\t(ParamFloat<px4::params::SENS_GPS1_OFFY>)  _param_gps1_offy,\n'
        '\t\t(ParamInt<px4::params::SIM_GPS_SPF_EN>)    _spf_en,\n'
        '\t\t(ParamInt<px4::params::SIM_GPS_SPF_VEL>)   _spf_vel,\n'
        '\t\t(ParamFloat<px4::params::SIM_GPS_SPF_JN>)  _spf_jn,\n'
        '\t\t(ParamFloat<px4::params::SIM_GPS_SPF_JE>)  _spf_je,\n'
        '\t\t(ParamFloat<px4::params::SIM_GPS_SPF_VN>)  _spf_vn,\n'
        '\t\t(ParamFloat<px4::params::SIM_GPS_SPF_VE>)  _spf_ve,\n'
        '\t\t(ParamFloat<px4::params::SIM_GPS_SPF_VD>)  _spf_vd,\n'
        '\t\t(ParamFloat<px4::params::SIM_GPS_SPF_MAX>) _spf_max\n'
        '\t)\n',
        'SensorGpsSim.hpp')
    HPP.write_text(hpp)
    print('header patched')

# ---------------------------------------------------------------- source
cpp = CPP.read_text()
if '_spf_en.get()' in cpp:
    print('source already patched')
else:
    cpp = must_replace(
        cpp,
        '\t\tconst double latitude = gpos.lat + math::degrees((double)_gps_pos_noise_n / CONSTANTS_RADIUS_OF_EARTH);\n'
        '\t\tconst double longitude = gpos.lon + math::degrees((double)_gps_pos_noise_e / CONSTANTS_RADIUS_OF_EARTH);\n'
        '\t\tconst double altitude = (double)(gpos.alt + _gps_pos_noise_d);\n',
        '\t\tdouble latitude = gpos.lat + math::degrees((double)_gps_pos_noise_n / CONSTANTS_RADIUS_OF_EARTH);\n'
        '\t\tdouble longitude = gpos.lon + math::degrees((double)_gps_pos_noise_e / CONSTANTS_RADIUS_OF_EARTH);\n'
        '\t\tdouble altitude = (double)(gpos.alt + _gps_pos_noise_d);\n',
        'SensorGpsSim.cpp (position block)')

    cpp = must_replace(
        cpp,
        '\t\tconst Vector3f gps_vel = Vector3f{lpos.vx + _gps_vel_noise_n, lpos.vy + _gps_vel_noise_e, lpos.vz + _gps_vel_noise_d};\n',
        '\t\tVector3f gps_vel = Vector3f{lpos.vx + _gps_vel_noise_n, lpos.vy + _gps_vel_noise_e, lpos.vz + _gps_vel_noise_d};\n'
        '\n'
        '\t\t// ---- GNSS spoof injector (research patch): jump + linear drift applied to the reported fix ----\n'
        '\t\tif (_spf_en.get() == 1) {\n'
        '\t\t\tif (!_spf_active) {\n'
        '\t\t\t\t_spf_active = true;\n'
        '\t\t\t\t_spf_t0 = gpos.timestamp_sample;\n'
        '\t\t\t}\n'
        '\n'
        '\t\t\tconst float t = (float)(gpos.timestamp_sample - _spf_t0) * 1e-6f;\n'
        '\t\t\tfloat off_n = _spf_jn.get() + _spf_vn.get() * t;\n'
        '\t\t\tfloat off_e = _spf_je.get() + _spf_ve.get() * t;\n'
        '\t\t\tconst float off_d = _spf_vd.get() * t;\n'
        '\t\t\tconst float mag = sqrtf(off_n * off_n + off_e * off_e);\n'
        '\t\t\tconst float cap = _spf_max.get();\n'
        '\t\t\tbool ramping = true;\n'
        '\n'
        '\t\t\tif (cap > 0.f && mag > cap) {\n'
        '\t\t\t\toff_n *= cap / mag;\n'
        '\t\t\t\toff_e *= cap / mag;\n'
        '\t\t\t\tramping = false;\n'
        '\t\t\t}\n'
        '\n'
        '\t\t\tlatitude  += math::degrees((double)off_n / CONSTANTS_RADIUS_OF_EARTH);\n'
        '\t\t\tlongitude += math::degrees((double)off_e / CONSTANTS_RADIUS_OF_EARTH) / cos(gpos.lat * M_PI / 180.0);\n'
        '\t\t\taltitude  -= (double)off_d;\n'
        '\n'
        '\t\t\tif (_spf_vel.get() == 1 && ramping) {\n'
        '\t\t\t\tgps_vel(0) += _spf_vn.get();\n'
        '\t\t\t\tgps_vel(1) += _spf_ve.get();\n'
        '\t\t\t\tgps_vel(2) += _spf_vd.get();\n'
        '\t\t\t}\n'
        '\n'
        '\t\t} else {\n'
        '\t\t\t_spf_active = false;\n'
        '\t\t}\n',
        'SensorGpsSim.cpp (velocity block)')
    CPP.write_text(cpp)
    print('source patched')

# ---------------------------------------------------------------- parameters
yml = YML.read_text()
if MARK in yml:
    print('parameters already patched')
else:
    yml = yml.rstrip('\n') + '''
    SIM_GPS_SPF_EN:
      description:
        short: Enable simulated GNSS spoof injector
      type: enum
      values:
        0: Disabled
        1: Enabled
      default: 0
      min: 0
      max: 1
    SIM_GPS_SPF_VEL:
      description:
        short: Spoof injector velocity coherence
        long: 0 keeps the reported velocity truthful (incoherent spoof). 1 adds the drift rate to the reported velocity while the offset is ramping (coherent spoof).
      type: enum
      values:
        0: Incoherent
        1: Coherent
      default: 0
      min: 0
      max: 1
    SIM_GPS_SPF_JN:
      description:
        short: Spoof initial offset north at onset
      type: float
      unit: m
      default: 0.0
    SIM_GPS_SPF_JE:
      description:
        short: Spoof initial offset east at onset
      type: float
      unit: m
      default: 0.0
    SIM_GPS_SPF_VN:
      description:
        short: Spoof drift rate north
      type: float
      unit: m/s
      default: 0.0
    SIM_GPS_SPF_VE:
      description:
        short: Spoof drift rate east
      type: float
      unit: m/s
      default: 0.0
    SIM_GPS_SPF_VD:
      description:
        short: Spoof drift rate down
      type: float
      unit: m/s
      default: 0.0
    SIM_GPS_SPF_MAX:
      description:
        short: Spoof horizontal offset cap (0 = no cap)
      type: float
      unit: m
      default: 200.0
'''
    YML.write_text(yml)
    print('parameters patched')

# ---------------------------------------------------------------- noise scale (v2)
NOISE_MARK = 'SIM_GPS_NOISE_K'
hpp = HPP.read_text()
if NOISE_MARK in hpp:
    print('header noise scale already patched')
else:
    hpp = must_replace(
        hpp,
        '\t\t(ParamFloat<px4::params::SIM_GPS_SPF_MAX>) _spf_max\n\t)\n',
        '\t\t(ParamFloat<px4::params::SIM_GPS_SPF_MAX>) _spf_max,\n'
        '\t\t(ParamFloat<px4::params::SIM_GPS_NOISE_K>) _noise_k,\n'
        '\t\t(ParamFloat<px4::params::SIM_GPS_NOISEV_K>) _noisev_k\n'
        '\t)\n',
        'SensorGpsSim.hpp (noise scale)')
    HPP.write_text(hpp)
    print('header noise scale patched')

cpp = CPP.read_text()
if '_noise_k.get()' in cpp:
    print('source noise scale already patched')
else:
    pos_old = '_pos_random_walk * generate_wgn() * _pos_noise_amplitude'
    vel_old = '_vel_noise_density * generate_wgn() * _vel_noise_amplitude'
    if cpp.count(pos_old) != 3 or cpp.count(vel_old) != 3:
        raise SystemExit('PATCH FAILED: expected 3 position and 3 velocity noise lines in SensorGpsSim.cpp')
    cpp = cpp.replace(pos_old, pos_old + ' * _noise_k.get()')
    cpp = cpp.replace(vel_old, vel_old + ' * _noisev_k.get()')
    CPP.write_text(cpp)
    print('source noise scale patched')

yml = YML.read_text()
if NOISE_MARK in yml:
    print('parameters noise scale already patched')
else:
    yml = yml.rstrip('\n') + '''
    SIM_GPS_NOISE_K:
      description:
        short: Simulated GPS position noise scale (1 = PX4 default, about 2.5 cm; 40 = about 1 m)
      type: float
      default: 1.0
      min: 0.0
      max: 500.0
    SIM_GPS_NOISEV_K:
      description:
        short: Simulated GPS velocity noise scale (1 = PX4 default, about 2.4 cm/s; 4 = about 0.1 m/s)
      type: float
      default: 1.0
      min: 0.0
      max: 100.0
'''
    YML.write_text(yml)
    print('parameters noise scale patched')

# ---------------------------------------------------------------- receiver-realism noise model (v3)
# Real receivers show a slowly varying position bias (tens of seconds correlation, metre level) with
# centimetre-level epoch-to-epoch jitter; PX4's stock model has a ~0.5 s correlation time, which inflates
# every derivative-based consistency feature. SIM_GPS_NOISE_T > 0 switches the position noise to an
# Ornstein-Uhlenbeck bias with correlation time T (s) and stationary std SIM_GPS_NOISE_P (m), plus white
# jitter SIM_GPS_NOISE_J (m). T = 0 keeps the stock model (scaled by SIM_GPS_NOISE_K).
REAL_MARK = 'SIM_GPS_NOISE_T'
hpp = HPP.read_text()
if REAL_MARK in hpp:
    print('header realism model already patched')
else:
    hpp = must_replace(
        hpp,
        '\t\t(ParamFloat<px4::params::SIM_GPS_NOISEV_K>) _noisev_k\n\t)\n',
        '\t\t(ParamFloat<px4::params::SIM_GPS_NOISEV_K>) _noisev_k,\n'
        '\t\t(ParamFloat<px4::params::SIM_GPS_NOISE_T>) _noise_t,\n'
        '\t\t(ParamFloat<px4::params::SIM_GPS_NOISE_P>) _noise_p,\n'
        '\t\t(ParamFloat<px4::params::SIM_GPS_NOISE_J>) _noise_j\n'
        '\t)\n',
        'SensorGpsSim.hpp (realism model)')
    hpp = must_replace(
        hpp,
        '\t// GNSS spoof injector state (research patch)\n',
        '\t// receiver-realism noise state (research patch)\n'
        '\tuint64_t _noise_last_ts{0};\n\n'
        '\t// GNSS spoof injector state (research patch)\n',
        'SensorGpsSim.hpp (realism state)')
    HPP.write_text(hpp)
    print('header realism model patched')

cpp = CPP.read_text()
if '#include <lib/mathlib/mathlib.h>' not in cpp:
    cpp = must_replace(cpp, '#include <lib/geo/geo.h>\n', '#include <lib/geo/geo.h>\n#include <lib/mathlib/mathlib.h>\n',
                       'SensorGpsSim.cpp (mathlib include)')
    CPP.write_text(cpp)
    print('mathlib include added')
if '_noise_t.get()' in cpp:
    print('source realism model already patched')
else:
    cpp = must_replace(
        cpp,
        '\t\t// Correlated Markov process position noise (matching GZBridge model)\n'
        '\t\t_gps_pos_noise_n = _pos_markov_time * _gps_pos_noise_n +\n'
        '\t\t\t\t   _pos_random_walk * generate_wgn() * _pos_noise_amplitude * _noise_k.get();\n'
        '\n'
        '\t\t_gps_pos_noise_e = _pos_markov_time * _gps_pos_noise_e +\n'
        '\t\t\t\t   _pos_random_walk * generate_wgn() * _pos_noise_amplitude * _noise_k.get();\n'
        '\n'
        '\t\t_gps_pos_noise_d = _pos_markov_time * _gps_pos_noise_d +\n'
        '\t\t\t\t   _pos_random_walk * generate_wgn() * _pos_noise_amplitude * _noise_k.get() * 1.5f;\n'
        '\n'
        '\t\tdouble latitude = gpos.lat + math::degrees((double)_gps_pos_noise_n / CONSTANTS_RADIUS_OF_EARTH);\n'
        '\t\tdouble longitude = gpos.lon + math::degrees((double)_gps_pos_noise_e / CONSTANTS_RADIUS_OF_EARTH);\n'
        '\t\tdouble altitude = (double)(gpos.alt + _gps_pos_noise_d);\n',
        '\t\tfloat jit_n = 0.f, jit_e = 0.f, jit_d = 0.f;\n'
        '\n'
        '\t\tif (_noise_t.get() > 0.f) {\n'
        '\t\t\t// Receiver-realism model (research patch): slow Ornstein-Uhlenbeck bias + white jitter\n'
        '\t\t\tconst float dt = (_noise_last_ts > 0) ? math::constrain((float)(gpos.timestamp_sample - _noise_last_ts) * 1e-6f, 0.001f, 1.0f) : 0.125f;\n'
        '\t\t\t_noise_last_ts = gpos.timestamp_sample;\n'
        '\t\t\tconst float rho = expf(-dt / _noise_t.get());\n'
        '\t\t\tconst float step = sqrtf(math::max(0.f, 1.f - rho * rho)) * _noise_p.get();\n'
        '\t\t\t_gps_pos_noise_n = rho * _gps_pos_noise_n + step * generate_wgn();\n'
        '\t\t\t_gps_pos_noise_e = rho * _gps_pos_noise_e + step * generate_wgn();\n'
        '\t\t\t_gps_pos_noise_d = rho * _gps_pos_noise_d + step * generate_wgn() * 1.5f;\n'
        '\t\t\tjit_n = _noise_j.get() * generate_wgn();\n'
        '\t\t\tjit_e = _noise_j.get() * generate_wgn();\n'
        '\t\t\tjit_d = _noise_j.get() * 1.5f * generate_wgn();\n'
        '\n'
        '\t\t} else {\n'
        '\t\t\t// Correlated Markov process position noise (matching GZBridge model)\n'
        '\t\t\t_gps_pos_noise_n = _pos_markov_time * _gps_pos_noise_n +\n'
        '\t\t\t\t\t   _pos_random_walk * generate_wgn() * _pos_noise_amplitude * _noise_k.get();\n'
        '\n'
        '\t\t\t_gps_pos_noise_e = _pos_markov_time * _gps_pos_noise_e +\n'
        '\t\t\t\t\t   _pos_random_walk * generate_wgn() * _pos_noise_amplitude * _noise_k.get();\n'
        '\n'
        '\t\t\t_gps_pos_noise_d = _pos_markov_time * _gps_pos_noise_d +\n'
        '\t\t\t\t\t   _pos_random_walk * generate_wgn() * _pos_noise_amplitude * _noise_k.get() * 1.5f;\n'
        '\t\t}\n'
        '\n'
        '\t\tdouble latitude = gpos.lat + math::degrees((double)(_gps_pos_noise_n + jit_n) / CONSTANTS_RADIUS_OF_EARTH);\n'
        '\t\tdouble longitude = gpos.lon + math::degrees((double)(_gps_pos_noise_e + jit_e) / CONSTANTS_RADIUS_OF_EARTH);\n'
        '\t\tdouble altitude = (double)(gpos.alt + _gps_pos_noise_d + jit_d);\n',
        'SensorGpsSim.cpp (realism model)')
    CPP.write_text(cpp)
    print('source realism model patched')

yml = YML.read_text()
if REAL_MARK in yml:
    print('parameters realism model already patched')
else:
    yml = yml.rstrip('\n') + '''
    SIM_GPS_NOISE_T:
      description:
        short: Simulated GPS position-bias correlation time (0 = stock PX4 noise model)
      type: float
      unit: s
      default: 0.0
      min: 0.0
      max: 1000.0
    SIM_GPS_NOISE_P:
      description:
        short: Simulated GPS position-bias stationary standard deviation (used when SIM_GPS_NOISE_T > 0)
      type: float
      unit: m
      default: 1.5
      min: 0.0
      max: 50.0
    SIM_GPS_NOISE_J:
      description:
        short: Simulated GPS white position jitter standard deviation (used when SIM_GPS_NOISE_T > 0)
      type: float
      unit: m
      default: 0.03
      min: 0.0
      max: 5.0
'''
    YML.write_text(yml)
    print('parameters realism model patched')

print('OK. Now rebuild:  cd', PX4, '&& make px4_sitl_default')
