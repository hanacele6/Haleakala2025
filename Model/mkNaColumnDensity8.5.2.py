# -*- coding: utf-8 -*-
"""
==============================================================================
概要 (安定化検証版 v2: 平衡更新緩和なし)
==============================================================================
水星ナトリウム大気 3次元モンテカルロシミュレーション

【適用している修正】
1. フラックスの時間平均 (Smoothing)
   - 入射フラックスに対して指数移動平均(EMA, alpha=0.05)を適用。
   - これにより、粒子が来ないステップでの0落ちや、1個来た時のスパイクを防ぎます。
   - 計算された「滑らかなフラックス」を使って平衡密度を算出します。

2. ターゲットSPSの増加
   - TARGET_SPS を 2000 に設定し、ベースの統計精度を上げます。

【適用していない修正】
- 平衡密度の更新ダンピング（徐々に近づける処理）は行いません。
- 計算された平衡密度をそのままそのステップの密度として代入します。

保存先: ./SimulationResult_202510/Stabilized_v2_...
==============================================================================
"""

import numpy as np
import sys
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

# ==============================================================================
# 物理定数 (SI単位系)
# ==============================================================================
PHYSICAL_CONSTANTS = {
    'PI': np.pi,
    'AU': 1.496e11,
    'MASS_NA': 22.98976928 * 1.66054e-27,
    'K_BOLTZMANN': 1.380649e-23,
    'GM_MERCURY': 2.2032e13,
    'RM': 2.440e6,
    'C': 299792458.0,
    'H': 6.62607015e-34,
    'E_CHARGE': 1.602176634e-19,
    'ME': 9.1093897e-31,
    'EPSILON_0': 8.854187817e-12,
    'G': 6.6743e-11,
    'MASS_SUN': 1.989e30,
    'EV_TO_JOULE': 1.602176634e-19,
    'ROTATION_PERIOD': 58.6462 * 24 * 3600,
    'ORBITAL_PERIOD': 87.969 * 24 * 3600,
}


# ==============================================================================
# ヘルパー関数
# ==============================================================================

def calculate_surface_temperature_leblanc(lon_fixed_rad, lat_rad, AU, subsolar_lon_rad):
    """表面温度計算 (Leblanc 2003)"""
    T_night = 100.0
    cos_theta = np.cos(lat_rad) * np.cos(lon_fixed_rad - subsolar_lon_rad)

    if cos_theta <= 0:
        return T_night

    T0_peri = 600.0
    T0_aph = 475.0
    AU_peri = 0.307
    AU_aph = 0.467
    T0 = np.interp(AU, [AU_peri, AU_aph], [T0_peri, T0_aph])
    T1 = 100.0
    return T0 + T1 * (cos_theta ** 0.25)


def calculate_sticking_probability(surface_temp_K):
    """吸着確率"""
    A = 0.0804
    B = 458.0
    porosity = 0.8
    if surface_temp_K <= 0: return 1.0
    p_stick = A * np.exp(B / surface_temp_K)
    p_stick_eff = p_stick / (1.0 - (1.0 - p_stick) * porosity)
    return min(p_stick_eff, 1.0)


def calculate_thermal_desorption_rate(surface_temp_K):
    """熱脱離率 [1/s]"""
    if surface_temp_K < 10.0: return 0.0
    VIB_FREQ = 1e13
    BINDING_ENERGY_EV = 1.85
    BINDING_ENERGY_J = BINDING_ENERGY_EV * PHYSICAL_CONSTANTS['EV_TO_JOULE']
    k_B = PHYSICAL_CONSTANTS['K_BOLTZMANN']
    exponent = -BINDING_ENERGY_J / (k_B * surface_temp_K)
    if exponent < -700: return 0.0
    return VIB_FREQ * np.exp(exponent)


def calculate_mmv_flux(AU):
    """MMVフラックス [atoms/m^2/s]"""
    TOTAL_FLUX_AT_PERI = 5e23
    PERIHELION_AU = 0.307
    AREA = 4 * PHYSICAL_CONSTANTS['PI'] * (PHYSICAL_CONSTANTS['RM'] ** 2)
    avg_flux_peri = TOTAL_FLUX_AT_PERI / AREA
    C = avg_flux_peri * (PERIHELION_AU ** 1.9)
    return C * (AU ** (-1.9))


def sample_speed_from_flux_distribution(mass_kg, temp_k):
    """フラックスマクスウェル分布からの速度サンプリング"""
    kT = PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_k
    E = np.random.gamma(2.0, kT)
    return np.sqrt(2.0 * E / mass_kg)


def sample_thompson_sigmund_energy(U_eV, E_max_eV=5.0):
    """SWS用エネルギーサンプリング"""
    f_max = (U_eV / 2.0) / (U_eV / 2.0 + U_eV) ** 3
    while True:
        E_try = np.random.uniform(0, E_max_eV)
        f_val = E_try / (E_try + U_eV) ** 3
        if np.random.uniform(0, f_max) <= f_val:
            return E_try


def sample_lambertian_direction_local():
    """ランバート分布 (cos則) の方向ベクトル"""
    u1, u2 = np.random.random(2)
    phi = 2 * PHYSICAL_CONSTANTS['PI'] * u1
    cos_theta = np.sqrt(1 - u2)
    sin_theta = np.sqrt(u2)
    return np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])


def transform_local_to_world(local_vec, normal_vector):
    """ローカル座標系 -> ワールド座標系"""
    local_z = normal_vector / np.linalg.norm(normal_vector)
    world_up = np.array([0., 0., 1.])
    if np.abs(np.dot(local_z, world_up)) > 0.99:
        world_up = np.array([0., 1., 0.])
    local_x = np.cross(world_up, local_z)
    local_x /= np.linalg.norm(local_x)
    local_y = np.cross(local_z, local_x)
    return local_vec[0] * local_x + local_vec[1] * local_y + local_vec[2] * local_z


def get_orbital_params_cyclic(time_sec, orbit_data, t_perihelion_file):
    """軌道パラメータ取得"""
    cycle_sec = PHYSICAL_CONSTANTS['ORBITAL_PERIOD']

    dt_from_peri = (time_sec - t_perihelion_file)
    time_in_cycle = dt_from_peri % cycle_sec

    time_col_original = orbit_data[:, 2]
    idx_peri = np.argmin(np.abs(orbit_data[:, 0]))
    t_peri_in_file = time_col_original[idx_peri]

    t_lookup = t_peri_in_file + time_in_cycle

    taa_deg = np.interp(t_lookup, time_col_original, orbit_data[:, 0])
    au = np.interp(t_lookup, time_col_original, orbit_data[:, 1])
    v_rad = np.interp(t_lookup, time_col_original, orbit_data[:, 3])
    v_tan = np.interp(t_lookup, time_col_original, orbit_data[:, 4])

    taa_rad = np.deg2rad(taa_deg)
    omega_rot = 2 * np.pi / PHYSICAL_CONSTANTS['ROTATION_PERIOD']
    rotation_angle = omega_rot * (time_sec - t_perihelion_file)
    subsolar_lon_rad = taa_rad - rotation_angle
    subsolar_lon_rad = (subsolar_lon_rad + np.pi) % (2 * np.pi) - np.pi

    return taa_deg, au, v_rad, v_tan, subsolar_lon_rad


def lonlat_to_xyz(lon_rad, lat_rad, radius):
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return np.array([x, y, z])


def _calculate_acceleration(pos, vel, V_radial_ms, V_tangential_ms, AU, spec_data, settings):
    """加速度計算"""
    x, y, z = pos
    r0 = AU * PHYSICAL_CONSTANTS['AU']
    velocity_for_doppler = vel[0] + V_radial_ms
    w_na_d2 = 589.1582e-9 * (1.0 + velocity_for_doppler / PHYSICAL_CONSTANTS['C'])
    w_na_d1 = 589.7558e-9 * (1.0 + velocity_for_doppler / PHYSICAL_CONSTANTS['C'])

    b = 0.0
    wl, gamma, sigma0_perdnu2, sigma0_perdnu1, JL = spec_data.values()

    if (wl[0] * 1e-9 <= w_na_d2 < wl[-1] * 1e-9) and (wl[0] * 1e-9 <= w_na_d1 < wl[-1] * 1e-9):
        gamma2 = np.interp(w_na_d2 * 1e9, wl, gamma)
        gamma1 = np.interp(w_na_d1 * 1e9, wl, gamma)
        F_lambda_1AU = JL * 1e13
        F_at_Merc = F_lambda_1AU / (AU ** 2)

        b = 1 / PHYSICAL_CONSTANTS['MASS_NA'] * (
                (PHYSICAL_CONSTANTS['H'] / w_na_d1) * sigma0_perdnu1 * (
                F_at_Merc * gamma1 * w_na_d1 ** 2 / PHYSICAL_CONSTANTS['C']) +
                (PHYSICAL_CONSTANTS['H'] / w_na_d2) * sigma0_perdnu2 * (
                        F_at_Merc * gamma2 * w_na_d2 ** 2 / PHYSICAL_CONSTANTS['C']))

    if x < 0 and np.sqrt(y ** 2 + z ** 2) < PHYSICAL_CONSTANTS['RM']:
        b = 0.0

    accel_srp = np.array([-b, 0.0, 0.0])
    r_sq = np.sum(pos ** 2)
    accel_g = -PHYSICAL_CONSTANTS['GM_MERCURY'] * pos / (r_sq ** 1.5) if r_sq > 0 else np.zeros(3)

    accel_sun = np.zeros(3)
    if settings.get('USE_SOLAR_GRAVITY', False):
        G = PHYSICAL_CONSTANTS['G']
        M_SUN = PHYSICAL_CONSTANTS['MASS_SUN']
        r_ps_vec = np.array([r0 - pos[0], -pos[1], -pos[2]])
        r_ps_mag_sq = np.sum(r_ps_vec ** 2)
        if r_ps_mag_sq > 0:
            accel_sun = (G * M_SUN * r_ps_vec) / (r_ps_mag_sq ** 1.5)

    accel_cor = np.zeros(3)
    accel_cen = np.zeros(3)
    if settings.get('USE_CORIOLIS_FORCES', False):
        if r0 > 0:
            omega_val = V_tangential_ms / r0
            omega_sq = omega_val ** 2
            accel_cen = np.array([(omega_val ** 2) * (pos[0] - r0), omega_sq * pos[1], 0.0])
            two_omega = 2 * omega_val
            accel_cor = np.array([two_omega * vel[1], -two_omega * vel[0], 0.0])

    return accel_srp + accel_g + accel_sun + accel_cen + accel_cor


def simulate_particle_for_one_step(args):
    """1ステップ分の粒子追跡"""
    settings, spec_data = args['settings'], args['spec']
    TAA, AU, V_rad, V_tan, subsolar_lon = args['orbit']
    total_duration = args['duration']

    DT_INTEGRATION = 500.0
    if total_duration <= 0:
        return {'status': 'alive', 'final_state': args['particle_state']}

    num_steps = int(np.ceil(total_duration / DT_INTEGRATION))
    dt_per_step = total_duration / num_steps

    pos = args['particle_state']['pos'].copy()
    vel = args['particle_state']['vel'].copy()
    weight = args['particle_state']['weight']

    tau_ion = settings['T1AU'] * AU ** 2
    RM = PHYSICAL_CONSTANTS['RM']
    R_MAX = RM * settings['GRID_RADIUS_RM']

    pos_start = pos.copy()

    for _ in range(num_steps):
        current_dt = dt_per_step

        if pos[0] > 0:
            if np.random.random() < (1.0 - np.exp(-current_dt / tau_ion)):
                return {'status': 'ionized', 'final_state': None}

        k1_v = current_dt * _calculate_acceleration(pos, vel, V_rad, V_tan, AU, spec_data, settings)
        k1_p = current_dt * vel
        k2_v = current_dt * _calculate_acceleration(pos + 0.5 * k1_p, vel + 0.5 * k1_v, V_rad, V_tan, AU, spec_data,
                                                    settings)
        k2_p = current_dt * (vel + 0.5 * k1_v)
        k3_v = current_dt * _calculate_acceleration(pos + 0.5 * k2_p, vel + 0.5 * k2_v, V_rad, V_tan, AU, spec_data,
                                                    settings)
        k3_p = current_dt * (vel + 0.5 * k2_v)
        k4_v = current_dt * _calculate_acceleration(pos + k3_p, vel + k3_v, V_rad, V_tan, AU, spec_data, settings)
        k4_p = current_dt * (vel + k3_v)

        pos += (k1_p + 2 * k2_p + 2 * k3_p + k4_p) / 6.0
        vel += (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6.0

        r_now = np.linalg.norm(pos)

        if r_now > R_MAX:
            return {'status': 'escaped', 'final_state': None}

        if r_now <= RM:
            pos_impact = pos_start
            lon_rot = np.arctan2(pos_impact[1], pos_impact[0])
            lat_rot = np.arcsin(np.clip(pos_impact[2] / np.linalg.norm(pos_impact), -1, 1))
            lon_fixed = (lon_rot + subsolar_lon + np.pi) % (2 * np.pi) - np.pi
            temp_impact = calculate_surface_temperature_leblanc(lon_fixed, lat_rot, AU, subsolar_lon)

            if np.random.random() < calculate_sticking_probability(temp_impact):
                return {'status': 'stuck', 'pos_at_impact': pos_impact, 'weight': weight}
            else:
                E_in = 0.5 * PHYSICAL_CONSTANTS['MASS_NA'] * np.sum(vel ** 2)
                E_T = PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_impact
                E_out = settings['BETA'] * E_T + (1.0 - settings['BETA']) * E_in
                v_out = np.sqrt(2 * E_out / PHYSICAL_CONSTANTS['MASS_NA']) if E_out > 0 else 0

                norm = pos_impact / np.linalg.norm(pos_impact)
                rebound_dir = transform_local_to_world(sample_lambertian_direction_local(), norm)

                pos = (RM + 1.0) * norm
                vel = v_out * rebound_dir

        pos_start = pos.copy()

    return {'status': 'alive', 'final_state': {'pos': pos, 'vel': vel, 'weight': weight}}


# ==============================================================================
# メイン処理
# ==============================================================================
def main_snapshot_simulation():
    start_time = time.time()

    OUTPUT_DIRECTORY = r"./SimulationResult_202511"

    N_LON_FIXED, N_LAT = 72, 36
    INIT_SURF_DENS = 7.5e14 * (100 ** 2) * 0.0053
    MAIN_DT = 500.0

    SPIN_UP_YEARS = 1.0
    TOTAL_SIM_YEARS = 1.0
    TARGET_TAA = np.arange(0, 360, 1)

    # ★ 安定化策1: ターゲットSPSを増加 (2000)
    TARGET_SPS = {'TD': 1000, 'PSD': 1000, 'SWS': 1000, 'MMV': 1000}

    # ★ 安定化策2: フラックス平滑化の係数
    FLUX_SMOOTHING_ALPHA = 0.05

    GRID_RESOLUTION = 101
    GRID_MAX_RM = 5.0

    F_UV_1AU = 1.5e14 * (100 ** 2)
    Q_PSD = 2.0e-20 / (100 ** 2)
    TEMP_PSD = 1500.0
    TEMP_MMV = 3000.0

    SWS_PARAMS = {
        'FLUX_1AU': 10.0 * 100 ** 3 * 400e3,
        'YIELD': 0.06,
        'U_eV': 0.27,
        'REF_DENS': 7.5e14 * 100 ** 2,
        'LON_RANGE': np.deg2rad([-40, 40]),
        'LAT_N_RANGE': np.deg2rad([30, 60]),
        'LAT_S_RANGE': np.deg2rad([-60, -30]),
    }

    settings = {
        'BETA': 0.5, 'T1AU': 168918.0, 'DT': MAIN_DT,
        'GRID_RADIUS_RM': GRID_MAX_RM + 1.0,
        'USE_SOLAR_GRAVITY': True, 'USE_CORIOLIS_FORCES': True
    }

    run_name = f"Stabilized_v2_Grid{N_LON_FIXED}x{N_LAT}_SPS{TARGET_SPS['TD']}"
    target_output_dir = os.path.join(OUTPUT_DIRECTORY, run_name)
    os.makedirs(target_output_dir, exist_ok=True)
    print(f"結果は '{target_output_dir}' に保存されます。")

    lon_edges = np.linspace(-np.pi, np.pi, N_LON_FIXED + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    dlon = lon_edges[1] - lon_edges[0]
    cell_areas = (PHYSICAL_CONSTANTS['RM'] ** 2) * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))
    surface_density = np.full((N_LON_FIXED, N_LAT), INIT_SURF_DENS, dtype=np.float64)

    # ★ フラックスの移動平均保存用グリッド
    smoothed_incoming_flux = np.zeros((N_LON_FIXED, N_LAT), dtype=np.float64)

    try:
        spec_np = np.loadtxt('SolarSpectrum_Na0.txt', usecols=(0, 3))
        orbit_data = np.loadtxt('orbit2025_v6.txt')
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    wl, gamma = spec_np[:, 0], spec_np[:, 1]
    if wl[1] < wl[0]:
        idx = np.argsort(wl)
        wl, gamma = wl[idx], gamma[idx]

    const_sigma = PHYSICAL_CONSTANTS['E_CHARGE'] ** 2 / (
            4 * PHYSICAL_CONSTANTS['ME'] * PHYSICAL_CONSTANTS['C'] * PHYSICAL_CONSTANTS['EPSILON_0'])
    spec_dict = {
        'wl': wl, 'gamma': gamma,
        'sigma0_perdnu2': const_sigma * 0.641,
        'sigma0_perdnu1': const_sigma * 0.320,
        'JL': 5.18e14
    }

    MERCURY_YEAR = PHYSICAL_CONSTANTS['ORBITAL_PERIOD']
    idx_peri = np.argmin(np.abs(orbit_data[:, 0]))
    t_peri_file = orbit_data[idx_peri, 2]

    t_start_run = t_peri_file
    t_end_run = t_start_run + TOTAL_SIM_YEARS * MERCURY_YEAR
    t_start_spinup = t_start_run - SPIN_UP_YEARS * MERCURY_YEAR

    t_curr = t_start_spinup

    active_particles = []
    prev_taa = -999

    prev_gained_grid_raw = np.zeros_like(surface_density)

    total_steps = int((t_end_run - t_start_spinup) / MAIN_DT)

    with tqdm(total=total_steps, desc="Simulation") as pbar:
        while t_curr < t_end_run:

            TAA, AU, V_rad, V_tan, sub_lon = get_orbital_params_cyclic(t_curr, orbit_data, t_peri_file)

            new_particles = []
            loss_grid = np.zeros_like(surface_density)
            rate_psd = np.zeros_like(surface_density)
            rate_td = np.zeros_like(surface_density)
            rate_sws = np.zeros_like(surface_density)

            # 平衡解適用のためのマスクと値
            na_eq_target_mask = np.zeros((N_LON_FIXED, N_LAT), dtype=bool)
            na_eq_values = np.zeros((N_LON_FIXED, N_LAT), dtype=np.float64)

            f_uv = F_UV_1AU / (AU ** 2)
            sw_flux = SWS_PARAMS['FLUX_1AU'] / (AU ** 2)
            mmv_flux = calculate_mmv_flux(AU)

            # MMV
            n_mmv = mmv_flux * 4 * np.pi * PHYSICAL_CONSTANTS['RM'] ** 2 * MAIN_DT
            w_mmv = max(1.0, n_mmv / TARGET_SPS['MMV'])
            if n_mmv > 0:
                num_p = int(n_mmv / w_mmv)
                if np.random.random() < (n_mmv / w_mmv - num_p): num_p += 1
                for _ in range(num_p):
                    dt_init = MAIN_DT * np.random.random()
                    while True:
                        lr = np.random.uniform(-np.pi, np.pi)
                        if np.random.random() < (1 - 1 / 3 * np.sin(lr)) * 0.75: break
                    latr = np.arcsin(np.random.uniform(-1, 1))
                    pos = lonlat_to_xyz(lr, latr, PHYSICAL_CONSTANTS['RM'])
                    norm = pos / PHYSICAL_CONSTANTS['RM']
                    spd = sample_speed_from_flux_distribution(PHYSICAL_CONSTANTS['MASS_NA'], TEMP_MMV)
                    vel = spd * transform_local_to_world(sample_lambertian_direction_local(), norm)
                    new_particles.append({
                        'pos': pos, 'vel': vel, 'weight': w_mmv, 'dt_remaining': dt_init
                    })

            atoms_psd_step = 0
            atoms_td_step = 0
            atoms_sws_step = 0

            # --- ★ フラックス平滑化 (Smoothing) ---
            # 1. 瞬間的なフラックス (atoms/m^2/s)
            current_instant_flux = (prev_gained_grid_raw / cell_areas[:, np.newaxis].T) / MAIN_DT

            # 2. 指数移動平均 (EMA) で更新
            if t_curr == t_start_spinup:
                smoothed_incoming_flux = current_instant_flux
            else:
                smoothed_incoming_flux = (1.0 - FLUX_SMOOTHING_ALPHA) * smoothed_incoming_flux + \
                                         FLUX_SMOOTHING_ALPHA * current_instant_flux

            # グリッドループ
            for i in range(N_LON_FIXED):
                for j in range(N_LAT):
                    lon_f = (lon_edges[i] + lon_edges[i + 1]) / 2
                    lat_f = (lat_edges[j] + lat_edges[j + 1]) / 2

                    cos_z = np.cos(lat_f) * np.cos(lon_f - sub_lon)
                    if cos_z > 0:
                        rate_psd[i, j] = f_uv * Q_PSD * cos_z

                    temp = calculate_surface_temperature_leblanc(lon_f, lat_f, AU, sub_lon)
                    rate_td[i, j] = calculate_thermal_desorption_rate(temp)

                    lon_sun = (lon_f - sub_lon + np.pi) % (2 * np.pi) - np.pi
                    in_lon = SWS_PARAMS['LON_RANGE'][0] <= lon_sun <= SWS_PARAMS['LON_RANGE'][1]
                    in_lat = (SWS_PARAMS['LAT_N_RANGE'][0] <= lat_f <= SWS_PARAMS['LAT_N_RANGE'][1]) or \
                             (SWS_PARAMS['LAT_S_RANGE'][0] <= lat_f <= SWS_PARAMS['LAT_S_RANGE'][1])
                    if in_lon and in_lat:
                        rate_sws[i, j] = (sw_flux * SWS_PARAMS['YIELD']) / SWS_PARAMS['REF_DENS']

                    rate_total = rate_psd[i, j] + rate_td[i, j] + rate_sws[i, j]
                    if rate_total <= 0: continue

                    # --- ★ 平衡解ロジック (Smooth Fluxを使用) ---
                    timescale = 1.0 / rate_total
                    dens = surface_density[i, j]

                    if timescale <= MAIN_DT and t_curr > t_start_spinup:
                        # ここで平滑化されたフラックスを使用
                        flux_in_smoothed = smoothed_incoming_flux[i, j]
                        dens_eq = flux_in_smoothed / rate_total

                        na_eq_target_mask[i, j] = True
                        na_eq_values[i, j] = dens_eq

                        # そのステップの密度計算には平衡値を一時的に使用
                        dens = dens_eq

                    n_avail = dens * cell_areas[j]
                    n_lost = min(n_avail, n_avail * rate_total * MAIN_DT)

                    loss_grid[i, j] = n_lost

                    atoms_psd_step += n_lost * (rate_psd[i, j] / rate_total)
                    atoms_td_step += n_lost * (rate_td[i, j] / rate_total)
                    atoms_sws_step += n_lost * (rate_sws[i, j] / rate_total)

            w_psd = max(1.0, atoms_psd_step / TARGET_SPS['PSD'])
            w_td = max(1.0, atoms_td_step / TARGET_SPS['TD'])
            w_sws = max(1.0, atoms_sws_step / TARGET_SPS['SWS'])

            # 表面粒子生成
            for i in range(N_LON_FIXED):
                for j in range(N_LAT):
                    n_lost = loss_grid[i, j]
                    if n_lost <= 0: continue

                    r_p, r_t, r_s = rate_psd[i, j], rate_td[i, j], rate_sws[i, j]
                    tot = r_p + r_t + r_s
                    if tot == 0: continue

                    lon_f = (lon_edges[i] + lon_edges[i + 1]) / 2
                    lat_f = (lat_edges[j] + lat_edges[j + 1]) / 2
                    temp = calculate_surface_temperature_leblanc(lon_f, lat_f, AU, sub_lon)

                    params = [
                        ('PSD', n_lost * (r_p / tot), TEMP_PSD, w_psd),
                        ('TD', n_lost * (r_t / tot), temp, w_td),
                        ('SWS', n_lost * (r_s / tot), None, w_sws)
                    ]

                    for p_type, n_amount, T_or_none, w in params:
                        if n_amount <= 0: continue
                        num = int(n_amount / w)
                        if np.random.random() < (n_amount / w - num): num += 1
                        for _ in range(num):
                            dt_init = MAIN_DT * np.random.random()
                            if p_type == 'SWS':
                                E = sample_thompson_sigmund_energy(SWS_PARAMS['U_eV'])
                                spd = np.sqrt(2 * E * PHYSICAL_CONSTANTS['EV_TO_JOULE'] / PHYSICAL_CONSTANTS['MASS_NA'])
                            else:
                                spd = sample_speed_from_flux_distribution(PHYSICAL_CONSTANTS['MASS_NA'], T_or_none)

                            lon_rot = lon_f - sub_lon
                            pos = lonlat_to_xyz(lon_rot, lat_f, PHYSICAL_CONSTANTS['RM'])
                            norm = pos / PHYSICAL_CONSTANTS['RM']
                            vel = spd * transform_local_to_world(sample_lambertian_direction_local(), norm)

                            new_particles.append({
                                'pos': pos, 'vel': vel, 'weight': w, 'dt_remaining': dt_init
                            })

            active_particles.extend(new_particles)

            # 移動計算
            tasks = []
            for p in active_particles:
                dur = p.pop('dt_remaining', MAIN_DT)
                tasks.append({
                    'settings': settings, 'spec': spec_dict, 'particle_state': p,
                    'orbit': (TAA, AU, V_rad, V_tan, sub_lon),
                    'duration': dur
                })

            next_particles = []
            gained_grid_raw = np.zeros_like(surface_density)

            if tasks:
                with Pool(cpu_count() - 1) as pool:
                    results = pool.map(simulate_particle_for_one_step, tasks)

                for res in results:
                    if res['status'] == 'alive':
                        next_particles.append(res['final_state'])
                    elif res['status'] == 'stuck':
                        pos = res['pos_at_impact']
                        w = res['weight']
                        ln = np.arctan2(pos[1], pos[0])
                        lt = np.arcsin(np.clip(pos[2] / np.linalg.norm(pos), -1, 1))
                        ln_fix = (ln + sub_lon + np.pi) % (2 * np.pi) - np.pi
                        ix = np.searchsorted(lon_edges, ln_fix) - 1
                        iy = np.searchsorted(lat_edges, lt) - 1
                        if 0 <= ix < N_LON_FIXED and 0 <= iy < N_LAT:
                            gained_grid_raw[ix, iy] += w

            active_particles = next_particles
            prev_gained_grid_raw = gained_grid_raw.copy()

            # 4. 表面密度更新 (通常計算)
            loss_dens = loss_grid / cell_areas
            gain_dens = gained_grid_raw / cell_areas
            dens_next = surface_density + gain_dens - loss_dens

            # ★ 平衡解適用 (ダンピングなしの直接代入)
            dens_next[na_eq_target_mask] = na_eq_values[na_eq_target_mask]

            surface_density = np.clip(dens_next, 0, None)

            # 5. 保存
            if prev_taa != -999:
                passed = False
                for tgt in TARGET_TAA:
                    if (prev_taa < tgt <= TAA) or (prev_taa > 350 and TAA < 10 and tgt == 0):
                        passed = True
                        break

                if passed and t_curr >= t_start_run:
                    rel_h = (t_curr - t_start_run) / 3600.0
                    print(f" Saving TAA={TAA:.1f}, Time={rel_h:.1f}h, Particles={len(active_particles)}")

                    dgrid = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION, GRID_RESOLUTION), dtype=np.float32)
                    gmin, gmax = -GRID_MAX_RM * PHYSICAL_CONSTANTS['RM'], GRID_MAX_RM * PHYSICAL_CONSTANTS['RM']
                    csize = (gmax - gmin) / GRID_RESOLUTION
                    cvol = csize ** 3

                    pos_arr = np.array([p['pos'] for p in active_particles])
                    weights_arr = np.array([p['weight'] for p in active_particles])

                    if len(pos_arr) > 0:
                        H, _ = np.histogramdd(pos_arr, bins=(GRID_RESOLUTION, GRID_RESOLUTION, GRID_RESOLUTION),
                                              range=[(gmin, gmax), (gmin, gmax), (gmin, gmax)],
                                              weights=weights_arr)
                        dgrid = H.astype(np.float32) / cvol

                    fname_d = f"density_grid_t{int(rel_h):05d}_taa{int(round(TAA)):03d}.npy"
                    fname_s = f"surface_density_t{int(rel_h):05d}_taa{int(round(TAA)):03d}.npy"

                    np.save(os.path.join(target_output_dir, fname_d), dgrid)
                    np.save(os.path.join(target_output_dir, fname_s), surface_density)

            prev_taa = TAA
            t_curr += MAIN_DT
            pbar.update(1)

    print("Done.")


if __name__ == '__main__':
    main_snapshot_simulation()