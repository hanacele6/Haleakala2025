# -*- coding: utf-8 -*-
"""
==============================================================================
プロジェクト: 水星ナトリウム外気圏 3次元モンテカルロシミュレーション
(SSH/ログ出力対応版: TQDM排除)
==============================================================================
"""

import numpy as np
import sys
import os
from multiprocessing import Pool, cpu_count
import time

# ==============================================================================
# 1. 物理定数・天文定数 (SI単位系)
# ==============================================================================
PHYSICAL_CONSTANTS = {
    'PI': np.pi,
    'AU': 1.496e11,
    'MASS_NA': 3.8175e-26,
    'K_BOLTZMANN': 1.380649e-23,
    'GM_MERCURY': 2.2032e13,
    'RM': 2.440e6,
    'C': 299792458.0,
    'H': 6.62607015e-34,
    'E_CHARGE': 1.602e-19,
    'ME': 9.109e-31,
    'EPSILON_0': 8.854e-12,
    'G': 6.6743e-11,
    'MASS_SUN': 1.989e30,
    'EV_TO_JOULE': 1.602e-19,
    'ROTATION_PERIOD': 58.6462 * 86400,
    'ORBITAL_PERIOD': 87.969 * 86400,
}


# ==============================================================================
# 2. 物理モデル・ヘルパー関数群
# ==============================================================================

def calculate_surface_temperature_leblanc(lon_rad, lat_rad, AU, subsolar_lon_rad):
    T0 = 100.0
    T1 = 600.0

    # 修正前: ((0.306 / AU) ** 2)  <-- これが間違いかも（フラックスの減衰率）
    # 修正後: np.sqrt(0.306 / AU)  <-- 温度は距離の平方根に反比例

    scaling = np.sqrt(0.306 / AU)

    cos_theta = np.cos(lat_rad) * np.cos(lon_rad - subsolar_lon_rad)
    if cos_theta <= 0:
        return T0
    # return T0 + T1 * (cos_theta ** 0.25) * ((0.306 / AU) ** 2)
    return T0 + T1 * (cos_theta ** 0.25) * scaling


def calculate_sticking_probability(surface_temp_K):
    A = 0.0804
    B = 458.0
    porosity = 0.8
    if surface_temp_K <= 0: return 1.0
    p_stick = A * np.exp(B / surface_temp_K)
    p_stick_eff = p_stick / (1.0 - (1.0 - p_stick) * porosity)
    return min(p_stick_eff, 1.0)


def calculate_thermal_desorption_rate(surface_temp_K):
    """
    ガウス分布を持つ結合エネルギーUに基づく熱脱離率の実効値を計算する
    Leblanc & Johnson (2003): U = 1.4 ~ 2.7 eV, Mean = 1.85 eV
    """
    if surface_temp_K < 10.0: return 0.0

    # --- 定数設定 ---
    VIB_FREQ = 1e13
    KB = PHYSICAL_CONSTANTS['K_BOLTZMANN']
    EV_J = PHYSICAL_CONSTANTS['EV_TO_JOULE']

    # ガウス分布のパラメータ
    U_MEAN = 1.85
    U_MIN = 1.40
    U_MAX = 2.70
    # 範囲(1.3eV)が概ね 6*sigma (±3sigma) に収まると仮定
    SIGMA = 0.20

    # --- 数値積分 (ベクトル化) ---
    # Uの範囲を分割して計算（50分割あれば十分な精度が出ます）
    u_ev_grid = np.linspace(U_MIN, U_MAX, 50)
    u_joule_grid = u_ev_grid * EV_J

    # 1. 確率密度関数 (Gaussian PDF) の計算
    # P(U) = exp( - (U - mean)^2 / (2 * sigma^2) )
    pdf = np.exp(- (u_ev_grid - U_MEAN) ** 2 / (2 * SIGMA ** 2))

    # 2. 確率の正規化 (合計が1になるようにする)
    pdf_sum = np.sum(pdf)
    if pdf_sum == 0: return 0.0
    norm_pdf = pdf / pdf_sum

    # 3. 各エネルギーでの脱離レート R(U) = v * exp(-U / kT)
    exponent = -u_joule_grid / (KB * surface_temp_K)

    # 指数が小さすぎる(-700以下)とUnderflowするのでマスク処理
    # (ただしnumpyは0になるだけなので、そのままでも概ね動きますが念のため)
    rates = np.zeros_like(u_ev_grid)
    mask = exponent > -700
    rates[mask] = VIB_FREQ * np.exp(exponent[mask])

    # 4. 重み付き平均 (期待値) を計算
    # これが「分布全体としての実効的な放出レート」になります
    effective_rate = np.sum(rates * norm_pdf)

    return effective_rate


def calculate_mmv_flux(AU):
    TOTAL_FLUX_AT_PERI = 5e23
    PERIHELION_AU = 0.307
    AREA = 4 * PHYSICAL_CONSTANTS['PI'] * (PHYSICAL_CONSTANTS['RM'] ** 2)
    avg_flux_peri = TOTAL_FLUX_AT_PERI / AREA
    C = avg_flux_peri * (PERIHELION_AU ** 1.9)
    return C * (AU ** (-1.9))


def sample_speed_from_flux_distribution(mass_kg, temp_k):
    kT = PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_k
    E = np.random.gamma(2.0, kT)
    return np.sqrt(2.0 * E / mass_kg)


def sample_thompson_sigmund_energy(U_eV, E_max_eV=5.0):
    f_max = (U_eV / 2.0) / (U_eV / 2.0 + U_eV) ** 3
    while True:
        E_try = np.random.uniform(0, E_max_eV)
        f_val = E_try / (E_try + U_eV) ** 3
        if np.random.uniform(0, f_max) <= f_val:
            return E_try


def sample_lambertian_direction_local():
    u1, u2 = np.random.random(2)
    phi = 2 * PHYSICAL_CONSTANTS['PI'] * u1
    cos_theta = np.sqrt(1 - u2)
    sin_theta = np.sqrt(u2)
    return np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])


def transform_local_to_world(local_vec, normal_vector):
    local_z = normal_vector / np.linalg.norm(normal_vector)
    world_up = np.array([0., 0., 1.])
    if np.abs(np.dot(local_z, world_up)) > 0.99:
        world_up = np.array([0., 1., 0.])
    local_x = np.cross(world_up, local_z)
    local_x /= np.linalg.norm(local_x)
    local_y = np.cross(local_z, local_x)
    return local_vec[0] * local_x + local_vec[1] * local_y + local_vec[2] * local_z


def get_orbital_params_cyclic(time_sec, orbit_data, t_perihelion_file):
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


# ==============================================================================
# 3. 粒子運動計算エンジン
# ==============================================================================

def _calculate_acceleration(pos, vel, V_radial_ms, V_tangential_ms, AU, spec_data, settings):
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
        F_at_Merc = (JL * 1e13) / (AU ** 2)
        term_d1 = (PHYSICAL_CONSTANTS['H'] / w_na_d1) * sigma0_perdnu1 * \
                  (F_at_Merc * gamma1 * w_na_d1 ** 2 / PHYSICAL_CONSTANTS['C'])
        term_d2 = (PHYSICAL_CONSTANTS['H'] / w_na_d2) * sigma0_perdnu2 * \
                  (F_at_Merc * gamma2 * w_na_d2 ** 2 / PHYSICAL_CONSTANTS['C'])
        b = (term_d1 + term_d2) / PHYSICAL_CONSTANTS['MASS_NA']

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
                E_T = 2 * PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_impact
                E_out = settings['BETA'] * E_T + (1.0 - settings['BETA']) * E_in
                v_out = np.sqrt(2 * E_out / PHYSICAL_CONSTANTS['MASS_NA']) if E_out > 0 else 0
                norm = pos_impact / np.linalg.norm(pos_impact)
                rebound_dir = transform_local_to_world(sample_lambertian_direction_local(), norm)
                pos = (RM + 1.0) * norm
                vel = v_out * rebound_dir
        pos_start = pos.copy()

    return {'status': 'alive', 'final_state': {'pos': pos, 'vel': vel, 'weight': weight}}


# ==============================================================================
# 4. メインルーチン (ログ出力修正版)
# ==============================================================================
def main_snapshot_simulation():
    start_time = time.time()

    # --- 設定 ---
    OUTPUT_DIRECTORY = r"./SimulationResult_202511"
    DT_MOVE = 500.0
    DT_RATE_UPDATE = 500.0
    N_LON_FIXED, N_LAT = 72, 36
    INIT_SURF_DENS = 7.5e14 * (100 ** 2) * 0.0053
    SPIN_UP_YEARS = 2.0
    TOTAL_SIM_YEARS = 1.0
    TARGET_TAA = np.arange(0, 360, 1)
    TARGET_SPS = {'TD': 1000, 'PSD': 1000, 'SWS': 1000, 'MMV': 1000}
    GRID_RESOLUTION = 101
    GRID_MAX_RM = 5.0
    F_UV_1AU = 1.5e14 * (100 ** 2)
    Q_PSD = 1.0e-20 / (100 ** 2)  # ここ変えた
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
        'BETA': 0.5, 'T1AU': 54500.0,  # ここ変えた
        'GRID_RADIUS_RM': GRID_MAX_RM + 1.0,
        'USE_SOLAR_GRAVITY': True, 'USE_CORIOLIS_FORCES': True
    }

    # --- 初期化 ---
    run_name = f"DynamicGrid{N_LON_FIXED}x{N_LAT}_17.1"
    target_output_dir = os.path.join(OUTPUT_DIRECTORY, run_name)
    os.makedirs(target_output_dir, exist_ok=True)
    print(f"Simulation Start. Results: {target_output_dir}")
    print(f"Time Step: Motion={DT_MOVE}s, SurfaceUpdate={DT_RATE_UPDATE}s")

    lon_edges = np.linspace(-np.pi, np.pi, N_LON_FIXED + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    dlon = lon_edges[1] - lon_edges[0]
    cell_areas = (PHYSICAL_CONSTANTS['RM'] ** 2) * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))

    surface_density = np.full((N_LON_FIXED, N_LAT), INIT_SURF_DENS, dtype=np.float64)

    try:
        spec_np = np.loadtxt('SolarSpectrum_Na0.txt', usecols=(0, 3))
        orbit_data = np.loadtxt('orbit2025_v6.txt')
        orbit_data[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(orbit_data[:, 0])))
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

    # キャッシュ & アキュムレータ
    cached_rate_psd = np.zeros(surface_density.shape)
    cached_rate_td = np.zeros(surface_density.shape)
    cached_rate_sws = np.zeros(surface_density.shape)
    cached_loss_rate_grid = np.zeros(surface_density.shape)
    accumulated_gained_grid = np.zeros_like(surface_density)
    time_since_last_update = DT_RATE_UPDATE * 2.0

    # ステップカウンタ設定
    total_steps = int((t_end_run - t_start_spinup) / DT_MOVE)
    step_count = 0
    print(f"Total steps to run: {total_steps}")

    # === ループ開始 (tqdm削除) ===
    while t_curr < t_end_run:
        step_count += 1

        # 1. 軌道更新
        TAA, AU, V_rad, V_tan, sub_lon = get_orbital_params_cyclic(t_curr, orbit_data, t_peri_file)

        # 2. フラックス・表面更新
        time_since_last_update += DT_MOVE
        if time_since_last_update >= DT_RATE_UPDATE:
            dt_accumulated = time_since_last_update
            f_uv = F_UV_1AU / (AU ** 2)
            sw_flux = SWS_PARAMS['FLUX_1AU'] / (AU ** 2)
            mmv_flux = calculate_mmv_flux(AU)

            avg_flux_in = (accumulated_gained_grid / cell_areas) / dt_accumulated
            supply_dens = mmv_flux * dt_accumulated

            temp_rate_psd = np.zeros_like(surface_density)
            temp_rate_td = np.zeros_like(surface_density)
            temp_rate_sws = np.zeros_like(surface_density)
            temp_loss_per_sec = np.zeros_like(surface_density)
            na_eq_record = np.full((N_LON_FIXED, N_LAT), np.nan)

            for i in range(N_LON_FIXED):
                for j in range(N_LAT):
                    lon_f = (lon_edges[i] + lon_edges[i + 1]) / 2
                    lat_f = (lat_edges[j] + lat_edges[j + 1]) / 2
                    cos_z = np.cos(lat_f) * np.cos(lon_f - sub_lon)
                    if cos_z > 0:
                        temp_rate_psd[i, j] = f_uv * Q_PSD * cos_z

                    temp_val = calculate_surface_temperature_leblanc(lon_f, lat_f, AU, sub_lon)
                    temp_rate_td[i, j] = calculate_thermal_desorption_rate(temp_val)

                    lon_sun = (lon_f - sub_lon + np.pi) % (2 * np.pi) - np.pi
                    in_lon = SWS_PARAMS['LON_RANGE'][0] <= lon_sun <= SWS_PARAMS['LON_RANGE'][1]
                    in_lat = (SWS_PARAMS['LAT_N_RANGE'][0] <= lat_f <= SWS_PARAMS['LAT_N_RANGE'][1]) or \
                             (SWS_PARAMS['LAT_S_RANGE'][0] <= lat_f <= SWS_PARAMS['LAT_S_RANGE'][1])
                    if in_lon and in_lat:
                        temp_rate_sws[i, j] = (sw_flux * SWS_PARAMS['YIELD']) / SWS_PARAMS['REF_DENS']

                    rate_total = temp_rate_psd[i, j] + temp_rate_td[i, j] + temp_rate_sws[i, j]
                    dens = surface_density[i, j]
                    timescale = 1.0 / rate_total if rate_total > 0 else float('inf')

                    if timescale <= dt_accumulated and t_curr > t_start_spinup:
                        dens_eq = avg_flux_in[i, j] / rate_total if rate_total > 0 else 0
                        na_eq_record[i, j] = dens_eq
                        dens = dens_eq

                    if not np.isnan(na_eq_record[i, j]):
                        surface_density[i, j] = na_eq_record[i, j]
                    else:
                        loss_dens = dens * rate_total * dt_accumulated  # [atoms/m^2]
                        gain_dens = accumulated_gained_grid[i, j] / cell_areas[j]  # [atoms/m^2]

                        surface_density[i, j] += gain_dens - loss_dens + supply_dens

                    if surface_density[i, j] < 0: surface_density[i, j] = 0
                    temp_loss_per_sec[i, j] = surface_density[i, j] * cell_areas[j] * rate_total

            cached_rate_psd = temp_rate_psd
            cached_rate_td = temp_rate_td
            cached_rate_sws = temp_rate_sws
            cached_loss_rate_grid = temp_loss_per_sec
            accumulated_gained_grid.fill(0.0)
            time_since_last_update = 0.0

        # 3. 粒子生成
        new_particles = []
        # MMV
        mmv_flux = calculate_mmv_flux(AU)
        n_mmv = mmv_flux * 4 * np.pi * PHYSICAL_CONSTANTS['RM'] ** 2 * DT_MOVE
        w_mmv = max(1.0, n_mmv / (TARGET_SPS['MMV'] * (DT_MOVE / DT_RATE_UPDATE)))
        if n_mmv > 0:
            num_p = int(n_mmv / w_mmv)
            if np.random.random() < (n_mmv / w_mmv - num_p): num_p += 1
            for _ in range(num_p):
                dt_init = DT_MOVE * np.random.random()
                while True:
                    lr = np.random.uniform(-np.pi, np.pi)
                    if np.random.random() < (1 - 1 / 3 * np.sin(lr)) * 0.75: break
                latr = np.arcsin(np.random.uniform(-1, 1))
                pos = lonlat_to_xyz(lr, latr, PHYSICAL_CONSTANTS['RM'])
                norm = pos / PHYSICAL_CONSTANTS['RM']
                spd = sample_speed_from_flux_distribution(PHYSICAL_CONSTANTS['MASS_NA'], TEMP_MMV)
                vel = spd * transform_local_to_world(sample_lambertian_direction_local(), norm)
                new_particles.append({'pos': pos, 'vel': vel, 'weight': w_mmv, 'dt_remaining': dt_init})

        # Surface
        total_loss_step = cached_loss_rate_grid * DT_MOVE
        with np.errstate(divide='ignore', invalid='ignore'):
            rate_tot = cached_rate_psd + cached_rate_td + cached_rate_sws
            frac_psd = np.where(rate_tot > 0, cached_rate_psd / rate_tot, 0)
            frac_td = np.where(rate_tot > 0, cached_rate_td / rate_tot, 0)
            frac_sws = np.where(rate_tot > 0, cached_rate_sws / rate_tot, 0)

        atoms_psd_step = np.sum(total_loss_step * frac_psd)
        atoms_td_step = np.sum(total_loss_step * frac_td)
        atoms_sws_step = np.sum(total_loss_step * frac_sws)
        scale_factor = DT_MOVE / DT_RATE_UPDATE
        w_psd = max(1.0, atoms_psd_step / (TARGET_SPS['PSD'] * scale_factor))
        w_td = max(1.0, atoms_td_step / (TARGET_SPS['TD'] * scale_factor))
        w_sws = max(1.0, atoms_sws_step / (TARGET_SPS['SWS'] * scale_factor))

        for i in range(N_LON_FIXED):
            for j in range(N_LAT):
                n_lost = total_loss_step[i, j]
                if n_lost <= 0: continue
                lon_f = (lon_edges[i] + lon_edges[i + 1]) / 2
                lat_f = (lat_edges[j] + lat_edges[j + 1]) / 2
                temp = calculate_surface_temperature_leblanc(lon_f, lat_f, AU, sub_lon)
                params = [
                    ('PSD', n_lost * frac_psd[i, j], TEMP_PSD, w_psd),
                    ('TD', n_lost * frac_td[i, j], temp, w_td),
                    ('SWS', n_lost * frac_sws[i, j], None, w_sws)
                ]
                for p_type, n_amount, T_or_none, w in params:
                    if n_amount <= 0: continue
                    num = int(n_amount / w)
                    if np.random.random() < (n_amount / w - num): num += 1
                    for _ in range(num):
                        dt_init = DT_MOVE * np.random.random()
                        if p_type == 'SWS':
                            E = sample_thompson_sigmund_energy(SWS_PARAMS['U_eV'])
                            spd = np.sqrt(2 * E * PHYSICAL_CONSTANTS['EV_TO_JOULE'] / PHYSICAL_CONSTANTS['MASS_NA'])
                        else:
                            spd = sample_speed_from_flux_distribution(PHYSICAL_CONSTANTS['MASS_NA'], T_or_none)
                        lon_rot = lon_f - sub_lon
                        pos = lonlat_to_xyz(lon_rot, lat_f, PHYSICAL_CONSTANTS['RM'])
                        norm = pos / PHYSICAL_CONSTANTS['RM']
                        vel = spd * transform_local_to_world(sample_lambertian_direction_local(), norm)
                        new_particles.append({'pos': pos, 'vel': vel, 'weight': w, 'dt_remaining': dt_init})

        active_particles.extend(new_particles)

        # 4. 粒子移動
        tasks = []
        for p in active_particles:
            dur = p.pop('dt_remaining', DT_MOVE)
            tasks.append({
                'settings': settings, 'spec': spec_dict, 'particle_state': p,
                'orbit': (TAA, AU, V_rad, V_tan, sub_lon),
                'duration': dur
            })

        next_particles = []
        step_gained_grid = np.zeros_like(surface_density)
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
                        step_gained_grid[ix, iy] += w

        active_particles = next_particles
        accumulated_gained_grid += step_gained_grid

        # 5. 保存
        if prev_taa != -999:
            passed = False
            for tgt in TARGET_TAA:
                if (prev_taa < tgt <= TAA) or (prev_taa > 350 and TAA < 10 and tgt == 0):
                    passed = True;
                    break
            if passed and t_curr >= t_start_run:
                rel_h = (t_curr - t_start_run) / 3600.0
                print(f"[SAVE] TAA={TAA:.1f}, Time={rel_h:.1f}h, Particles={len(active_particles)}")

                dgrid = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION, GRID_RESOLUTION), dtype=np.float32)
                gmin, gmax = -GRID_MAX_RM * PHYSICAL_CONSTANTS['RM'], GRID_MAX_RM * PHYSICAL_CONSTANTS['RM']
                cvol = ((gmax - gmin) / GRID_RESOLUTION) ** 3
                pos_arr = np.array([p['pos'] for p in active_particles])
                weights_arr = np.array([p['weight'] for p in active_particles])
                if len(pos_arr) > 0:
                    H, _ = np.histogramdd(pos_arr, bins=GRID_RESOLUTION, range=[(gmin, gmax)] * 3, weights=weights_arr)
                    dgrid = H.astype(np.float32) / cvol

                fname_d = f"density_grid_t{int(rel_h):05d}_taa{int(round(TAA)):03d}.npy"
                np.save(os.path.join(target_output_dir, fname_d), dgrid)
                np.save(os.path.join(target_output_dir, f"surface_density_t{int(rel_h):05d}.npy"), surface_density)

        # === 進捗ログ出力 (100ステップごと) ===
        if step_count % 100 == 0:
            elapsed = time.time() - start_time
            progress_pct = (step_count / total_steps) * 100
            print(
                f"Step {step_count}/{total_steps} ({progress_pct:.1f}%) | TAA={TAA:.2f} | Particles={len(active_particles)} | Elapsed={elapsed:.1f}s")

        prev_taa = TAA
        t_curr += DT_MOVE

    print("Done. Simulation Completed.")


if __name__ == '__main__':
    sys.modules['__main__'].__spec__ = None
    main_snapshot_simulation()