# -*- coding: utf-8 -*-
"""
==============================================================================
プロジェクト: 水星ナトリウム外気圏 3次元モンテカルロシミュレーション
              (Mercury Sodium Exosphere 3D Monte-Carlo Simulation)

修正履歴 (2025/12/31):
    1. MMV放出率計算(Suzukiモデル)の修正: スケーリングを廃止し、論文の物理式に基づく絶対量計算を実装。
    2. scipy.integrate.quad の導入。
    3. 設定: MMVモデルをデフォルトで 'suzuki' に変更。

作成者: Koki Masaki (Rikkyo Univ.) / Updated by Assistant
==============================================================================
"""

import numpy as np
import sys
import os
from multiprocessing import Pool, cpu_count, freeze_support
import time
from typing import Dict, Tuple, List, Optional, Any
from scipy.integrate import quad  # 追加: 積分用

# ==============================================================================
# 0. シミュレーション設定・物理定数
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
    'A_ORBIT_AU': 0.387098,
}

SIMULATION_SETTINGS = {
    # タイムステップ設定 (秒)
    'DT_MOVE': 500.0,
    'DT_RATE_UPDATE': 500.0,
    'DT_INTEGRATION': 500.0,

    # 温度モデル設定
    'TEMP_BASE': 100.0,
    'TEMP_AMP': 600.0,
    'TEMP_NIGHT': 100.0,

    # グリッド設定
    'N_LON': 72,
    'N_LAT': 36,
    'GRID_RESOLUTION': 101,
    'GRID_MAX_RM': 5.0,
    'GRID_RADIUS_RM': 6.0,

    # 物理プロセス設定
    'BETA': 1.0,
    'T1AU': 54500.0,
    'USE_SOLAR_GRAVITY': True,
    'USE_CORIOLIS_FORCES': True,

    'USE_EQUILIBRIUM_MODE': True,
    'USE_AREA_WEIGHTED_FLUX': False,
    'USE_SUBGRID_SMOOTHING': False,

    # マイクロメテオロイド (MMV) 設定
    # Suzukiモデルの絶対量計算と空間分布を使用する設定に変更
    'MMV_SPATIAL_MODEL': 'uniform',  # 'uniform', 'leblanc', 'suzuki'
    'MMV_FLUX_MODEL': 'suzuki',  # 'leblanc', 'suzuki'
}


# ==============================================================================
# 1. 物理モデル・ヘルパー関数群
# ==============================================================================

def calculate_surface_temperature_leblanc(lon_rad: float, lat_rad: float, AU: float, subsolar_lon_rad: float) -> float:
    """Leblanc et al. モデルに基づく表面温度計算"""
    T_BASE = SIMULATION_SETTINGS['TEMP_BASE']
    T_AMP = SIMULATION_SETTINGS['TEMP_AMP']
    T_NIGHT = SIMULATION_SETTINGS['TEMP_NIGHT']

    scaling = np.sqrt(0.306 / AU)
    cos_theta = np.cos(lat_rad) * np.cos(lon_rad - subsolar_lon_rad)

    if cos_theta <= 0:
        return T_NIGHT
    return T_BASE + T_AMP * (cos_theta ** 0.25) * scaling


def calculate_sticking_probability(surface_temp_K: float) -> float:
    """表面付着確率"""
    A = 0.0804
    B = 458.0
    porosity = 0.8
    if surface_temp_K <= 0: return 1.0
    p_stick = A * np.exp(B / surface_temp_K)
    p_stick_eff = p_stick / (1.0 - (1.0 - p_stick) * porosity)
    return min(p_stick_eff, 1.0)


def calculate_thermal_desorption_rate(surface_temp_K: float) -> float:
    """
    熱脱離率 (TD) 計算: U = 1.85 eV 固定
    """
    if surface_temp_K < 10.0: return 0.0

    VIB_FREQ = 1e13
    KB = PHYSICAL_CONSTANTS['K_BOLTZMANN']
    EV_J = PHYSICAL_CONSTANTS['EV_TO_JOULE']

    U_VAL = 1.85
    exponent = -(U_VAL * EV_J) / (KB * surface_temp_K)

    if exponent < -700:
        return 0.0
    return VIB_FREQ * np.exp(exponent)


def calculate_mmv_total_rate_leblanc(AU: float) -> float:
    """Leblanc et al. (2003) に基づくMMV総放出率 (近日点で5e23に正規化)"""
    TOTAL_FLUX_AT_PERI = 5.0e23
    PERIHELION_AU = 0.307
    return TOTAL_FLUX_AT_PERI * ((PERIHELION_AU / AU) ** 1.9)


def calculate_mmv_total_rate_suzuki(AU: float) -> float:
    """
    Suzuki et al. (2020) に基づくMMV総放出率の絶対値計算
    Eq.8, Eq.9, Eq.10 を使用し、物理パラメータから直接算出する。
    正規化(Scaling)は行わない。
    """
    # 物理定数
    AU_M = PHYSICAL_CONSTANTS['AU']
    GM_SUN = PHYSICAL_CONSTANTS['G'] * PHYSICAL_CONSTANTS['MASS_SUN']
    R_MERCURY = PHYSICAL_CONSTANTS['RM']
    M_NA_KG = PHYSICAL_CONSTANTS['MASS_NA']
    C_NA = 0.0053  # レゴリス中のNa含有率
    A_ORBIT_AU = PHYSICAL_CONSTANTS['A_ORBIT_AU']

    # 1. 衝突速度 v0 [m/s] の計算 (Vis-viva equation)
    r_m = AU * AU_M
    v0 = np.sqrt(GM_SUN * (2.0 / r_m - 1.0 / (A_ORBIT_AU * AU_M)))

    # 2. ダスト数密度 n_mm [m^-3] の計算 (Eq.9 & 10)
    # パラメータ: [JFC, HTC, OCC]
    f = [0.45, 0.50, 0.05]
    chi = [1.00, 1.45, 2.00]
    sigma_deg = [7.0, 33.0, 0.0]
    c_const = [10.3, 2.19, 0.0]

    integrals = []
    # j=1 (JFC)
    s1_rad = np.deg2rad(sigma_deg[0])
    val1, _ = quad(lambda i: c_const[0] * np.exp(-(i) ** 2 / (2 * s1_rad ** 2)), 0, np.pi)
    integrals.append(val1)

    # j=2 (HTC)
    s2_rad = np.deg2rad(sigma_deg[1])
    val2, _ = quad(lambda i: c_const[1] * np.exp(-(i) ** 2 / (2 * s2_rad ** 2)), 0, np.pi)
    integrals.append(val2)

    # j=3 (OCC) 等方性 -> integral = pi/2
    integrals.append(0.5 * np.pi)

    n_mm = 0.0
    for j in range(3):
        # n_mm = sum( f_j * R^-chi_j * Integral ) * 10^-4
        term = f[j] * (AU ** -chi[j]) * integrals[j]
        n_mm += term
    n_mm *= 1.0e-4  # 単位調整 (/m^3)

    # 3. 平均蒸発質量 M_vapor [kg] (Eq.7 近似: 7e-15 * R[au])
    m_vapor = 7.0e-15 * AU

    # 4. 総放出率 Rate [atoms/s] (Eq.8)
    # Rate = (Flux) * (CrossSection) * (Yield)
    flux_events = n_mm * v0
    cross_section = np.pi * R_MERCURY ** 2
    atoms_per_event = m_vapor * C_NA / M_NA_KG

    total_rate = flux_events * cross_section * atoms_per_event
    return total_rate


def get_mmv_spatial_probability(lon_sun_coords: float, lat_rad: float, model: str) -> float:
    """
    MMV粒子の発生位置確率 (空間分布)
    """
    # Ram方向（明け方）からの角度差
    phi_ram = lon_sun_coords - (-np.pi / 2)

    if model == 'suzuki':
        # Suzukiモデル: cos(Z') = cos(lat) * cos(lon_ram) に比例
        # Leading hemisphere (cos Z' > 0) のみ
        cos_Z_prime = np.cos(lat_rad) * np.cos(phi_ram)
        return cos_Z_prime if cos_Z_prime > 0 else 0.0

    elif model == 'leblanc':
        # Leblancモデル: 1 + 1/3 * cos(phi_ram)
        return 1.0 + (1.0 / 3.0) * np.cos(phi_ram)

    else:  # uniform
        return 1.0


def sample_speed_from_flux_distribution(mass_kg: float, temp_k: float) -> float:
    # Maxwell-Boltzmann Flux Distribution
    kT = PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_k
    E = np.random.gamma(2.0, kT)
    return np.sqrt(2.0 * E / mass_kg)


def sample_thompson_sigmund_energy(U_eV: float, E_max_eV: float = 5.0) -> float:
    f_max = (U_eV / 2.0) / (U_eV / 2.0 + U_eV) ** 3
    while True:
        E_try = np.random.uniform(0, E_max_eV)
        f_val = E_try / (E_try + U_eV) ** 3
        if np.random.uniform(0, f_max) <= f_val:
            return E_try


def sample_lambertian_direction_local() -> np.ndarray:
    u1, u2 = np.random.random(2)
    phi = 2 * PHYSICAL_CONSTANTS['PI'] * u1
    cos_theta = np.sqrt(1 - u2)
    sin_theta = np.sqrt(u2)
    return np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])


def transform_local_to_world(local_vec: np.ndarray, normal_vector: np.ndarray) -> np.ndarray:
    local_z = normal_vector / np.linalg.norm(normal_vector)
    world_up = np.array([0., 0., 1.])
    if np.abs(np.dot(local_z, world_up)) > 0.99:
        world_up = np.array([0., 1., 0.])
    local_x = np.cross(world_up, local_z)
    local_x /= np.linalg.norm(local_x)
    local_y = np.cross(local_z, local_x)
    return local_vec[0] * local_x + local_vec[1] * local_y + local_vec[2] * local_z


def get_orbital_params_linear(time_sec: float, orbit_data: np.ndarray, t_perihelion_file: float) -> Tuple[
    float, float, float, float, float]:
    time_col_original = orbit_data[:, 2]
    t_lookup = np.clip(time_sec, time_col_original[0], time_col_original[-1])
    taa_deg = np.interp(t_lookup, time_col_original, orbit_data[:, 0])
    au = np.interp(t_lookup, time_col_original, orbit_data[:, 1])
    v_rad = np.interp(t_lookup, time_col_original, orbit_data[:, 3])
    v_tan = np.interp(t_lookup, time_col_original, orbit_data[:, 4])
    sub_lon_deg = np.interp(t_lookup, time_col_original, orbit_data[:, 5])
    subsolar_lon_rad = np.deg2rad(sub_lon_deg)
    return taa_deg, au, v_rad, v_tan, subsolar_lon_rad


def lonlat_to_xyz(lon_rad: float, lat_rad: float, radius: float) -> np.ndarray:
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return np.array([x, y, z])


# ==============================================================================
# 2. 粒子運動計算エンジン
# ==============================================================================

def _calculate_acceleration(pos: np.ndarray, vel: np.ndarray, V_radial_ms: float, V_tangential_ms: float, AU: float,
                            spec_data: Dict, settings: Dict) -> np.ndarray:
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

    accel_cen = np.zeros(3)
    accel_cor = np.zeros(3)
    if settings.get('USE_CORIOLIS_FORCES', False):
        if r0 > 0:
            omega_val = V_tangential_ms / r0
            omega_sq = omega_val ** 2
            accel_cen = np.array([(omega_val ** 2) * (pos[0] - r0), omega_sq * pos[1], 0.0])
            two_omega = 2 * omega_val
            accel_cor = np.array([two_omega * vel[1], -two_omega * vel[0], 0.0])

    return accel_srp + accel_g + accel_sun + accel_cen + accel_cor


def simulate_particle_for_one_step(args: Dict) -> Dict:
    # 1粒子のRK4積分 (並列処理用関数)
    settings, spec_data = args['settings'], args['spec']
    TAA, AU, V_rad, V_tan, subsolar_lon = args['orbit']
    time_remaining = args['duration']
    MAX_DT_STEP = settings['DT_INTEGRATION']
    pos = args['particle_state']['pos'].copy()
    vel = args['particle_state']['vel'].copy()
    weight = args['particle_state']['weight']
    RM = PHYSICAL_CONSTANTS['RM']
    R_MAX_COEFF = settings.get('GRID_RADIUS_RM', 6.0)
    R_MAX = RM * R_MAX_COEFF
    tau_ion = settings['T1AU'] * AU ** 2

    while time_remaining > 1e-6:
        dt_this_loop = min(time_remaining, MAX_DT_STEP)

        # 光イオン化
        if pos[0] > 0 and np.random.random() < (1.0 - np.exp(-dt_this_loop / tau_ion)):
            return {'status': 'ionized', 'final_state': None}

        # 衝突予測
        r_vec_now = pos
        r_mag_now = np.linalg.norm(r_vec_now)
        n_vec = r_vec_now / r_mag_now
        g_mag = PHYSICAL_CONSTANTS['GM_MERCURY'] / (r_mag_now ** 2)
        v_rad = np.dot(vel, n_vec)
        t_hit_est = float('inf')

        if v_rad > 0:
            t_hit_est = 2.0 * v_rad / g_mag
        elif r_mag_now > RM:
            val_c = r_mag_now - RM
            if val_c < 1000.0:
                term_sq = v_rad ** 2 + 2.0 * g_mag * val_c
                if term_sq >= 0:
                    t_hit_est = (np.abs(v_rad) + np.sqrt(term_sq)) / g_mag

        if t_hit_est < dt_this_loop:
            t_flight = t_hit_est
            acc_vec_approx = -n_vec * g_mag
            pos_hit = pos + vel * t_flight + 0.5 * acc_vec_approx * (t_flight ** 2)
            vel_hit = vel + acc_vec_approx * t_flight
            pos_hit = pos_hit * (RM / np.linalg.norm(pos_hit))
            time_remaining -= t_flight
            if time_remaining < 0: time_remaining = 0.0

            lon_rot = np.arctan2(pos_hit[1], pos_hit[0])
            lat_rot = np.arcsin(np.clip(pos_hit[2] / RM, -1, 1))
            lon_fixed = (lon_rot + subsolar_lon + np.pi) % (2 * np.pi) - np.pi
            temp_impact = calculate_surface_temperature_leblanc(lon_fixed, lat_rot, AU, subsolar_lon)

            if np.random.random() < calculate_sticking_probability(temp_impact):
                return {'status': 'stuck', 'pos_at_impact': pos_hit, 'weight': weight}

            E_in = 0.5 * PHYSICAL_CONSTANTS['MASS_NA'] * np.sum(vel_hit ** 2)
            E_T = 2 * PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_impact
            E_out = settings['BETA'] * E_T + (1.0 - settings['BETA']) * E_in
            v_out = np.sqrt(2 * E_out / PHYSICAL_CONSTANTS['MASS_NA'])
            norm_hit = pos_hit / RM
            rebound_dir = transform_local_to_world(sample_lambertian_direction_local(), norm_hit)
            pos = (RM + 1.0) * norm_hit
            vel = v_out * rebound_dir
            continue

        else:
            dt = dt_this_loop
            k1_v = dt * _calculate_acceleration(pos, vel, V_rad, V_tan, AU, spec_data, settings)
            k1_p = dt * vel
            k2_v = dt * _calculate_acceleration(pos + 0.5 * k1_p, vel + 0.5 * k1_v, V_rad, V_tan, AU, spec_data,
                                                settings)
            k2_p = dt * (vel + 0.5 * k1_v)
            k3_v = dt * _calculate_acceleration(pos + 0.5 * k2_p, vel + 0.5 * k2_v, V_rad, V_tan, AU, spec_data,
                                                settings)
            k3_p = dt * (vel + 0.5 * k2_v)
            k4_v = dt * _calculate_acceleration(pos + k3_p, vel + k3_v, V_rad, V_tan, AU, spec_data, settings)
            k4_p = dt * (vel + k3_v)

            pos_next = pos + (k1_p + 2 * k2_p + 2 * k3_p + k4_p) / 6.0
            vel_next = vel + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6.0
            r_next = np.linalg.norm(pos_next)

            if r_next > R_MAX:
                return {'status': 'escaped', 'final_state': None}

            if r_next <= RM:
                time_remaining -= dt
                pos_hit = pos_next * (RM / r_next)
                lon_rot = np.arctan2(pos_hit[1], pos_hit[0])
                lat_rot = np.arcsin(np.clip(pos_hit[2] / RM, -1, 1))
                lon_fixed = (lon_rot + subsolar_lon + np.pi) % (2 * np.pi) - np.pi
                temp_impact = calculate_surface_temperature_leblanc(lon_fixed, lat_rot, AU, subsolar_lon)

                if np.random.random() < calculate_sticking_probability(temp_impact):
                    return {'status': 'stuck', 'pos_at_impact': pos_hit, 'weight': weight}

                E_in = 0.5 * PHYSICAL_CONSTANTS['MASS_NA'] * np.sum(vel_next ** 2)
                E_T = 2 * PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_impact
                E_out = settings['BETA'] * E_T + (1.0 - settings['BETA']) * E_in
                v_out = np.sqrt(2 * E_out / PHYSICAL_CONSTANTS['MASS_NA'])
                norm_hit = pos_hit / RM
                rebound_dir = transform_local_to_world(sample_lambertian_direction_local(), norm_hit)
                pos = (RM + 1.0) * norm_hit
                vel = v_out * rebound_dir
                continue

            pos = pos_next
            vel = vel_next
            time_remaining -= dt

    return {'status': 'alive', 'final_state': {'pos': pos, 'vel': vel, 'weight': weight}}


# ==============================================================================
# 3. メインルーチン
# ==============================================================================
def main_snapshot_simulation():
    start_time = time.time()
    DT_MOVE = SIMULATION_SETTINGS['DT_MOVE']
    DT_RATE_UPDATE = SIMULATION_SETTINGS['DT_RATE_UPDATE']
    N_LON_FIXED = SIMULATION_SETTINGS['N_LON']
    N_LAT = SIMULATION_SETTINGS['N_LAT']
    GRID_RESOLUTION = SIMULATION_SETTINGS['GRID_RESOLUTION']
    GRID_MAX_RM = SIMULATION_SETTINGS['GRID_MAX_RM']
    OUTPUT_DIRECTORY = r"./SimulationResult_202512"

    INIT_SURF_DENS = 7.5e14 * (100 ** 2) * 0.0053
    SPIN_UP_YEARS = 2.0
    TOTAL_SIM_YEARS = 1.0
    TARGET_TAA = np.arange(0, 360, 1)

    TARGET_SPS = {'TD': 1000, 'PSD': 1000, 'SWS': 1000, 'MMV': 1000}
    F_UV_1AU = 1.5e14 * (100 ** 2)
    Q_PSD = 1.0e-20 / (100 ** 2)
    TEMP_PSD = 1500.0
    TEMP_MMV = 3000.0
    SWS_PARAMS = {
        'FLUX_1AU': 10.0 * 100 ** 3 * 400e3 * 4,
        'YIELD': 0.06,
        'U_eV': 0.27,
        'REF_DENS': 7.5e14 * 100 ** 2,
        'LON_RANGE': np.deg2rad([-40, 40]),
        'LAT_N_RANGE': np.deg2rad([20, 80]),
        'LAT_S_RANGE': np.deg2rad([-20, -80]),
    }

    mode_str = "EqMode" if SIMULATION_SETTINGS['USE_EQUILIBRIUM_MODE'] else "NoEq"
    run_name = f"ParabolicHop_{N_LON_FIXED}x{N_LAT}_{mode_str}_DT{int(DT_MOVE)}_PSuzuki_DKillen"
    target_output_dir = os.path.join(OUTPUT_DIRECTORY, run_name)
    os.makedirs(target_output_dir, exist_ok=True)
    print(f"Simulation Start. Results: {target_output_dir}")
    print(f"Updates: MMV Model = Suzuki (Absolute Calculation with No Scaling), TD U=1.85eV")

    lon_edges = np.linspace(-np.pi, np.pi, N_LON_FIXED + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    dlon = lon_edges[1] - lon_edges[0]
    cell_areas = (PHYSICAL_CONSTANTS['RM'] ** 2) * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))

    surface_density = np.full((N_LON_FIXED, N_LAT), INIT_SURF_DENS, dtype=np.float64)

    try:
        spec_np = np.loadtxt('SolarSpectrum_Na0.txt', usecols=(0, 3))
        orbit_data = np.loadtxt('orbit2025_spice_unwrapped.txt')
        orbit_data[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(orbit_data[:, 0])))
        orbit_data[:, 5] = np.rad2deg(np.unwrap(np.deg2rad(orbit_data[:, 5])))
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
    t_file_start = orbit_data[0, 2]
    t_start_spinup = t_file_start
    t_start_run = t_start_spinup + SPIN_UP_YEARS * MERCURY_YEAR
    t_end_run = t_start_run + TOTAL_SIM_YEARS * MERCURY_YEAR
    t_curr = t_start_spinup
    t_peri_file = t_file_start

    active_particles = []
    prev_taa = -999

    cached_rate_psd = np.zeros(surface_density.shape)
    cached_rate_td = np.zeros(surface_density.shape)
    cached_rate_sws = np.zeros(surface_density.shape)
    cached_loss_rate_grid = np.zeros(surface_density.shape)
    accumulated_gained_grid = np.zeros_like(surface_density)

    time_since_last_update = DT_RATE_UPDATE * 2.0
    total_steps = int((t_end_run - t_start_spinup) / DT_MOVE)
    step_count = 0

    half_grid_width_rad = dlon / 2.0
    sin_half_width = np.sin(half_grid_width_rad)

    num_processes = max(1, cpu_count() - 1)
    print(f"Starting Multiprocessing Pool with {num_processes} workers...")

    with Pool(processes=num_processes) as pool:

        while t_curr < t_end_run:
            step_count += 1
            TAA_raw, AU, V_rad, V_tan, sub_lon = get_orbital_params_linear(t_curr, orbit_data, t_peri_file)
            TAA = TAA_raw % 360.0
            time_since_last_update += DT_MOVE

            # ==================================================================
            # A. 表面放出率マップの更新
            # ==================================================================
            if time_since_last_update >= DT_RATE_UPDATE:
                dt_accumulated = time_since_last_update
                f_uv = F_UV_1AU / (AU ** 2)
                sw_flux = SWS_PARAMS['FLUX_1AU'] / (AU ** 2)
                supply_dens = 0

                scaling = np.sqrt(0.306 / AU)
                temp_rate_psd = np.zeros_like(surface_density)
                temp_rate_td = np.zeros_like(surface_density)
                temp_rate_sws = np.zeros_like(surface_density)
                temp_loss_per_sec = np.zeros_like(surface_density)

                for i in range(N_LON_FIXED):
                    for j in range(N_LAT):
                        lon_f = (lon_edges[i] + lon_edges[i + 1]) / 2
                        lat_f = (lat_edges[j] + lat_edges[j + 1]) / 2
                        cos_z_center = np.cos(lat_f) * np.cos(lon_f - sub_lon)

                        illum_frac = 0.0
                        if cos_z_center > sin_half_width:
                            illum_frac = 1.0
                        elif cos_z_center < -sin_half_width:
                            illum_frac = 0.0
                        else:
                            illum_frac = np.clip((cos_z_center + sin_half_width) / (2 * sin_half_width), 0, 1)

                        if not SIMULATION_SETTINGS['USE_AREA_WEIGHTED_FLUX'] and not SIMULATION_SETTINGS[
                            'USE_SUBGRID_SMOOTHING']:
                            illum_frac = 1.0 if cos_z_center > 0 else 0.0

                        eff_cos = max(0.0, cos_z_center)
                        T_day_potential = SIMULATION_SETTINGS['TEMP_BASE'] + \
                                          SIMULATION_SETTINGS['TEMP_AMP'] * (eff_cos ** 0.25) * scaling

                        if illum_frac > 0:
                            temp_rate_psd[i, j] = f_uv * Q_PSD * eff_cos * illum_frac

                        if SIMULATION_SETTINGS['USE_AREA_WEIGHTED_FLUX']:
                            rate_day = calculate_thermal_desorption_rate(T_day_potential)
                            rate_night = calculate_thermal_desorption_rate(SIMULATION_SETTINGS['TEMP_NIGHT'])
                            temp_rate_td[i, j] = rate_day * illum_frac + rate_night * (1.0 - illum_frac)
                        else:
                            temp_val = T_day_potential if illum_frac > 0.5 else SIMULATION_SETTINGS['TEMP_NIGHT']
                            temp_rate_td[i, j] = calculate_thermal_desorption_rate(temp_val)

                        lon_sun = (lon_f - sub_lon + np.pi) % (2 * np.pi) - np.pi
                        in_lon = SWS_PARAMS['LON_RANGE'][0] <= lon_sun <= SWS_PARAMS['LON_RANGE'][1]
                        in_lat = (SWS_PARAMS['LAT_N_RANGE'][0] <= lat_f <= SWS_PARAMS['LAT_N_RANGE'][1]) or \
                                 (SWS_PARAMS['LAT_S_RANGE'][0] <= lat_f <= SWS_PARAMS['LAT_S_RANGE'][1])
                        if in_lon and in_lat:
                            temp_rate_sws[i, j] = (sw_flux * SWS_PARAMS['YIELD']) / SWS_PARAMS['REF_DENS']

                        rate_total = temp_rate_psd[i, j] + temp_rate_td[i, j] + temp_rate_sws[i, j]
                        current_dens = surface_density[i, j]
                        gain_dens = accumulated_gained_grid[i, j] / cell_areas[j]
                        total_input_dens = gain_dens + supply_dens

                        decay_factor = np.exp(-rate_total * dt_accumulated)
                        term_decay = current_dens * decay_factor

                        if rate_total > 1e-30:
                            gain_per_sec = total_input_dens / dt_accumulated
                            term_source = (gain_per_sec / rate_total) * (1.0 - decay_factor)
                        else:
                            term_source = total_input_dens

                        new_dens = term_decay + term_source
                        actual_loss_dens = (current_dens + total_input_dens) - new_dens

                        surface_density[i, j] = max(0.0, new_dens)
                        temp_loss_per_sec[i, j] = actual_loss_dens * cell_areas[j] / dt_accumulated

                cached_rate_psd = temp_rate_psd
                cached_rate_td = temp_rate_td
                cached_rate_sws = temp_rate_sws
                cached_loss_rate_grid = temp_loss_per_sec
                accumulated_gained_grid.fill(0.0)
                time_since_last_update = 0.0

            # ==================================================================
            # B. 粒子の生成
            # ==================================================================
            new_particles = []

            # MMVの放出量計算 (モデル切り替え対応)
            if SIMULATION_SETTINGS['MMV_FLUX_MODEL'] == 'suzuki':
                total_mmv_rate = calculate_mmv_total_rate_suzuki(AU)
            else:
                total_mmv_rate = calculate_mmv_total_rate_leblanc(AU)

            n_mmv = total_mmv_rate * DT_MOVE
            w_mmv = max(1.0, n_mmv / (TARGET_SPS['MMV'] * (DT_MOVE / DT_RATE_UPDATE)))

            if n_mmv > 0:
                num_p = int(n_mmv / w_mmv)
                if np.random.random() < (n_mmv / w_mmv - num_p): num_p += 1
                for _ in range(num_p):
                    dt_init = DT_MOVE * np.random.random()
                    while True:
                        lr = np.random.uniform(-np.pi, np.pi)
                        latr = np.arcsin(np.random.uniform(-1, 1))
                        # 空間分布判定 (Suzukiモデル対応)
                        lon_from_sun = lr
                        prob = get_mmv_spatial_probability(lon_from_sun, latr, SIMULATION_SETTINGS['MMV_SPATIAL_MODEL'])

                        # 棄却法: 確率は最大1.0なので、probより大きければ採用
                        # Leblanc(1+1/3cos)の場合は最大4/3なので1.5で割るなど調整が必要
                        # Suzuki(cos*cos)は最大1.0
                        norm_factor = 1.5 if SIMULATION_SETTINGS['MMV_SPATIAL_MODEL'] == 'leblanc' else 1.0
                        if np.random.random() < prob / norm_factor:
                            break

                    pos = lonlat_to_xyz(lr, latr, PHYSICAL_CONSTANTS['RM'])
                    norm = pos / PHYSICAL_CONSTANTS['RM']
                    spd = sample_speed_from_flux_distribution(PHYSICAL_CONSTANTS['MASS_NA'], TEMP_MMV)
                    vel = spd * transform_local_to_world(sample_lambertian_direction_local(), norm)
                    new_particles.append({'pos': pos, 'vel': vel, 'weight': w_mmv, 'dt_remaining': dt_init})

            total_loss_step = cached_loss_rate_grid * DT_MOVE
            with np.errstate(divide='ignore', invalid='ignore'):
                rate_tot = cached_rate_psd + cached_rate_td + cached_rate_sws
                frac_psd = np.where(rate_tot > 0, cached_rate_psd / rate_tot, 0)
                frac_td = np.where(rate_tot > 0, cached_rate_td / rate_tot, 0)
                frac_sws = np.where(rate_tot > 0, cached_rate_sws / rate_tot, 0)

            scale_factor = DT_MOVE / DT_RATE_UPDATE
            atoms_psd = np.sum(total_loss_step * frac_psd)
            atoms_td = np.sum(total_loss_step * frac_td)
            atoms_sws = np.sum(total_loss_step * frac_sws)

            w_psd = max(1.0, atoms_psd / (TARGET_SPS['PSD'] * scale_factor))
            w_td = max(1.0, atoms_td / (TARGET_SPS['TD'] * scale_factor))
            w_sws = max(1.0, atoms_sws / (TARGET_SPS['SWS'] * scale_factor))

            for i in range(N_LON_FIXED):
                for j in range(N_LAT):
                    n_lost = total_loss_step[i, j]
                    if n_lost <= 0: continue
                    lon_f = (lon_edges[i] + lon_edges[i + 1]) / 2
                    lat_f = (lat_edges[j] + lat_edges[j + 1]) / 2
                    cos_z = np.cos(lat_f) * np.cos(lon_f - sub_lon)
                    eff_cos = max(0.0, cos_z)
                    T_day_potential = SIMULATION_SETTINGS['TEMP_BASE'] + \
                                      SIMULATION_SETTINGS['TEMP_AMP'] * (eff_cos ** 0.25) * scaling
                    illum_frac = 1.0 if cos_z > 0 else 0.0

                    if SIMULATION_SETTINGS['USE_AREA_WEIGHTED_FLUX']:
                        temp_eff_for_vel = T_day_potential if illum_frac > 0 else SIMULATION_SETTINGS['TEMP_NIGHT']
                    else:
                        temp_eff_for_vel = T_day_potential if illum_frac > 0.5 else SIMULATION_SETTINGS['TEMP_NIGHT']

                    params = [
                        ('PSD', n_lost * frac_psd[i, j], TEMP_PSD, w_psd),
                        ('TD', n_lost * frac_td[i, j], temp_eff_for_vel, w_td),
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

            # ==================================================================
            # C. 粒子の移動 (Multiprocessing)
            # ==================================================================
            tasks = []
            for p in active_particles:
                dur = p.pop('dt_remaining', DT_MOVE)
                tasks.append({
                    'settings': SIMULATION_SETTINGS, 'spec': spec_dict, 'particle_state': p,
                    'orbit': (TAA, AU, V_rad, V_tan, sub_lon),
                    'duration': dur
                })

            next_particles = []
            step_gained_grid = np.zeros_like(surface_density)

            if tasks:
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

            # ==================================================================
            # D. データ保存
            # ==================================================================
            if prev_taa != -999:
                passed = False
                for tgt in TARGET_TAA:
                    if (prev_taa < tgt <= TAA) or (prev_taa > 350 and TAA < 10 and tgt == 0):
                        passed = True
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
                        H, _ = np.histogramdd(pos_arr, bins=GRID_RESOLUTION, range=[(gmin, gmax)] * 3,
                                              weights=weights_arr)
                        dgrid = H.astype(np.float32) / cvol

                    fname_d = f"density_grid_t{int(rel_h):05d}_taa{int(round(TAA)):03d}.npy"
                    np.save(os.path.join(target_output_dir, fname_d), dgrid)
                    np.save(os.path.join(target_output_dir, f"surface_density_t{int(rel_h):05d}.npy"), surface_density)

            if step_count % 100 == 0:
                elapsed = time.time() - start_time
                progress_pct = (step_count / total_steps) * 100
                print(
                    f"Step {step_count}/{total_steps} ({progress_pct:.1f}%) | TAA={TAA:.2f} | Particles={len(active_particles)} | Elapsed={elapsed:.1f}s")

            prev_taa = TAA
            t_curr += DT_MOVE

    print("Done. Simulation Completed.")


if __name__ == '__main__':
    freeze_support()
    sys.modules['__main__'].__spec__ = None
    main_snapshot_simulation()