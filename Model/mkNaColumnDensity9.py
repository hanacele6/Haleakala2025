# -*- coding: utf-8 -*-
"""
==============================================================================
プロジェクト: 水星ナトリウム外気圏 3次元モンテカルロシミュレーション
              (Mercury Sodium Exosphere 3D Monte-Carlo Simulation)

概要:
    水星表面からのナトリウム放出と、外気圏における粒子の運動を計算する。

更新内容:
    - Budget Analysis (生成・消滅の内訳集計) 機能を追加
      TAAごとの生成(PSD/TD/SWS/MMV)と消滅(Stuck/Ionized/Escaped)をCSV出力。
    - [New] Multi-Bin Binding Energy Model (マルチビン束縛エネルギーモデル) の導入
      表面のナトリウムを複数の束縛エネルギー(U)のビンとして管理し、それぞれで
      独立した放出率(PSD/TD)を計算。
    - [New] 物理的な滞在時間(tau_TD)に基づく自然なマルチバウンド(即時脱離)モデルへ移行。

作成者: Koki Masaki (Rikkyo Univ.)
日付: 2026/01/28 (Updated: Multi-Bin Model Integration)
==============================================================================
"""

import numpy as np
import sys
import os
from multiprocessing import Pool, cpu_count
import time
from typing import Dict, Tuple, List, Optional, Any
import csv  # [追加] CSV出力用

# ==============================================================================
# 0. シミュレーション設定・物理定数 (一元管理)
# ==============================================================================

# 物理定数
PHYSICAL_CONSTANTS = {
    'PI': np.pi,
    'AU': 1.496e11,  # 1天文単位 [m]
    'MASS_NA': 3.8175e-26,  # ナトリウム原子質量 [kg]
    'K_BOLTZMANN': 1.380649e-23,  # ボルツマン定数 [J/K]
    'GM_MERCURY': 2.2032e13,  # 水星重力定数 (G * M_Mercury) [m^3/s^2]
    'RM': 2.440e6,  # 水星半径 [m]
    'C': 299792458.0,  # 光速 [m/s]
    'H': 6.62607015e-34,  # プランク定数 [J s]
    'E_CHARGE': 1.602e-19,  # 素電荷 [C]
    'ME': 9.109e-31,  # 電子質量 [kg]
    'EPSILON_0': 8.854e-12,  # 真空の誘電率 [F/m]
    'G': 6.6743e-11,  # 万有引力定数 [m^3 kg^-1 s^-2]
    'MASS_SUN': 1.989e30,  # 太陽質量 [kg]
    'EV_TO_JOULE': 1.602e-19,  # eV -> Joule 変換係数
    'ROTATION_PERIOD': 58.6462 * 86400,  # 自転周期 [s]
    'ORBITAL_PERIOD': 87.969 * 86400,  # 公転周期 [s]

    # [追加] 軌道計算用 (基準距離算出のため)
    'MERCURY_SEMI_MAJOR_AXIS_AU': 0.387098,
    'MERCURY_ECCENTRICITY': 0.205630,
}

# 統合シミュレーション設定
SIMULATION_SETTINGS = {
    # --- 時間ステップ設定 ---
    'DT_MOVE': 100.0,  # 粒子の位置更新ステップ [s]
    'DT_RATE_UPDATE': 100.0,  # 表面放出率の再計算ステップ [s]
    'DT_INTEGRATION': 100.0,  # 粒子軌跡計算の内部積分ステップ

    # --- 温度モデル設定 (Leblanc et al.) ---
    'TEMP_BASE': 100.0,
    'TEMP_AMP': 600.0,
    'TEMP_NIGHT': 100.0,

    # --- グリッド・領域設定 ---
    'N_LON': 72,
    'N_LAT': 36,
    'GRID_RESOLUTION': 101,
    'GRID_MAX_RM': 5.0,
    'GRID_RADIUS_RM': 6.0,

    # --- 物理フラグ ---
    'BETA': 1.0,
    'T1AU': 134000.0,
    'USE_SOLAR_GRAVITY': True,
    'USE_CORIOLIS_FORCES': True,

    # --- 計算モード ---
    'USE_EQUILIBRIUM_MODE': False,
    'USE_AREA_WEIGHTED_FLUX': False,
    'USE_SUBGRID_SMOOTHING': False,

    # --- [拡散モデル設定] ---
    'USE_STD_DIFFUSION': True,  # 標準拡散 (距離に応じてフル変動)
    'USE_CLAMPED_DIFFUSION': False,  # 近日点ピークカット拡散 (現在は基本使用していない)

    # ==========================================================================
    # マルチビン束縛エネルギー設定 (Multi-Bin Binding Energy)
    # 拡張性を持たせるため、束縛エネルギーとPSD断面積を配列で定義します。
    # インデックス 0: 浅いサイト (吸着用)
    # インデックス 1: 深いサイト (拡散供給用)
    # ==========================================================================
    'U_BINS': np.array([1.85, 2.7]),
    # 'Q_PSD_BINS': np.array([1.0e-20 / (100 ** 2), 2.7e-21 / (100 ** 2)]),
    'Q_PSD_BINS': np.array([2.7e-21 / (100 ** 2), 2.7e-21 / (100 ** 2)]),
}

# サイトのインデックス定義
IDX_SHALLOW = 0
IDX_DEEP = 1
N_BINS = len(SIMULATION_SETTINGS['U_BINS'])

# 定数計算用
KB_EV_CONST = 8.617e-5  # ボルツマン定数 [eV/K]

# ==============================================================================
# [A] Diffusion Model Parameters
# ==============================================================================
DIFF_REF_FLUX = 2.0e7 * (100.0 ** 2)
DIFF_REF_TEMP = 700.0  # 基準温度 [K]
DIFF_E_A_EV = 0.4  # 活性化エネルギー [eV]
Target_Grain_Radius = 100.0e-6  # [m]

# 頻度因子 A (J0) の事前計算
DIFF_PRE_FACTOR = DIFF_REF_FLUX / np.exp(-DIFF_E_A_EV / (KB_EV_CONST * DIFF_REF_TEMP))
print(f"Diffusion Parameters: Ea={DIFF_E_A_EV}eV, RefFlux={DIFF_REF_FLUX:.1e} at {DIFF_REF_TEMP}K")

# ==============================================================================
# [B] Clamped (Peak-Cut) Diffusion Settings
# ==============================================================================
TAA_CLAMP_START = 70.0  # これより小さいTAA (0~70) は 70度の距離に固定
TAA_CLAMP_END = 290.0  # これより大きいTAA (290~360) は 290度の距離に固定


def calculate_au_at_taa(taa_deg: float) -> float:
    a = PHYSICAL_CONSTANTS['MERCURY_SEMI_MAJOR_AXIS_AU']
    e = PHYSICAL_CONSTANTS['MERCURY_ECCENTRICITY']
    rad = np.deg2rad(taa_deg)
    r = a * (1 - e ** 2) / (1 + e * np.cos(rad))
    return r


AU_AT_CUTOFF = calculate_au_at_taa(TAA_CLAMP_START)
FORCED_INJECTION_EVENTS = []


# ==============================================================================
# 1. 物理モデル・ヘルパー関数群
# ==============================================================================

def assign_sticking_bin() -> int:
    """
    [New] 吸着サイト決定関数
    外気圏から落下してきた粒子がどの束縛エネルギーのビンに入るかを決定する。
    現在はすべて浅いサイト（1.6 eV）に入ると仮定。
    将来的に分布を持たせたい場合は、この関数内でルーレット処理などを記述する。
    """
    return IDX_SHALLOW


def calculate_surface_temperature_leblanc(lon_rad: float, lat_rad: float, AU: float, subsolar_lon_rad: float) -> float:
    T_BASE = SIMULATION_SETTINGS['TEMP_BASE']
    T_AMP = SIMULATION_SETTINGS['TEMP_AMP']
    T_NIGHT = SIMULATION_SETTINGS['TEMP_NIGHT']

    scaling = np.sqrt(0.306 / AU)
    cos_theta = np.cos(lat_rad) * np.cos(lon_rad - subsolar_lon_rad)

    if cos_theta <= 0: return T_NIGHT
    return T_BASE + T_AMP * (cos_theta ** 0.25) * scaling


def calculate_thermal_desorption_rate(surface_temp_K: float, U_eff_eV: float) -> float:
    """
    [Update] 引数として束縛エネルギー(U_eff_eV)を受け取り、ビンごとに計算できるように変更
    """
    if surface_temp_K < 10.0: return 0.0
    VIB_FREQ = 1e13
    KB = PHYSICAL_CONSTANTS['K_BOLTZMANN']
    EV_J = PHYSICAL_CONSTANTS['EV_TO_JOULE']

    U_JOULE = U_eff_eV * EV_J
    exponent = -U_JOULE / (KB * surface_temp_K)
    if exponent < -700: return 0.0
    return VIB_FREQ * np.exp(exponent)


def calculate_mmv_flux(AU: float) -> float:
    TOTAL_FLUX_AT_PERI = 5e23
    PERIHELION_AU = 0.307
    AREA = 4 * PHYSICAL_CONSTANTS['PI'] * (PHYSICAL_CONSTANTS['RM'] ** 2)
    avg_flux_peri = TOTAL_FLUX_AT_PERI / AREA
    C = avg_flux_peri * (PERIHELION_AU ** 1.9)
    return C * (AU ** (-1.9))


def sample_speed_from_flux_distribution(mass_kg: float, temp_k: float) -> float:
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
    return taa_deg, au, v_rad, v_tan, np.deg2rad(sub_lon_deg)


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

    # 1. 放射圧
    velocity_for_doppler = vel[0] - V_radial_ms
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

    # 2. 水星重力
    r_sq = np.sum(pos ** 2)
    accel_g = -PHYSICAL_CONSTANTS['GM_MERCURY'] * pos / (r_sq ** 1.5) if r_sq > 0 else np.zeros(3)

    # 3. 太陽重力
    accel_sun = np.zeros(3)
    if settings.get('USE_SOLAR_GRAVITY', False):
        G = PHYSICAL_CONSTANTS['G']
        M_SUN = PHYSICAL_CONSTANTS['MASS_SUN']
        r_ps_vec = np.array([r0 - pos[0], -pos[1], -pos[2]])
        r_ps_mag_sq = np.sum(r_ps_vec ** 2)
        if r_ps_mag_sq > 0:
            accel_sun = (G * M_SUN * r_ps_vec) / (r_ps_mag_sq ** 1.5)

    # 4. コリオリ力
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


def simulate_particle_for_one_step(args: Dict) -> Dict:
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
        # 1. 光電離判定
        dt_this_loop = min(time_remaining, MAX_DT_STEP)

        if pos[0] > 0 and np.random.random() < (1.0 - np.exp(-dt_this_loop / tau_ion)):
            altitude = np.linalg.norm(pos) - PHYSICAL_CONSTANTS['RM']
            if altitude < 10000.0:
                bin_idx = assign_sticking_bin()
                return {'status': 'stuck', 'pos_at_impact': pos, 'weight': weight, 'bin_idx': bin_idx}
            else:
                return {'status': 'ionized', 'final_state': None, 'weight': weight}

        # 衝突予測
        r_vec_now = pos
        r_mag_now = np.linalg.norm(r_vec_now)
        n_vec = r_vec_now / r_mag_now
        g_mag = PHYSICAL_CONSTANTS['GM_MERCURY'] / (r_mag_now ** 2)
        v_rad_local = np.dot(vel, n_vec)

        t_hit_est = float('inf')
        if v_rad_local > 0:
            t_hit_est = 2.0 * v_rad_local / g_mag
        elif r_mag_now > RM:
            val_c = r_mag_now - RM
            if val_c < 1000.0:
                term_sq = v_rad_local ** 2 + 2.0 * g_mag * val_c
                if term_sq >= 0:
                    t_hit_est = (np.abs(v_rad_local) + np.sqrt(term_sq)) / g_mag

        is_hit = False
        pos_hit = pos
        if t_hit_est < dt_this_loop:
            # === 直線近似予測での衝突 ===
            t_flight = t_hit_est
            acc_vec_approx = -n_vec * g_mag
            pos_hit = pos + vel * t_flight + 0.5 * acc_vec_approx * (t_flight ** 2)
            vel_hit_in = vel + acc_vec_approx * t_flight
            pos_hit = pos_hit * (RM / np.linalg.norm(pos_hit))

            time_remaining -= t_flight
            is_hit = True

        else:
            # === RK4移動 ===
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
                return {'status': 'escaped', 'final_state': None, 'weight': weight}

            if r_next <= RM:
                # === 移動後のめり込みによる衝突 ===
                time_remaining -= dt
                pos_hit = pos_next * (RM / r_next)
                vel_hit_in = vel_next
                is_hit = True
            else:
                pos = pos_next
                vel = vel_next
                time_remaining -= dt

        if is_hit:
            time_remaining = max(0.0, time_remaining)
            lon_rot = np.arctan2(pos_hit[1], pos_hit[0])
            lat_rot = np.arcsin(np.clip(pos_hit[2] / RM, -1, 1))
            lon_fixed = (lon_rot + subsolar_lon + np.pi) % (2 * np.pi) - np.pi
            temp_impact = calculate_surface_temperature_leblanc(lon_fixed, lat_rot, AU, subsolar_lon)

            # [New] 吸着先のビンを決定し、その束縛エネルギーを取得
            bin_idx = assign_sticking_bin()
            U_assigned = settings['U_BINS'][bin_idx]

            # [New] 滞在時間(tau_TD)の計算
            td_rate = calculate_thermal_desorption_rate(temp_impact, U_assigned)
            tau_td = 1.0 / td_rate if td_rate > 1e-30 else float('inf')

            # --- ここから変更 ---
            HOP_TAU_THRESHOLD = 30.0  # 即時バウンドを許す最大の滞在時間 [秒]

            # 「滞在時間が閾値(30秒)以下」かつ「残り時間内で飛べる」場合のみ即時ホップ
            if tau_td <= HOP_TAU_THRESHOLD and time_remaining > tau_td:
                time_remaining -= tau_td
                spd = sample_speed_from_flux_distribution(PHYSICAL_CONSTANTS['MASS_NA'], temp_impact)
                norm_hit = pos_hit / RM
                rebound_dir = transform_local_to_world(sample_lambertian_direction_local(), norm_hit)
                pos = (RM + 1.0) * norm_hit
                vel = spd * rebound_dir
                continue
            else:
                # 滞在時間が30秒を超える、あるいは残り時間が足りない場合は一旦吸着(Stuck)
                # → 次のステップで、表面密度マップ(surface_density)の確率計算によって正しくTD放出される
                return {'status': 'stuck', 'pos_at_impact': pos_hit, 'weight': weight, 'bin_idx': bin_idx}

    return {'status': 'alive', 'final_state': {'pos': pos, 'vel': vel, 'weight': weight}}


# ==============================================================================
# 3. メインルーチン
# ==============================================================================
def main_snapshot_simulation():
    start_time = time.time()

    # --- 設定読み込み ---
    DT_MOVE = SIMULATION_SETTINGS['DT_MOVE']
    DT_RATE_UPDATE = SIMULATION_SETTINGS['DT_RATE_UPDATE']
    N_LON_FIXED = SIMULATION_SETTINGS['N_LON']
    N_LAT = SIMULATION_SETTINGS['N_LAT']
    GRID_RESOLUTION = SIMULATION_SETTINGS['GRID_RESOLUTION']
    GRID_MAX_RM = SIMULATION_SETTINGS['GRID_MAX_RM']

    OUTPUT_DIRECTORY = r"./SimulationResult_202603"

    # 実行パラメータ
    INIT_SURF_DENS = 7.5e14 * (100 ** 2) * 0.0053
    SPIN_UP_YEARS = 2.0
    TOTAL_SIM_YEARS = 1.0
    TARGET_TAA = np.arange(0, 360, 1)

    # スーパーパーティクル数
    TARGET_SPS = {'TD': 100, 'PSD': 100, 'SWS': 100, 'MMV': 100}

    # ソースプロセス物理定数
    F_UV_1AU = 1.5e14 * (100 ** 2)
    TEMP_PSD = 1500.0
    TEMP_MMV = 3000.0

    SWS_PARAMS = {
        'FLUX_1AU': 10.0 * 100 ** 3 * 400e3 * 4,
        'YIELD': 0.06,
        'U_eV': 0.27,
        'REF_DENS': 7.5e14 * 100 ** 2,
        'LON_RANGE': np.deg2rad([-40, 40]),
        'LAT_N_RANGE': np.deg2rad([20, 80]),
        'LAT_S_RANGE': np.deg2rad([-80, -20]),
    }

    # === 初期化処理 ===
    mode_str = "EqMode" if SIMULATION_SETTINGS['USE_EQUILIBRIUM_MODE'] else "NoEq"

    # ファイル名 (最新版に_MultiBinを付与)
    run_name = f"ParabolicHop_{N_LON_FIXED}x{N_LAT}_{mode_str}_DT{int(DT_MOVE)}_0317_Multi_0.4Denabled_1.85&2.7_OnlyLowestQ_Bouncetau30s_A2.0_LongLT"
    target_output_dir = os.path.join(OUTPUT_DIRECTORY, run_name)
    os.makedirs(target_output_dir, exist_ok=True)
    print(f"Simulation Start. Results: {target_output_dir}")
    print(f"Settings: DT_MOVE={DT_MOVE}s")

    # 表面グリッド定義
    lon_edges = np.linspace(-np.pi, np.pi, N_LON_FIXED + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    dlon = lon_edges[1] - lon_edges[0]
    cell_areas = (PHYSICAL_CONSTANTS['RM'] ** 2) * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))

    # [New] 表面密度マップの3次元化
    surface_density = np.zeros((N_LON_FIXED, N_LAT, N_BINS), dtype=np.float64)
    # 初期値は内部拡散用の深いサイト(IDX_DEEP)に与える
    surface_density[:, :, IDX_DEEP] = INIT_SURF_DENS

    # 外部データ読み込み
    try:
        spec_np = np.loadtxt('SolarSpectrum_Na0.txt', usecols=(0, 3))
        orbit_data = np.loadtxt('orbit2025_spice_unwrapped.txt')
        orbit_data[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(orbit_data[:, 0])))
        orbit_data[:, 5] = np.rad2deg(np.unwrap(np.deg2rad(orbit_data[:, 5])))
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # スペクトルデータ準備
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

    # 時間管理変数
    MERCURY_YEAR = PHYSICAL_CONSTANTS['ORBITAL_PERIOD']
    t_file_start = orbit_data[0, 2]
    t_start_spinup = t_file_start
    t_start_run = t_start_spinup + SPIN_UP_YEARS * MERCURY_YEAR
    t_end_run = t_start_run + TOTAL_SIM_YEARS * MERCURY_YEAR
    t_curr = t_start_spinup
    t_peri_file = t_file_start

    active_particles = []
    prev_taa = -999

    # マップの3次元化
    cached_rate_psd = np.zeros_like(surface_density)
    cached_rate_td = np.zeros_like(surface_density)
    cached_rate_sws = np.zeros_like(surface_density)
    cached_loss_rate_grid = np.zeros_like(surface_density)
    accumulated_gained_grid = np.zeros_like(surface_density)

    time_since_last_update = DT_RATE_UPDATE * 2.0
    total_steps = int((t_end_run - t_start_spinup) / DT_MOVE)
    step_count = 0

    half_grid_width_rad = dlon / 2.0
    sin_half_width = np.sin(half_grid_width_rad)

    stats_data = {}
    for deg in range(360):
        stats_data[deg] = {
            'Gen_PSD': 0.0, 'Gen_TD': 0.0, 'Gen_SWS': 0.0, 'Gen_MMV': 0.0,
            'Loss_Stuck': 0.0, 'Loss_Ionized': 0.0, 'Loss_Escaped': 0.0,
            'Step_Count': 0
        }

    # === メインループ ===
    while t_curr < t_end_run:
        step_count += 1

        TAA_raw, AU, V_rad, V_tan, sub_lon = get_orbital_params_linear(t_curr, orbit_data, t_peri_file)
        TAA = TAA_raw % 360.0
        time_since_last_update += DT_MOVE

        is_recording_phase = (t_curr >= t_start_run)
        current_taa_bin = int(TAA) % 360
        step_stats = {k: 0.0 for k in stats_data[0].keys()}

        # ----------------------------------------------------------------------
        # A. 表面放出率マップの更新
        # ----------------------------------------------------------------------
        if time_since_last_update >= DT_RATE_UPDATE:
            dt_accumulated = time_since_last_update
            f_uv = F_UV_1AU / (AU ** 2)
            sw_flux = SWS_PARAMS['FLUX_1AU'] / (AU ** 2)
            mmv_flux = calculate_mmv_flux(AU)

            scaling = np.sqrt(0.306 / AU)
            temp_rate_psd = np.zeros_like(surface_density)
            temp_rate_td = np.zeros_like(surface_density)
            temp_rate_sws = np.zeros_like(surface_density)
            temp_loss_per_sec = np.zeros_like(surface_density)

            target_au_for_diff = AU if TAA_CLAMP_START <= TAA <= TAA_CLAMP_END else AU_AT_CUTOFF
            scaling_clamped = np.sqrt(0.306 / target_au_for_diff)

            for i in range(N_LON_FIXED):
                for j in range(N_LAT):
                    lon_f = (lon_edges[i] + lon_edges[i + 1]) / 2
                    lat_f = (lat_edges[j] + lat_edges[j + 1]) / 2
                    cos_z_center = np.cos(lat_f) * np.cos(lon_f - sub_lon)

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

                    supply_dens_std_diff = 0.0
                    if SIMULATION_SETTINGS['USE_STD_DIFFUSION'] and T_day_potential > 100.0:
                        flux_val = DIFF_PRE_FACTOR * np.exp(-DIFF_E_A_EV / (KB_EV_CONST * T_day_potential))
                        supply_dens_std_diff = flux_val * dt_accumulated

                    supply_dens_clamped = 0.0
                    if SIMULATION_SETTINGS['USE_CLAMPED_DIFFUSION']:
                        T_day_clamped = SIMULATION_SETTINGS['TEMP_BASE'] + \
                                        SIMULATION_SETTINGS['TEMP_AMP'] * (eff_cos ** 0.25) * scaling_clamped
                        if T_day_clamped > 100.0:
                            flux_val = DIFF_PRE_FACTOR * np.exp(-DIFF_E_A_EV / (KB_EV_CONST * T_day_clamped))
                            supply_dens_clamped = flux_val * dt_accumulated

                    supply_dens_forced = 0.0
                    diff_rad = (lon_f - sub_lon + np.pi) % (2 * np.pi) - np.pi
                    lt_hour = (12.0 + np.rad2deg(diff_rad) / 15.0) % 24.0
                    curr_lat_deg = np.rad2deg(lat_f)

                    for event in FORCED_INJECTION_EVENTS:
                        if event['taa_range'][0] <= TAA <= event['taa_range'][1]:
                            if not (event['lat_range'][0] <= curr_lat_deg <= event['lat_range'][1]): continue
                            start_lt, end_lt = event['lt_range']
                            is_in_lt = False
                            if start_lt <= end_lt:
                                if start_lt <= lt_hour <= end_lt: is_in_lt = True
                            else:
                                if start_lt <= lt_hour or lt_hour <= end_lt: is_in_lt = True

                            if is_in_lt: supply_dens_forced += event['flux'] * dt_accumulated

                    # [New] 拡散などの内部供給源は、すべて深いサイト(IDX_DEEP)にのみ追加する
                    supply_dens_total = np.zeros(N_BINS)
                    supply_dens_total[IDX_DEEP] = supply_dens_std_diff + supply_dens_clamped + supply_dens_forced

                    lon_sun = (lon_f - sub_lon + np.pi) % (2 * np.pi) - np.pi
                    in_lon = SWS_PARAMS['LON_RANGE'][0] <= lon_sun <= SWS_PARAMS['LON_RANGE'][1]
                    in_lat = (SWS_PARAMS['LAT_N_RANGE'][0] <= lat_f <= SWS_PARAMS['LAT_N_RANGE'][1]) or \
                             (SWS_PARAMS['LAT_S_RANGE'][0] <= lat_f <= SWS_PARAMS['LAT_S_RANGE'][1])

                    for b in range(N_BINS):
                        # ビンごとに個別のPSD断面積(Q)を使用
                        if illum_frac > 0:
                            temp_rate_psd[i, j, b] = f_uv * SIMULATION_SETTINGS['Q_PSD_BINS'][b] * eff_cos * illum_frac

                        # ビンごとに個別の束縛エネルギー(U)を使用してTDを計算
                        if SIMULATION_SETTINGS['USE_AREA_WEIGHTED_FLUX']:
                            rate_day = calculate_thermal_desorption_rate(T_day_potential,
                                                                         SIMULATION_SETTINGS['U_BINS'][b])
                            rate_night = calculate_thermal_desorption_rate(SIMULATION_SETTINGS['TEMP_NIGHT'],
                                                                           SIMULATION_SETTINGS['U_BINS'][b])
                            temp_rate_td[i, j, b] = rate_day * illum_frac + rate_night * (1.0 - illum_frac)
                        else:
                            temp_val = T_day_potential if illum_frac > 0.5 else SIMULATION_SETTINGS['TEMP_NIGHT']
                            temp_rate_td[i, j, b] = calculate_thermal_desorption_rate(temp_val,
                                                                                      SIMULATION_SETTINGS['U_BINS'][b])

                        if in_lon and in_lat:
                            temp_rate_sws[i, j, b] = (sw_flux * SWS_PARAMS['YIELD']) / SWS_PARAMS['REF_DENS']

                        rate_total = temp_rate_psd[i, j, b] + temp_rate_td[i, j, b] + temp_rate_sws[i, j, b]
                        current_dens = surface_density[i, j, b]
                        gain_dens = accumulated_gained_grid[i, j, b] / cell_areas[j]
                        total_input_dens = gain_dens + supply_dens_total[b]

                        timescale = 1.0 / rate_total if rate_total > 1e-30 else float('inf')
                        allow_eq_mode = (step_count > 1)

                        if SIMULATION_SETTINGS['USE_EQUILIBRIUM_MODE'] and allow_eq_mode and (
                                timescale <= dt_accumulated):
                            if rate_total > 1e-30:
                                dens_eq = (total_input_dens / dt_accumulated) / rate_total
                            else:
                                dens_eq = current_dens + total_input_dens
                            surface_density[i, j, b] = dens_eq
                            actual_loss_dens = total_input_dens
                        else:
                            decay_factor = np.exp(-rate_total * dt_accumulated)
                            actual_loss_dens = current_dens * (1.0 - decay_factor)
                            surface_density[i, j, b] = (current_dens - actual_loss_dens) + total_input_dens

                        if surface_density[i, j, b] < 0: surface_density[i, j, b] = 0
                        temp_loss_per_sec[i, j, b] = actual_loss_dens * cell_areas[j] / dt_accumulated

            cached_rate_psd = temp_rate_psd
            cached_rate_td = temp_rate_td
            cached_rate_sws = temp_rate_sws
            cached_loss_rate_grid = temp_loss_per_sec
            accumulated_gained_grid.fill(0.0)
            time_since_last_update = 0.0

        # ----------------------------------------------------------------------
        # B. 粒子の生成
        # ----------------------------------------------------------------------
        new_particles = []

        mmv_flux = calculate_mmv_flux(AU)
        n_mmv = mmv_flux * 4 * np.pi * PHYSICAL_CONSTANTS['RM'] ** 2 * DT_MOVE
        w_mmv = max(1.0, n_mmv / (TARGET_SPS['MMV'] * (DT_MOVE / DT_RATE_UPDATE)))
        if n_mmv > 0:
            num_p = int(n_mmv / w_mmv)
            if np.random.random() < (n_mmv / w_mmv - num_p): num_p += 1

            step_stats['Gen_MMV'] += w_mmv * num_p

            for _ in range(num_p):
                dt_init = DT_MOVE * np.random.random()
                lr = np.random.uniform(-np.pi, np.pi)
                latr = np.arcsin(np.random.uniform(-1, 1))
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

                for b in range(N_BINS):
                    n_lost_bin = total_loss_step[i, j, b]
                    if n_lost_bin <= 0: continue

                    params = [
                        ('PSD', n_lost_bin * frac_psd[i, j, b], TEMP_PSD, w_psd),
                        ('TD', n_lost_bin * frac_td[i, j, b], temp_eff_for_vel, w_td),
                        ('SWS', n_lost_bin * frac_sws[i, j, b], None, w_sws)
                    ]

                    for p_type, n_amount, T_or_none, w in params:
                        if n_amount <= 0: continue
                        num = int(n_amount / w)
                        if np.random.random() < (n_amount / w - num): num += 1

                        if p_type == 'PSD':
                            step_stats['Gen_PSD'] += w * num
                        elif p_type == 'TD':
                            step_stats['Gen_TD'] += w * num
                        elif p_type == 'SWS':
                            step_stats['Gen_SWS'] += w * num

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

        # ----------------------------------------------------------------------
        # C. 粒子の移動 (並列計算) & 消滅理由の集計
        # ----------------------------------------------------------------------
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
            with Pool(cpu_count() - 1) as pool:
                results = pool.map(simulate_particle_for_one_step, tasks)

            for res in results:
                if res['status'] == 'alive':
                    next_particles.append(res['final_state'])
                else:
                    lost_weight = res.get('weight', 0.0)

                    if res['status'] == 'stuck':
                        step_stats['Loss_Stuck'] += lost_weight
                        pos = res['pos_at_impact']
                        w = res['weight']
                        b_idx = res['bin_idx']  # 戻ってきた先のビンインデックス

                        ln = np.arctan2(pos[1], pos[0])
                        lt = np.arcsin(np.clip(pos[2] / np.linalg.norm(pos), -1, 1))
                        ln_fix = (ln + sub_lon + np.pi) % (2 * np.pi) - np.pi
                        ix = np.searchsorted(lon_edges, ln_fix) - 1
                        iy = np.searchsorted(lat_edges, lt) - 1
                        if 0 <= ix < N_LON_FIXED and 0 <= iy < N_LAT:
                            step_gained_grid[ix, iy, b_idx] += w

                    elif res['status'] == 'ionized':
                        step_stats['Loss_Ionized'] += lost_weight
                    elif res['status'] == 'escaped':
                        step_stats['Loss_Escaped'] += lost_weight

        active_particles = next_particles
        accumulated_gained_grid += step_gained_grid

        # ----------------------------------------------------------------------
        # 統計データの蓄積 (スピンアップ後のみ)
        # ----------------------------------------------------------------------
        if is_recording_phase:
            tgt = stats_data[current_taa_bin]
            for key in step_stats:
                if key != 'Step_Count':
                    tgt[key] += step_stats[key]
            tgt['Step_Count'] += 1

        # ----------------------------------------------------------------------
        # D. データ保存
        # ----------------------------------------------------------------------
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
                    H, _ = np.histogramdd(pos_arr, bins=GRID_RESOLUTION, range=[(gmin, gmax)] * 3, weights=weights_arr)
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

    # === ループ終了後：CSVへの書き出し ===
    print("Saving TAA-binned statistics...")
    csv_filename = os.path.join(target_output_dir, "budget_statistics_per_taa.csv")

    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ['TAA_Bin',
                   'Gen_Total', 'Gen_PSD', 'Gen_TD', 'Gen_SWS', 'Gen_MMV',
                   'Pct_PSD', 'Pct_TD', 'Pct_SWS', 'Pct_MMV',
                   'Loss_Total', 'Loss_Stuck', 'Loss_Ionized', 'Loss_Escaped',
                   'Pct_Stuck', 'Pct_Ionized', 'Pct_Escaped']
        writer.writerow(headers)

        for deg in range(360):
            d = stats_data[deg]

            gen_total = d['Gen_PSD'] + d['Gen_TD'] + d['Gen_SWS'] + d['Gen_MMV']
            loss_total = d['Loss_Stuck'] + d['Loss_Ionized'] + d['Loss_Escaped']

            def safe_pct(val, total):
                return (val / total * 100.0) if total > 0 else 0.0

            row = [
                deg,
                f"{gen_total:.4e}",
                f"{d['Gen_PSD']:.4e}", f"{d['Gen_TD']:.4e}", f"{d['Gen_SWS']:.4e}", f"{d['Gen_MMV']:.4e}",
                f"{safe_pct(d['Gen_PSD'], gen_total):.1f}",
                f"{safe_pct(d['Gen_TD'], gen_total):.1f}",
                f"{safe_pct(d['Gen_SWS'], gen_total):.1f}",
                f"{safe_pct(d['Gen_MMV'], gen_total):.1f}",

                f"{loss_total:.4e}",
                f"{d['Loss_Stuck']:.4e}", f"{d['Loss_Ionized']:.4e}", f"{d['Loss_Escaped']:.4e}",
                f"{safe_pct(d['Loss_Stuck'], loss_total):.1f}",
                f"{safe_pct(d['Loss_Ionized'], loss_total):.1f}",
                f"{safe_pct(d['Loss_Escaped'], loss_total):.1f}"
            ]
            writer.writerow(row)

    print(f"Statistics saved to {csv_filename}")
    print("Done. Simulation Completed.")


if __name__ == '__main__':
    sys.modules['__main__'].__spec__ = None
    main_snapshot_simulation()