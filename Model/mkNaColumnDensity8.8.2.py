# -*- coding: utf-8 -*-
"""
==============================================================================
プロジェクト: 水星ナトリウム外気圏 3次元モンテカルロシミュレーション
              (Mercury Sodium Exosphere 3D Monte-Carlo Simulation)

概要:
    水星表面からのナトリウム放出と、外気圏における粒子の運動を計算する。
    重力、太陽放射圧、コリオリ力、光電離などを考慮し、4次のルンゲ・クッタ法で
    粒子の軌跡を追跡する。

主な機能:
    1. 放出過程 (Source Processes): PSD, TD, SWS, MMV
    2. グリッド計算: 表面密度管理、Equilibrium Mode
    3. 並列化: multiprocessing使用
    4. マルチバウンド処理 (Parabolic Approx):
       1ステップ内の「跳躍→着地」を放物線近似で正確に計算し、微小ステップのスタックを解消。

作成者: Koki Masaki (Rikkyo Univ.)
日付: 2025/12/22 (Updated: Fixed 'stuck' issue with parabolic approximation)
==============================================================================
"""

import numpy as np
import sys
import os
from multiprocessing import Pool, cpu_count
import time
from typing import Dict, Tuple, List, Optional, Any

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
}

# 統合シミュレーション設定
SIMULATION_SETTINGS = {
    # --- 時間ステップ設定 ---
    'DT_MOVE': 500.0,  # 粒子の位置更新ステップ (メインループの刻み) [s]
    'DT_RATE_UPDATE': 500.0,  # 表面放出率の再計算ステップ [s]
    'DT_INTEGRATION': 500.0,  # 粒子軌跡計算の内部積分ステップ (精度用)

    # --- 温度モデル設定 (Leblanc et al.) ---
    # T = T_BASE + T_AMP * (cos(theta))^0.25 * scaling
    'TEMP_BASE': 100.0,
    'TEMP_AMP': 600.0,
    'TEMP_NIGHT': 100.0,

    # --- グリッド・領域設定 ---
    'N_LON': 72,
    'N_LAT': 36,
    'GRID_RESOLUTION': 101,  # 出力用3Dグリッド
    'GRID_MAX_RM': 5.0,  # 計算領域半径 [RM]
    'GRID_RADIUS_RM': 6.0,

    # --- 物理フラグ ---
    'BETA': 1.0,  # エネルギー適応係数 (1.0 = 完全順応)
    'T1AU': 54500.0,  # 光電離寿命 @ 1AU [s]
    'USE_SOLAR_GRAVITY': True,
    'USE_CORIOLIS_FORCES': True,

    # --- 計算モード ---
    'USE_EQUILIBRIUM_MODE': True,
    'USE_AREA_WEIGHTED_FLUX': False,
    'USE_SUBGRID_SMOOTHING': False,
}


# ==============================================================================
# 1. 物理モデル・ヘルパー関数群
# ==============================================================================

def calculate_surface_temperature_leblanc(lon_rad: float, lat_rad: float, AU: float, subsolar_lon_rad: float) -> float:
    T_BASE = SIMULATION_SETTINGS['TEMP_BASE']
    T_AMP = SIMULATION_SETTINGS['TEMP_AMP']
    T_NIGHT = SIMULATION_SETTINGS['TEMP_NIGHT']

    scaling = np.sqrt(0.306 / AU)
    cos_theta = np.cos(lat_rad) * np.cos(lon_rad - subsolar_lon_rad)

    if cos_theta <= 0:
        return T_NIGHT
    return T_BASE + T_AMP * (cos_theta ** 0.25) * scaling


def calculate_sticking_probability(surface_temp_K: float) -> float:
    A = 0.0804
    B = 458.0
    porosity = 0.8

    if surface_temp_K <= 0: return 1.0

    p_stick = A * np.exp(B / surface_temp_K)
    p_stick_eff = p_stick / (1.0 - (1.0 - p_stick) * porosity)

    return min(p_stick_eff, 1.0)


def calculate_thermal_desorption_rate(surface_temp_K: float) -> float:
    if surface_temp_K < 10.0: return 0.0

    VIB_FREQ = 1e13
    KB = PHYSICAL_CONSTANTS['K_BOLTZMANN']
    EV_J = PHYSICAL_CONSTANTS['EV_TO_JOULE']

    U_MEAN = 1.85
    U_MIN = 1.40
    U_MAX = 2.70
    SIGMA = 0.20

    u_ev_grid = np.linspace(U_MIN, U_MAX, 50)
    u_joule_grid = u_ev_grid * EV_J

    pdf = np.exp(- (u_ev_grid - U_MEAN) ** 2 / (2 * SIGMA ** 2))
    pdf_sum = np.sum(pdf)
    if pdf_sum == 0: return 0.0
    norm_pdf = pdf / pdf_sum

    exponent = -u_joule_grid / (KB * surface_temp_K)
    rates = np.zeros_like(u_ev_grid)
    mask = exponent > -700
    rates[mask] = VIB_FREQ * np.exp(exponent[mask])

    effective_rate = np.sum(rates * norm_pdf)
    return effective_rate


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
    subsolar_lon_rad = np.deg2rad(sub_lon_deg)
    return taa_deg, au, v_rad, v_tan, subsolar_lon_rad


def lonlat_to_xyz(lon_rad: float, lat_rad: float, radius: float) -> np.ndarray:
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return np.array([x, y, z])


# ==============================================================================
# 2. 粒子運動計算エンジン (放物線近似・マルチバウンド版)
# ==============================================================================

def _calculate_acceleration(pos: np.ndarray, vel: np.ndarray, V_radial_ms: float, V_tangential_ms: float, AU: float,
                            spec_data: Dict, settings: Dict) -> np.ndarray:
    """粒子にかかる総加速度を計算する"""
    x, y, z = pos
    r0 = AU * PHYSICAL_CONSTANTS['AU']

    # --- 1. 放射圧 ---
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

    # --- 2. 水星重力 ---
    r_sq = np.sum(pos ** 2)
    accel_g = -PHYSICAL_CONSTANTS['GM_MERCURY'] * pos / (r_sq ** 1.5) if r_sq > 0 else np.zeros(3)

    # --- 3. 太陽重力 ---
    accel_sun = np.zeros(3)
    if settings.get('USE_SOLAR_GRAVITY', False):
        G = PHYSICAL_CONSTANTS['G']
        M_SUN = PHYSICAL_CONSTANTS['MASS_SUN']
        r_ps_vec = np.array([r0 - pos[0], -pos[1], -pos[2]])
        r_ps_mag_sq = np.sum(r_ps_vec ** 2)
        if r_ps_mag_sq > 0:
            accel_sun = (G * M_SUN * r_ps_vec) / (r_ps_mag_sq ** 1.5)

    # --- 4. コリオリ力・遠心力 ---
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
    """
    1つの粒子を計算する。
    【改良版】放物線近似による衝突判定を実装。
    線形補間では捉えきれない「1ステップ内のジャンプ(Short Hop)」を正確に処理する。
    """
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

    # --- マルチバウンドループ ---
    while time_remaining > 1e-6:

        # 1. 光電離判定 (今回はdt分を一括判定ではなく、ステップ毎に処理するためにループ内へ)
        #    ただし簡易的に、最小ステップでの判定とする
        dt_this_loop = min(time_remaining, MAX_DT_STEP)

        if pos[0] > 0 and np.random.random() < (1.0 - np.exp(-dt_this_loop / tau_ion)):
            return {'status': 'ionized', 'final_state': None}

        # === 衝突予測 (Parabolic Approximation Check) ===
        # 現在位置での重力と、垂直方向(動径方向)の速度を取得
        r_vec_now = pos
        r_mag_now = np.linalg.norm(r_vec_now)
        n_vec = r_vec_now / r_mag_now

        # 地表での重力加速度の大きさ (簡易計算)
        g_mag = PHYSICAL_CONSTANTS['GM_MERCURY'] / (r_mag_now ** 2)

        # 動径方向速度 (正なら上昇、負なら下降)
        v_rad = np.dot(vel, n_vec)

        t_hit_est = float('inf')

        # [Case A] ショートホップ判定: 地表付近から上昇中 (v_rad > 0)
        # 滞空時間 t = 2 * v_perp / g
        if v_rad > 0:
            t_hit_est = 2.0 * v_rad / g_mag

        # [Case B] 落下中だが地表に近い場合 (v_rad <= 0)
        # 2次方程式: (RM - r) + v_rad*t - 0.5*g*t^2 = 0  => 0.5gt^2 - v_rad*t + (r - RM) = 0
        elif r_mag_now > RM:
            # 判別式 D = b^2 - 4ac
            # a = 0.5*g, b = -v_rad (positive term), c = r_now - RM
            val_c = r_mag_now - RM
            # 落下速度が十分ある、または距離が近ければ解を持つ
            # ※遠すぎる場合は t_hit_est は inf のままでRK4に任せる
            if val_c < 1000.0:  # 1km以内なら近似計算発動
                term_sq = v_rad ** 2 + 2.0 * g_mag * val_c
                if term_sq >= 0:
                    # 解の公式 (-b +/- sqrt(D)) / 2a
                    # ここでは v_rad < 0 なので、b = -v_rad > 0.
                    # t = (v_rad + sqrt(v^2 + 2gh)) / g  <-- v_radは負なので注意
                    # t = (-(-v_rad) + sqrt(...)) / g
                    # 正しい形: t = (v_rad + np.sqrt(term_sq)) / g は負になる(過去)。
                    # 未来の解は t = (v_rad + np.sqrt(term_sq)) / g だと v_radが負で項が消え合う?
                    # 物理的に: t = ( |v_rad| + sqrt(v^2 + 2gh) ) / g
                    t_hit_est = (np.abs(v_rad) + np.sqrt(term_sq)) / g_mag

        # --- 分岐処理 ---
        # 「予測された衝突時間」が「今回のステップ幅」より短い場合 -> 衝突処理へ
        if t_hit_est < dt_this_loop:
            t_flight = t_hit_est

            # 放物運動近似で位置を進める (重力のみ考慮)
            # x(t) = x0 + v0*t - 0.5*g*n*t^2
            acc_vec_approx = -n_vec * g_mag
            pos_hit = pos + vel * t_flight + 0.5 * acc_vec_approx * (t_flight ** 2)

            # 速度更新 v(t) = v0 + a*t
            vel_hit = vel + acc_vec_approx * t_flight

            # 強制的に表面位置へ補正
            pos_hit = pos_hit * (RM / np.linalg.norm(pos_hit))

            # 時間消費
            time_remaining -= t_flight
            if time_remaining < 0: time_remaining = 0.0

            # === 反射処理 (Bounce) ===
            lon_rot = np.arctan2(pos_hit[1], pos_hit[0])
            lat_rot = np.arcsin(np.clip(pos_hit[2] / RM, -1, 1))
            lon_fixed = (lon_rot + subsolar_lon + np.pi) % (2 * np.pi) - np.pi
            temp_impact = calculate_surface_temperature_leblanc(lon_fixed, lat_rot, AU, subsolar_lon)

            # 付着判定
            if np.random.random() < calculate_sticking_probability(temp_impact):
                return {'status': 'stuck', 'pos_at_impact': pos_hit, 'weight': weight}

            # 反射
            E_in = 0.5 * PHYSICAL_CONSTANTS['MASS_NA'] * np.sum(vel_hit ** 2)
            E_T = 2 * PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_impact
            E_out = settings['BETA'] * E_T + (1.0 - settings['BETA']) * E_in
            v_out = np.sqrt(2 * E_out / PHYSICAL_CONSTANTS['MASS_NA'])

            norm_hit = pos_hit / RM
            rebound_dir = transform_local_to_world(sample_lambertian_direction_local(), norm_hit)

            # 状態更新
            pos = (RM + 1.0) * norm_hit  # 表面のわずか上
            vel = v_out * rebound_dir

            # 次のループへ (残り時間で再計算)
            continue

        else:
            # --- 通常のRK4移動 (衝突しない、または遠い) ---
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

            # 散逸判定
            if r_next > R_MAX:
                return {'status': 'escaped', 'final_state': None}

            # 念のため、RK4計算でも衝突してしまった場合のバックアップ (高高度からの落下など)
            # ここでは線形補間を使わず、2次方程式で時間を解くのがベストだが、
            # 上記の予測ロジックでほとんどカバーできるため、稀なケースとして簡易処理(あるいは再計算)する
            if r_next <= RM:
                # 予測が外れて衝突した場合（放射圧などの影響）
                # 簡易的に、このステップの最後で衝突したことにする
                time_remaining -= dt
                pos_hit = pos_next * (RM / r_next)  # 強制補正

                # --- 反射処理 (コード重複になるが記述) ---
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

            # 空中移動確定
            pos = pos_next
            vel = vel_next
            time_remaining -= dt

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

    OUTPUT_DIRECTORY = r"./SimulationResult_202512"

    # 実行パラメータ
    INIT_SURF_DENS = 7.5e14 * (100 ** 2) * 0.0053
    SPIN_UP_YEARS = 2.0
    TOTAL_SIM_YEARS = 1.0
    TARGET_TAA = np.arange(0, 360, 1)

    # スーパーパーティクル数
    TARGET_SPS = {'TD': 1000, 'PSD': 1000, 'SWS': 1000, 'MMV': 1000}

    # ソースプロセス物理定数
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

    # === 初期化処理 ===
    mode_str = "EqMode" if SIMULATION_SETTINGS['USE_EQUILIBRIUM_MODE'] else "NoEq"
    smooth_str = "FluxW" if SIMULATION_SETTINGS['USE_AREA_WEIGHTED_FLUX'] else "Hard"
    # ファイル名
    run_name = f"ParabolicHop_{N_LON_FIXED}x{N_LAT}_{mode_str}_DT{int(DT_MOVE)}_1223_s1"
    target_output_dir = os.path.join(OUTPUT_DIRECTORY, run_name)
    os.makedirs(target_output_dir, exist_ok=True)
    print(f"Simulation Start. Results: {target_output_dir}")
    print(f"Settings: DT_MOVE={DT_MOVE}s (Parabolic Approximation Enabled)")

    # 表面グリッド定義
    lon_edges = np.linspace(-np.pi, np.pi, N_LON_FIXED + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    dlon = lon_edges[1] - lon_edges[0]
    cell_areas = (PHYSICAL_CONSTANTS['RM'] ** 2) * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))

    surface_density = np.full((N_LON_FIXED, N_LAT), INIT_SURF_DENS, dtype=np.float64)

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

    # === メインループ ===
    while t_curr < t_end_run:
        step_count += 1

        # 3-1. 軌道情報の更新
        TAA_raw, AU, V_rad, V_tan, sub_lon = get_orbital_params_linear(t_curr, orbit_data, t_peri_file)
        TAA = TAA_raw % 360.0
        time_since_last_update += DT_MOVE

        # ----------------------------------------------------------------------
        # A. 表面放出率マップの更新
        # ----------------------------------------------------------------------
        if time_since_last_update >= DT_RATE_UPDATE:
            dt_accumulated = time_since_last_update
            f_uv = F_UV_1AU / (AU ** 2)
            sw_flux = SWS_PARAMS['FLUX_1AU'] / (AU ** 2)
            mmv_flux = calculate_mmv_flux(AU)
            #supply_dens = mmv_flux * dt_accumulated
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

                    timescale = 1.0 / rate_total if rate_total > 1e-30 else float('inf')
                    allow_eq_mode = (step_count > 1)

                    if SIMULATION_SETTINGS['USE_EQUILIBRIUM_MODE'] and allow_eq_mode and (timescale <= dt_accumulated):
                        if rate_total > 1e-30:
                            dens_eq = (total_input_dens / dt_accumulated) / rate_total
                        else:
                            dens_eq = current_dens + total_input_dens
                        surface_density[i, j] = dens_eq
                        actual_loss_dens = total_input_dens
                    else:
                        calculated_loss = current_dens * rate_total * dt_accumulated
                        actual_loss_dens = min(current_dens, calculated_loss)
                        surface_density[i, j] += total_input_dens - actual_loss_dens

                    if surface_density[i, j] < 0: surface_density[i, j] = 0
                    temp_loss_per_sec[i, j] = actual_loss_dens * cell_areas[j] / dt_accumulated

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

        # ----------------------------------------------------------------------
        # C. 粒子の移動 (並列計算)
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

    print("Done. Simulation Completed.")


if __name__ == '__main__':
    sys.modules['__main__'].__spec__ = None
    main_snapshot_simulation()