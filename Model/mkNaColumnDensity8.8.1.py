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
    1. 放出過程 (Source Processes):
       - PSD (Photon-Stimulated Desorption): 紫外光励起脱離
       - TD (Thermal Desorption): 熱脱離
       - SWS (Solar Wind Sputtering): 太陽風スパッタリング
       - MMV (Micrometeoroid Vaporization): 微小隕石衝突気化
    2. グリッド計算:
       - 表面を緯度経度グリッドに分割し、放出フラックスと表面密度を管理。
       - Equilibrium Mode: 時定数が短い場合に平衡状態を解析的に解くモードを搭載。
       - Sub-grid Smoothing: グリッド内の昼夜境界を考慮した温度ブレンド処理。
    3. 並列化:
       - multiprocessingを用いた粒子追跡の並列計算。

作成者: Koki Masaki (Rikkyo Univ.)
日付: 2025/12/16 (Updated)
==============================================================================
"""

import numpy as np
import sys
import os
from multiprocessing import Pool, cpu_count
import time
from typing import Dict, Tuple, List, Optional, Any

# ==============================================================================
# 1. 物理定数・天文定数 (SI単位系)
# ==============================================================================
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


# ==============================================================================
# 2. 物理モデル・ヘルパー関数群
# ==============================================================================

def calculate_surface_temperature_leblanc(lon_rad: float, lat_rad: float, AU: float, subsolar_lon_rad: float) -> float:
    """
    指定された座標における表面温度を計算する (LeBlanc et al. モデルベース)。

    数式:
        T = T_night                                (夜間)
        T = T_base * (0.306/r)^0.5 + T_amp * (cos(theta))^0.25 (昼間)

    Args:
        lon_rad (float): 経度 [rad]
        lat_rad (float): 緯度 [rad]
        AU (float): 太陽からの距離 [AU]
        subsolar_lon_rad (float): 太陽直下点経度 [rad]

    Returns:
        float: 表面温度 [K]
    """
    # 設定: ユーザー指定の高温モデル (Base 600, Amp 100)
    T_BASE = 100.0
    T_ANGLE = 600.0
    T_NIGHT = 100.0

    scaling = (0.306 / AU) ** 2
    # 太陽天頂角のコサイン
    cos_theta = np.cos(lat_rad) * np.cos(lon_rad - subsolar_lon_rad)

    if cos_theta <= 0:
        return T_NIGHT

    # 放射平衡に近い温度分布モデル
    # return T_BASE * scaling + T_ANGLE * (cos_theta ** 0.25)
    return T_BASE + T_ANGLE * (cos_theta ** 0.25) * scaling


def calculate_sticking_probability(surface_temp_K: float) -> float:
    """
    ナトリウム原子が表面に衝突した際の付着確率を計算する。
    Yakshinskiy & Madey (1999) の実験式に基づく。

    Args:
        surface_temp_K (float): 表面温度 [K]

    Returns:
        float: 付着確率 (0.0 - 1.0)
    """
    A = 0.0804
    B = 458.0
    porosity = 0.8  # 表面の多孔質度（レゴリスの効果）

    if surface_temp_K <= 0: return 1.0

    # 平滑表面での付着確率
    p_stick = A * np.exp(B / surface_temp_K)
    # 多孔質表面での実効付着確率 (複数回衝突を考慮)
    p_stick_eff = p_stick / (1.0 - (1.0 - p_stick) * porosity)

    return min(p_stick_eff, 1.0)


def calculate_thermal_desorption_rate(surface_temp_K: float) -> float:
    """
    熱脱離(TD)の放出率を計算する。
    Polanyi-Wigner方程式に基づき、活性化エネルギーにガウス分布を持たせている。

    Args:
        surface_temp_K (float): 表面温度 [K]

    Returns:
        float: 放出率係数 [s^-1] (原子1個あたりの脱離確率)
    """
    if surface_temp_K < 10.0: return 0.0

    VIB_FREQ = 1e13  # 格子振動数 [Hz]
    KB = PHYSICAL_CONSTANTS['K_BOLTZMANN']
    EV_J = PHYSICAL_CONSTANTS['EV_TO_JOULE']

    # 活性化エネルギー分布パラメータ (eV)
    U_MEAN = 1.85
    U_MIN = 1.40
    U_MAX = 2.70
    SIGMA = 0.20

    # エネルギー分布の積分計算（離散化）
    u_ev_grid = np.linspace(U_MIN, U_MAX, 50)
    u_joule_grid = u_ev_grid * EV_J

    pdf = np.exp(- (u_ev_grid - U_MEAN) ** 2 / (2 * SIGMA ** 2))
    pdf_sum = np.sum(pdf)
    if pdf_sum == 0: return 0.0
    norm_pdf = pdf / pdf_sum

    exponent = -u_joule_grid / (KB * surface_temp_K)
    rates = np.zeros_like(u_ev_grid)
    mask = exponent > -700  # オーバーフロー防止
    rates[mask] = VIB_FREQ * np.exp(exponent[mask])

    effective_rate = np.sum(rates * norm_pdf)
    return effective_rate


def calculate_mmv_flux(AU: float) -> float:
    """
    微小隕石衝突気化(MMV)によるNa供給フラックスを計算する。
    Cintala (1992) 等に基づき、太陽距離の-1.9乗に比例すると仮定。

    Args:
        AU (float): 太陽距離 [AU]

    Returns:
        float: 供給フラックス [atoms m^-2 s^-1]
    """
    TOTAL_FLUX_AT_PERI = 5e23  # 近日点での全球総放出量 [atoms/s]
    PERIHELION_AU = 0.307
    AREA = 4 * PHYSICAL_CONSTANTS['PI'] * (PHYSICAL_CONSTANTS['RM'] ** 2)

    avg_flux_peri = TOTAL_FLUX_AT_PERI / AREA
    # 距離依存性の係数C
    C = avg_flux_peri * (PERIHELION_AU ** 1.9)

    return C * (AU ** (-1.9))


def sample_speed_from_flux_distribution(mass_kg: float, temp_k: float) -> float:
    """
    熱的なマクスウェル=ボルツマン分布（フラックス分布）から速度をサンプリングする。
    f(v) ~ v^3 * exp(-mv^2/2kT) に従う (Gamma分布で近似可能)。

    Args:
        mass_kg (float): 粒子質量 [kg]
        temp_k (float): 温度 [K]

    Returns:
        float: 速度 [m/s]
    """
    kT = PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_k
    # エネルギーEをサンプリング (Gamma分布形状パラメータ2.0)
    E = np.random.gamma(2.0, kT)
    return np.sqrt(2.0 * E / mass_kg)


def sample_thompson_sigmund_energy(U_eV: float, E_max_eV: float = 5.0) -> float:
    """
    スパッタリング粒子のエネルギー分布 (Thompson-Sigmund理論) からサンプリング。
    f(E) ~ E / (E + U)^3

    Args:
        U_eV (float): 表面結合エネルギー [eV]
        E_max_eV (float): カットオフエネルギー [eV]

    Returns:
        float: エネルギー [eV]
    """
    f_max = (U_eV / 2.0) / (U_eV / 2.0 + U_eV) ** 3
    # 棄却法によるサンプリング
    while True:
        E_try = np.random.uniform(0, E_max_eV)
        f_val = E_try / (E_try + U_eV) ** 3
        if np.random.uniform(0, f_max) <= f_val:
            return E_try


def sample_lambertian_direction_local() -> np.ndarray:
    """
    ランベルト余弦則 (Cos則) に従う放出方向ベクトルをローカル座標系で生成する。

    Returns:
        np.ndarray: ローカル座標系での単位ベクトル (x, y, z) [z軸が法線方向]
    """
    u1, u2 = np.random.random(2)
    phi = 2 * PHYSICAL_CONSTANTS['PI'] * u1
    cos_theta = np.sqrt(1 - u2)  # ランベルト分布の極角
    sin_theta = np.sqrt(u2)
    return np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])


def transform_local_to_world(local_vec: np.ndarray, normal_vector: np.ndarray) -> np.ndarray:
    """
    ローカル座標系のベクトルをワールド座標系に変換する。

    Args:
        local_vec (np.ndarray): ローカルベクトル (z軸が法線基準)
        normal_vector (np.ndarray): ワールド座標系での表面法線ベクトル

    Returns:
        np.ndarray: ワールド座標系でのベクトル
    """
    local_z = normal_vector / np.linalg.norm(normal_vector)
    world_up = np.array([0., 0., 1.])

    # ジンバルロック回避
    if np.abs(np.dot(local_z, world_up)) > 0.99:
        world_up = np.array([0., 1., 0.])

    local_x = np.cross(world_up, local_z)
    local_x /= np.linalg.norm(local_x)
    local_y = np.cross(local_z, local_x)

    return local_vec[0] * local_x + local_vec[1] * local_y + local_vec[2] * local_z


def get_orbital_params_linear(time_sec: float, orbit_data: np.ndarray, t_perihelion_file: float) -> Tuple[
    float, float, float, float, float]:
    """
    軌道データファイルから線形補間で現在の水星の位置・速度情報を取得する。

    Args:
        time_sec (float): 時刻 [s]
        orbit_data (np.ndarray): 軌道データ配列
        t_perihelion_file (float): ファイル開始時刻 (使用されていないが互換性のため維持)

    Returns:
        Tuple: (TAA[deg], 距離[AU], 動径速度[m/s], 接線速度[m/s], 太陽直下点経度[rad])
    """
    time_col_original = orbit_data[:, 2]
    # 範囲外参照を防ぐクリップ
    t_lookup = np.clip(time_sec, time_col_original[0], time_col_original[-1])

    taa_deg = np.interp(t_lookup, time_col_original, orbit_data[:, 0])
    au = np.interp(t_lookup, time_col_original, orbit_data[:, 1])
    v_rad = np.interp(t_lookup, time_col_original, orbit_data[:, 3])
    v_tan = np.interp(t_lookup, time_col_original, orbit_data[:, 4])
    sub_lon_deg = np.interp(t_lookup, time_col_original, orbit_data[:, 5])
    subsolar_lon_rad = np.deg2rad(sub_lon_deg)

    return taa_deg, au, v_rad, v_tan, subsolar_lon_rad


def lonlat_to_xyz(lon_rad: float, lat_rad: float, radius: float) -> np.ndarray:
    """
    球座標 (経度, 緯度, 半径) をデカルト座標 (x, y, z) に変換する。
    """
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return np.array([x, y, z])


# ==============================================================================
# 3. 粒子運動計算エンジン
# ==============================================================================

def _calculate_acceleration(pos: np.ndarray, vel: np.ndarray, V_radial_ms: float, V_tangential_ms: float, AU: float,
                            spec_data: Dict, settings: Dict) -> np.ndarray:
    """
    粒子にかかる総加速度を計算する。

    考慮される力:
    1. 放射圧 (Radiation Pressure): ドップラーシフトを考慮したNaの共鳴散乱。
    2. 水星重力
    3. 太陽重力 (オプション)
    4. コリオリ力・遠心力 (回転系座標への変換項、オプション)

    Returns:
        np.ndarray: 加速度ベクトル [m/s^2]
    """
    x, y, z = pos
    r0 = AU * PHYSICAL_CONSTANTS['AU']
    # 太陽視線方向の相対速度 (水星の公転速度V_rad + 粒子のx成分速度)
    velocity_for_doppler = vel[0] + V_radial_ms

    # ドップラーシフトした波長
    w_na_d2 = 589.1582e-9 * (1.0 + velocity_for_doppler / PHYSICAL_CONSTANTS['C'])
    w_na_d1 = 589.7558e-9 * (1.0 + velocity_for_doppler / PHYSICAL_CONSTANTS['C'])

    b = 0.0  # 放射圧加速度の大きさ
    wl, gamma, sigma0_perdnu2, sigma0_perdnu1, JL = spec_data.values()

    # フラウンホーファー線内の太陽フラックス強度を補間計算
    if (wl[0] * 1e-9 <= w_na_d2 < wl[-1] * 1e-9) and (wl[0] * 1e-9 <= w_na_d1 < wl[-1] * 1e-9):
        gamma2 = np.interp(w_na_d2 * 1e9, wl, gamma)
        gamma1 = np.interp(w_na_d1 * 1e9, wl, gamma)
        F_at_Merc = (JL * 1e13) / (AU ** 2)

        # g-factor (加速度) の計算
        term_d1 = (PHYSICAL_CONSTANTS['H'] / w_na_d1) * sigma0_perdnu1 * \
                  (F_at_Merc * gamma1 * w_na_d1 ** 2 / PHYSICAL_CONSTANTS['C'])
        term_d2 = (PHYSICAL_CONSTANTS['H'] / w_na_d2) * sigma0_perdnu2 * \
                  (F_at_Merc * gamma2 * w_na_d2 ** 2 / PHYSICAL_CONSTANTS['C'])
        b = (term_d1 + term_d2) / PHYSICAL_CONSTANTS['MASS_NA']

    # 影の判定 (水星本体による日食)
    if x < 0 and np.sqrt(y ** 2 + z ** 2) < PHYSICAL_CONSTANTS['RM']:
        b = 0.0

    accel_srp = np.array([-b, 0.0, 0.0])  # 放射圧は常に反太陽方向(-x)

    # 水星重力
    r_sq = np.sum(pos ** 2)
    accel_g = -PHYSICAL_CONSTANTS['GM_MERCURY'] * pos / (r_sq ** 1.5) if r_sq > 0 else np.zeros(3)

    # 太陽重力 (潮汐力項として重要)
    accel_sun = np.zeros(3)
    if settings.get('USE_SOLAR_GRAVITY', False):
        G = PHYSICAL_CONSTANTS['G']
        M_SUN = PHYSICAL_CONSTANTS['MASS_SUN']
        r_ps_vec = np.array([r0 - pos[0], -pos[1], -pos[2]])  # 太陽へのベクトル
        r_ps_mag_sq = np.sum(r_ps_vec ** 2)
        if r_ps_mag_sq > 0:
            accel_sun = (G * M_SUN * r_ps_vec) / (r_ps_mag_sq ** 1.5)

    # 非慣性系（水星と共に公転する座標系）の慣性力
    accel_cor = np.zeros(3)
    accel_cen = np.zeros(3)
    if settings.get('USE_CORIOLIS_FORCES', False):
        if r0 > 0:
            omega_val = V_tangential_ms / r0
            omega_sq = omega_val ** 2
            # 遠心力
            accel_cen = np.array([(omega_val ** 2) * (pos[0] - r0), omega_sq * pos[1], 0.0])
            # コリオリ力 (2 * v x omega)
            two_omega = 2 * omega_val
            accel_cor = np.array([two_omega * vel[1], -two_omega * vel[0], 0.0])

    return accel_srp + accel_g + accel_sun + accel_cen + accel_cor


def simulate_particle_for_one_step(args: Dict) -> Dict:
    """
    1つの粒子（またはスーパーパーティクル）を指定時間分だけ積分計算する。
    Runge-Kutta 4th Order (RK4) を使用。

    Args:
        args (Dict): 並列化のためにパックされた引数辞書
            - settings: 設定
            - spec: スペクトルデータ
            - orbit: 軌道パラメータ
            - particle_state: 現在の位置・速度
            - duration: 積分時間

    Returns:
        Dict: 計算後の状態とステータス
            - status: 'alive', 'ionized', 'escaped', 'stuck'
            - final_state: 位置・速度 (aliveの場合)
            - pos_at_impact: 衝突位置 (stuckの場合)
    """
    settings, spec_data = args['settings'], args['spec']
    TAA, AU, V_rad, V_tan, subsolar_lon = args['orbit']
    total_duration = args['duration']
    DT_INTEGRATION = 500.0  # 数値積分の刻み幅 [s]

    if total_duration <= 0:
        return {'status': 'alive', 'final_state': args['particle_state']}

    num_steps = int(np.ceil(total_duration / DT_INTEGRATION))
    dt_per_step = total_duration / num_steps

    pos = args['particle_state']['pos'].copy()
    vel = args['particle_state']['vel'].copy()
    weight = args['particle_state']['weight']

    # 光電離寿命 (1AUでの寿命 * r^2)
    tau_ion = settings['T1AU'] * AU ** 2
    RM = PHYSICAL_CONSTANTS['RM']
    R_MAX = RM * settings['GRID_RADIUS_RM']
    pos_start = pos.copy()

    for _ in range(num_steps):
        vel_start = vel.copy()
        current_dt = dt_per_step

        # 光電離判定 (確率的消滅)
        if pos[0] > 0:  # 影に入っていない場合のみ
            if np.random.random() < (1.0 - np.exp(-current_dt / tau_ion)):
                return {'status': 'ionized', 'final_state': None}

        # --- RK4 積分ステップ ---
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
        # 領域外への散逸判定
        if r_now > R_MAX:
            return {'status': 'escaped', 'final_state': None}

        # 表面への衝突判定
        if r_now <= RM:
            # === 正確な衝突点と時間の計算 ===
            p_now = pos_start
            v_now = vel_start
            r_sq_now = np.sum(p_now ** 2)
            g_mag = PHYSICAL_CONSTANTS['GM_MERCURY'] / r_sq_now
            r_mag_now = np.sqrt(r_sq_now)
            n_vec = p_now / r_mag_now
            v_perp = np.dot(v_now, n_vec)

            # 放物運動近似で落下時間を推定
            t_flight = 0.0
            if v_perp > 0:  # 一度上がって落ちてきた場合
                t_flight = 2.0 * v_perp / g_mag
            if t_flight > current_dt or t_flight < 0:
                t_flight = current_dt

            acc_vec = -n_vec * g_mag
            hit_pos_est = p_now + v_now * t_flight + 0.5 * acc_vec * (t_flight ** 2)

            # 強制的に表面上に補正
            hit_dist = np.linalg.norm(hit_pos_est)
            if hit_dist > 0:
                pos_impact = hit_pos_est * (PHYSICAL_CONSTANTS['RM'] / hit_dist)
            else:
                pos_impact = pos_start

            # 衝突点の経緯度計算
            lon_rot = np.arctan2(pos_impact[1], pos_impact[0])
            lat_rot = np.arcsin(np.clip(pos_impact[2] / np.linalg.norm(pos_impact), -1, 1))
            lon_fixed = (lon_rot + subsolar_lon + np.pi) % (2 * np.pi) - np.pi

            # 衝突点の温度計算 (着地判定用)
            temp_impact = calculate_surface_temperature_leblanc(lon_fixed, lat_rot, AU, subsolar_lon)

            # 付着判定 (Sticking)
            if np.random.random() < calculate_sticking_probability(temp_impact):
                return {'status': 'stuck', 'pos_at_impact': pos_impact, 'weight': weight}
            else:
                # 反射 (Thermal Accommodation)
                E_in = 0.5 * PHYSICAL_CONSTANTS['MASS_NA'] * np.sum(vel ** 2)
                E_T = 2 * PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_impact
                # エネルギー適応係数 beta による反射エネルギー
                E_out = settings['BETA'] * E_T + (1.0 - settings['BETA']) * E_in
                v_out = np.sqrt(2 * E_out / PHYSICAL_CONSTANTS['MASS_NA']) if E_out > 0 else 0

                norm = pos_impact / np.linalg.norm(pos_impact)
                rebound_dir = transform_local_to_world(sample_lambertian_direction_local(), norm)

                # 表面のわずか上空へ再配置
                pos = (RM + 1.0) * norm
                vel = v_out * rebound_dir

        pos_start = pos.copy()

    return {'status': 'alive', 'final_state': {'pos': pos, 'vel': vel, 'weight': weight}}


# ==============================================================================
# 4. メインルーチン (サブグリッドスムージング版)
# ==============================================================================
def main_snapshot_simulation():
    start_time = time.time()

    # ==========================================================================
    # 1. シミュレーション設定パラメータ
    # ==========================================================================
    OUTPUT_DIRECTORY = r"./SimulationResult_202512"
    DT_MOVE = 500.0  # 粒子の位置更新ステップ [s]
    DT_RATE_UPDATE = 500.0  # 表面放出率の再計算ステップ [s]

    # グリッド解像度設定 (必要に応じて変更可)
    N_LON_FIXED, N_LAT = 72, 36

    INIT_SURF_DENS = 7.5e14 * (100 ** 2) * 0.0053  # 初期表面密度 [atoms/m^2]
    SPIN_UP_YEARS = 2.0  # スピンアップ期間 (定常状態待ち)
    TOTAL_SIM_YEARS = 1.0  # データ取得期間
    TARGET_TAA = np.arange(0, 360, 1)  # 出力するTAAタイミング

    # スーパーパーティクル制御 (1ステップあたりの生成数目安)
    TARGET_SPS = {'TD': 1000, 'PSD': 1000, 'SWS': 1000, 'MMV': 1000}
    GRID_RESOLUTION = 101  # 空間密度出力用グリッド (101x101x101)
    GRID_MAX_RM = 5.0  # 計算領域半径 [RM]

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

    # === 動作モード設定フラグ ===
    settings = {
        'BETA': 1.0,
        'T1AU': 54500.0,
        'GRID_RADIUS_RM': GRID_MAX_RM + 1.0,
        'USE_SOLAR_GRAVITY': True,
        'USE_CORIOLIS_FORCES': True,
        'USE_EQUILIBRIUM_MODE': True,  # 高速放出時の平衡解モード

        # 面積比によるFlux重み付け
        # グリッド境界で「昼のFlux」と「夜のFlux」を面積比で線形結合します。
        # これにより、温度平均による過小評価を防ぎ、かつ滑らかな立ち上がりを実現します。
        'USE_AREA_WEIGHTED_FLUX': False,

        # 温度スムージング
        # 温度そのものを平均化してからFlux計算を行うモード。Flux重み付けを使う場合はオフにしてください。
        'USE_SUBGRID_SMOOTHING': False
    }

    # ==========================================================================
    # 2. 初期化処理
    # ==========================================================================
    # 実行名とディレクトリ作成
    mode_str = "EqMode" if settings['USE_EQUILIBRIUM_MODE'] else "NoEq"
    smooth_str = "FluxW" if settings['USE_AREA_WEIGHTED_FLUX'] else (
        "SmoothT" if settings['USE_SUBGRID_SMOOTHING'] else "Hard")
    run_name = f"DynamicGrid{N_LON_FIXED}x{N_LAT}_{mode_str}_{smooth_str}_DT500_suzuki_1.0"
    target_output_dir = os.path.join(OUTPUT_DIRECTORY, run_name)
    os.makedirs(target_output_dir, exist_ok=True)
    print(f"Simulation Start. Results: {target_output_dir}")

    # 表面グリッド定義
    lon_edges = np.linspace(-np.pi, np.pi, N_LON_FIXED + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    dlon = lon_edges[1] - lon_edges[0]
    # 各緯度帯のセル面積計算
    cell_areas = (PHYSICAL_CONSTANTS['RM'] ** 2) * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))

    # 表面密度マップ初期化
    surface_density = np.full((N_LON_FIXED, N_LAT), INIT_SURF_DENS, dtype=np.float64)

    # 外部データ読み込み
    try:
        spec_np = np.loadtxt('SolarSpectrum_Na0.txt', usecols=(0, 3))
        orbit_data = np.loadtxt('orbit2025_spice_unwrapped.txt')
        # 角度データの正規化 (rad -> deg -> unwrap -> deg)
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

    # シミュレーション状態変数
    active_particles = []
    prev_taa = -999

    # キャッシュ用配列 (計算高速化のため)
    cached_rate_psd = np.zeros(surface_density.shape)
    cached_rate_td = np.zeros(surface_density.shape)
    cached_rate_sws = np.zeros(surface_density.shape)
    cached_loss_rate_grid = np.zeros(surface_density.shape)
    accumulated_gained_grid = np.zeros_like(surface_density)

    time_since_last_update = DT_RATE_UPDATE * 2.0
    total_steps = int((t_end_run - t_start_spinup) / DT_MOVE)
    step_count = 0

    # 表面温度モデル定数★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    scaling_params_t_base = 100.0
    scaling_params_t_amp = 600.0
    scaling_params_t_night = 100.0
    half_grid_width_rad = dlon / 2.0
    sin_half_width = np.sin(half_grid_width_rad)

    # ==========================================================================
    # 3. メインループ
    # ==========================================================================
    while t_curr < t_end_run:
        step_count += 1

        # 3-1. 軌道情報の更新
        TAA_raw, AU, V_rad, V_tan, sub_lon = get_orbital_params_linear(t_curr, orbit_data, t_peri_file)
        TAA = TAA_raw % 360.0
        time_since_last_update += DT_MOVE

        # ----------------------------------------------------------------------
        # A. 表面放出率マップの更新 (DT_RATE_UPDATE ごとに実行)
        # ----------------------------------------------------------------------
        if time_since_last_update >= DT_RATE_UPDATE:
            dt_accumulated = time_since_last_update
            f_uv = F_UV_1AU / (AU ** 2)
            sw_flux = SWS_PARAMS['FLUX_1AU'] / (AU ** 2)
            mmv_flux = calculate_mmv_flux(AU)
            supply_dens = mmv_flux * dt_accumulated

            scaling = (0.306 / AU) ** 2
            temp_rate_psd = np.zeros_like(surface_density)
            temp_rate_td = np.zeros_like(surface_density)
            temp_rate_sws = np.zeros_like(surface_density)
            temp_loss_per_sec = np.zeros_like(surface_density)

            # グリッドループ
            for i in range(N_LON_FIXED):
                for j in range(N_LAT):
                    lon_f = (lon_edges[i] + lon_edges[i + 1]) / 2
                    lat_f = (lat_edges[j] + lat_edges[j + 1]) / 2

                    # --- 幾何学計算: 照明率 (illum_frac) ---
                    # 中心座標でのcos
                    cos_z_center = np.cos(lat_f) * np.cos(lon_f - sub_lon)

                    illum_frac = 0.0
                    if cos_z_center > sin_half_width:
                        illum_frac = 1.0
                    elif cos_z_center < -sin_half_width:
                        illum_frac = 0.0
                    else:
                        # 境界線形補間
                        illum_frac = (cos_z_center - (-sin_half_width)) / (2 * sin_half_width)
                        illum_frac = np.clip(illum_frac, 0.0, 1.0)

                    # フラグが両方FalseならBinary動作 (0 or 1)
                    if not settings['USE_AREA_WEIGHTED_FLUX'] and not settings['USE_SUBGRID_SMOOTHING']:
                        illum_frac = 1.0 if cos_z_center > 0 else 0.0

                    # 昼側の理論最大温度 (コサイン項は正の値のみ)
                    eff_cos = max(0.0, cos_z_center)
                    # T_day_potential = scaling_params_t_base * scaling + \
                    #                  scaling_params_t_amp * (eff_cos ** 0.25)
                    T_day_potential = scaling_params_t_base + \
                                      scaling_params_t_amp * (eff_cos ** 0.25) * scaling

                    # --- 1. PSD計算 (面積比を適用) ---
                    if illum_frac > 0:
                        temp_rate_psd[i, j] = f_uv * Q_PSD * eff_cos * illum_frac

                    # --- 2. TD計算 (フラグ分岐) ---
                    if settings['USE_AREA_WEIGHTED_FLUX']:
                        # === 新機能: Flux重み付け (推奨) ===
                        # 昼温度でのFlux、夜温度でのFluxをそれぞれ計算し、面積比で混ぜる
                        rate_day = calculate_thermal_desorption_rate(T_day_potential)
                        rate_night = calculate_thermal_desorption_rate(scaling_params_t_night)

                        temp_rate_td[i, j] = rate_day * illum_frac + rate_night * (1.0 - illum_frac)

                    elif settings['USE_SUBGRID_SMOOTHING']:
                        # === 旧機能: 温度スムージング (非推奨) ===
                        # 温度を混ぜてからFlux計算 (ぬるま湯問題によりFluxが過小評価される)
                        temp_mix = T_day_potential * illum_frac + \
                                   scaling_params_t_night * (1.0 - illum_frac)
                        temp_rate_td[i, j] = calculate_thermal_desorption_rate(temp_mix)

                    else:
                        # === Binary Mode ===
                        temp_val = T_day_potential if illum_frac > 0.5 else scaling_params_t_night
                        temp_rate_td[i, j] = calculate_thermal_desorption_rate(temp_val)

                    # --- 3. SWS計算 (簡易領域判定) ---
                    lon_sun = (lon_f - sub_lon + np.pi) % (2 * np.pi) - np.pi
                    in_lon = SWS_PARAMS['LON_RANGE'][0] <= lon_sun <= SWS_PARAMS['LON_RANGE'][1]
                    in_lat = (SWS_PARAMS['LAT_N_RANGE'][0] <= lat_f <= SWS_PARAMS['LAT_N_RANGE'][1]) or \
                             (SWS_PARAMS['LAT_S_RANGE'][0] <= lat_f <= SWS_PARAMS['LAT_S_RANGE'][1])
                    if in_lon and in_lat:
                        temp_rate_sws[i, j] = (sw_flux * SWS_PARAMS['YIELD']) / SWS_PARAMS['REF_DENS']

                    # --- 物質収支計算 (Equilibrium Mode対応) ---
                    rate_total = temp_rate_psd[i, j] + temp_rate_td[i, j] + temp_rate_sws[i, j]
                    dens = surface_density[i, j]
                    gain_dens = accumulated_gained_grid[i, j] / cell_areas[j]

                    timescale = 1.0 / rate_total if rate_total > 1e-30 else float('inf')

                    # === 修正版: Step 1 だけ強制的に時間発展(D*R*dt)にする ===

                    current_dens = surface_density[i, j]
                    gain_dens = accumulated_gained_grid[i, j] / cell_areas[j]
                    total_input_dens = gain_dens + supply_dens
                    loss_dens = 0.0

                    # ★ここが変更点★
                    # 「2ステップ目以降」かつ「時定数が短い」場合のみ平衡モード
                    # つまり、Step 1 は必ず else に行きます
                    allow_eq_mode = (step_count > 1)

                    if settings['USE_EQUILIBRIUM_MODE'] and allow_eq_mode and (timescale <= dt_accumulated):
                        # --- [Step 2以降] 平衡モード ---
                        if rate_total > 1e-30:
                            # 平衡密度 = 入力フラックス / 放出率
                            # (Input / dt) / Rate
                            dens_eq = (total_input_dens / dt_accumulated) / rate_total
                        else:
                            dens_eq = current_dens + total_input_dens

                        surface_density[i, j] = dens_eq

                        actual_loss_dens = total_input_dens

                    else:
                        # --- [Step 1 or Slow] 通常の時間発展 ---
                        # 理論上の放出量
                        calculated_loss = current_dens * rate_total * dt_accumulated

                        # 在庫以上は出せない
                        actual_loss_dens = min(current_dens, calculated_loss)

                        # 密度の更新
                        surface_density[i, j] += total_input_dens - actual_loss_dens

                    if surface_density[i, j] < 0: surface_density[i, j] = 0

                    # [atoms/m^2] * [m^2] / [s] = [atoms/s]
                    temp_loss_per_sec[i, j] = actual_loss_dens * cell_areas[j] / dt_accumulated

            # キャッシュ更新
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

        # --- MMVソース (全球一様) ---
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

        # --- 表面依存ソース (PSD, TD, SWS) ---
        total_loss_step = cached_loss_rate_grid * DT_MOVE
        with np.errstate(divide='ignore', invalid='ignore'):
            rate_tot = cached_rate_psd + cached_rate_td + cached_rate_sws
            frac_psd = np.where(rate_tot > 0, cached_rate_psd / rate_tot, 0)
            frac_td = np.where(rate_tot > 0, cached_rate_td / rate_tot, 0)
            frac_sws = np.where(rate_tot > 0, cached_rate_sws / rate_tot, 0)

        # 重み計算
        atoms_psd_step = np.sum(total_loss_step * frac_psd)
        atoms_td_step = np.sum(total_loss_step * frac_td)
        atoms_sws_step = np.sum(total_loss_step * frac_sws)
        scale_factor = DT_MOVE / DT_RATE_UPDATE
        w_psd = max(1.0, atoms_psd_step / (TARGET_SPS['PSD'] * scale_factor))
        w_td = max(1.0, atoms_td_step / (TARGET_SPS['TD'] * scale_factor))
        w_sws = max(1.0, atoms_sws_step / (TARGET_SPS['SWS'] * scale_factor))

        # グリッドループ
        for i in range(N_LON_FIXED):
            for j in range(N_LAT):
                n_lost = total_loss_step[i, j]
                if n_lost <= 0: continue
                lon_f = (lon_edges[i] + lon_edges[i + 1]) / 2
                lat_f = (lat_edges[j] + lat_edges[j + 1]) / 2

                # === 生成粒子の温度決定ロジック ===
                cos_z_center = np.cos(lat_f) * np.cos(lon_f - sub_lon)
                eff_cos = max(0.0, cos_z_center)
                T_day_potential = scaling_params_t_base * scaling + \
                                  scaling_params_t_amp * (eff_cos ** 0.25)

                # illum_frac 再計算
                illum_frac = 0.0
                if cos_z_center > sin_half_width:
                    illum_frac = 1.0
                elif cos_z_center < -sin_half_width:
                    illum_frac = 0.0
                else:
                    illum_frac = np.clip((cos_z_center + sin_half_width) / (2 * sin_half_width), 0, 1)

                if not settings['USE_AREA_WEIGHTED_FLUX'] and not settings['USE_SUBGRID_SMOOTHING']:
                    illum_frac = 1.0 if cos_z_center > 0 else 0.0

                # 速度分布用の実効温度 (temp_eff_for_vel)
                temp_eff_for_vel = scaling_params_t_night

                if settings['USE_AREA_WEIGHTED_FLUX']:
                    # ★Flux重み付けモードの場合
                    # Fluxの大半は昼側領域から出るため、粒子の初速度も「昼温度」で近似するのが物理的に妥当
                    if illum_frac > 0.0:
                        temp_eff_for_vel = T_day_potential
                    else:
                        temp_eff_for_vel = scaling_params_t_night

                elif settings['USE_SUBGRID_SMOOTHING']:
                    temp_eff_for_vel = T_day_potential * illum_frac + scaling_params_t_night * (1.0 - illum_frac)

                else:
                    temp_eff_for_vel = T_day_potential if illum_frac > 0.5 else scaling_params_t_night

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

                # 3次元密度グリッド作成
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

        # 進捗ログ
        if step_count % 100 == 0:
            elapsed = time.time() - start_time
            progress_pct = (step_count / total_steps) * 100
            print(
                f"Step {step_count}/{total_steps} ({progress_pct:.1f}%) | TAA={TAA:.2f} | Particles={len(active_particles)} | Elapsed={elapsed:.1f}s")

        prev_taa = TAA
        t_curr += DT_MOVE

    print("Done. Simulation Completed.")


if __name__ == '__main__':
    # multiprocessingのWindows環境バグ回避
    sys.modules['__main__'].__spec__ = None
    main_snapshot_simulation()