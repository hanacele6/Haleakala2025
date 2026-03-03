# -*- coding: utf-8 -*-
"""
水星ナトリウム大気 3次元時間発展モンテカルロシミュレーションコード
[SWS (CPS) 単体バージョン]

==============================================================================
概要
==============================================================================
このスクリプトは、水星のナトリウム大気のふるまいを、時間発展を考慮して
シミュレートする3次元モンテカルロ法に基づいたプログラムです。

粒子生成源として、太陽風スパッタリング (SWS / CPS) のみ を考慮します。

==============================================================================
座標系
==============================================================================
このシミュレーションは、「水星中心・太陽固定回転座標系」を採用しています。

-   **原点 (0, 0, 0)**: 水星の中心
-   **+X 軸**: 常に太陽の方向を指します (Sun-Mercury line)
-   **-Y 軸**: 水星の公転軌道面に含まれ、公転の進行方向を指します
-   **+Z 軸**: 軌道面に垂直な方向（公転の角運動量ベクトル方向）

==============================================================================
主な物理モデル
==============================================================================
1.  **粒子生成 (Solar Wind Sputtering, SWS)**:
    -   太陽風フラックス、スパッタリング収率、表面密度に基づき、
        日照側の特定領域（高緯度帯）から粒子が生成されます。
    -   エネルギー分布は Thompson-Sigmund 分布に従います。
    -   本バージョンでは、表面密度は無限供給源（常に一定）と仮定しています。

2.  **初期速度**:
    -   SWS: 表面束縛エネルギー U=0.27eV の Thompson-Sigmund 分布に従います。
    -   放出角度: 表面の法線方向を基準としたランバート（余弦則）分布に従います。

3.  **軌道計算 (4次ルンゲ＝クッタ法)**:
    -   粒子にかかる力として以下を考慮します。
        1.  水星の重力 (中心力)
        2.  太陽光の放射圧 (SRP, -X方向の力, ドップラーシフト考慮)
        3.  太陽の重力 (潮汐力として作用)
        4.  遠心力 (回転座標系による見かけの力)
        5.  コリオリ力 (回転座標系による見かけの力)
    -   軌道積分には4次のルンゲ＝クッタ（RK4）法を使用します。

4.  **消滅過程**:
    -   光電離: 太陽光に照らされている領域（+X側）を飛行する粒子は、
                 確率的にイオン化され、シミュレーションから除去されます。
    -   表面衝突: 表面に再衝突した粒子は、衝突地点の局所表面温度に応じた
                 吸着確率(sticking probability)で吸着（消滅）します。
                 吸着しなかった場合は、エネルギーを交換して熱的に再放出されます。

==============================================================================
必要な外部ファイル
==============================================================================
1.  **orbit2025_v5.txt**:
    -   水星の軌道パラメータ（TAA, AU, Time, V_radial, V_tangential）を
        時系列で格納したテキストファイル。
2.  **SolarSpectrum_Na0.txt**:
    -   太陽光のスペクトルデータ（波長 vs フラックス比）。
    -   放射圧のドップラーシフト計算に使用されます。

"""

import numpy as np
import sys
import os
from multiprocessing import Pool, cpu_count  # 並列処理のため
from tqdm import tqdm  # 進捗バー表示のため
import time

# ==============================================================================
# 物理定数 (SI単位系)
# ==============================================================================
PHYSICAL_CONSTANTS = {
    'PI': np.pi,  # 円周率
    'AU': 1.496e11,  # 天文単位 (Astronomical Unit) [m]
    'MASS_NA': 22.98976928 * 1.66054e-27,  # ナトリウム原子の質量 [kg]
    'K_BOLTZMANN': 1.380649e-23,  # ボルツマン定数 [J/K]
    'GM_MERCURY': 2.2032e13,  # 水星の重力定数 G * M_Mercury [m^3/s^2]
    'RM': 2.440e6,  # 水星の半径 (Radius of Mercury) [m]
    'C': 299792458.0,  # 光速 [m/s]
    'H': 6.62607015e-34,  # プランク定数 [J·s]
    'E_CHARGE': 1.602176634e-19,  # 電気素量 [C] (eV <-> J 変換用)
    'ME': 9.1093897e-31,  # 電子の質量 [kg]
    'EPSILON_0': 8.854187817e-12,  # 真空の誘電率 [F/m]
    'G': 6.6743e-11,  # 万有引力定数 [N m^2/kg^2]
    'MASS_SUN': 1.989e30,  # 太陽の質量 [kg]
}


# ==============================================================================
# 物理モデルに基づくヘルパー関数群
# ==============================================================================

def calculate_surface_temperature(lon_rad, lat_rad, AU, subsolar_lon_rad):
    """
    水星表面の局所的な温度を計算します。
    (既存のコードをそのまま使用)
    """
    T0 = 100.0  # 夜側の最低温度 [K]
    T1 = 600.0  # 日照による最大温度上昇の係数 [K]
    cos_theta = np.cos(lat_rad) * np.cos(lon_rad - subsolar_lon_rad)
    if cos_theta <= 0:
        return T0  # 夜側
    return T0 + T1 * (cos_theta ** 0.25) * ((0.306 / AU) ** 2)


def calculate_sticking_probability(surface_temp_K):
    """
    表面温度に基づき、ナトリウム原子が表面に吸着する確率を計算します。
    (既存のコードをそのまま使用)
    """
    A = 0.08
    B = 458.0
    porosity = 0.8  # 表面の多孔性 (0-1)
    if surface_temp_K <= 0:
        return 1.0
    p_stick = A * np.exp(B / surface_temp_K)
    p_stick_eff = p_stick / (1.0 - (1.0 - p_stick) * porosity)
    return min(p_stick_eff, 1.0)


def sample_maxwellian_speed(mass_kg, temp_k):
    """
    マクスウェル分布に従う速さをサンプリングします。
    (既存のコードをそのまま使用。SWSでは使わないが、将来のTD実装のため残置)
    """
    scale_param = np.sqrt(PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_k / mass_kg)
    vx, vy, vz = np.random.normal(0, scale_param, 3)
    return np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)


def sample_lambertian_direction_local():
    """
    ランバート（余弦則）分布に従う方向ベクトルをローカル座標系で生成します。
    (既存のコードをそのまま使用)
    """
    u1, u2 = np.random.random(2)
    phi = 2 * PHYSICAL_CONSTANTS['PI'] * u1
    cos_theta = np.sqrt(1 - u2)
    sin_theta = np.sqrt(u2)
    return np.array([sin_theta * np.cos(phi),
                     sin_theta * np.sin(phi),
                     cos_theta])


def transform_local_to_world(local_vec, normal_vector):
    """
    ローカル座標系（法線がZ軸）のベクトルをワールド座標系に変換します。
    (既存のコードをそのまま使用)
    """
    local_z_axis = normal_vector / np.linalg.norm(normal_vector)
    world_up = np.array([0., 0., 1.])
    if np.allclose(local_z_axis, world_up) or np.allclose(local_z_axis, -world_up):
        world_up = np.array([0., 1., 0.])
    local_x_axis = np.cross(world_up, local_z_axis)
    local_x_axis /= np.linalg.norm(local_x_axis)
    local_y_axis = np.cross(local_z_axis, local_x_axis)
    return (local_vec[0] * local_x_axis +
            local_vec[1] * local_y_axis +
            local_vec[2] * local_z_axis)


def get_orbital_params(time_sec, orbit_data, mercury_year_sec):
    """
    指定された時刻における水星の軌道パラメータと太陽直下点経度を取得します。
    (既存のコードをそのまま使用)
    """
    ROTATION_PERIOD_SEC = 58.646 * 24 * 3600  # 水星の自転周期 [s]
    current_time_in_orbit = time_sec % mercury_year_sec
    time_col = orbit_data[:, 2]
    taa = np.interp(current_time_in_orbit, time_col, orbit_data[:, 0])
    au = np.interp(current_time_in_orbit, time_col, orbit_data[:, 1])
    v_radial = np.interp(current_time_in_orbit, time_col, orbit_data[:, 3])
    v_tangential = np.interp(current_time_in_orbit, time_col, orbit_data[:, 4])
    subsolar_lon_rad = (2 * np.pi * time_sec / ROTATION_PERIOD_SEC) % (2 * np.pi)
    return taa, au, v_radial, v_tangential, subsolar_lon_rad


def lonlat_to_xyz(lon_rad, lat_rad, radius):
    """
    惑星中心の経度・緯度 [rad] を三次元直交座標 [m] に変換します。
    (既存のコードをそのまま使用)
    """
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return np.array([x, y, z])


# ==============================================================================
# ### SWS (CPS) 用の新しい関数 1 ###
# ==============================================================================

def sample_thompson_sigmund_energy(U_eV, E_max_eV=5.0):
    """
    Thompson-Sigmund 分布 f(E) ∝ E / (E + U)^3 に従うエネルギーを
    棄却サンプリング法 (Rejection Sampling) を用いて生成します。

    Args:
        U_eV (float): 表面束縛エネルギー [eV]
        E_max_eV (float): サンプリングするエネルギーの最大値 [eV]

    Returns:
        float: サンプリングされたエネルギー [eV]
    """
    # f(E) = E / (E + U)^3
    # f(E) は E = U/2 で最大値 f_max = (U/2) / (U/2 + U)^3 = 0.148 / U^2 をとる
    f_max = (U_eV / 2.0) / (U_eV / 2.0 + U_eV) ** 3

    while True:
        # 1. 0からE_maxの範囲でランダムなエネルギー E を試行
        E_try = np.random.uniform(0, E_max_eV)

        # 2. E_try における分布の値を計算
        f_E_try = E_try / (E_try + U_eV) ** 3

        # 3. 0からf_maxの間の乱数 y を生成
        y_try = np.random.uniform(0, f_max)

        # 4. y が f(E) より小さければ、そのEを採用 (Accept)
        if y_try <= f_E_try:
            return E_try


# ==============================================================================
# ### SWS (CPS) 用の新しい関数 2 ###
# ==============================================================================

def generate_particles_sws(current_time_sec, orbital_params, SWS_PARAMS, grid_params, sim_constants, phys_const):
    """
    太陽風スパッタリング (SWS / CPS) によって生成される
    新しいナトリウム粒子（スーパーパーティクル）のリストを生成します。

    Args:
        current_time_sec (float): 現在のシミュレーション時刻 [s]
        orbital_params (tuple): (TAA, AU, V_radial_ms, V_tangential_ms, subsolar_lon_rad)
        SWS_PARAMS (dict): SWS固有のパラメータ辞書
        grid_params (tuple): (lon_edges, lat_edges, cell_areas_m2)
        sim_constants (tuple): (ATOMS_PER_SUPERPARTICLE, CONSTANT_SURFACE_DENSITY, TIME_STEP_SEC)
        phys_const (dict): PHYSICAL_CONSTANTS 辞書

    Returns:
        list: SWSによって新たに生成された粒子状態辞書のリスト
    """

    # ---------------------------------
    # 1. パラメータの展開
    # ---------------------------------
    ATOMS_PER_SUPERPARTICLE, CONSTANT_SURFACE_DENSITY, TIME_STEP_SEC = sim_constants
    TAA, AU, V_radial_ms, V_tangential_ms, subsolar_lon_rad = orbital_params
    lon_edges, lat_edges, cell_areas_m2 = grid_params
    N_LON, N_LAT = len(lon_edges) - 1, len(lat_edges) - 1
    MASS_NA = phys_const['MASS_NA']
    E_CHARGE = phys_const['E_CHARGE']  # eVからJへの変換用
    RM = phys_const['RM']

    # ---------------------------------
    # 2. SWS 固有のパラメータ (SWS_PARAMS 辞書から)
    # ---------------------------------
    SW_DENSITY_1AU = SWS_PARAMS['SW_DENSITY_1AU']  # [particles/m^3]
    SW_VELOCITY = SWS_PARAMS['SW_VELOCITY']  # [m/s]
    YIELD_EFF = SWS_PARAMS['YIELD_EFF']  # [atoms/ion]
    U_eV = SWS_PARAMS['U_eV']  # [eV]
    DENSITY_REF_M2 = SWS_PARAMS['DENSITY_REF_M2']  # [atoms/m^2]

    # SWS 発生領域
    # 太陽直下点経度 (lon=0, subsolar_lon_rad ではない) を中心とする
    SPUTTER_LON_MIN_RAD = np.deg2rad(SWS_PARAMS['LON_MIN_DEG'])
    SPUTTER_LON_MAX_RAD = np.deg2rad(SWS_PARAMS['LON_MAX_DEG'])
    SPUTTER_LAT_N_MIN_RAD = np.deg2rad(SWS_PARAMS['LAT_N_MIN_DEG'])
    SPUTTER_LAT_N_MAX_RAD = np.deg2rad(SWS_PARAMS['LAT_N_MAX_DEG'])
    SPUTTER_LAT_S_MIN_RAD = np.deg2rad(SWS_PARAMS['LAT_S_MIN_DEG'])
    SPUTTER_LAT_S_MAX_RAD = np.deg2rad(SWS_PARAMS['LAT_S_MAX_DEG'])

    # ---------------------------------
    # 3. 太陽風フラックスと生成率の計算
    # ---------------------------------

    # 太陽風フラックス [particles / m^2 / s]
    flux_sw_1au = SW_DENSITY_1AU * SW_VELOCITY  # 1AUでのフラックス
    current_flux_sw = flux_sw_1au / (AU ** 2)

    # 30分周期の変動
    # 周期 T = 1800s (30 min)
    period_sec = 1800.0
    # (1 + cos(2*pi*t/T)) / 2 の形で変動 (平均値が 0.5 になるように)
    temporal_variation_factor = (1.0 + np.cos(2 * np.pi * current_time_sec / period_sec)) / 2.0

    # 実効的な太陽風フラックス
    effective_flux_sw = current_flux_sw #* temporal_variation_factor

    # SWSの生成率 [atoms / m^2 / s]
    concentration_ratio = CONSTANT_SURFACE_DENSITY / DENSITY_REF_M2

    # 基準となるスパッタリング率 (この領域内では一様と仮定)
    sputtering_rate_per_m2_s = effective_flux_sw * YIELD_EFF * concentration_ratio

    if sputtering_rate_per_m2_s <= 0:
        return []

    # ---------------------------------
    # 4. 粒子生成ループ
    # ---------------------------------

    newly_launched_particles = []

    for i_lon in range(N_LON):
        for i_lat in range(N_LAT):

            lon_center_rad = (lon_edges[i_lon] + lon_edges[i_lon + 1]) / 2
            lat_center_rad = (lat_edges[i_lat] + lat_edges[i_lat + 1]) / 2

            # このセルがSWSの発生領域内にあるか判定
            is_in_lon_band = (SPUTTER_LON_MIN_RAD <= lon_center_rad <= SPUTTER_LON_MAX_RAD)
            is_in_lat_band_N = (SPUTTER_LAT_N_MIN_RAD <= lat_center_rad <= SPUTTER_LAT_N_MAX_RAD)
            is_in_lat_band_S = (SPUTTER_LAT_S_MIN_RAD <= lat_center_rad <= SPUTTER_LAT_S_MAX_RAD)

            # 領域外ならスキップ
            if not (is_in_lon_band and (is_in_lat_band_N or is_in_lat_band_S)):
                continue

            # このセルから放出されるべき原子の総数
            n_atoms_to_sputter = sputtering_rate_per_m2_s * cell_areas_m2[i_lat] * TIME_STEP_SEC
            if n_atoms_to_sputter <= 0:
                continue

            # スーパーパーティクル数に変換
            num_sps_to_launch_float = n_atoms_to_sputter / ATOMS_PER_SUPERPARTICLE
            num_to_launch_int = int(num_sps_to_launch_float)
            if np.random.random() < (num_sps_to_launch_float - num_to_launch_int):
                num_to_launch_int += 1
            if num_to_launch_int == 0:
                continue

            # --- 指定された数のSPを生成 ---
            for _ in range(num_to_launch_int):
                # (1) 生成位置 (セル内でランダム)
                random_lon_rad = np.random.uniform(lon_edges[i_lon], lon_edges[i_lon + 1])
                sin_lat_min, sin_lat_max = np.sin(lat_edges[i_lat]), np.sin(lat_edges[i_lat + 1])
                random_lat_rad = np.arcsin(np.random.uniform(sin_lat_min, sin_lat_max))

                initial_pos = lonlat_to_xyz(random_lon_rad, random_lat_rad, RM)
                surface_normal = initial_pos / np.linalg.norm(initial_pos)

                # (2) 初期速度 (Thompson-Sigmund + Lambert)
                energy_eV = sample_thompson_sigmund_energy(U_eV)
                energy_J = energy_eV * E_CHARGE
                speed = np.sqrt(2.0 * energy_J / MASS_NA)
                direction = transform_local_to_world(sample_lambertian_direction_local(), surface_normal)
                initial_vel = speed * direction

                # (3) 粒子をリストに追加
                newly_launched_particles.append({
                    'pos': initial_pos,
                    'vel': initial_vel,
                    'weight': ATOMS_PER_SUPERPARTICLE
                })

    return newly_launched_particles


# ==============================================================================
# コア追跡関数 (並列処理の対象)
# ==============================================================================

def _calculate_acceleration(pos, vel, V_radial_ms, V_tangential_ms, AU, spec_data, settings):
    """
    【シミュレーションの核】
    粒子にかかる総加速度（重力＋放射圧＋見かけの力）を計算します。
    (既存のコードをそのまま使用)
    """
    x, y, z = pos
    r0 = AU * PHYSICAL_CONSTANTS['AU']  # 水星-太陽間距離 [m]

    # --- 1. 太陽放射圧 (Solar Radiation Pressure, SRP) ---
    velocity_for_doppler = vel[0] + V_radial_ms
    w_na_d2 = 589.1582e-9 * (1.0 + velocity_for_doppler / PHYSICAL_CONSTANTS['C'])
    w_na_d1 = 589.7558e-9 * (1.0 + velocity_for_doppler / PHYSICAL_CONSTANTS['C'])
    b = 0.0
    wl, gamma, sigma0_perdnu2, sigma0_perdnu1, JL = spec_data.values()
    if (wl[0] * 1e-9 <= w_na_d2 < wl[-1] * 1e-9) and \
            (wl[0] * 1e-9 <= w_na_d1 < wl[-1] * 1e-9):
        gamma2 = np.interp(w_na_d2 * 1e9, wl, gamma)
        gamma1 = np.interp(w_na_d1 * 1e9, wl, gamma)
        F_lambda_1AU_m = JL * 1e4 * 1e9
        F_lambda_d1_at_Mercury = F_lambda_1AU_m / (AU ** 2) * gamma1
        F_nu_d1 = F_lambda_d1_at_Mercury * (w_na_d1 ** 2) / PHYSICAL_CONSTANTS['C']
        J1 = sigma0_perdnu1 * F_nu_d1
        F_lambda_d2_at_Mercury = F_lambda_1AU_m / (AU ** 2) * gamma2
        F_nu_d2 = F_lambda_d2_at_Mercury * (w_na_d2 ** 2) / PHYSICAL_CONSTANTS['C']
        J2 = sigma0_perdnu2 * F_nu_d2
        b = 1 / PHYSICAL_CONSTANTS['MASS_NA'] * (
                (PHYSICAL_CONSTANTS['H'] / w_na_d1) * J1 + (PHYSICAL_CONSTANTS['H'] / w_na_d2) * J2)
    if x < 0 and np.sqrt(y ** 2 + z ** 2) < PHYSICAL_CONSTANTS['RM']:
        b = 0.0
    accel_srp = np.array([-b, 0.0, 0.0])

    # --- 2. 水星の重力 ---
    r_sq = np.sum(pos ** 2)
    accel_g = -PHYSICAL_CONSTANTS['GM_MERCURY'] * pos / (r_sq ** 1.5) if r_sq > 0 else np.array([0., 0., 0.])

    # --- 3. 太陽の重力 ---
    accel_sun = np.array([0.0, 0.0, 0.0])
    if settings.get('USE_SOLAR_GRAVITY', False):
        G = PHYSICAL_CONSTANTS['G']
        M_SUN = PHYSICAL_CONSTANTS['MASS_SUN']
        r_ps_vec = np.array([r0 - pos[0], -pos[1], -pos[2]])
        r_ps_mag_sq = np.sum(r_ps_vec ** 2)
        if r_ps_mag_sq > 0:
            accel_sun = (G * M_SUN * r_ps_vec) / (r_ps_mag_sq ** 1.5)

    # --- 4. 見かけの力 (コリオリ力・遠心力) ---
    accel_coriolis = np.array([0.0, 0.0, 0.0])
    accel_centrifugal = np.array([0.0, 0.0, 0.0])
    if settings.get('USE_CORIOLIS_FORCES', False):
        if r0 > 0:
            omega_val = V_tangential_ms / r0
            omega_sq = omega_val ** 2
            accel_centrifugal = np.array([
                omega_val ** 2 * (pos[0] - r0),
                omega_sq * pos[1],
                0.0
            ])
            two_omega = 2 * omega_val
            accel_coriolis = np.array([
                two_omega * vel[1],
                -two_omega * vel[0],
                0.0
            ])

    return accel_srp + accel_g + accel_sun + accel_centrifugal + accel_coriolis


def simulate_particle_for_one_step(args):
    """
    一個のスーパーパーティクルを、指定された時間 (duration) だけ追跡します。
    (既存のコードをそのまま使用)
    """
    # ---------------------------------
    # 1. 引数の展開
    # ---------------------------------
    settings, spec_data = args['settings'], args['spec']
    TAA, AU, V_radial_ms, V_tangential_ms, subsolar_lon_rad = args['orbit']
    duration, DT = args['duration'], settings['DT']
    RM = PHYSICAL_CONSTANTS['RM']
    R_MAX = RM * settings['GRID_RADIUS_RM']
    pos, vel, weight = args['particle_state']['pos'], args['particle_state']['vel'], args['particle_state']['weight']
    tau_ionization = settings['T1AU'] * AU ** 2
    num_steps = int(duration / DT)

    # ---------------------------------
    # 2. 時間積分ループ (RK4)
    # ---------------------------------
    for _ in range(num_steps):
        if pos[0] > 0:
            if np.random.random() < (1.0 - np.exp(-DT / tau_ionization)):
                return {'status': 'ionized', 'final_state': None}

        pos_prev = pos.copy()

        k1_vel = DT * _calculate_acceleration(pos, vel, V_radial_ms, V_tangential_ms, AU, spec_data, settings)
        k1_pos = DT * vel
        k2_vel = DT * _calculate_acceleration(pos + 0.5 * k1_pos, vel + 0.5 * k1_vel, V_radial_ms, V_tangential_ms, AU,
                                              spec_data, settings)
        k2_pos = DT * (vel + 0.5 * k1_vel)
        k3_vel = DT * _calculate_acceleration(pos + 0.5 * k2_pos, vel + 0.5 * k2_vel, V_radial_ms, V_tangential_ms, AU,
                                              spec_data, settings)
        k3_pos = DT * (vel + 0.5 * k2_vel)
        k4_vel = DT * _calculate_acceleration(pos + k3_pos, vel + k3_vel, V_radial_ms, V_tangential_ms, AU, spec_data,
                                              settings)
        k4_pos = DT * (vel + k3_vel)

        pos += (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos) / 6.0
        vel += (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel) / 6.0

        r_current = np.linalg.norm(pos)

        if r_current > R_MAX:
            return {'status': 'escaped', 'final_state': None}

        if r_current <= RM:
            impact_lon = np.arctan2(pos_prev[1], pos_prev[0])
            impact_lat = np.arcsin(np.clip(pos_prev[2] / np.linalg.norm(pos_prev), -1.0, 1.0))

            # (注意: 太陽直下点は 0.0 固定で呼び出し)
            temp_at_impact = calculate_surface_temperature(impact_lon, impact_lat, AU, 0.0)

            if np.random.random() < calculate_sticking_probability(temp_at_impact):
                return {'status': 'stuck', 'final_state': None}
            else:
                E_in = 0.5 * PHYSICAL_CONSTANTS['MASS_NA'] * np.sum(vel ** 2)
                E_T = PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_at_impact
                E_out = settings['BETA'] * E_T + (1.0 - settings['BETA']) * E_in
                v_out_speed = np.sqrt(2 * E_out / PHYSICAL_CONSTANTS['MASS_NA']) if E_out > 0 else 0.0
                impact_normal = pos_prev / np.linalg.norm(pos_prev)
                rebound_direction = transform_local_to_world(sample_lambertian_direction_local(), impact_normal)
                vel = v_out_speed * rebound_direction
                pos = RM * impact_normal
                continue

    return {'status': 'alive', 'final_state': {'pos': pos, 'vel': vel, 'weight': weight}}


# ==============================================================================
# メイン制御関数 (★SWS単体バージョンに修正★)
# ==============================================================================

def main_snapshot_simulation():
    """
    シミュレーション全体を制御するメイン関数。
    """
    start_time = time.time()

    # --- 1. シミュレーション設定 ---

    OUTPUT_DIRECTORY = r"./SimulationResult_202510"
    N_LON, N_LAT = 48, 24
    # 表面密度 [atoms/m^2] (7.5e14 atoms/cm^2 * 0.0053)
    CONSTANT_SURFACE_DENSITY = 7.5e14 * (100) ** 2 * 0.0053
    SPIN_UP_YEARS = 0.1
    TIME_STEP_SEC = 1000
    TOTAL_SIM_YEARS = 1.0
    TARGET_TAA_DEGREES = np.arange(0, 360, 1)
    ATOMS_PER_SUPERPARTICLE = 5e21

    # --- (PSDのパラメータはSWS単体版では不要) ---

    # --- ★ SWS (CPS) 用のパラメータ辞書 ★ ---
    SWS_PARAMS = {
        # 1AUでの太陽風平均密度 [particles/m^3] (10 /cm^3)
        'SW_DENSITY_1AU': 10.0 * (100.0) ** 3,
        # 太陽風平均速度 [m/s] (400 km/s)
        'SW_VELOCITY': 400.0 * 1000.0,
        # 実効スパッタリング収率 (多孔性を考慮) [atoms/ion]
        'YIELD_EFF': 0.06,
        # 表面束縛エネルギー (Thompson-Sigmund分布用) [eV]
        'U_eV': 0.27,
        # 表面密度の基準値 [atoms/m^2] (7.5e14 atoms/cm^2)
        'DENSITY_REF_M2': 7.5e14 * (100.0) ** 2,
        # --- スパッタリング発生領域 (太陽固定座標系, 経度=0が太陽直下点) ---
        'LON_MIN_DEG': -40.0,  # (論文では 20-70 deg SZA)
        'LON_MAX_DEG': 40.0,
        'LAT_N_MIN_DEG': 30.0,  # 北半球の帯
        'LAT_N_MAX_DEG': 60.0,
        'LAT_S_MIN_DEG': -60.0,  # 南半球の帯
        'LAT_S_MAX_DEG': -30.0,
    }

    # --- 出力グリッドの設定 ---
    GRID_RESOLUTION = 101
    GRID_MAX_RM = 5.0

    # --- 物理モデルのフラグ ---
    USE_SOLAR_GRAVITY = True
    USE_CORIOLIS_FORCES = True

    # --- その他の設定 (settings辞書にまとめる) ---
    settings = {
        'BETA': 0.5,
        'T1AU': 168918.0,  # (既存コードの値を維持)
        'DT': 1000.0,
        'N_LON': N_LON, 'N_LAT': N_LAT,
        'GRID_RADIUS_RM': GRID_MAX_RM + 1.0,
        'USE_SOLAR_GRAVITY': USE_SOLAR_GRAVITY,
        'USE_CORIOLIS_FORCES': USE_CORIOLIS_FORCES
    }

    # 出力ディレクトリの準備 (★run_name を変更)
    run_name = f"Grid{GRID_RESOLUTION}_Range{int(GRID_MAX_RM)}RM_SP{ATOMS_PER_SUPERPARTICLE:.0e}_SWS_COSOFF"
    target_output_dir = os.path.join(OUTPUT_DIRECTORY, run_name)
    os.makedirs(target_output_dir, exist_ok=True)
    print(f"結果は '{target_output_dir}' に保存されます。")
    print(f"--- 物理モデル設定 ---")
    print(f"Solar Gravity: {USE_SOLAR_GRAVITY}")
    print(f"Coriolis/Centrifugal: {USE_CORIOLIS_FORCES}")
    print(f"Source Model: SWS (CPS) only")
    print(f"----------------------")

    # --- 2. シミュレーションの初期化 ---
    MERCURY_YEAR_SEC = 87.97 * 24 * 3600
    lon_edges = np.linspace(-np.pi, np.pi, N_LON + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    dlon = lon_edges[1] - lon_edges[0]
    cell_areas_m2 = (PHYSICAL_CONSTANTS['RM'] ** 2) * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))

    # --- 3. 外部ファイルの読み込み ---
    try:
        spec_data_np = np.loadtxt('SolarSpectrum_Na0.txt', usecols=(0, 3))
        orbit_file_name = 'orbit2025_v5.txt'
        orbit_data = np.loadtxt(orbit_file_name)
    except FileNotFoundError as e:
        print(f"エラー: データファイル '{e.filename}' が見つかりません。");
        sys.exit()
    if orbit_data.shape[1] < 5:
        print(f"エラー: '{orbit_file_name}' の列が不足しています。")
        sys.exit()

    # --- RUN開始時刻 (TAA=0) の調整 (既存コードのロジックを維持) ---
    taa_col = orbit_data[:, 0]
    time_col = orbit_data[:, 2]
    idx_perihelion = np.argmin(np.abs(taa_col))
    t_start_run = time_col[idx_perihelion]
    t_end_run = t_start_run + (TOTAL_SIM_YEARS * MERCURY_YEAR_SEC)
    t_start_spinup = t_start_run - (SPIN_UP_YEARS * MERCURY_YEAR_SEC)
    time_steps = np.arange(t_start_spinup, t_end_run, TIME_STEP_SEC)

    print(f"--- 時間設定 ---")
    print(f"軌道ファイル上のTAA=0 (近日点) 時刻: {t_start_run:.1f} s")
    print(f"スピンアップ開始時刻: {t_start_spinup:.1f} s ({-SPIN_UP_YEARS} 年前)")
    print(f"RUN開始時刻 (TAA=0): {t_start_run:.1f} s")
    print(f"RUN終了時刻: {t_end_run:.1f} s (+{TOTAL_SIM_YEARS} 年後)")
    print(f"------------------")

    # スペクトルデータの前処理
    wl, gamma = spec_data_np[:, 0], spec_data_np[:, 1]
    if not np.all(np.diff(wl) > 0):
        sort_indices = np.argsort(wl)
        wl, gamma = wl[sort_indices], gamma[sort_indices]
    sigma_const = PHYSICAL_CONSTANTS['E_CHARGE'] ** 2 / (
            4 * PHYSICAL_CONSTANTS['ME'] * PHYSICAL_CONSTANTS['C'] * PHYSICAL_CONSTANTS['EPSILON_0'])
    spec_data_dict = {'wl': wl, 'gamma': gamma,
                      'sigma0_perdnu2': sigma_const * 0.641,
                      'sigma0_perdnu1': sigma_const * 0.320,
                      'JL': 5.18e14}

    # --- 4. メインループ (時間発展) ---
    active_particles = []
    previous_taa = -1
    target_taa_idx = 0

    with tqdm(total=len(time_steps), desc="Time Evolution") as pbar:
        for t_sec in time_steps:

            # --- 4a. 現在時刻の軌道パラメータを取得 ---
            TAA, AU, V_radial_ms, V_tangential_ms, subsolar_lon_rad = get_orbital_params(
                t_sec, orbit_data, MERCURY_YEAR_SEC
            )
            current_orbital_params = (TAA, AU, V_radial_ms, V_tangential_ms, subsolar_lon_rad)

            run_phase = "Spin-up" if t_sec < t_start_run else "Run"
            pbar.set_description(f"[{run_phase}] TAA={TAA:.1f} | N_particles={len(active_particles)}")

            # --- 4b. 表面から新しい粒子を生成 (★SWSのみに修正★) ---

            # (i) SWS (CPS) による生成
            current_grid_params = (lon_edges, lat_edges, cell_areas_m2)
            current_sim_constants = (ATOMS_PER_SUPERPARTICLE, CONSTANT_SURFACE_DENSITY, TIME_STEP_SEC)

            newly_launched_sws = generate_particles_sws(
                t_sec,
                current_orbital_params,
                SWS_PARAMS,
                current_grid_params,
                current_sim_constants,
                PHYSICAL_CONSTANTS
            )

            # --- 4c. 全ての新しい粒子をアクティブリストに追加 ---
            active_particles.extend(newly_launched_sws)

            # --- 4d. 全ての粒子を1ステップ進める (並列処理) ---
            # (元コードの 4c. に相当)
            tasks = [{'settings': settings, 'spec': spec_data_dict, 'particle_state': p,
                      'orbit': (TAA, AU, V_radial_ms, V_tangential_ms, subsolar_lon_rad),
                      'duration': TIME_STEP_SEC} for p in
                     active_particles]
            next_active_particles = []
            if tasks:
                # (既存のコードの並列処理ロジックを維持)
                with Pool(processes=max(1, cpu_count() - 1)) as pool:
                    results = list(pool.imap(simulate_particle_for_one_step, tasks, chunksize=100))
                for res in results:
                    if res['status'] == 'alive':
                        next_active_particles.append(res['final_state'])
            active_particles = next_active_particles

            # --- 4e. スナップショット保存判定 (既存コードのロジックを維持) ---
            save_this_step = False
            if TAA < previous_taa:
                target_taa_idx = 0
            if target_taa_idx < len(TARGET_TAA_DEGREES):
                current_target_taa = TARGET_TAA_DEGREES[target_taa_idx]
                is_crossing_zero = (current_target_taa == 0) and \
                                   ((TAA < previous_taa) or (TAA >= 0 and previous_taa < 0))
                is_crossing_normal = (previous_taa < current_target_taa <= TAA)
                if is_crossing_normal or is_crossing_zero:
                    save_this_step = True
                    target_taa_idx += 1

            # --- 4f. 立方体グリッドに集計して保存 (既存コードのロジックを維持) ---
            if save_this_step and t_sec >= t_start_run:
                pbar.write(f"\n>>> [Run] Saving grid snapshot at TAA={TAA:.1f} ({len(active_particles)} particles) <<<")

                density_grid = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION, GRID_RESOLUTION), dtype=np.float32)
                grid_min = -GRID_MAX_RM * PHYSICAL_CONSTANTS['RM']
                grid_max = GRID_MAX_RM * PHYSICAL_CONSTANTS['RM']
                cell_size = (grid_max - grid_min) / GRID_RESOLUTION
                cell_volume_m3 = cell_size ** 3

                for p in active_particles:
                    pos = p['pos']
                    ix = int((pos[0] - grid_min) / cell_size)
                    iy = int((pos[1] - grid_min) / cell_size)
                    iz = int((pos[2] - grid_min) / cell_size)
                    if 0 <= ix < GRID_RESOLUTION and 0 <= iy < GRID_RESOLUTION and 0 <= iz < GRID_RESOLUTION:
                        density_grid[ix, iy, iz] += p['weight']

                density_grid /= cell_volume_m3

                relative_time_sec = t_sec - t_start_run
                save_time_h = relative_time_sec / 3600
                filename = f"density_grid_t{int(save_time_h):05d}_taa{int(round(TAA)):03d}.npy"
                np.save(os.path.join(target_output_dir, filename), density_grid)

            previous_taa = TAA
            pbar.update(1)

    # --- 5. 終了 ---
    end_time = time.time()
    print(f"\n★★★ シミュレーションが完了しました ★★★")
    print(f"総計算時間: {(end_time - start_time) / 3600:.2f} 時間")


if __name__ == '__main__':
    print("必須ファイルを確認しています...")
    for f in ['orbit2025_v5.txt', 'SolarSpectrum_Na0.txt']:
        if not os.path.exists(f):
            print(f"エラー: 必須ファイル '{f}' が見つかりません。スクリプトと同じディレクトリに配置してください。")
            if f == 'orbit2025_v5.txt':
                print("（orbit2025_v5.txt がない場合は、軌道生成スクリプトを先に実行してください）")
            sys.exit()

    print("ファイルOK。シミュレーションを開始します。")
    main_snapshot_simulation()