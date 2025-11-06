# -*- coding: utf-8 -*-
"""
==============================================================================
概要
==============================================================================
このスクリプトは、水星のナトリウム大気のふるまいを、時間発展を考慮して
3次元モンテカルロ法に基づいてシミュレーションを行うことを目的にしたコードです。

Leblanc (2003) の論文に基づき、以下の特徴を持ちます。

1.  動的表面密度:
    - 水星表面を惑星固定座標系のグリッドで管理します。
    - 各セルのナトリウム密度は、放出によって減少し、吸着によって増加します。

2.  複数の生成過程:
    - 光刺激脱離 (PSD): 表面密度と太陽光に比例。
    - 熱脱離 (TD): 表面密度と表面温度に比例。
    - 微小隕石蒸発 (MIV/MMV): 表面密度に依存しない補給源。
    - 太陽風イオンスパッタリング (SWS): 表面密度に比例し、太陽風フラックスに依存。

==============================================================================
座標系
==============================================================================
このシミュレーションは、2つの座標系を併用します。

1.  惑星固定座標系（水星自転座標系）:
    - 表面グリッド(surface_density_grid)、表面温度計算に使用。
    - 経度 0 は水星の特定の地点に固定。

2.  水星中心・太陽固定回転座標系:
    - 粒子の軌道追跡、空間密度グリッドの集計に使用。
    - +X方向: 常に太陽の方向。
    - -Y方向: 自転の進行方向。
    - メインループで計算される `subsolar_lon_rad`（惑星固定座標系での
      太陽直下点経度）を用いて、2つの座標系をマッピングします。

==============================================================================
主な物理モデル (Leblanc 2003 準拠)
==============================================================================
1.  粒子生成:
    - PSD, TD, SWS: 表面のナトリウム貯蔵庫を消費します。
    - MMV: 外部からの補給源として機能します 。

2.  初期速度:
    - PSD (T=1500K), TD (T=表面温度), MMV (T=3000K)
      それぞれの温度におけるマクスウェル「フラックス」分布に従う速度。
    - SWS (U=0.27eV): Thompson-Sigmund 分B布に従うエネルギー。
    - 放出角度: ランバート（余弦則）分布。

3.  軌道計算 (4次ルンゲ＝クッタ法):
    - (変更なし：水星重み力、SRP、太陽重み力、見かけの力)

4.  消滅過程:
    - 光電離: (変更なし)
    - 表面衝突: (変更なし)

==============================================================================
必要な外部ファイル
==============================================================================
1.  orbit2025_v5.txt: 視線速度、惑星間距離などの情報を計算したもの
2.  SolarSpectrum_Na0.txt: 放射圧計算用の波長ごとのgammaが記録されたもの
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
    'AU': 1.496e11,  # [m]
    'MASS_NA': 22.98976928 * 1.66054e-27,  # [kg]
    'K_BOLTZMANN': 1.380649e-23,  # [J/K]
    'GM_MERCURY': 2.2032e13,  # [m^3/s^2]
    'RM': 2.440e6,  # [m]
    'C': 299792458.0,  # [m/s]
    'H': 6.62607015e-34,  # [J·s]
    'E_CHARGE': 1.602176634e-19,  # [C]
    'ME': 9.1093897e-31,  # [kg]
    'EPSILON_0': 8.854187817e-12,  # [F/m]
    'G': 6.6743e-11,  # [N m^2/kg^2]
    'MASS_SUN': 1.989e30,  # [kg]
    'EV_TO_JOULE': 1.602176634e-19,  # [J / eV]
}


# ==============================================================================
# 物理モデルに基づくヘルパー関数群
# ==============================================================================

def calculate_surface_temperature_leblanc(lon_fixed_rad, lat_rad, AU, subsolar_lon_rad):
    """ (ベースコードから変更なし) """
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
    """ (ベースコードから変更なし) """
    A = 0.0804
    B = 458.0
    porosity = 0.8
    if surface_temp_K <= 0:
        return 1.0
    p_stick = A * np.exp(B / surface_temp_K)
    p_stick_eff = p_stick / (1.0 - (1.0 - p_stick) * porosity)
    return min(p_stick_eff, 1.0)


def calculate_thermal_desorption_rate(surface_temp_K):
    """ (ベースコードから変更なし) """
    if surface_temp_K < 350.0:
        return 0.0
    VIB_FREQ = 1e13
    BINDING_ENERGY_EV = 1.85
    BINDING_ENERGY_J = BINDING_ENERGY_EV * PHYSICAL_CONSTANTS['EV_TO_JOULE']
    k_B = PHYSICAL_CONSTANTS['K_BOLTZMANN']
    exponent = -BINDING_ENERGY_J / (k_B * surface_temp_K)
    if exponent < -700:
        return 0.0
    return VIB_FREQ * np.exp(exponent)


def calculate_mmv_flux(AU):
    """ (ベースコードから変更なし) """
    TOTAL_FLUX_AT_PERI_NA_S = 5e23
    PERIHELION_AU = 0.307
    MERCURY_SURFACE_AREA_M2 = 4 * PHYSICAL_CONSTANTS['PI'] * (PHYSICAL_CONSTANTS['RM'] ** 2)
    avg_flux_at_peri = TOTAL_FLUX_AT_PERI_NA_S / MERCURY_SURFACE_AREA_M2
    C = avg_flux_at_peri * (PERIHELION_AU ** 1.9)
    return C * (AU ** (-1.9))


def sample_speed_from_flux_distribution(mass_kg, temp_k):
    """ (ベースコードから変更なし) """
    kT = PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_k
    E = np.random.gamma(2.0, kT)
    return np.sqrt(2.0 * E / mass_kg)


# ==============================================================================
# ### SWS (CPS) 用の新しい関数 (SWSコードより) ###
# ==============================================================================

def sample_thompson_sigmund_energy(U_eV, E_max_eV=5.0):
    """
    Thompson-Sigmund 分布 f(E) ∝ E / (E + U)^3 に従うエネルギーを
    棄却サンプリング法 (Rejection Sampling) を用いて生成します。

    Args:
        U_eV (float): 表面束縛エネルギー [eV]
        E_max_eV (float): サンプリングするエネルギーの最大値 [eV] (論文に基づき~5eVで十分)

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
# (ここまでSWS用ヘルパー)
# ==============================================================================


def sample_lambertian_direction_local():
    """ (ベースコードから変更なし) """
    u1, u2 = np.random.random(2)
    phi = 2 * PHYSICAL_CONSTANTS['PI'] * u1
    cos_theta = np.sqrt(1 - u2)
    sin_theta = np.sqrt(u2)
    return np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])


def transform_local_to_world(local_vec, normal_vector):
    """ (ベースコードから変更なし) """
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
    """ (ベースコードから変更なし) """
    ROTATION_PERIOD_SEC = 58.646 * 24 * 3600
    current_time_in_orbit = time_sec % mercury_year_sec
    time_col = orbit_data[:, 2]
    taa = np.interp(current_time_in_orbit, time_col, orbit_data[:, 0])
    au = np.interp(current_time_in_orbit, time_col, orbit_data[:, 1])
    v_radial = np.interp(current_time_in_orbit, time_col, orbit_data[:, 3])
    v_tangential = np.interp(current_time_in_orbit, time_col, orbit_data[:, 4])
    subsolar_lon_rad = (2 * np.pi * time_sec / ROTATION_PERIOD_SEC) % (2 * np.pi)
    return taa, au, v_radial, v_tangential, subsolar_lon_rad


def lonlat_to_xyz(lon_rad, lat_rad, radius):
    """ (ベースコードから変更なし) """
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return np.array([x, y, z])


def xyz_to_lonlat_idx(pos_vec, lon_edges_fixed, lat_edges_fixed, N_LON_FIXED, N_LAT):
    """ (ベースコードから変更なし) """
    r = np.linalg.norm(pos_vec)
    if r == 0: return -1, -1
    lon_rot = np.arctan2(pos_vec[1], pos_vec[0])
    lat_rot = np.arcsin(np.clip(pos_vec[2] / r, -1.0, 1.0))
    i_lon = np.searchsorted(lon_edges_fixed, lon_rot) - 1
    i_lat = np.searchsorted(lat_edges_fixed, lat_rot) - 1
    if 0 <= i_lon < N_LON_FIXED and 0 <= i_lat < N_LAT:
        return i_lon, i_lat
    else:
        return -1, -1


def _calculate_acceleration(pos, vel, V_radial_ms, V_tangential_ms, AU, spec_data, settings):
    """ (ベースコードから変更なし) """
    x, y, z = pos
    r0 = AU * PHYSICAL_CONSTANTS['AU']
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
    r_sq = np.sum(pos ** 2)
    accel_g = -PHYSICAL_CONSTANTS['GM_MERCURY'] * pos / (r_sq ** 1.5) if r_sq > 0 else np.array([0., 0., 0.])
    accel_sun = np.array([0.0, 0.0, 0.0])
    if settings.get('USE_SOLAR_GRAVITY', False):
        G = PHYSICAL_CONSTANTS['G']
        M_SUN = PHYSICAL_CONSTANTS['MASS_SUN']
        r_ps_vec = np.array([r0 - pos[0], -pos[1], -pos[2]])
        r_ps_mag_sq = np.sum(r_ps_vec ** 2)
        if r_ps_mag_sq > 0:
            accel_sun = (G * M_SUN * r_ps_vec) / (r_ps_mag_sq ** 1.5)
    accel_coriolis = np.array([0.0, 0.0, 0.0])
    accel_centrifugal = np.array([0.0, 0.0, 0.0])
    if settings.get('USE_CORIOLIS_FORCES', False):
        if r0 > 0:
            omega_val = V_tangential_ms / r0
            omega_sq = omega_val ** 2
            accel_centrifugal = np.array([
                omega_val ** 2 * (pos[0] - r0),
                omega_sq * pos[1],
                0.0])
            two_omega = 2 * omega_val
            accel_coriolis = np.array([two_omega * vel[1], -two_omega * vel[0], 0.0])
    return accel_srp + accel_g + accel_sun + accel_centrifugal + accel_coriolis


def simulate_particle_for_one_step(args):
    """ (ベースコードから変更なし) """
    settings, spec_data = args['settings'], args['spec']
    TAA, AU, V_radial_ms, V_tangential_ms, subsolar_lon_rad_fixed = args['orbit']
    duration, DT = args['duration'], settings['DT']
    RM = PHYSICAL_CONSTANTS['RM']
    R_MAX = RM * settings['GRID_RADIUS_RM']
    pos, vel, weight = args['particle_state']['pos'], args['particle_state']['vel'], args['particle_state']['weight']
    tau_ionization = settings['T1AU'] * AU ** 2
    num_steps = int(duration / DT)
    pos_at_start_of_step = pos.copy()
    for _ in range(num_steps):
        pos_at_start_of_step = pos.copy()
        if pos[0] > 0:
            if np.random.random() < (1.0 - np.exp(-DT / tau_ionization)):
                return {'status': 'ionized', 'final_state': None}
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
            lon_rot = np.arctan2(pos_at_start_of_step[1], pos_at_start_of_step[0])
            lat_rot = np.arcsin(np.clip(pos_at_start_of_step[2] / np.linalg.norm(pos_at_start_of_step), -1.0, 1.0))
            lon_fixed = (lon_rot + subsolar_lon_rad_fixed)
            lon_fixed = (lon_fixed + PHYSICAL_CONSTANTS['PI']) % (2 * PHYSICAL_CONSTANTS['PI']) - PHYSICAL_CONSTANTS[
                'PI']
            lat_fixed = lat_rot
            temp_at_impact = calculate_surface_temperature_leblanc(lon_fixed, lat_fixed, AU, subsolar_lon_rad_fixed)
            if np.random.random() < calculate_sticking_probability(temp_at_impact):
                return {'status': 'stuck', 'pos_at_impact': pos_at_start_of_step, 'weight': weight}
            else:
                E_in = 0.5 * PHYSICAL_CONSTANTS['MASS_NA'] * np.sum(vel ** 2)
                E_T = PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_at_impact
                E_out = settings['BETA'] * E_T + (1.0 - settings['BETA']) * E_in
                v_out_speed = np.sqrt(2 * E_out / PHYSICAL_CONSTANTS['MASS_NA']) if E_out > 0 else 0.0
                impact_normal = pos_at_start_of_step / np.linalg.norm(pos_at_start_of_step)
                rebound_direction = transform_local_to_world(sample_lambertian_direction_local(), impact_normal)
                vel = v_out_speed * rebound_direction
                pos = (RM + 1.0) * impact_normal
                continue
    return {'status': 'alive', 'final_state': {'pos': pos, 'vel': vel, 'weight': weight}}


# ==============================================================================
# メイン制御関数
# ==============================================================================

def main_snapshot_simulation():
    """
    シミュレーション全体を制御するメイン関数。
    （動的表面密度モデル + SWS）
    """
    start_time = time.time()

    # --- 1. シミュレーション設定 ---
    OUTPUT_DIRECTORY = r"./SimulationResult_202510"
    N_LON_FIXED, N_LAT = 72, 36  # 経度 72 (5度毎), 緯度 36 (5度毎)
    # 論文Source 138 (7.5e14 atoms/cm^2 * 1) -> 単位を m^2 に
    INITIAL_SURFACE_DENSITY_PER_M2 = 7.5e14 * (100.0 ** 2)

    SPIN_UP_YEARS = 3.0
    TIME_STEP_SEC = 1000
    TOTAL_SIM_YEARS = 1.0
    TARGET_TAA_DEGREES = np.arange(0, 360, 1)

    # TD (熱脱離) は可変重み（予算配分方式）
    TARGET_SPS_TD = 1000  # (例: TDで毎ステップ1000個生成)

    # PSD, SWS, MMV は固定重み
    # !! 注意 !! これらの値は適切に調整してください。
    WEIGHT_PSD = 1e23  # (要調整: PSDの固定重み)
    WEIGHT_SWS = 5e21  # (要調整: SWSの固定重み)
    WEIGHT_MMV = 1e23  # (要調整: MMVの固定重み)

    # --- 粒子生成モデル (Leblanc 2003 準拠) ---
    # (PSD)
    F_UV_1AU_PER_M2 = 1.5e14 * (100.0 ** 2)  # Source 261 (1.5e14 /cm^2)
    Q_PSD_M2 = 1.0e-20 / (100.0 ** 2)  # Source 264 (1.0e-20 cm^2)
    TEMP_PSD = 1500.0  # Source 269
    # (MMV)
    TEMP_MMV = 3000.0  # Source 338
    # (TD)
    # (関数 'calculate_thermal_desorption_rate' 内で定義)

    # --- ★ SWS (CPS) 用のパラメータ辞書 (SWSコードより) ★ ---
    SWS_PARAMS = {
        # 1AUでの太陽風平均密度 [particles/m^3] (10 /cm^3) (Source 317)
        'SW_DENSITY_1AU': 10.0 * (100.0) ** 3,
        # 太陽風平均速度 [m/s] (400 km/s) (Source 315)
        'SW_VELOCITY': 400.0 * 1000.0,
        # 実効スパッタリング収率 (多孔性を考慮) [atoms/ion] (Source 313)
        'YIELD_EFF': 0.06,
        # 表面束縛エネルギー (Thompson-Sigmund分布用) [eV] (Source 307)
        'U_eV': 0.27,
        # 表面密度の基準値 [atoms/m^2] (7.5e14 atoms/cm^2) (Source 138)
        'DENSITY_REF_M2': 7.5e14 * (100.0) ** 2,
        # --- スパッタリング発生領域 (太陽固定座標系, 経度=0が太陽直下点) ---
        'LON_MIN_RAD': np.deg2rad(-40.0),
        'LON_MAX_RAD': np.deg2rad(40.0),
        'LAT_N_MIN_RAD': np.deg2rad(30.0),
        'LAT_N_MAX_RAD': np.deg2rad(60.0),
        'LAT_S_MIN_RAD': np.deg2rad(-60.0),
        'LAT_S_MAX_RAD': np.deg2rad(-30.0),
    }
    # ★★★ 修正ここまで ★★★

    # --- 出力グリッドの設定 ---
    GRID_RESOLUTION = 101
    GRID_MAX_RM = 5.0

    # --- 物理モデルのフラグ ---
    USE_SOLAR_GRAVITY = True
    USE_CORIOLIS_FORCES = True

    settings = {
        'BETA': 0.5,
        'T1AU': 168918.0,
        'DT': 1000.0,
        'GRID_RADIUS_RM': GRID_MAX_RM + 1.0,
        'USE_SOLAR_GRAVITY': USE_SOLAR_GRAVITY,
        'USE_CORIOLIS_FORCES': USE_CORIOLIS_FORCES
    }

    # 出力ディレクトリの準備
    run_name = f"DynamicGrid{N_LON_FIXED}x{N_LAT}_HybridWeight_PSD_TD_MMV_SWS"
    target_output_dir = os.path.join(OUTPUT_DIRECTORY, run_name)
    os.makedirs(target_output_dir, exist_ok=True)
    print(f"結果は '{target_output_dir}' に保存されます。")
    print(f"--- 物理モデル設定 (ハイブリッド重み方式) ---")
    print(f"Dynamic Surface Grid: {N_LON_FIXED}x{N_LAT}")
    print(f"Processes: PSD, Thermal Desorption, Micrometeoroid Vaporization, SWS")
    print(f"TD (Variable): Target SPs/Step = {TARGET_SPS_TD}")
    print(f"PSD (Fixed): Weight = {WEIGHT_PSD:.1e}")
    print(f"SWS (Fixed): Weight = {WEIGHT_SWS:.1e}")
    print(f"MMV (Fixed): Weight = {WEIGHT_MMV:.1e}")
    print(f"Solar Gravity: {USE_SOLAR_GRAVITY}")
    print(f"Coriolis/Centrifugal: {USE_CORIOLIS_FORCES}")
    print(f"----------------------")

    # --- 2. シミュレーションの初期化 ---
    MERCURY_YEAR_SEC = 87.97 * 24 * 3600

    lon_edges_fixed = np.linspace(-np.pi, np.pi, N_LON_FIXED + 1)
    lat_edges_fixed = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    dlon_fixed = lon_edges_fixed[1] - lon_edges_fixed[0]
    cell_areas_m2 = (PHYSICAL_CONSTANTS['RM'] ** 2) * dlon_fixed * \
                    (np.sin(lat_edges_fixed[1:]) - np.sin(lat_edges_fixed[:-1]))
    surface_density_grid = np.full((N_LON_FIXED, N_LAT), INITIAL_SURFACE_DENSITY_PER_M2, dtype=np.float64)

    # --- 3. 外部ファイルの読み込み ---
    try:
        spec_data_np = np.loadtxt('SolarSpectrum_Na0.txt', usecols=(0, 3))
        orbit_file_name = 'orbit2025_v5.txt'
        orbit_data = np.loadtxt(orbit_file_name)
    except FileNotFoundError as e:
        print(f"エラー: データファイル '{e.filename}' が見つかりません。");
        sys.exit()
    if orbit_data.shape[1] < 5:
        print(f"エラー: '{orbit_file_name}' の列が不足しています。");
        sys.exit()

    # --- RUN開始時刻 (TAA=0) の調整 ---
    taa_col = orbit_data[:, 0]
    time_col = orbit_data[:, 2]
    idx_perihelion = np.argmin(np.abs(taa_col))
    t_start_run = time_col[idx_perihelion]
    t_end_run = t_start_run + (TOTAL_SIM_YEARS * MERCURY_YEAR_SEC)
    t_start_spinup = t_start_run - (SPIN_UP_YEARS * MERCURY_YEAR_SEC)
    time_steps = np.arange(t_start_spinup, t_end_run, TIME_STEP_SEC)
    print(f"--- 時間設定 (動的モデル) ---");
    print(f"軌道ファイル上のTAA=0 (近日点) 時刻: {t_start_run:.1f} s");
    print(f"スピンアップ開始時刻: {t_start_spinup:.1f} s ({-SPIN_UP_YEARS} 年前)");
    print(f"RUN開始時刻 (TAA=0): {t_start_run:.1f} s");
    print(f"RUN終了時刻: {t_end_run:.1f} s (+{TOTAL_SIM_YEARS} 年後)");
    print(f"------------------")

    # (スペクトルデータの前処理)
    sigma_const = PHYSICAL_CONSTANTS['E_CHARGE'] ** 2 / (
            4 * PHYSICAL_CONSTANTS['ME'] * PHYSICAL_CONSTANTS['C'] * PHYSICAL_CONSTANTS['EPSILON_0'])
    wl, gamma = spec_data_np[:, 0], spec_data_np[:, 1]
    if not np.all(np.diff(wl) > 0):
        sort_indices = np.argsort(wl)
        wl, gamma = wl[sort_indices], gamma[sort_indices]
    spec_data_dict = {'wl': wl, 'gamma': gamma,
                      'sigma0_perdnu2': sigma_const * 0.641,
                      'sigma0_perdnu1': sigma_const * 0.320,
                      'JL': 5.18e14}

    # --- 4. メインループ (時間発展) ---
    active_particles = []
    previous_taa = -1
    target_taa_idx = 0
    current_step_weight_td = 0.0  # ★ TD専用の可変重み

    with tqdm(total=len(time_steps), desc="Time Evolution") as pbar:
        for t_sec in time_steps:
            # --- 4a. 現在時刻の軌道パラメータを取得 ---
            TAA, AU, V_radial_ms, V_tangential_ms, subsolar_lon_rad_fixed = get_orbital_params(
                t_sec, orbit_data, MERCURY_YEAR_SEC
            )

            run_phase = "Spin-up" if t_sec < t_start_run else "Run"
            pbar.set_description(
                f"[{run_phase}] TAA={TAA:.1f} | N_act={len(active_particles)} | W_TD={current_step_weight_td:.1e}"
            )

            # --- 4b. 表面から新しい粒子を生成 (★★★ ハイブリッド重み方式 ★★★) ---

            # (A) 事前ループ: 惑星全体の「総放出原子数」をプロセスごとに計算する
            total_atoms_td_this_step = 0.0  # ★ TDの総原子数（可変重み計算用）
            n_atoms_mmv = 0.0
            n_atoms_psd_grid = np.zeros_like(surface_density_grid)
            n_atoms_td_grid = np.zeros_like(surface_density_grid)
            n_atoms_sws_grid = np.zeros_like(surface_density_grid)

            # (A-1) MMV の総原子数を計算
            flux_mmv_per_m2_s = calculate_mmv_flux(AU)
            MERCURY_SURFACE_AREA_M2 = 4 * PHYSICAL_CONSTANTS['PI'] * (PHYSICAL_CONSTANTS['RM'] ** 2)
            n_atoms_mmv = flux_mmv_per_m2_s * MERCURY_SURFACE_AREA_M2 * TIME_STEP_SEC

            # (A-2) PSD, TD, SWS の総原子数を計算 (グリッドをループ)
            F_UV_current_per_m2 = F_UV_1AU_PER_M2 / (AU ** 2)
            flux_sw_1au = SWS_PARAMS['SW_DENSITY_1AU'] * SWS_PARAMS['SW_VELOCITY']
            current_flux_sw = flux_sw_1au / (AU ** 2)
            period_sec = 1800.0
            temporal_variation_factor = (1.0 + np.cos(2 * np.pi * t_sec / period_sec)) / 2.0
            effective_flux_sw = current_flux_sw * temporal_variation_factor
            base_sputtering_rate_per_m2_s = effective_flux_sw * SWS_PARAMS['YIELD_EFF']
            DENSITY_REF_M2 = SWS_PARAMS['DENSITY_REF_M2']

            for i_lon in range(N_LON_FIXED):
                for i_lat in range(N_LAT):
                    current_density_per_m2 = surface_density_grid[i_lon, i_lat]
                    if current_density_per_m2 <= 0:
                        continue

                    lon_fixed_rad = (lon_edges_fixed[i_lon] + lon_edges_fixed[i_lon + 1]) / 2
                    lat_rad = (lat_edges_fixed[i_lat] + lat_edges_fixed[i_lat + 1]) / 2
                    area_m2 = cell_areas_m2[i_lat]

                    cos_Z = np.cos(lat_rad) * np.cos(lon_fixed_rad - subsolar_lon_rad_fixed)

                    # (1) PSD
                    if cos_Z > 0:
                        rate_psd_per_s = F_UV_current_per_m2 * Q_PSD_M2 * cos_Z
                        n_atoms_psd = rate_psd_per_s * current_density_per_m2 * area_m2 * TIME_STEP_SEC
                        if n_atoms_psd > 0:
                            n_atoms_psd_grid[i_lon, i_lat] = n_atoms_psd

                    # (2) TD
                    temp_k = calculate_surface_temperature_leblanc(lon_fixed_rad, lat_rad, AU, subsolar_lon_rad_fixed)
                    rate_td_per_s = calculate_thermal_desorption_rate(temp_k)
                    n_atoms_td = rate_td_per_s * current_density_per_m2 * area_m2 * TIME_STEP_SEC
                    if n_atoms_td > 0:
                        n_atoms_td_grid[i_lon, i_lat] = n_atoms_td
                        total_atoms_td_this_step += n_atoms_td  # ★ TDの総和を計算

                    # (3) ★ SWS ★
                    # このセルの太陽固定座標系での位置を計算
                    lon_sun_fixed_rad = (lon_fixed_rad - subsolar_lon_rad_fixed)
                    lon_sun_fixed_rad = (lon_sun_fixed_rad + PHYSICAL_CONSTANTS['PI']) % (
                            2 * PHYSICAL_CONSTANTS['PI']) - PHYSICAL_CONSTANTS['PI']
                    lat_sun_fixed_rad = lat_rad

                    # 領域判定 (元のロジック)
                    is_in_lon_band = (SWS_PARAMS['LON_MIN_RAD'] <= lon_sun_fixed_rad <= SWS_PARAMS['LON_MAX_RAD'])
                    is_in_lat_n_band = (SWS_PARAMS['LAT_N_MIN_RAD'] <= lat_sun_fixed_rad <= SWS_PARAMS['LAT_N_MAX_RAD'])
                    is_in_lat_s_band = (SWS_PARAMS['LAT_S_MIN_RAD'] <= lat_sun_fixed_rad <= SWS_PARAMS['LAT_S_MAX_RAD'])

                    if is_in_lon_band and (is_in_lat_n_band or is_in_lat_s_band):
                        current_concentration_ratio = current_density_per_m2 / DENSITY_REF_M2
                        current_sputtering_rate_per_m2_s = base_sputtering_rate_per_m2_s * current_concentration_ratio
                        n_atoms_sws = current_sputtering_rate_per_m2_s * area_m2 * TIME_STEP_SEC

                        if n_atoms_sws > 0:
                            n_atoms_sws_grid[i_lon, i_lat] = n_atoms_sws

            # (B) 重みの決定 (TDのみ可変)
            current_step_weight_td = 1.0
            if total_atoms_td_this_step > 0 and TARGET_SPS_TD > 0:
                current_step_weight_td = total_atoms_td_this_step / TARGET_SPS_TD
            else:
                pass

                # (C) メインループ: 粒子を生成
            newly_launched_particles = []
            atoms_lost_grid = np.zeros_like(surface_density_grid)

            # (C-1) MMV の粒子を生成 (固定重み)
            if n_atoms_mmv > 0 and WEIGHT_MMV > 0:
                num_sps_float_mmv = n_atoms_mmv / WEIGHT_MMV
                num_sps_int_mmv = int(num_sps_float_mmv)
                if np.random.random() < (num_sps_float_mmv - num_sps_int_mmv):
                    num_sps_int_mmv += 1

                # MMVは枯渇させない

                if num_sps_int_mmv > 0:
                    M_rejection = 4.0 / 3.0
                    for _ in range(num_sps_int_mmv):
                        while True:
                            lon_rot_rad = np.random.uniform(-np.pi, np.pi)
                            prob_accept = (1.0 - (1.0 / 3.0) * np.sin(lon_rot_rad)) / M_rejection
                            if np.random.random() < prob_accept:
                                break
                        lat_rot_rad = np.arcsin(np.random.uniform(-1.0, 1.0))

                        initial_pos_rot = lonlat_to_xyz(lon_rot_rad, lat_rot_rad, PHYSICAL_CONSTANTS['RM'])
                        surface_normal_rot = initial_pos_rot / PHYSICAL_CONSTANTS['RM']
                        speed = sample_speed_from_flux_distribution(PHYSICAL_CONSTANTS['MASS_NA'], TEMP_MMV)
                        initial_vel_rot = speed * transform_local_to_world(sample_lambertian_direction_local(),
                                                                           surface_normal_rot)
                        newly_launched_particles.append({
                            'pos': initial_pos_rot, 'vel': initial_vel_rot,
                            'weight': WEIGHT_MMV  # ★ 固定重み
                        })

            # (C-2) PSD, TD, SWS の粒子を生成 (グリッドをループ)
            for i_lon in range(N_LON_FIXED):
                for i_lat in range(N_LAT):

                    n_atoms_psd = n_atoms_psd_grid[i_lon, i_lat]
                    n_atoms_td = n_atoms_td_grid[i_lon, i_lat]
                    n_atoms_sws = n_atoms_sws_grid[i_lon, i_lat]

                    if n_atoms_psd <= 0 and n_atoms_td <= 0 and n_atoms_sws <= 0:
                        continue

                    # (温度を再計算)
                    lon_fixed_rad = (lon_edges_fixed[i_lon] + lon_edges_fixed[i_lon + 1]) / 2
                    lat_rad = (lat_edges_fixed[i_lat] + lat_edges_fixed[i_lat + 1]) / 2
                    temp_k = calculate_surface_temperature_leblanc(lon_fixed_rad, lat_rad, AU,
                                                                   subsolar_lon_rad_fixed)

                    procs = {
                        'PSD': {'n_atoms': n_atoms_psd, 'temp': TEMP_PSD, 'U_eV': None, 'weight': WEIGHT_PSD},  # ★ 固定重み
                        'TD': {'n_atoms': n_atoms_td, 'temp': temp_k, 'U_eV': None, 'weight': current_step_weight_td},
                        # ★ 可変重み
                        'SWS': {'n_atoms': n_atoms_sws, 'temp': None, 'U_eV': SWS_PARAMS['U_eV'], 'weight': WEIGHT_SWS}
                        # ★ 固定重み
                    }

                    for proc_name, p in procs.items():
                        if p['n_atoms'] <= 0 or p['weight'] <= 0: continue

                        # ★ 割り当てられた重みでSP数を計算 ★
                        weight_to_use = p['weight']
                        num_sps_float = p['n_atoms'] / weight_to_use
                        num_sps_int = int(num_sps_float)
                        if np.random.random() < (num_sps_float - num_sps_int):
                            num_sps_int += 1
                        if num_sps_int == 0: continue

                        # 枯渇量を計算 (P4修正ロジックは維持)
                        atoms_to_launch_and_deplete = num_sps_int * weight_to_use
                        atoms_lost_grid[i_lon, i_lat] += atoms_to_launch_and_deplete

                        # 粒子を生成
                        for _ in range(num_sps_int):
                            # --- 1. 速度を決定 ---
                            if proc_name == 'SWS':
                                energy_eV = sample_thompson_sigmund_energy(p['U_eV'])
                                energy_J = energy_eV * PHYSICAL_CONSTANTS['EV_TO_JOULE']
                                speed = np.sqrt(2.0 * energy_J / PHYSICAL_CONSTANTS['MASS_NA'])
                            else:  # PSD or TD
                                speed = sample_speed_from_flux_distribution(PHYSICAL_CONSTANTS['MASS_NA'],
                                                                            p['temp'])

                            # --- 2. 位置と角度を決定 (共通ロジック) ---
                            lon_rot_rad = lon_fixed_rad - subsolar_lon_rad_fixed
                            lat_rot_rad = lat_rad

                            initial_pos_rot = lonlat_to_xyz(lon_rot_rad, lat_rot_rad, PHYSICAL_CONSTANTS['RM'])
                            surface_normal_rot = initial_pos_rot / PHYSICAL_CONSTANTS['RM']
                            initial_vel_rot = speed * transform_local_to_world(sample_lambertian_direction_local(),
                                                                               surface_normal_rot)
                            newly_launched_particles.append({
                                'pos': initial_pos_rot, 'vel': initial_vel_rot,
                                'weight': weight_to_use  # ★ 割り当てられた重みを設定
                            })

            # --- (4b はここまで) ---

            # PSD/TD/SWSによって失われた密度を表面グリッドから減算
            surface_density_grid -= atoms_lost_grid / cell_areas_m2
            np.clip(surface_density_grid, 0, None, out=surface_density_grid)

            # アクティブ粒子リストに新粒子を追加
            active_particles.extend(newly_launched_particles)

            # --- 4c. 全ての粒子を1ステップ進め、結果を集計 ---
            tasks = [{'settings': settings, 'spec': spec_data_dict, 'particle_state': p,
                      'orbit': (TAA, AU, V_radial_ms, V_tangential_ms, subsolar_lon_rad_fixed),
                      'duration': TIME_STEP_SEC} for p in
                     active_particles]

            next_active_particles = []
            atoms_gained_grid = np.zeros_like(surface_density_grid)

            if tasks:
                with Pool(processes=max(1, cpu_count() - 1)) as pool:
                    results = list(pool.imap(simulate_particle_for_one_step, tasks, chunksize=100))

                for res in results:
                    if res['status'] == 'alive':
                        next_active_particles.append(res['final_state'])
                    elif res['status'] == 'stuck':
                        pos_rot = res['pos_at_impact']
                        weight = res['weight']  # ★ 重みは粒子が保持している
                        lon_rot = np.arctan2(pos_rot[1], pos_rot[0])
                        lat_rot = np.arcsin(np.clip(pos_rot[2] / np.linalg.norm(pos_rot), -1.0, 1.0))
                        # 太陽固定座標 -> 惑星固定座標 に変換
                        lon_fixed = (lon_rot + subsolar_lon_rad_fixed)
                        lon_fixed = (lon_fixed + PHYSICAL_CONSTANTS['PI']) % (2 * PHYSICAL_CONSTANTS['PI']) - \
                                    PHYSICAL_CONSTANTS['PI']
                        lat_fixed = lat_rot

                        i_lon = np.searchsorted(lon_edges_fixed, lon_fixed) - 1
                        i_lat = np.searchsorted(lat_edges_fixed, lat_fixed) - 1
                        if 0 <= i_lon < N_LON_FIXED and 0 <= i_lat < N_LAT:
                            atoms_gained_grid[i_lon, i_lat] += weight

            active_particles = next_active_particles
            # 吸着した粒子を表面グリッドに追加
            surface_density_grid += atoms_gained_grid / cell_areas_m2

            # --- 4d. スナップショット保存判定 ---
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

            # --- 4e. 立方体グリッドに集計して保存 ---
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

                # 表面密度も保存
                filename_surf = f"surface_density_t{int(save_time_h):05d}_taa{int(round(TAA)):03d}.npy"
                np.save(os.path.join(target_output_dir, filename_surf), surface_density_grid)

            previous_taa = TAA
            pbar.update(1)

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