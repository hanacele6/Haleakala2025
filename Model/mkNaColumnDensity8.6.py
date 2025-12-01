# -*- coding: utf-8 -*-
"""
==============================================================================
概要 (Sub-cycling版)
==============================================================================
このスクリプトは、水星のナトリウム大気のふるまいをシミュレーションします。

変更点 (2025/11):
- 表面密度の時間発展において「平衡近似 (Quasi-Steady State)」を廃止しました。
- 代わりに「サブサイクリング (Sub-cycling)」を導入し、軌道計算(重い)は500秒間隔、
  表面密度更新(軽い)は1秒間隔で行うことで、高速かつ近似のない計算を実現しています。
- これにより、TD（熱脱離）が激しい領域での密度の枯渇や、供給と放出のバランスを
  正確に追跡します。

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
    'AU': 1.496e11,  # 天文単位 [m]
    'MASS_NA': 22.98976928 * 1.66054e-27,  # ナトリウム原子質量 [kg]
    'K_BOLTZMANN': 1.380649e-23,  # ボルツマン定数 [J/K]
    'GM_MERCURY': 2.2032e13,  # 水星の万有引力定数と質山の積 [m^3/s^2]
    'RM': 2.440e6,  # 水星半径 [m]
    'C': 299792458.0,  # 光速 [m/s]
    'H': 6.62607015e-34,  # プランク定数 [J·s]
    'E_CHARGE': 1.602176634e-19,  # 電気素量 [C]
    'ME': 9.1093897e-31,  # 電子質量 [kg]
    'EPSILON_0': 8.854187817e-12,  # 真空の誘電率 [F/m]
    'G': 6.6743e-11,  # 万有引力定数 [N m^2/kg^2]
    'MASS_SUN': 1.989e30,  # 太陽質量 [kg]
    'EV_TO_JOULE': 1.602176634e-19,  # eV から Joule への換算係数 [J / eV]
}


# ==============================================================================
# 物理モデルに基づくヘルパー関数群
# ==============================================================================

def calculate_surface_temperature_leblanc(lon_fixed_rad, lat_rad, AU, subsolar_lon_rad):
    """
    惑星固定座標上の指定された地点の表面温度を計算 (Leblanc 2003)
    """
    T_night = 100.0
    # cos(Solar Zenith Angle)
    # lon_fixed_rad, lat_rad はグリッドのメッシュグリッドまたは単一値を想定
    cos_theta = np.cos(lat_rad) * np.cos(lon_fixed_rad - subsolar_lon_rad)

    # 昼側温度計算
    T0_peri = 600.0
    T0_aph = 475.0
    AU_peri = 0.307
    AU_aph = 0.467
    T0 = np.interp(AU, [AU_peri, AU_aph], [T0_peri, T0_aph])
    T1 = 100.0

    T_day = T0 * (np.maximum(0.0, cos_theta) ** 0.25) + T1 * (np.maximum(0.0, cos_theta) ** 0.25)  # 論文の式に準拠して修正
    # 実際は T = T_subsolar * cos(theta)^0.25
    # ここでは簡易的に元のコードのロジックを踏襲しつつ、マイナスにならないようmaximumを使用
    T_calc = T0 * (np.maximum(0.0, cos_theta) ** 0.25)

    # 夜側は T_night
    return np.maximum(T_night, T_calc)


def calculate_sticking_probability(surface_temp_K):
    """吸着確率"""
    A = 0.0804
    B = 458.0
    porosity = 0.8
    # surface_temp_K が配列の場合に対応
    p_stick = A * np.exp(B / np.maximum(surface_temp_K, 1e-10))  # ゼロ除算防止
    p_stick_eff = p_stick / (1.0 - (1.0 - p_stick) * porosity)
    return np.minimum(p_stick_eff, 1.0)


def calculate_thermal_desorption_rate(surface_temp_K):
    """熱脱離率 [1/s]"""
    VIB_FREQ = 1e13
    BINDING_ENERGY_EV = 1.85
    BINDING_ENERGY_J = BINDING_ENERGY_EV * PHYSICAL_CONSTANTS['EV_TO_JOULE']
    k_B = PHYSICAL_CONSTANTS['K_BOLTZMANN']

    # 低温でのオーバーフロー防止
    exponent = -BINDING_ENERGY_J / (k_B * np.maximum(surface_temp_K, 10.0))
    rate = np.zeros_like(surface_temp_K)
    mask = exponent > -700
    rate[mask] = VIB_FREQ * np.exp(exponent[mask])
    return rate


def calculate_mmv_flux(AU):
    """MMVフラックス [atoms/m^2/s]"""
    TOTAL_FLUX_AT_PERI_NA_S = 5e23
    PERIHELION_AU = 0.307
    MERCURY_SURFACE_AREA_M2 = 4 * PHYSICAL_CONSTANTS['PI'] * (PHYSICAL_CONSTANTS['RM'] ** 2)
    avg_flux_at_peri = TOTAL_FLUX_AT_PERI_NA_S / MERCURY_SURFACE_AREA_M2
    C = avg_flux_at_peri * (PERIHELION_AU ** 1.9)
    return C * (AU ** (-1.9))


def sample_speed_from_flux_distribution(mass_kg, temp_k):
    """Maxwell-Boltzmann Flux分布からの速度サンプリング"""
    kT = PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_k
    E = np.random.gamma(2.0, kT)
    return np.sqrt(2.0 * E / mass_kg)


def sample_thompson_sigmund_energy(U_eV, E_max_eV=5.0):
    """Thompson-Sigmund分布からのエネルギーサンプリング"""
    f_max = (U_eV / 2.0) / (U_eV / 2.0 + U_eV) ** 3
    while True:
        E_try = np.random.uniform(0, E_max_eV)
        f_E_try = E_try / (E_try + U_eV) ** 3
        y_try = np.random.uniform(0, f_max)
        if y_try <= f_E_try:
            return E_try


def sample_lambertian_direction_local():
    """ランバート分布（局所座標）"""
    u1, u2 = np.random.random(2)
    phi = 2 * PHYSICAL_CONSTANTS['PI'] * u1
    cos_theta = np.sqrt(1 - u2)
    sin_theta = np.sqrt(u2)
    return np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])


def transform_local_to_world(local_vec, normal_vector):
    """局所座標 -> ワールド座標変換"""
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
    """軌道パラメータ取得 (3:2共鳴対応版)"""

    # 何回目の公転（年）か計算
    orbit_cycle = int(time_sec // mercury_year_sec)

    # 軌道データ参照用の時刻（0〜88日に丸める）
    current_time_in_orbit = time_sec % mercury_year_sec
    time_col = orbit_data[:, 2]

    taa = np.interp(current_time_in_orbit, time_col, orbit_data[:, 0])
    au = np.interp(current_time_in_orbit, time_col, orbit_data[:, 1])
    v_radial = np.interp(current_time_in_orbit, time_col, orbit_data[:, 3])
    v_tangential = np.interp(current_time_in_orbit, time_col, orbit_data[:, 4])

    # 太陽直下点経度
    SUBSOLAR_LON_COL_IDX = 5
    subsolar_lon_deg_fixed = np.interp(current_time_in_orbit, time_col, orbit_data[:, SUBSOLAR_LON_COL_IDX])
    subsolar_lon_rad = np.deg2rad(subsolar_lon_deg_fixed)

    # 【重要修正】
    # 水星は2年で太陽に対する向きが一周する (3:2共鳴)。
    # 奇数年目 (Year 1, 3, ...) は、偶数年目 (Year 0, 2...) に対して
    # 太陽直下点が 180度 (pi) 反対側にくる。
    if orbit_cycle % 2 != 0:
        subsolar_lon_rad = (subsolar_lon_rad + np.pi) % (2 * np.pi)
        # -pi ~ pi の範囲に正規化
        if subsolar_lon_rad > np.pi:
            subsolar_lon_rad -= 2 * np.pi

    return taa, au, v_radial, v_tangential, subsolar_lon_rad


def lonlat_to_xyz(lon_rad, lat_rad, radius):
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return np.array([x, y, z])


def _calculate_acceleration(pos, vel, V_radial_ms, V_tangential_ms, AU, spec_data, settings):
    """加速度計算 (重力 + SRP + 慣性力)"""
    x, y, z = pos
    r0 = AU * PHYSICAL_CONSTANTS['AU']
    velocity_for_doppler = vel[0] + V_radial_ms
    w_na_d2 = 589.1582e-9 * (1.0 + velocity_for_doppler / PHYSICAL_CONSTANTS['C'])
    w_na_d1 = 589.7558e-9 * (1.0 + velocity_for_doppler / PHYSICAL_CONSTANTS['C'])

    b = 0.0
    wl, gamma, sigma0_perdnu2, sigma0_perdnu1, JL = spec_data.values()

    # SRP計算
    if (wl[0] * 1e-9 <= w_na_d2 < wl[-1] * 1e-9) and (wl[0] * 1e-9 <= w_na_d1 < wl[-1] * 1e-9):
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
            accel_centrifugal = np.array([omega_val ** 2 * (pos[0] - r0), omega_sq * pos[1], 0.0])
            two_omega = 2 * omega_val
            accel_coriolis = np.array([two_omega * vel[1], -two_omega * vel[0], 0.0])

    return accel_srp + accel_g + accel_sun + accel_centrifugal + accel_coriolis


def simulate_particle_for_one_step(args):
    """
    1個のスーパーパーティクルの時間発展 (RK4)
    duration はメインステップ幅 (500s)
    """
    settings, spec_data = args['settings'], args['spec']
    TAA, AU, V_radial_ms, V_tangential_ms, subsolar_lon_rad_fixed = args['orbit']
    duration = args['duration']
    DT = settings['DT']  # RK4のステップ幅 (メインステップと同じに設定)

    RM = PHYSICAL_CONSTANTS['RM']
    R_MAX = RM * settings['GRID_RADIUS_RM']
    pos, vel, weight = args['particle_state']['pos'], args['particle_state']['vel'], args['particle_state']['weight']
    tau_ionization = settings['T1AU'] * AU ** 2

    # duration / DT 回ループ (通常は1回)
    num_steps = int(duration / DT)
    if num_steps < 1:
        num_steps = 1
        DT = duration

    pos_at_start_of_step = pos.copy()

    for _ in range(num_steps):
        pos_at_start_of_step = pos.copy()

        # 光電離
        if pos[0] > 0:
            if np.random.random() < (1.0 - np.exp(-DT / tau_ionization)):
                return {'status': 'ionized', 'final_state': None}

        # RK4
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
            # 衝突
            lon_rot = np.arctan2(pos_at_start_of_step[1], pos_at_start_of_step[0])
            lat_rot = np.arcsin(np.clip(pos_at_start_of_step[2] / np.linalg.norm(pos_at_start_of_step), -1.0, 1.0))

            lon_fixed = (lon_rot + subsolar_lon_rad_fixed)
            lon_fixed = (lon_fixed + np.pi) % (2 * np.pi) - np.pi
            lat_fixed = lat_rot

            temp_at_impact = calculate_surface_temperature_leblanc(lon_fixed, lat_fixed, AU, subsolar_lon_rad_fixed)

            if np.random.random() < calculate_sticking_probability(temp_at_impact):
                return {'status': 'stuck', 'pos_at_impact': pos_at_start_of_step, 'weight': weight}
            else:
                # 反射
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


def main_snapshot_simulation():
    start_time = time.time()

    # --- 1. シミュレーション設定 ---
    OUTPUT_DIRECTORY = r"./SimulationResult_202511"
    N_LON_FIXED, N_LAT = 72, 36
    INITIAL_SURFACE_DENSITY_PER_M2 = 7.5e14 * (100.0 ** 2) * 0.0053

    # メイン時間ステップ (軌道計算用)
    MAIN_TIME_STEP_SEC = 500.0

    # 表面更新用サブ時間ステップ (★要件: 1秒更新)
    SURFACE_DT_SEC = 1.0

    SPIN_UP_YEARS = 3.0
    TOTAL_SIM_YEARS = 1.0
    TARGET_TAA_DEGREES = np.arange(0, 360, 1)

    TARGET_SPS_TD = 1000
    TARGET_SPS_PSD = 1000
    TARGET_SPS_SWS = 1000
    TARGET_SPS_MMV = 1000

    # 物理パラメータ
    F_UV_1AU_PER_M2 = 1.5e14 * (100.0 ** 2)
    Q_PSD_M2 = 2.0e-20 / (100.0 ** 2)
    TEMP_PSD = 1500.0
    TEMP_MMV = 3000.0

    SWS_PARAMS = {
        'SW_DENSITY_1AU': 10.0 * (100.0) ** 3,
        'SW_VELOCITY': 400.0 * 1000.0,
        'YIELD_EFF': 0.06,
        'U_eV': 0.27,
        'DENSITY_REF_M2': 7.5e14 * (100.0) ** 2,
        'LON_MIN_RAD': np.deg2rad(-40.0), 'LON_MAX_RAD': np.deg2rad(40.0),
        'LAT_N_MIN_RAD': np.deg2rad(30.0), 'LAT_N_MAX_RAD': np.deg2rad(60.0),
        'LAT_S_MIN_RAD': np.deg2rad(-60.0), 'LAT_S_MAX_RAD': np.deg2rad(-30.0),
    }

    GRID_RESOLUTION = 101
    GRID_MAX_RM = 5.0

    settings = {
        'BETA': 0.5,
        'T1AU': 168918.0,
        'DT': MAIN_TIME_STEP_SEC,
        'GRID_RADIUS_RM': GRID_MAX_RM + 1.0,
        'USE_SOLAR_GRAVITY': True,
        'USE_CORIOLIS_FORCES': True
    }

    # --- 2. 初期化 ---
    run_name = f"SubCycle_{N_LON_FIXED}x{N_LAT}_1.0"
    target_output_dir = os.path.join(OUTPUT_DIRECTORY, run_name)
    os.makedirs(target_output_dir, exist_ok=True)
    print(f"結果保存先: '{target_output_dir}'")

    MERCURY_YEAR_SEC = 87.97 * 24 * 3600

    # グリッド
    lon_edges_fixed = np.linspace(-np.pi, np.pi, N_LON_FIXED + 1)
    lat_edges_fixed = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)

    # メッシュグリッド作成 (ベクトル化用)
    lon_centers = (lon_edges_fixed[:-1] + lon_edges_fixed[1:]) / 2
    lat_centers = (lat_edges_fixed[:-1] + lat_edges_fixed[1:]) / 2
    LON_GRID, LAT_GRID = np.meshgrid(lon_centers, lat_centers, indexing='ij')

    # セル面積 [m^2]
    dlon_fixed = lon_edges_fixed[1] - lon_edges_fixed[0]
    cell_areas_1d = (PHYSICAL_CONSTANTS['RM'] ** 2) * dlon_fixed * (
                np.sin(lat_edges_fixed[1:]) - np.sin(lat_edges_fixed[:-1]))
    CELL_AREAS_GRID = np.tile(cell_areas_1d, (N_LON_FIXED, 1))  # (N_LON, N_LAT)

    surface_density_grid = np.full((N_LON_FIXED, N_LAT), INITIAL_SURFACE_DENSITY_PER_M2, dtype=np.float64)

    # --- 3. ファイル読み込み ---
    try:
        spec_data_np = np.loadtxt('SolarSpectrum_Na0.txt', usecols=(0, 3))
        orbit_data = np.loadtxt('orbit2025_v6.txt')

        # === 重要な修正: 角度の連続化 ===
        # どんなに短い期間のデータでも、補間(interp)する前には必ずunwrapすべきです。
        SUBSOLAR_LON_COL_IDX = 5

        # 1. 度数法 -> ラジアン
        raw_lon_rad = np.deg2rad(orbit_data[:, SUBSOLAR_LON_COL_IDX])

        # 2. 不連続を滑らかにする (unwrap)
        unwrapped_lon_rad = np.unwrap(raw_lon_rad)

        # 3. 再び度数法に戻してデータを上書き
        # (この値は -180~180 を超えることがありますが、cos/sinに入れるので問題ありません)
        orbit_data[:, SUBSOLAR_LON_COL_IDX] = np.rad2deg(unwrapped_lon_rad)
        # =============================

    except FileNotFoundError:
        print("必要なファイル(SolarSpectrum_Na0.txt, orbit2025_v6.txt)が見つかりません。")
        sys.exit()

    taa_col = orbit_data[:, 0]
    time_col = orbit_data[:, 2]
    idx_perihelion = np.argmin(np.abs(taa_col))
    t_start_run = time_col[idx_perihelion]
    t_end_run = t_start_run + (TOTAL_SIM_YEARS * MERCURY_YEAR_SEC)
    t_start_spinup = t_start_run - (SPIN_UP_YEARS * MERCURY_YEAR_SEC)

    sigma_const = PHYSICAL_CONSTANTS['E_CHARGE'] ** 2 / (
                4 * PHYSICAL_CONSTANTS['ME'] * PHYSICAL_CONSTANTS['C'] * PHYSICAL_CONSTANTS['EPSILON_0'])
    wl, gamma = spec_data_np[:, 0], spec_data_np[:, 1]
    if not np.all(np.diff(wl) > 0):
        sidx = np.argsort(wl)
        wl, gamma = wl[sidx], gamma[sidx]
    spec_data_dict = {'wl': wl, 'gamma': gamma,
                      'sigma0_perdnu2': sigma_const * 0.641,
                      'sigma0_perdnu1': sigma_const * 0.320,
                      'JL': 5.18e14}

    # --- 4. メインループ ---
    active_particles = []
    previous_taa = -1
    target_taa_idx = 0

    previous_atoms_gained_grid = np.zeros_like(surface_density_grid)
    t_sec = t_start_spinup

    with tqdm(total=int(t_end_run - t_start_spinup), desc="Simulating") as pbar:
        while t_sec < t_end_run:

            current_dt_main = MAIN_TIME_STEP_SEC
            TAA, AU, V_radial_ms, V_tangential_ms, subsolar_lon_rad_fixed = get_orbital_params(
                t_sec, orbit_data, MERCURY_YEAR_SEC
            )

            # ==============================================================================
            # A. 表面密度の更新 (サブサイクリング / 1秒刻み)
            # ==============================================================================

            # --- A-1. 物理量の事前計算 (ベクトル化) ---
            # 500秒間での温度変化は無視できるため、ループ外で一括計算

            # 温度分布 [K]
            temp_grid_k = calculate_surface_temperature_leblanc(LON_GRID, LAT_GRID, AU, subsolar_lon_rad_fixed)

            # 各放出プロセスのレート係数 k [1/s] を計算
            # (密度を掛けると放出フラックスになる値)

            # 1. PSD Rate Constant
            cos_Z_grid = np.cos(LAT_GRID) * np.cos(LON_GRID - subsolar_lon_rad_fixed)
            F_UV_current = F_UV_1AU_PER_M2 / (AU ** 2)
            k_psd_grid = np.maximum(0.0, F_UV_current * Q_PSD_M2 * cos_Z_grid)

            # 2. TD Rate Constant
            k_td_grid = calculate_thermal_desorption_rate(temp_grid_k)

            # 3. SWS Rate Constant
            # 太陽固定座標系での経度
            lon_sun_fixed_grid = (LON_GRID - subsolar_lon_rad_fixed + np.pi) % (2 * np.pi) - np.pi

            flux_sw_1au = SWS_PARAMS['SW_DENSITY_1AU'] * SWS_PARAMS['SW_VELOCITY']
            base_sputtering_rate = (flux_sw_1au / (AU ** 2)) * SWS_PARAMS['YIELD_EFF']
            k_sws_val = base_sputtering_rate / SWS_PARAMS['DENSITY_REF_M2']

            mask_sws_lon = (lon_sun_fixed_grid >= SWS_PARAMS['LON_MIN_RAD']) & (
                        lon_sun_fixed_grid <= SWS_PARAMS['LON_MAX_RAD'])
            mask_sws_lat_n = (LAT_GRID >= SWS_PARAMS['LAT_N_MIN_RAD']) & (LAT_GRID <= SWS_PARAMS['LAT_N_MAX_RAD'])
            mask_sws_lat_s = (LAT_GRID >= SWS_PARAMS['LAT_S_MIN_RAD']) & (LAT_GRID <= SWS_PARAMS['LAT_S_MAX_RAD'])
            mask_sws = mask_sws_lon & (mask_sws_lat_n | mask_sws_lat_s)

            k_sws_grid = np.zeros_like(surface_density_grid)
            k_sws_grid[mask_sws] = k_sws_val

            # 総放出レート定数 k_total [1/s]
            k_total_grid = k_psd_grid + k_td_grid + k_sws_grid

            # --- A-2. 流入フラックスの準備 ---
            # 前のステップで戻ってきた原子が、500秒かけて均等に降ってくると仮定
            # [atoms] -> [atoms/m^2/s]
            flux_in_per_s = (previous_atoms_gained_grid / CELL_AREAS_GRID) / current_dt_main

            # --- A-3. サブサイクリング実行 ---
            num_substeps = int(current_dt_main / SURFACE_DT_SEC)  # 500回

            # 累積放出原子数 [atoms] を記録する配列 (粒子生成用)
            accumulated_loss_psd_atoms = np.zeros_like(surface_density_grid)
            accumulated_loss_td_atoms = np.zeros_like(surface_density_grid)
            accumulated_loss_sws_atoms = np.zeros_like(surface_density_grid)

            # 現在の密度を作業用変数に
            na_current = surface_density_grid.copy()

            for _ in range(num_substeps):
                # 現在の密度に基づく放出フラックス [atoms/m^2/s]
                # Out = N * k
                loss_flux_total = na_current * k_total_grid

                # 更新量 dN/dt = In - Out
                d_na = flux_in_per_s - loss_flux_total

                # オイラー積分
                na_next = na_current + d_na * SURFACE_DT_SEC

                # 負値の補正 (放出が多すぎてマイナスになった場合、在庫ゼロにする)
                # 実際に放出できた量 = 元の在庫 + 流入分
                mask_neg = na_next < 0

                # 実際にこの1秒で失われた密度 [atoms/m^2] (統計用)
                # 基本: loss_flux * dt
                # 枯渇時: na_current + flux_in * dt (在庫全放出)
                actual_loss_density = loss_flux_total * SURFACE_DT_SEC
                if np.any(mask_neg):
                    actual_loss_density[mask_neg] = na_current[mask_neg] + flux_in_per_s[mask_neg] * SURFACE_DT_SEC
                    na_next[mask_neg] = 0.0

                # 在庫更新
                na_current = na_next

                # 放出内訳を記録
                # (k_total が 0 の場所は 0除算になるので注意。k_total=0ならlossも0なので計算不要)
                mask_active = k_total_grid > 0

                # 各プロセスの割合に応じて分配
                # Loss_i = Total_Loss * (k_i / k_total)
                # さらに [atoms/m^2] -> [atoms] に変換して蓄積
                loss_atoms = actual_loss_density * CELL_AREAS_GRID

                accumulated_loss_psd_atoms[mask_active] += loss_atoms[mask_active] * (
                            k_psd_grid[mask_active] / k_total_grid[mask_active])
                accumulated_loss_td_atoms[mask_active] += loss_atoms[mask_active] * (
                            k_td_grid[mask_active] / k_total_grid[mask_active])
                accumulated_loss_sws_atoms[mask_active] += loss_atoms[mask_active] * (
                            k_sws_grid[mask_active] / k_total_grid[mask_active])

            # メインループの密度を更新
            surface_density_grid = na_current

            # 粒子生成用に合計ロスのグリッドを作成
            total_atoms_lost_grid = accumulated_loss_psd_atoms + accumulated_loss_td_atoms + accumulated_loss_sws_atoms

            # ==============================================================================
            # B. 粒子生成 (蓄積されたLossに基づく)
            # ==============================================================================

            # MMV
            flux_mmv = calculate_mmv_flux(AU)
            n_atoms_mmv = flux_mmv * (4 * np.pi * PHYSICAL_CONSTANTS['RM'] ** 2) * current_dt_main

            # 各プロセスの総原子数（重み計算用）
            total_atoms_psd = np.sum(accumulated_loss_psd_atoms)
            total_atoms_td = np.sum(accumulated_loss_td_atoms)
            total_atoms_sws = np.sum(accumulated_loss_sws_atoms)

            # 重み計算
            w_td = total_atoms_td / TARGET_SPS_TD if total_atoms_td > 0 else 1.0
            w_psd = total_atoms_psd / TARGET_SPS_PSD if total_atoms_psd > 0 else 1.0
            w_sws = total_atoms_sws / TARGET_SPS_SWS if total_atoms_sws > 0 else 1.0
            w_mmv = n_atoms_mmv / TARGET_SPS_MMV if n_atoms_mmv > 0 else 1.0

            newly_launched_particles = []

            # B-1. MMV
            if n_atoms_mmv > 0:
                num_sps = int(n_atoms_mmv / w_mmv)
                if np.random.random() < (n_atoms_mmv / w_mmv - num_sps): num_sps += 1
                for _ in range(num_sps):
                    # (簡易化のため位置ランダムサンプリング)
                    while True:
                        lon_rot = np.random.uniform(-np.pi, np.pi)
                        if np.random.random() < (1.0 - (1.0 / 3.0) * np.sin(lon_rot)) / (4.0 / 3.0): break
                    lat_rot = np.arcsin(np.random.uniform(-1.0, 1.0))
                    pos_rot = lonlat_to_xyz(lon_rot, lat_rot, PHYSICAL_CONSTANTS['RM'])
                    norm = pos_rot / PHYSICAL_CONSTANTS['RM']
                    speed = sample_speed_from_flux_distribution(PHYSICAL_CONSTANTS['MASS_NA'], TEMP_MMV)
                    vel_rot = speed * transform_local_to_world(sample_lambertian_direction_local(), norm)
                    newly_launched_particles.append({'pos': pos_rot, 'vel': vel_rot, 'weight': w_mmv})

            # B-2. PSD, TD, SWS
            # グリッドごとに、蓄積された放出数に基づいて生成
            # ベクトル化された変数をループで参照
            for i_lon in range(N_LON_FIXED):
                for i_lat in range(N_LAT):
                    n_psd = accumulated_loss_psd_atoms[i_lon, i_lat]
                    n_td = accumulated_loss_td_atoms[i_lon, i_lat]
                    n_sws = accumulated_loss_sws_atoms[i_lon, i_lat]

                    if (n_psd + n_td + n_sws) <= 0: continue

                    # 温度はそのセルの代表値を使用
                    T_cell = temp_grid_k[i_lon, i_lat]

                    # 3つのプロセスをループ処理
                    # (プロセス名, 原子数, 温度, SWSエネルギー, 重み)
                    procs = [('PSD', n_psd, TEMP_PSD, None, w_psd),
                             ('TD', n_td, T_cell, None, w_td),
                             ('SWS', n_sws, None, SWS_PARAMS['U_eV'], w_sws)]

                    lon_f = lon_centers[i_lon]
                    lat_f = lat_centers[i_lat]

                    for pname, n_atoms, T, U, w in procs:
                        if n_atoms <= 0: continue

                        num_sps = int(n_atoms / w)
                        if np.random.random() < (n_atoms / w - num_sps): num_sps += 1

                        for _ in range(num_sps):
                            if pname == 'SWS':
                                E_ev = sample_thompson_sigmund_energy(U)
                                spd = np.sqrt(
                                    2.0 * E_ev * PHYSICAL_CONSTANTS['EV_TO_JOULE'] / PHYSICAL_CONSTANTS['MASS_NA'])
                            else:
                                spd = sample_speed_from_flux_distribution(PHYSICAL_CONSTANTS['MASS_NA'], T)

                            # 低速カット (0.0なら全生成)
                            if spd < 0.0: continue

                            # 位置と速度
                            # 太陽固定座標系へ変換
                            lon_rot = lon_f - subsolar_lon_rad_fixed
                            pos_rot = lonlat_to_xyz(lon_rot, lat_f, PHYSICAL_CONSTANTS['RM'])
                            norm = pos_rot / PHYSICAL_CONSTANTS['RM']
                            vel_rot = spd * transform_local_to_world(sample_lambertian_direction_local(), norm)

                            newly_launched_particles.append({'pos': pos_rot, 'vel': vel_rot, 'weight': w})

            active_particles.extend(newly_launched_particles)

            # ==============================================================================
            # C. 粒子追跡 (Multiprocessing)
            # ==============================================================================
            tasks = [{'settings': settings, 'spec': spec_data_dict, 'particle_state': p,
                      'orbit': (TAA, AU, V_radial_ms, V_tangential_ms, subsolar_lon_rad_fixed),
                      'duration': current_dt_main} for p in active_particles]

            next_active_particles = []
            atoms_gained_grid_this_step = np.zeros_like(surface_density_grid)

            if tasks:
                with Pool(processes=max(1, cpu_count() - 1)) as pool:
                    # chunksizeを調整してオーバーヘッド削減
                    results = list(pool.imap(simulate_particle_for_one_step, tasks, chunksize=500))

                for res in results:
                    if res['status'] == 'alive':
                        next_active_particles.append(res['final_state'])
                    elif res['status'] == 'stuck':
                        # 吸着した場所を特定して記録
                        pos = res['pos_at_impact']
                        lon = np.arctan2(pos[1], pos[0])
                        lat = np.arcsin(np.clip(pos[2] / np.linalg.norm(pos), -1, 1))

                        # 惑星固定座標系へ
                        lon_fix = (lon + subsolar_lon_rad_fixed + np.pi) % (2 * np.pi) - np.pi
                        ix = np.searchsorted(lon_edges_fixed, lon_fix) - 1
                        iy = np.searchsorted(lat_edges_fixed, lat) - 1

                        if 0 <= ix < N_LON_FIXED and 0 <= iy < N_LAT:
                            atoms_gained_grid_this_step[ix, iy] += res['weight']

            active_particles = next_active_particles
            previous_atoms_gained_grid = atoms_gained_grid_this_step.copy()

            # ==============================================================================
            # D. 結果保存
            # ==============================================================================
            if TAA < previous_taa: target_taa_idx = 0
            if target_taa_idx < len(TARGET_TAA_DEGREES):
                tgt = TARGET_TAA_DEGREES[target_taa_idx]
                if (previous_taa < tgt <= TAA) or (
                        (tgt == 0) and (TAA < previous_taa or (TAA >= 0 and previous_taa < 0))):
                    if t_sec >= t_start_run:
                        pbar.write(f"\n>>> [Run] Saving at TAA={TAA:.1f} ({len(active_particles)} particles) <<<")

                        # 3D密度グリッド生成 (簡易版)
                        dgrid = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION, GRID_RESOLUTION), dtype=np.float32)
                        gmin, gmax = -GRID_MAX_RM * PHYSICAL_CONSTANTS['RM'], GRID_MAX_RM * PHYSICAL_CONSTANTS['RM']
                        csize = (gmax - gmin) / GRID_RESOLUTION

                        # ヒストグラム化 (ループより高速)
                        pos_arr = np.array([p['pos'] for p in active_particles])
                        weights_arr = np.array([p['weight'] for p in active_particles])
                        if len(pos_arr) > 0:
                            H, edges = np.histogramdd(pos_arr, bins=(GRID_RESOLUTION, GRID_RESOLUTION, GRID_RESOLUTION),
                                                      range=[(gmin, gmax), (gmin, gmax), (gmin, gmax)],
                                                      weights=weights_arr)
                            dgrid = H.astype(np.float32) / (csize ** 3)

                        rel_t = t_sec - t_start_run
                        fname = f"density_grid_t{int(rel_t / 3600):05d}_taa{int(round(TAA)):03d}.npy"
                        np.save(os.path.join(target_output_dir, fname), dgrid)

                        fname_s = f"surface_density_t{int(rel_t / 3600):05d}_taa{int(round(TAA)):03d}.npy"
                        np.save(os.path.join(target_output_dir, fname_s), surface_density_grid)

                    target_taa_idx += 1
            previous_taa = TAA

            pbar.update(current_dt_main)
            t_sec += current_dt_main

    print(f"Simulation Finished. Total Time: {(time.time() - start_time) / 3600:.2f}h")


if __name__ == '__main__':
    main_snapshot_simulation()