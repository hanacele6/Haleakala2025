import numpy as np
import sys
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time


# --- 物理モデルに基づく基本関数 (変更なし) ---

def calculate_surface_temperature(lon_rad, lat_rad, AU, subsolar_lon_rad):
    """水星表面の局所的な温度を計算する。"""
    T0 = 100.0
    T1 = 600.0
    # 太陽天頂角のコサインを計算
    cos_theta = np.cos(lat_rad) * np.cos(lon_rad - subsolar_lon_rad)
    if cos_theta <= 0:
        return T0
    return T0 + T1 * (cos_theta ** 0.25) * ((0.306 / AU) ** 2)


def calculate_sticking_probability(surface_temp_K):
    """表面温度に基づいて、ナトリウム原子の表面への吸着確率を計算する。"""
    A = 0.08
    B = 458.0
    porosity = 0.8
    if surface_temp_K <= 0: return 1.0
    p_stick = A * np.exp(B / surface_temp_K)
    p_stick_eff = p_stick / (1.0 - (1.0 - p_stick) * porosity)
    return min(p_stick_eff, 1.0)


def sample_maxwellian_speed(mass_kg, temp_k):
    """マクスウェル分布に従う速さをサンプリングする。"""
    K_BOLTZMANN = 1.380649e-23
    sigma = np.sqrt(K_BOLTZMANN * temp_k / mass_kg)
    vx, vy, vz = np.random.normal(0, sigma, 3)
    return np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)


def sample_cosine_direction(normal_vector):
    """コサイン則に従う3D方向ベクトルをサンプリングする。"""
    norm = np.linalg.norm(normal_vector)
    if norm == 0:
        # ゼロベクトルの場合はランダムな方向を返す
        vec = np.random.randn(3)
        return vec / np.linalg.norm(vec)

    normal_vector = normal_vector / norm
    # 法線ベクトルと直交する接ベクトルt1, t2を生成
    if np.abs(normal_vector[0]) > np.abs(normal_vector[1]):
        inv_len = 1.0 / np.sqrt(normal_vector[0] ** 2 + normal_vector[2] ** 2)
        t1 = np.array([-normal_vector[2] * inv_len, 0, normal_vector[0] * inv_len])
    else:
        inv_len = 1.0 / np.sqrt(normal_vector[1] ** 2 + normal_vector[2] ** 2)
        t1 = np.array([0, normal_vector[2] * inv_len, -normal_vector[1] * inv_len])
    t2 = np.cross(normal_vector, t1)

    p1, p2 = np.random.random(), np.random.random()
    sin_theta = np.sqrt(p1)
    cos_theta = np.sqrt(1.0 - p1)
    phi = 2 * np.pi * p2

    return t1 * sin_theta * np.cos(phi) + t2 * sin_theta * np.sin(phi) + normal_vector * cos_theta


# --- 時間発展モデル用のヘルパー関数 (変更なし) ---

def preprocess_orbit_data(filename, mercury_year_sec):
    """orbit360.txtを読み込み、時間軸を追加する。"""
    data = np.loadtxt(filename)
    num_points = len(data)
    time_axis = np.linspace(0, mercury_year_sec, num_points, endpoint=False)
    return np.column_stack((time_axis, data))


def get_orbital_params(time_sec, orbit_data, mercury_year_sec):
    """経過時間から、軌道パラメータと自転状態を返す。"""
    ROTATION_PERIOD_SEC = 58.646 * 24 * 3600
    current_time_in_orbit = time_sec % mercury_year_sec

    # 軌道データを線形補間
    taa = np.interp(current_time_in_orbit, orbit_data[:, 0], orbit_data[:, 1])
    au = np.interp(current_time_in_orbit, orbit_data[:, 0], orbit_data[:, 2])
    vms = np.interp(current_time_in_orbit, orbit_data[:, 0], orbit_data[:, 5])  # 論文に合わせてVms_msを利用

    # 太陽直下点の経度を計算
    subsolar_lon_rad = (2 * np.pi * time_sec / ROTATION_PERIOD_SEC) % (2 * np.pi)
    return taa, au, vms, subsolar_lon_rad


def lonlat_to_xyz(lon_rad, lat_rad, radius):
    """経度・緯度から三次元座標へ変換"""
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return np.array([x, y, z])


def xyz_to_lonlat_idx(x, y, z, N_LON, N_LAT):
    """三次元座標から表面グリッドのインデックスへ変換"""
    radius = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    lon_rad = np.arctan2(y, x)
    lat_rad = np.arcsin(np.clip(z / radius, -1.0, 1.0))

    lon_idx = int((lon_rad + np.pi) / (2 * np.pi) * N_LON) % N_LON
    lat_idx = int((lat_rad + np.pi / 2) / np.pi * N_LAT)

    # 範囲内に収める
    lat_idx = np.clip(lat_idx, 0, N_LAT - 1)

    return lon_idx, lat_idx


# --- コアとなる粒子追跡関数 ---
def simulate_and_track_particle(args):
    """
    一個の粒子を追跡し、最終状態、衝突位置、および大気密度グリッドへの貢献を返す。
    """
    # 引数を展開
    constants = args['constants']
    settings = args['settings']
    spec_data = args['spec']
    grid_params = args['grid_params']
    initial_pos, initial_vel = args['particle_data']
    TAA, AU, Vms_ms, subsolar_lon_rad = args['orbit']

    # 定数と設定
    C, PI, H, MASS_NA, RM, GM_MERCURY = constants['C'], constants['PI'], constants['H'], constants['MASS_NA'], \
        constants['RM'], constants['GM_MERCURY']
    T1AU, DT = settings['T1AU'], settings['DT']
    N_R, N_THETA, N_PHI, R_MAX = grid_params['N_R'], grid_params['N_THETA'], grid_params['N_PHI'], grid_params['R_MAX']
    N_LON, N_LAT = settings['N_LON'], settings['N_LAT']

    # 太陽スペクトルデータ
    wl, gamma, sigma0_perdnu2, JL = spec_data['wl'], spec_data['gamma'], spec_data['sigma0_perdnu2'], spec_data['JL']

    # この粒子専用のローカルな密度グリッドを初期化
    local_density_grid = np.zeros((N_R, N_THETA, N_PHI), dtype=np.float32)

    x, y, z = initial_pos
    vx_ms, vy_ms, vz_ms = initial_vel

    tau = T1AU * AU ** 2
    # 粒子がシミュレーション範囲を大きく超えない程度の最大ステップ数を設定 (安全装置)
    itmax = int((grid_params['R_MAX'] * 2 / 1000) / DT)  # 平均速度1km/sを仮定

    death_reason = 'max_time'
    x_prev, y_prev, z_prev = x, y, z

    for _ in range(itmax):
        # --- 電離プロセス ---
        if x > 0 and np.random.random() < (1.0 - np.exp(-DT / tau)):
            death_reason = 'ionized'
            break

        # --- 放射圧の計算 ---
        velocity_for_doppler = vx_ms + Vms_ms
        w_na_d2 = 589.1582e-9 * (1.0 - velocity_for_doppler / C)
        b = 0.0
        if wl[0] * 1e-9 <= w_na_d2 < wl[-1] * 1e-9:
            gamma2 = np.interp(w_na_d2 * 1e9, wl, gamma)
            F_lambda_1AU_m = JL * 1e4
            F_lambda_d2_at_Mercury = F_lambda_1AU_m / (AU ** 2) * gamma2
            F_nu_d2 = F_lambda_d2_at_Mercury * (w_na_d2 ** 2) / C
            J2 = sigma0_perdnu2 * F_nu_d2
            b = 1 / MASS_NA * ((H / w_na_d2) * J2)

        if x < 0 and np.sqrt(y ** 2 + z ** 2) < RM: b = 0.0

        # --- 運動方程式 ---
        r_sq = x ** 2 + y ** 2 + z ** 2
        if r_sq == 0: continue
        r = np.sqrt(r_sq)

        grav_accel_total = -GM_MERCURY / r_sq
        accel_gx = grav_accel_total * (x / r)
        accel_gy = grav_accel_total * (y / r)
        accel_gz = grav_accel_total * (z / r)

        total_accel_x = accel_gx - b
        total_accel_y = accel_gy
        total_accel_z = accel_gz

        vx_ms += total_accel_x * DT
        vy_ms += total_accel_y * DT
        vz_ms += total_accel_z * DT
        x_prev, y_prev, z_prev = x, y, z
        x += vx_ms * DT
        y += vy_ms * DT
        z += vz_ms * DT

        # --- 大気密度グリッドへの貢献を記録 ---
        if r < R_MAX:
            theta = np.arccos(np.clip(z / r, -1.0, 1.0))
            phi = np.arctan2(y, x)
            ir = int(r / (R_MAX / N_R))
            itheta = int(theta / (PI / N_THETA))
            iphi = int((phi + PI) / (2 * PI / N_PHI)) % N_PHI

            if 0 <= ir < N_R and 0 <= itheta < N_THETA:
                local_density_grid[ir, itheta, iphi] += DT

        # --- 終了条件の判定 ---
        if r > R_MAX:
            death_reason = 'escaped'
            break

        if r <= RM:
            temp_at_impact = calculate_surface_temperature(
                np.arctan2(y_prev, x_prev), np.arcsin(np.clip(z_prev / RM, -1.0, 1.0)), AU, subsolar_lon_rad
            )
            if np.random.random() < calculate_sticking_probability(temp_at_impact):
                death_reason = 'stuck'
            else:
                death_reason = 'bounced'
            break

    impact_lon_idx, impact_lat_idx = (None, None)
    if death_reason == 'stuck':
        impact_lon_idx, impact_lat_idx = xyz_to_lonlat_idx(x_prev, y_prev, z_prev, N_LON, N_LAT)

    return (death_reason, (impact_lon_idx, impact_lat_idx), local_density_grid)


# --- メインの制御関数 ---
def main_time_evolution_with_density_output():
    start_time = time.time()

    # --- シミュレーション設定 ---
    OUTPUT_DIRECTORY = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult3D_test"

    N_LON = 24
    N_LAT = 24
    INITIAL_CNA_PER_M2 = 1.5e17

    N_R = 50
    N_THETA = 24
    N_PHI = 24
    GRID_RADIUS_RM = 5.0

    TIME_STEP_SEC = 3600
    TOTAL_SIM_YEARS = 1.0
    SAVE_INTERVAL_STEPS = 22

    ATOMS_PER_SUPERPARTICLE = 1e25

    settings = {
        'T1AU': 61728.4,
        'DT': 10.0,
        'N_LON': N_LON,
        'N_LAT': N_LAT
    }

    constants = {
        'C': 299792458.0, 'PI': np.pi, 'H': 6.62607015e-34,
        'MASS_NA': 22.98976928 * 1.66054e-27, 'RM': 2439.7e3,
        'GM_MERCURY': 2.2032e13
    }

    grid_params = {
        'N_R': N_R, 'N_THETA': N_THETA, 'N_PHI': N_PHI,
        'R_MAX': constants['RM'] * GRID_RADIUS_RM
    }

    run_name = f"PSD_atm_R{N_R}_YEAR{TOTAL_SIM_YEARS:.1f}_SP{ATOMS_PER_SUPERPARTICLE:.0e}"
    target_output_dir = os.path.join(OUTPUT_DIRECTORY, run_name)
    os.makedirs(target_output_dir, exist_ok=True)
    print(f"結果は '{target_output_dir}' に保存されます。")

    MERCURY_YEAR_SEC = 87.97 * 24 * 3600
    TOTAL_SIM_SEC = TOTAL_SIM_YEARS * MERCURY_YEAR_SEC
    time_steps = np.arange(0, TOTAL_SIM_SEC, TIME_STEP_SEC)

    lon_edges = np.linspace(-np.pi, np.pi, N_LON + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    cell_areas_m2 = (constants['RM'] ** 2) * (lon_edges[1] - lon_edges[0]) * \
                    (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))
    surface_grid_density = np.full((N_LON, N_LAT), INITIAL_CNA_PER_M2)

    try:
        spec_data_np = np.loadtxt('SolarSpectrum_Na0.txt', usecols=(0, 3))
        orbit_data = preprocess_orbit_data('orbit360.txt', MERCURY_YEAR_SEC)
    except FileNotFoundError as e:
        print(f"エラー: データファイル '{e.filename}' が見つかりません。");
        sys.exit()

    wl, gamma = spec_data_np[:, 0], spec_data_np[:, 1]
    if not np.all(np.diff(wl) > 0):
        sort_indices = np.argsort(wl);
        wl, gamma = wl[sort_indices], gamma[sort_indices]

    ME, E_CHARGE = 9.1093897e-31, 1.60217733e-19
    epsilon_0 = 8.854187817e-12
    sigma_const = E_CHARGE ** 2 / (4 * ME * constants['C'] * epsilon_0)
    spec_data_dict = {'wl': wl, 'gamma': gamma, 'sigma0_perdnu2': sigma_const * 0.641, 'JL': 5.18e14}

    # --- メイン時間発展ループ ---
    with tqdm(total=len(time_steps), desc="Time Evolution") as pbar:
        for step_idx, t_sec in enumerate(time_steps):
            TAA, AU, Vms_ms, subsolar_lon_rad = get_orbital_params(t_sec, orbit_data, MERCURY_YEAR_SEC)

            pbar.set_description(f"Time Step {step_idx + 1}/{len(time_steps)} (TAA={TAA:.1f}) | Generating particles")

            particles_to_launch_args = []
            total_desorbed_atoms_grid = np.zeros_like(surface_grid_density)

            # --- 1. 放出する粒子を決定 (変更なし) ---
            lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
            lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
            for i_lon in range(N_LON):
                for i_lat in range(N_LAT):
                    cos_Z = np.cos(lat_centers[i_lat]) * np.cos(lon_centers[i_lon] - subsolar_lon_rad)
                    if cos_Z <= 0: continue
                    F_UV_current_per_m2 = (1.5e14 * 1e4) / (AU ** 2)
                    Q_PSD_m2 = 2.0e-20 / 1e4
                    current_density = surface_grid_density[i_lon, i_lat]
                    desorption_rate_per_m2 = F_UV_current_per_m2 * Q_PSD_m2 * cos_Z * current_density
                    n_atoms_to_desorb = desorption_rate_per_m2 * cell_areas_m2[i_lat] * TIME_STEP_SEC
                    total_desorbed_atoms_grid[i_lon, i_lat] = n_atoms_to_desorb
                    n_super_particles = n_atoms_to_desorb / ATOMS_PER_SUPERPARTICLE
                    num_to_launch = int(n_super_particles)
                    if np.random.random() < (n_super_particles - num_to_launch):
                        num_to_launch += 1
                    for _ in range(num_to_launch):
                        initial_pos = lonlat_to_xyz(lon_centers[i_lon], lat_centers[i_lat], constants['RM'])
                        normal_vec = initial_pos / np.linalg.norm(initial_pos)
                        speed = sample_maxwellian_speed(constants['MASS_NA'], 1500.0)
                        direction = sample_cosine_direction(normal_vec)
                        initial_vel = speed * direction
                        task_args = {
                            'constants': constants, 'settings': settings, 'spec': spec_data_dict,
                            'grid_params': grid_params,
                            'particle_data': (initial_pos, initial_vel),
                            'orbit': (TAA, AU, Vms_ms, subsolar_lon_rad)
                        }
                        particles_to_launch_args.append(task_args)

            # --- 2. 表面密度を更新 (変更なし) ---
            surface_grid_density -= total_desorbed_atoms_grid / cell_areas_m2[np.newaxis, :]
            surface_grid_density[surface_grid_density < 0] = 0

            # --- 3. 粒子追跡を実行 ---
            if particles_to_launch_args:
                pbar.set_description(
                    f"Time Step {step_idx + 1}/{len(time_steps)} (TAA={TAA:.1f}) | Tracking {len(particles_to_launch_args)} particles")
                with Pool(processes=cpu_count()) as pool:
                    # imapの結果をそのままリストに変換して処理を待つ（内側プログレスバーはなし）
                    results = list(pool.imap(simulate_and_track_particle, particles_to_launch_args, chunksize=100))
            else:
                results = []  # 粒子がない場合は空のリスト

            # --- 4. 結果を集計 ---
            pbar.set_description(f"Time Step {step_idx + 1}/{len(time_steps)} (TAA={TAA:.1f}) | Aggregating results")
            master_density_grid = np.zeros((N_R, N_THETA, N_PHI), dtype=np.float32)
            if results:
                for reason, (imp_lon, imp_lat), local_grid in results:
                    if reason == 'stuck':
                        surface_grid_density[imp_lon, imp_lat] += ATOMS_PER_SUPERPARTICLE / cell_areas_m2[imp_lat]
                    master_density_grid += local_grid

            # --- 5. 大気密度を計算して保存 ---
            if (step_idx + 1) % SAVE_INTERVAL_STEPS == 0:
                # ★★★ print() の代わりに pbar.write() を使う ★★★
                pbar.write(f"\nCalculating and saving snapshot at t={t_sec / 3600:.1f}h (TAA={TAA:.1f})")
                r_edges = np.linspace(0, grid_params['R_MAX'], N_R + 1)
                theta_edges = np.linspace(0, np.pi, N_THETA + 1)
                delta_r3_3 = (r_edges[1:] ** 3 - r_edges[:-1] ** 3) / 3.0
                delta_cos_theta = np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:])
                cell_volumes_m3 = delta_r3_3[:, np.newaxis, np.newaxis] * \
                                  delta_cos_theta[np.newaxis, :, np.newaxis] * \
                                  (2 * np.pi / N_PHI)
                cell_volumes_m3[cell_volumes_m3 == 0] = 1e-30
                number_density_m3 = (master_density_grid * ATOMS_PER_SUPERPARTICLE) / (cell_volumes_m3 * TIME_STEP_SEC)
                number_density_cm3 = number_density_m3 / 1e6
                save_time_h = t_sec / 3600
                filename = f"atmospheric_density_t{int(save_time_h):05d}_taa{int(TAA):03d}.npy"
                np.save(os.path.join(target_output_dir, filename), number_density_cm3)
                np.save(os.path.join(target_output_dir, f"surface_density_t{int(save_time_h):05d}.npy"),
                        surface_grid_density)

            pbar.update(1)  # メインループのプログレスバーを1ステップ進める

    end_time = time.time()
    print(f"\n★★★ 時間発展シミュレーションが完了しました ★★★")
    print(f"総計算時間: {(end_time - start_time) / 3600:.2f} 時間")


if __name__ == '__main__':
    for f in ['orbit360.txt', 'SolarSpectrum_Na0.txt']:
        if not os.path.exists(f):
            print(f"エラー: 必須ファイル '{f}' が見つかりません。プログラムを終了します。");
            sys.exit()
    main_time_evolution_with_density_output()
