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
    vms = np.interp(current_time_in_orbit, orbit_data[:, 0], orbit_data[:, 5])

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
    weight = args['weight']

    # 定数と設定
    C, PI, H, MASS_NA, RM, GM_MERCURY, K_BOLTZMANN = constants.values()
    BETA, T1AU, DT = settings['BETA'], settings['T1AU'], settings['DT']
    N_R, N_THETA, N_PHI, R_MAX = grid_params.values()
    N_LON, N_LAT = settings['N_LON'], settings['N_LAT']

    # 太陽スペクトルデータ
    wl, gamma, sigma0_perdnu2, sigma0_perdnu1, JL = spec_data.values()

    # この粒子専用のローカルな密度グリッドを初期化
    local_density_grid = np.zeros((N_R, N_THETA, N_PHI), dtype=np.float32)

    x, y, z = initial_pos
    vx_ms, vy_ms, vz_ms = initial_vel

    tau = T1AU * AU ** 2
    # itmaxを少し長めに設定（粒子が地表付近で何度もバウンドすることを考慮）
    itmax = int(tau * 2.0 / DT)

    death_reason = 'max_time'

    for _ in range(itmax):
        # --- ループの最初に衝突前の位置と速度を保存 ---
        x_prev, y_prev, z_prev = x, y, z
        vx_ms_prev, vy_ms_prev, vz_ms_prev = vx_ms, vy_ms, vz_ms

        # --- 電離プロセス ---
        # 太陽に照らされている側（簡単化のためx>0で判定）でのみ電離を考慮
        if x > 0 and np.random.random() < (1.0 - np.exp(-DT / tau)):
            death_reason = 'ionized'
            break

        # --- 放射圧の計算 ---
        # 太陽方向（x）への速度成分を考慮
        velocity_for_doppler = vx_ms - Vms_ms  # 水星の公転速度と粒子の速度の相対速度
        w_na_d2 = 589.1582e-9 * (1.0 + velocity_for_doppler / C)
        w_na_d1 = 589.7558e-9 * (1.0 + velocity_for_doppler / C)
        b = 0.0
        # スペクトルの範囲内かチェック
        if (wl[0] * 1e-9 <= w_na_d2 < wl[-1] * 1e-9) and \
                (wl[0] * 1e-9 <= w_na_d1 < wl[-1] * 1e-9):
            gamma2 = np.interp(w_na_d2 * 1e9, wl, gamma)
            gamma1 = np.interp(w_na_d1 * 1e9, wl, gamma)
            F_lambda_1AU_m = JL * 1e4 * 1e9  # [phs/s/m2/m]

            F_lambda_d1_at_Mercury = F_lambda_1AU_m / (AU ** 2) * gamma1
            F_nu_d1 = F_lambda_d1_at_Mercury * (w_na_d1 ** 2) / C
            J1 = sigma0_perdnu1 * F_nu_d1

            F_lambda_d2_at_Mercury = F_lambda_1AU_m / (AU ** 2) * gamma2
            F_nu_d2 = F_lambda_d2_at_Mercury * (w_na_d2 ** 2) / C
            J2 = sigma0_perdnu2 * F_nu_d2

            b = 1 / MASS_NA * ((H / w_na_d1) * J1 + (H / w_na_d2) * J2)

        # 水星の影に入ったら放射圧は0
        if x < 0 and np.sqrt(y ** 2 + z ** 2) < RM: b = 0.0

        # --- 運動方程式 ---
        r_sq = x ** 2 + y ** 2 + z ** 2
        if r_sq == 0: continue
        r = np.sqrt(r_sq)

        grav_accel_total = -GM_MERCURY / r_sq
        accel_gx = grav_accel_total * (x / r)
        accel_gy = grav_accel_total * (y / r)
        accel_gz = grav_accel_total * (z / r)

        # 放射圧は太陽と反対方向 (-x方向) に働く
        total_accel_x = accel_gx - b
        total_accel_y = accel_gy
        total_accel_z = accel_gz

        vx_ms += total_accel_x * DT
        vy_ms += total_accel_y * DT
        vz_ms += total_accel_z * DT

        x += ((vx_ms_prev + vx_ms) / 2.0) * DT
        y += ((vy_ms_prev + vy_ms) / 2.0) * DT
        z += ((z_prev + vz_ms) / 2.0) * DT

        r_current = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        # --- 大気密度グリッドへの貢献を記録 ---
        if r_current < R_MAX:
            # 座標変換 (安全のためclip)
            theta = np.arccos(np.clip(z / r_current, -1.0, 1.0))
            phi = np.arctan2(y, x)
            ir = int(r_current / (R_MAX / N_R))
            itheta = int(theta / (PI / N_THETA))
            iphi = int((phi + PI) / (2 * PI / N_PHI)) % N_PHI

            if 0 <= ir < N_R and 0 <= itheta < N_THETA:
                local_density_grid[ir, itheta, iphi] += weight * DT

        # --- 終了条件の判定 ---
        if r_current > R_MAX:
            death_reason = 'escaped'
            break

        if r_current <= RM:
            # 衝突地点の温度を計算
            # 衝突直前の位置(x_prev, y_prev, z_prev)を使う
            impact_radius = np.sqrt(x_prev ** 2 + y_prev ** 2 + z_prev ** 2)
            impact_lon = np.arctan2(y_prev, x_prev)
            impact_lat = np.arcsin(np.clip(z_prev / impact_radius, -1.0, 1.0))

            temp_at_impact = calculate_surface_temperature(
                impact_lon, impact_lat, AU, subsolar_lon_rad
            )

            # 吸着確率に基づいて吸着するか判定
            if np.random.random() < calculate_sticking_probability(temp_at_impact):
                death_reason = 'stuck'
                # 吸着したので、衝突位置を記録してループを抜ける
                impact_lon_idx, impact_lat_idx = xyz_to_lonlat_idx(x_prev, y_prev, z_prev, N_LON, N_LAT)
                # ★★★ 戻り値のためにタプルを定義 ★★★
                impact_location = (impact_lon_idx, impact_lat_idx)
                return (death_reason, impact_location, local_density_grid, weight)

            else:  # 吸着しなかった場合：反発計算を行う
                # 衝突前の速度ベクトルから入射エネルギーを計算
                v_in_sq = vx_ms_prev ** 2 + vy_ms_prev ** 2 + vz_ms_prev ** 2
                E_in = 0.5 * MASS_NA * (v_in_sq)
                # 表面温度から熱エネルギーを計算
                E_T = K_BOLTZMANN * temp_at_impact
                # エネルギー交換モデルに基づいて放出エネルギーを計算
                E_out = BETA * E_T + (1.0 - BETA) * E_in
                v_out_sq = E_out / (0.5 * MASS_NA)
                v_out_speed = np.sqrt(v_out_sq) if v_out_sq > 0 else 0.0

                # 衝突地点の法線ベクトル（位置ベクトルと同じ）
                impact_normal = np.array([x_prev, y_prev, z_prev])
                # コサイン則に従って反発方向を決定
                rebound_direction = sample_cosine_direction(impact_normal)

                # 新しい速度ベクトルを計算
                vx_ms = v_out_speed * rebound_direction[0]
                vy_ms = v_out_speed * rebound_direction[1]
                vz_ms = v_out_speed * rebound_direction[2]

                # 粒子を正確に地表に戻す
                impact_pos_norm = np.linalg.norm(impact_normal)
                if impact_pos_norm > 0:
                    x = RM * impact_normal[0] / impact_pos_norm
                    y = RM * impact_normal[1] / impact_pos_norm
                    z = RM * impact_normal[2] / impact_pos_norm

                # ループを継続して次のステップへ
                continue


    # ループが正常に終了した場合（max_time, escaped, ionized）の戻り値
    impact_location = (None, None)  # 吸着していないので衝突位置はNone
    return (death_reason, impact_location, local_density_grid, weight)


# --- メインの制御関数 ---
def main_time_evolution_with_density_output():
    start_time = time.time()

    # --- シミュレーション設定 ---
    # 出力先ディレクトリを適宜変更してください
    OUTPUT_DIRECTORY = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult3D_test"

    # N_LON, N_LAT: 地表を分割する経度・緯度のグリッド数
    N_LON = 24
    N_LAT = 24
    # INITIAL_CNA_PER_M2: シミュレーション開始時の地表のナトリウム原子の初期カラム密度 [atoms/m^2]
    INITIAL_CNA_PER_M2 = 1.5e17  # suzuki (2019)

    # N_R, N_THETA, N_PHI: 大気密度を計算する球座標グリッドの分割数 (動径、天頂角、方位角)
    N_R = 50
    N_THETA = 24
    N_PHI = 24
    # GRID_RADIUS_RM: 大気密度グリッドの最大半径 (水星半径の何倍か)
    GRID_RADIUS_RM = 5.0

    # TIME_STEP_SEC: 時間発展の1ステップの時間 [秒]。ここでは1時間。
    TIME_STEP_SEC = 3600
    # TOTAL_SIM_YEARS: シミュレーションを実行する合計年数 (水星年)
    TOTAL_SIM_YEARS = 1.0

    # SUPERPARTICLES_PER_CELL: 地表の1セルから1タイムステップあたりに放出するスーパーパーティクルの数
    SUPERPARTICLES_PER_CELL = 10

    # TARGET_TAA_DEGREES: 結果を出力する水星の真近点角 [度]。ここでは1度ごと。
    TARGET_TAA_DEGREES = np.arange(0, 360, 1)

    settings = {
        'BETA': 0.5,  # 水星表面との衝突での係数 0で弾性衝突、1で完全にエネルギーを失う　
        # 理想的な石英表面において、ナトリウムではβ≈0.62、カリウムではβ≈0.26
        'T1AU': 61728.4,  # 電離寿命　実験値 [s]
        # 'T1AU': 168918.0, # 電離寿命 理論値[s]
        'DT': 10.0,  # 時間ステップ [s]
        'N_LON': N_LON,
        'N_LAT': N_LAT
    }

    constants = {
        'C': 299792458.0,  # 光速 [m/s]
        'PI': np.pi,
        'H': 6.62607015e-34,  # プランク定数 [kg・m^2/s] (J・s)
        'MASS_NA': 22.98976928 * 1.66054e-27,  # Na原子の質量 [kg]
        'RM': 2439.7e3,  # 水星の半径 [m]
        'GM_MERCURY': 2.2032e13,  # G*M_mercury [m^3/s^2]  (万有引力定数 * 水星の質量)
        'K_BOLTZMANN': 1.380649e-23  # ボルツマン定数[J/K]
    }

    grid_params = {
        'N_R': N_R, 'N_THETA': N_THETA, 'N_PHI': N_PHI,
        'R_MAX': constants['RM'] * GRID_RADIUS_RM
    }

    run_name = f"PSD_atm_R{N_R}_YEAR{TOTAL_SIM_YEARS:.1f}_SPperCell{SUPERPARTICLES_PER_CELL}_Q2.0_new"
    target_output_dir = os.path.join(OUTPUT_DIRECTORY, run_name)
    os.makedirs(target_output_dir, exist_ok=True)
    print(f"結果は '{target_output_dir}' に保存されます。")

    # MERCURY_YEAR_SEC: 1水星年の秒数 (87.97日)
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

    ME, E_CHARGE = 9.1093897e-31, 1.60217733e-19  # 電子の質量 [kg] # 電子の電荷 [C]
    epsilon_0 = 8.854187817e-12  # 真空の誘電率
    sigma_const = E_CHARGE ** 2 / (4 * ME * constants['C'] * epsilon_0)
    spec_data_dict = {'wl': wl,
                      'gamma': gamma,
                      'sigma0_perdnu2': sigma_const * 0.641,  # 0.641 = D2線の振動子強度
                      'sigma0_perdnu1': sigma_const * 0.320,  # 0.320 = D1線の振動子強度
                      'JL': 5.18e14# 1AUでの太陽フラックス [phs/s/cm2/nm]
                      }

    # previous_taa: TAAが360度から0度に戻ったことを検出するための変数。-1で初期化。
    previous_taa = -1
    target_taa_idx = 0

    with tqdm(total=len(time_steps), desc="Time Evolution") as pbar:
        for step_idx, t_sec in enumerate(time_steps):
            TAA, AU, Vms_ms, subsolar_lon_rad = get_orbital_params(t_sec, orbit_data, MERCURY_YEAR_SEC)

            if TAA < previous_taa:
                target_taa_idx = 0

            save_this_step = False
            if target_taa_idx < len(TARGET_TAA_DEGREES):
                current_target_taa = TARGET_TAA_DEGREES[target_taa_idx]
                if previous_taa < current_target_taa <= TAA:
                    save_this_step = True
                    target_taa_idx += 1

            pbar.set_description(f"Time Step {step_idx + 1}/{len(time_steps)} (TAA={TAA:.1f}) | Generating particles")

            particles_to_launch_args = []
            total_desorbed_atoms_grid = np.zeros_like(surface_grid_density)

            for i_lon in range(N_LON):
                for i_lat in range(N_LAT):
                    lon_center_rad = (lon_edges[i_lon] + lon_edges[i_lon + 1]) / 2
                    lat_center_rad = (lat_edges[i_lat] + lat_edges[i_lat + 1]) / 2
                    cos_Z = np.cos(lat_center_rad) * np.cos(lon_center_rad - subsolar_lon_rad)

                    if cos_Z <= 0: continue

                    F_UV_current_per_m2 = (1.5e14 * 1e4) / (AU ** 2) # 1天文単位での紫外線光子フラックス [photons/m^2/s]
                    #Q_PSD_m2 = 1.0e-20 / 1e4  # 光脱離断面積 [m^2]
                    Q_PSD_m2 = 2.0e-20 / 1e4 # suzukiが使ってたやつ
                    #Q_PSD_m2 = 3.0e-20 / 1e4  # YakshinskiyとMadey（1999）
                    #Q_PSD_m2 = 1.4e-21 / 1e4 # Killenら（2004）

                    current_density = surface_grid_density[i_lon, i_lat]
                    desorption_rate_per_m2 = F_UV_current_per_m2 * Q_PSD_m2 * cos_Z * current_density
                    n_atoms_to_desorb = desorption_rate_per_m2 * cell_areas_m2[i_lat] * TIME_STEP_SEC
                    total_desorbed_atoms_grid[i_lon, i_lat] = n_atoms_to_desorb

                    if n_atoms_to_desorb <= 0: continue

                    weight_for_this_cell = n_atoms_to_desorb / SUPERPARTICLES_PER_CELL

                    for _ in range(SUPERPARTICLES_PER_CELL):
                        random_lon_rad = np.random.uniform(lon_edges[i_lon], lon_edges[i_lon + 1])

                        sin_lat_min = np.sin(lat_edges[i_lat])
                        sin_lat_max = np.sin(lat_edges[i_lat + 1])
                        random_sin_lat = np.random.uniform(sin_lat_min, sin_lat_max)
                        random_lat_rad = np.arcsin(random_sin_lat)

                        initial_pos = lonlat_to_xyz(random_lon_rad, random_lat_rad, constants['RM'])

                        normal_vec = initial_pos / np.linalg.norm(initial_pos)
                        speed = sample_maxwellian_speed(constants['MASS_NA'], 1500.0)
                        direction = sample_cosine_direction(normal_vec)
                        initial_vel = speed * direction

                        task_args = {
                            'constants': constants, 'settings': settings, 'spec': spec_data_dict,
                            'grid_params': grid_params,
                            'particle_data': (initial_pos, initial_vel),
                            'orbit': (TAA, AU, Vms_ms, subsolar_lon_rad),
                            'weight': weight_for_this_cell
                        }
                        particles_to_launch_args.append(task_args)

            surface_grid_density -= total_desorbed_atoms_grid / cell_areas_m2[np.newaxis, :]
            surface_grid_density[surface_grid_density < 0] = 0

            if particles_to_launch_args:
                pbar.set_description(
                    f"Time Step {step_idx + 1}/{len(time_steps)} (TAA={TAA:.1f}) | Tracking {len(particles_to_launch_args)} particles")
                with Pool(processes=cpu_count()) as pool:
                    results = list(pool.imap(simulate_and_track_particle, particles_to_launch_args, chunksize=100))
            else:
                results = []

            pbar.set_description(f"Time Step {step_idx + 1}/{len(time_steps)} (TAA={TAA:.1f}) | Aggregating results")
            master_density_grid = np.zeros((N_R, N_THETA, N_PHI), dtype=np.float32)
            if results:
                for reason, (imp_lon, imp_lat), local_grid, weight in results:
                    if reason == 'stuck' and imp_lon is not None and imp_lat is not None:
                        surface_grid_density[imp_lon, imp_lat] += weight / cell_areas_m2[imp_lat]
                    master_density_grid += local_grid

            if save_this_step:
                pbar.write(f"\n>>> Saving snapshot at TAA={TAA:.1f} (t={t_sec / 3600:.1f}h) <<<")
                r_edges = np.linspace(0, grid_params['R_MAX'], N_R + 1)
                theta_edges = np.linspace(0, np.pi, N_THETA + 1)
                delta_r3_3 = (r_edges[1:] ** 3 - r_edges[:-1] ** 3) / 3.0
                delta_cos_theta = np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:])
                cell_volumes_m3 = delta_r3_3[:, np.newaxis, np.newaxis] * \
                                  delta_cos_theta[np.newaxis, :, np.newaxis] * \
                                  (2 * np.pi / N_PHI)
                cell_volumes_m3[cell_volumes_m3 == 0] = 1e-30

                number_density_m3 = master_density_grid / (cell_volumes_m3 * TIME_STEP_SEC)
                number_density_cm3 = number_density_m3 / 1e6
                save_time_h = t_sec / 3600
                filename = f"atmospheric_density_t{int(save_time_h):05d}_taa{int(TAA):03d}.npy"
                np.save(os.path.join(target_output_dir, filename), number_density_cm3)
                np.save(os.path.join(target_output_dir, f"surface_density_t{int(save_time_h):05d}.npy"),
                        surface_grid_density)

            previous_taa = TAA
            pbar.update(1)

    end_time = time.time()
    print(f"\n★★★ 時間発展シミュレーションが完了しました ★★★")
    print(f"総計算時間: {(end_time - start_time) / 3600:.2f} 時間")


if __name__ == '__main__':
    # 実行前に必要なファイルが存在するか確認
    for f in ['orbit360.txt', 'SolarSpectrum_Na0.txt']:
        if not os.path.exists(f):
            print(f"エラー: 必須ファイル '{f}' が見つかりません。プログラムを終了します。");
            sys.exit()
    # メイン関数を実行
    main_time_evolution_with_density_output()