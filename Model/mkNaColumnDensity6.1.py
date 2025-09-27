import numpy as np
import sys
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

# --- 物理定数 (グローバルスコープ) ---
PHYSICAL_CONSTANTS = {
    'C': 299792458.0,  # 光速 [m/s]
    'PI': np.pi,
    'H': 6.62607015e-34,  # プランク定数 [J・s]
    'MASS_NA': 22.98976928 * 1.66054e-27,  # Na原子の質量 [kg]
    'RM': 2439.7e3,  # 水星の半径 [m]
    'GM_MERCURY': 2.2032e13,  # 万有引力定数 * 水星の質量 [m^3/s^2]
    'K_BOLTZMANN': 1.380649e-23,  # ボルツマン定数 [J/K]
    'E_CHARGE': 1.60217733e-19,  # 素電荷 [C]
    'ME': 9.1093897e-31,  # 電子の質量 [kg]
    'EPSILON_0': 8.854187817e-12  # 真空の誘電率 [F/m]
}


# --- 物理モデルに基づく基本関数 ---

def calculate_surface_temperature(lon_rad, lat_rad, AU, subsolar_lon_rad):
    """
    水星表面の経緯度、太陽距離、太陽直下点経度から局所的な温度を計算する。

    Args:
        lon_rad (float): 経度 [rad]
        lat_rad (float): 緯度 [rad]
        AU (float): 太陽距離 [天文単位]
        subsolar_lon_rad (float): 太陽直下点の経度 [rad]

    Returns:
        float: 表面温度 [K]
    """
    T0 = 100.0  # 夜側最低温度
    T1 = 600.0  # 日側最大温度上昇
    cos_theta = np.cos(lat_rad) * np.cos(lon_rad - subsolar_lon_rad)
    if cos_theta <= 0:
        return T0
    return T0 + T1 * (cos_theta ** 0.25) * ((0.306 / AU) ** 2)


def calculate_sticking_probability(surface_temp_K):
    """
    表面温度に基づいて、ナトリウム原子の表面への吸着確率を計算する。

    Args:
        surface_temp_K (float): 表面温度 [K]

    Returns:
        float: 吸着確率 (0-1)
    """
    A = 0.08
    B = 458.0
    porosity = 0.8
    if surface_temp_K <= 0: return 1.0
    p_stick = A * np.exp(B / surface_temp_K)
    p_stick_eff = p_stick / (1.0 - (1.0 - p_stick) * porosity)
    return min(p_stick_eff, 1.0)


def sample_maxwellian_speed(mass_kg, temp_k):
    """
    マクスウェル分布に従う速さをサンプリングする。

    Args:
        mass_kg (float): 粒子の質量 [kg]
        temp_k (float): 温度 [K]

    Returns:
        float: サンプリングされた速さ [m/s]
    """
    sigma = np.sqrt(PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_k / mass_kg)
    vx, vy, vz = np.random.normal(0, sigma, 3)
    return np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)


def sample_cosine_direction(normal_vector):
    """
    法線ベクトル周りにコサイン分布に従う3D方向ベクトルをサンプリングする。

    Args:
        normal_vector (np.ndarray): 3次元の法線ベクトル

    Returns:
        np.ndarray: サンプリングされた3次元方向ベクトル (正規化済み)
    """
    norm = np.linalg.norm(normal_vector)
    if norm == 0:
        vec = np.random.randn(3)
        return vec / np.linalg.norm(vec)

    normal_vector = normal_vector / norm
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


def sample_isotropic_direction(normal_vector):
    """
    ★★★ 追加 ★★★
    法線ベクトルの半球上で等方的に分布する方向ベクトルを生成する。
    """
    vec = np.random.randn(3)
    vec /= np.linalg.norm(vec)
    if np.dot(vec, normal_vector) < 0:
        vec = -vec
    return vec


# --- 時間発展モデル用のヘルパー関数 ---

def preprocess_orbit_data(filename, mercury_year_sec):
    """軌道データファイルを読み込み、時間軸を追加して前処理する。"""
    data = np.loadtxt(filename)
    num_points = len(data)
    time_axis = np.linspace(0, mercury_year_sec, num_points, endpoint=False)
    return np.column_stack((time_axis, data))


def get_orbital_params(time_sec, orbit_data, mercury_year_sec):
    """指定された時刻における水星の軌道パラメータを取得する。"""
    ROTATION_PERIOD_SEC = 58.646 * 24 * 3600
    current_time_in_orbit = time_sec % mercury_year_sec
    # 線形補間で各パラメータを計算
    taa = np.interp(current_time_in_orbit, orbit_data[:, 0], orbit_data[:, 1])
    au = np.interp(current_time_in_orbit, orbit_data[:, 0], orbit_data[:, 2])
    vms = np.interp(current_time_in_orbit, orbit_data[:, 0], orbit_data[:, 5])
    # 太陽直下点経度は水星の自転に基づいて計算
    subsolar_lon_rad = (2 * np.pi * time_sec / ROTATION_PERIOD_SEC) % (2 * np.pi)
    return taa, au, vms, subsolar_lon_rad


def lonlat_to_xyz(lon_rad, lat_rad, radius):
    """経緯度（ラジアン）を三次元直交座標に変換する。"""
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return np.array([x, y, z])


def xyz_to_lonlat_idx(x, y, z, N_LON, N_LAT):
    """三次元直交座標を表面グリッドのインデックスに変換する。"""
    radius = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    lon_rad = np.arctan2(y, x)
    lat_rad = np.arcsin(np.clip(z / radius, -1.0, 1.0))
    # インデックス計算
    lon_idx = int((lon_rad + np.pi) / (2 * np.pi) * N_LON) % N_LON
    lat_idx = int((lat_rad + np.pi / 2) / np.pi * N_LAT)
    lat_idx = np.clip(lat_idx, 0, N_LAT - 1)
    return lon_idx, lat_idx


# --- ★★★ コア追跡関数 (RK4積分法に更新) ★★★ ---
def _calculate_acceleration(pos, vel, Vms_ms, AU, spec_data):
    """
    【RK4用ヘルパー】指定された位置と速度における粒子の加速度を計算する。
    """
    x, y, z = pos
    vx, vy, vz = vel

    # 放射圧計算 (太陽は-X方向から来ると仮定)
    velocity_for_doppler = vx - Vms_ms
    w_na_d2 = 589.1582e-9 * (1.0 + velocity_for_doppler / PHYSICAL_CONSTANTS['C'])
    w_na_d1 = 589.7558e-9 * (1.0 + velocity_for_doppler / PHYSICAL_CONSTANTS['C'])
    b = 0.0

    wl, gamma, sigma0_perdnu2, sigma0_perdnu1, JL = spec_data.values()
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

    # 水星の影に入っている場合は放射圧を0にする
    if x < 0 and np.sqrt(y ** 2 + z ** 2) < PHYSICAL_CONSTANTS['RM']:
        b = 0.0

    accel_srp = np.array([-b, 0.0, 0.0])

    # 重力加速度計算
    r_sq = x ** 2 + y ** 2 + z ** 2
    if r_sq == 0:
        accel_g = np.array([0.0, 0.0, 0.0])
    else:
        r = np.sqrt(r_sq)
        grav_accel_total = -PHYSICAL_CONSTANTS['GM_MERCURY'] / r_sq
        accel_g = grav_accel_total * (pos / r)

    return accel_srp + accel_g


def simulate_particle_for_one_step(args):
    """
    一個の粒子を、指定された時間 (duration) だけ追跡し、最終状態を返す。
    積分法を4次ルンゲ=クッタ法に更新。
    """
    # 引数展開
    settings, spec_data = args['settings'], args['spec']
    TAA, AU, Vms_ms, subsolar_lon_rad = args['orbit']
    duration, DT = args['duration'], settings['DT']

    # 定数展開
    RM = PHYSICAL_CONSTANTS['RM']
    R_MAX = RM * settings['GRID_RADIUS_RM']

    # 粒子状態
    pos, vel, weight = args['particle_state']['pos'], args['particle_state']['vel'], args['particle_state']['weight']

    # 電離寿命の計算
    tau = settings['T1AU'] * AU ** 2

    # 時間発展ループ
    num_steps = int(duration / DT)
    for _ in range(num_steps):
        # 1. 電離判定 (各ステップの最初に評価)
        # 日照側にいる場合のみ電離の可能性あり (x>0)
        if pos[0] > 0 and np.random.random() < (1.0 - np.exp(-DT / tau)):
            return {'status': 'ionized', 'final_state': None}

        # 2. 軌道計算 (4次ルンゲ＝クッタ法)
        pos_prev = pos.copy()  # 衝突判定用に更新前の位置を保持

        k1_vel = DT * _calculate_acceleration(pos, vel, Vms_ms, AU, spec_data)
        k1_pos = DT * vel
        k2_vel = DT * _calculate_acceleration(pos + 0.5 * k1_pos, vel + 0.5 * k1_vel, Vms_ms, AU, spec_data)
        k2_pos = DT * (vel + 0.5 * k1_vel)
        k3_vel = DT * _calculate_acceleration(pos + 0.5 * k2_pos, vel + 0.5 * k2_vel, Vms_ms, AU, spec_data)
        k3_pos = DT * (vel + 0.5 * k2_vel)
        k4_vel = DT * _calculate_acceleration(pos + k3_pos, vel + k3_vel, Vms_ms, AU, spec_data)
        k4_pos = DT * (vel + k3_vel)

        pos += (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos) / 6.0
        vel += (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel) / 6.0

        # 3. 境界条件の判定
        r_current = np.linalg.norm(pos)

        # 領域外への脱出
        if r_current > R_MAX:
            return {'status': 'escaped', 'final_state': None}

        # 表面への衝突
        if r_current <= RM:
            impact_lon = np.arctan2(pos_prev[1], pos_prev[0])
            impact_lat = np.arcsin(np.clip(pos_prev[2] / np.linalg.norm(pos_prev), -1.0, 1.0))
            temp_at_impact = calculate_surface_temperature(impact_lon, impact_lat, AU, subsolar_lon_rad)

            # 吸着判定
            if np.random.random() < calculate_sticking_probability(temp_at_impact):
                lon_idx, lat_idx = xyz_to_lonlat_idx(pos_prev[0], pos_prev[1], pos_prev[2], settings['N_LON'],
                                                     settings['N_LAT'])
                final_state = {'pos': pos_prev, 'weight': weight, 'lon_idx': lon_idx, 'lat_idx': lat_idx}
                return {'status': 'stuck', 'final_state': final_state}

            # 再放出処理
            else:
                E_in = 0.5 * PHYSICAL_CONSTANTS['MASS_NA'] * np.sum(vel ** 2)
                E_T = PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_at_impact
                E_out = settings['BETA'] * E_T + (1.0 - settings['BETA']) * E_in
                v_out_speed = np.sqrt(E_out / (0.5 * PHYSICAL_CONSTANTS['MASS_NA'])) if E_out > 0 else 0.0

                impact_normal = pos_prev / np.linalg.norm(pos_prev)
                # ★★★ 再放出モデルを等方性に変更 ★★★
                rebound_direction = sample_isotropic_direction(impact_normal)
                vel = v_out_speed * rebound_direction

                # 粒子位置を表面に補正
                pos = RM * impact_normal
                continue

    # durationを生き延びた場合
    final_state = {'pos': pos, 'vel': vel, 'weight': weight}
    return {'status': 'alive', 'final_state': final_state}


# --- メイン制御関数 ---
def main_snapshot_simulation():
    """
    時間発展シミュレーションを実行し、定期的に系全体の粒子スナップショットを保存する。
    """
    start_time = time.time()

    # --- シミュレーション設定 ---
    OUTPUT_DIRECTORY = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult3D_test"
    N_LON, N_LAT = 24, 24  # 表面グリッドの分割数
    INITIAL_CNA_PER_M2 = 1.5e17  # 初期状態での表面ナトリウム原子の面密度 [atoms/m^2]
    TIME_STEP_SEC = 3600  # メインループの時間ステップ [s]
    TOTAL_SIM_YEARS = 1.0  # 総シミュレーション時間 [水星年]
    SUPERPARTICLES_PER_CELL = 1000  # 1グリッドセルから1ステップで放出されるスーパーパーティクル数
    TARGET_TAA_DEGREES = np.arange(0, 360, 1)  # スナップショットを保存するTAA [度]

    settings = {
        'BETA': 0.5,  # 表面衝突時の熱 accomodation 係数
        'T1AU': 168918.0,  # 1AUでの電離寿命 [s] (理論値)
        'DT': 10.0,  # 粒子追跡の時間刻み [s]
        'N_LON': N_LON, 'N_LAT': N_LAT,
        'GRID_RADIUS_RM': 5.0  # シミュレーション空間の半径 (水星半径単位)
    }

    run_name = f"snapshot_SPperCell{SUPERPARTICLES_PER_CELL}_Q2.0"
    target_output_dir = os.path.join(OUTPUT_DIRECTORY, run_name)
    os.makedirs(target_output_dir, exist_ok=True)
    print(f"結果は '{target_output_dir}' に保存されます。")

    # --- 時間とグリッドの準備 ---
    MERCURY_YEAR_SEC = 87.97 * 24 * 3600
    TOTAL_SIM_SEC = TOTAL_SIM_YEARS * MERCURY_YEAR_SEC
    time_steps = np.arange(0, TOTAL_SIM_SEC, TIME_STEP_SEC)

    lon_edges = np.linspace(-np.pi, np.pi, N_LON + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    cell_areas_m2 = (PHYSICAL_CONSTANTS['RM'] ** 2) * (lon_edges[1] - lon_edges[0]) * (
            np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))
    surface_grid_density = np.full((N_LON, N_LAT), INITIAL_CNA_PER_M2)

    # --- 外部データの読み込み ---
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

    sigma_const = PHYSICAL_CONSTANTS['E_CHARGE'] ** 2 / (
                4 * PHYSICAL_CONSTANTS['ME'] * PHYSICAL_CONSTANTS['C'] * PHYSICAL_CONSTANTS['EPSILON_0'])
    spec_data_dict = {'wl': wl, 'gamma': gamma, 'sigma0_perdnu2': sigma_const * 0.641,
                      'sigma0_perdnu1': sigma_const * 0.320, 'JL': 5.18e14}

    # --- メインループ ---
    active_particles = []
    previous_taa = -1
    target_taa_idx = 0

    with tqdm(total=len(time_steps), desc="Time Evolution") as pbar:
        for t_sec in time_steps:
            # 現在時刻の軌道パラメータを取得
            TAA, AU, Vms_ms, subsolar_lon_rad = get_orbital_params(t_sec, orbit_data, MERCURY_YEAR_SEC)
            pbar.set_description(f"TAA={TAA:.1f} | N_particles={len(active_particles)}")

            # --- 1. 新しい粒子を表面から生成 ---
            desorption_map = np.zeros_like(surface_grid_density)
            newly_launched_particles = []

            for i_lon in range(N_LON):
                for i_lat in range(N_LAT):
                    # 各グリッドセルからの光脱離量を計算
                    lon_center_rad = (lon_edges[i_lon] + lon_edges[i_lon + 1]) / 2
                    lat_center_rad = (lat_edges[i_lat] + lat_edges[i_lat + 1]) / 2
                    cos_Z = np.cos(lat_center_rad) * np.cos(lon_center_rad - subsolar_lon_rad)
                    if cos_Z <= 0: continue

                    F_UV_current_per_m2 = (1.5e14 * 1e4) / (AU ** 2)  # 1AUでの紫外線光子フラックス [photons/m^2/s]
                    Q_PSD_m2 = 3.0e-20 / 1e4  # 光脱離断面積 [m^2]
                    # Q_PSD_m2 = 1.0e-20 / 1e4  # 光脱離断面積 [m^2]
                    # Q_PSD_m2 = 2.0e-20 / 1e4 # suzukiが使ってたやつ
                    # Q_PSD_m2 = 3.0e-20 / 1e4  # YakshinskiyとMadey（1999）
                    # Q_PSD_m2 = 1.4e-21 / 1e4 # Killenら（2004）
                    # cNa = 0.053 * 7.5e14 * 1e4 # Moroni (2023) 水星表面のナトリウム原子の割合
                    current_density = surface_grid_density[i_lon, i_lat]
                    desorption_rate_per_m2 = F_UV_current_per_m2 * Q_PSD_m2 * cos_Z * current_density
                    n_atoms_to_desorb = desorption_rate_per_m2 * cell_areas_m2[i_lat] * TIME_STEP_SEC
                    if n_atoms_to_desorb <= 0: continue
                    desorption_map[i_lon, i_lat] = n_atoms_to_desorb

                    # 指定された数のスーパーパーティクルを生成
                    weight_for_this_cell = n_atoms_to_desorb / SUPERPARTICLES_PER_CELL
                    if weight_for_this_cell <= 0: continue

                    for _ in range(SUPERPARTICLES_PER_CELL):
                        # セル内でランダムな位置から放出
                        random_lon_rad = np.random.uniform(lon_edges[i_lon], lon_edges[i_lon + 1])
                        sin_lat_min, sin_lat_max = np.sin(lat_edges[i_lat]), np.sin(lat_edges[i_lat + 1])
                        random_lat_rad = np.arcsin(np.random.uniform(sin_lat_min, sin_lat_max))

                        initial_pos = lonlat_to_xyz(random_lon_rad, random_lat_rad, PHYSICAL_CONSTANTS['RM'])
                        normal_vec = initial_pos / np.linalg.norm(initial_pos)
                        speed = sample_maxwellian_speed(PHYSICAL_CONSTANTS['MASS_NA'], 1500.0)
                        initial_vel = speed * sample_cosine_direction(normal_vec)
                        newly_launched_particles.append(
                            {'pos': initial_pos, 'vel': initial_vel, 'weight': weight_for_this_cell})

            # 地表密度を更新 (脱離した分を減算)
            surface_grid_density -= desorption_map / cell_areas_m2[np.newaxis, :]
            surface_grid_density[surface_grid_density < 0] = 0

            # 生成した粒子を既存の粒子リストに追加
            active_particles.extend(newly_launched_particles)

            # --- 2. 既存の粒子を1ステップ進める (並列処理) ---
            tasks = [{'settings': settings, 'spec': spec_data_dict,
                      'particle_state': p, 'orbit': (TAA, AU, Vms_ms, subsolar_lon_rad),
                      'duration': TIME_STEP_SEC} for p in active_particles]

            next_active_particles = []
            if tasks:
                with Pool(processes=cpu_count()) as pool:
                    results = list(pool.imap(simulate_particle_for_one_step, tasks, chunksize=100))

                # 結果を集計
                for res in results:
                    if res['status'] == 'alive':
                        next_active_particles.append(res['final_state'])
                    elif res['status'] == 'stuck':
                        # 表面に吸着した粒子は、その重みを地表密度に戻す
                        state = res['final_state']
                        surface_grid_density[state['lon_idx'], state['lat_idx']] += state['weight'] / cell_areas_m2[
                            state['lat_idx']]

            active_particles = next_active_particles

            # --- 3. スナップショットを保存 ---
            if TAA < previous_taa: target_taa_idx = 0  # TAAが一周したらリセット
            save_this_step = False
            if target_taa_idx < len(TARGET_TAA_DEGREES):
                current_target_taa = TARGET_TAA_DEGREES[target_taa_idx]
                if previous_taa < current_target_taa <= TAA:
                    save_this_step = True
                    target_taa_idx += 1

            if save_this_step:
                pbar.write(f"\n>>> Saving snapshot at TAA={TAA:.1f} ({len(active_particles)} particles) <<<")
                snapshot_data = np.zeros((len(active_particles), 7))  # x,y,z, vx,vy,vz, weight
                for i, p in enumerate(active_particles):
                    snapshot_data[i, 0:3] = p['pos']
                    snapshot_data[i, 3:6] = p['vel']
                    snapshot_data[i, 6] = p['weight']

                save_time_h = t_sec / 3600
                filename = f"snapshot_t{int(save_time_h):05d}_taa{int(TAA):03d}.npy"
                np.save(os.path.join(target_output_dir, filename), snapshot_data)
                np.save(
                    os.path.join(target_output_dir, f"surface_density_t{int(save_time_h):05d}_taa{int(TAA):03d}.npy"),
                    surface_grid_density)

            previous_taa = TAA
            pbar.update(1)

    end_time = time.time()
    print(f"\n★★★ シミュレーションが完了しました ★★★")
    print(f"総計算時間: {(end_time - start_time) / 3600:.2f} 時間")


if __name__ == '__main__':
    for f in ['orbit360.txt', 'SolarSpectrum_Na0.txt']:
        if not os.path.exists(f):
            print(f"エラー: 必須ファイル '{f}' が見つかりません。")
            sys.exit()
    main_snapshot_simulation()