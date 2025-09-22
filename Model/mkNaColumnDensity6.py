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


# --- 時間発展モデル用のヘルパー関数 (変更なし) ---

def preprocess_orbit_data(filename, mercury_year_sec):
    data = np.loadtxt(filename)
    num_points = len(data)
    time_axis = np.linspace(0, mercury_year_sec, num_points, endpoint=False)
    return np.column_stack((time_axis, data))


def get_orbital_params(time_sec, orbit_data, mercury_year_sec):
    ROTATION_PERIOD_SEC = 58.646 * 24 * 3600
    current_time_in_orbit = time_sec % mercury_year_sec
    taa = np.interp(current_time_in_orbit, orbit_data[:, 0], orbit_data[:, 1])
    au = np.interp(current_time_in_orbit, orbit_data[:, 0], orbit_data[:, 2])
    vms = np.interp(current_time_in_orbit, orbit_data[:, 0], orbit_data[:, 5])
    subsolar_lon_rad = (2 * np.pi * time_sec / ROTATION_PERIOD_SEC) % (2 * np.pi)
    return taa, au, vms, subsolar_lon_rad


def lonlat_to_xyz(lon_rad, lat_rad, radius):
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return np.array([x, y, z])


def xyz_to_lonlat_idx(x, y, z, N_LON, N_LAT):
    radius = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    lon_rad = np.arctan2(y, x)
    lat_rad = np.arcsin(np.clip(z / radius, -1.0, 1.0))
    lon_idx = int((lon_rad + np.pi) / (2 * np.pi) * N_LON) % N_LON
    lat_idx = int((lat_rad + np.pi / 2) / np.pi * N_LAT)
    lat_idx = np.clip(lat_idx, 0, N_LAT - 1)
    return lon_idx, lat_idx


# --- コア追跡関数 (変更なし) ---
def simulate_particle_for_one_step(args):
    """
    一個の粒子を、指定された時間 (duration) だけ追跡し、最終状態を返す。
    """
    constants = args['constants']
    settings = args['settings']
    spec_data = args['spec']
    particle_state = args['particle_state']
    TAA, AU, Vms_ms, subsolar_lon_rad = args['orbit']
    duration = args['duration']

    C, H, MASS_NA, RM, GM_MERCURY, K_BOLTZMANN = constants['C'], constants['H'], constants['MASS_NA'], constants['RM'], \
    constants['GM_MERCURY'], constants['K_BOLTZMANN']
    BETA, T1AU, DT, N_LON, N_LAT = settings['BETA'], settings['T1AU'], settings['DT'], settings['N_LON'], settings[
        'N_LAT']
    R_MAX = constants['RM'] * settings['GRID_RADIUS_RM']
    wl, gamma, sigma0_perdnu2, sigma0_perdnu1, JL = spec_data.values()

    pos, vel, weight = particle_state['pos'], particle_state['vel'], particle_state['weight']
    x, y, z = pos[0], pos[1], pos[2]
    vx_ms, vy_ms, vz_ms = vel[0], vel[1], vel[2]

    tau = T1AU * AU ** 2
    itmax = int(duration / DT)

    for _ in range(itmax):
        x_prev, y_prev, z_prev = x, y, z
        vx_ms_prev, vy_ms_prev, vz_ms_prev = vx_ms, vy_ms, vz_ms

        if x > 0 and np.random.random() < (1.0 - np.exp(-DT / tau)):
            return {'status': 'ionized', 'final_state': None}

        velocity_for_doppler = vx_ms - Vms_ms
        w_na_d2 = 589.1582e-9 * (1.0 + velocity_for_doppler / C)
        w_na_d1 = 589.7558e-9 * (1.0 + velocity_for_doppler / C)
        b = 0.0
        if (wl[0] * 1e-9 <= w_na_d2 < wl[-1] * 1e-9) and (wl[0] * 1e-9 <= w_na_d1 < wl[-1] * 1e-9):
            gamma2 = np.interp(w_na_d2 * 1e9, wl, gamma)
            gamma1 = np.interp(w_na_d1 * 1e9, wl, gamma)
            F_lambda_1AU_m = JL * 1e4 * 1e9
            F_lambda_d1_at_Mercury = F_lambda_1AU_m / (AU ** 2) * gamma1
            F_nu_d1 = F_lambda_d1_at_Mercury * (w_na_d1 ** 2) / C
            J1 = sigma0_perdnu1 * F_nu_d1
            F_lambda_d2_at_Mercury = F_lambda_1AU_m / (AU ** 2) * gamma2
            F_nu_d2 = F_lambda_d2_at_Mercury * (w_na_d2 ** 2) / C
            J2 = sigma0_perdnu2 * F_nu_d2
            b = 1 / MASS_NA * ((H / w_na_d1) * J1 + (H / w_na_d2) * J2)
        if x < 0 and np.sqrt(y ** 2 + z ** 2) < RM: b = 0.0

        r_sq = x ** 2 + y ** 2 + z ** 2
        if r_sq == 0: continue
        r = np.sqrt(r_sq)
        grav_accel_total = -GM_MERCURY / r_sq
        accel_gx, accel_gy, accel_gz = grav_accel_total * (x / r), grav_accel_total * (y / r), grav_accel_total * (
                    z / r)
        total_accel_x, total_accel_y, total_accel_z = accel_gx - b, accel_gy, accel_gz

        vx_ms += total_accel_x * DT
        vy_ms += total_accel_y * DT
        vz_ms += total_accel_z * DT
        x += ((vx_ms_prev + vx_ms) / 2.0) * DT
        y += ((vy_ms_prev + vy_ms) / 2.0) * DT
        z += ((vz_ms_prev + vz_ms) / 2.0) * DT

        r_current = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        if r_current > R_MAX:
            return {'status': 'escaped', 'final_state': None}

        if r_current <= RM:
            impact_radius = np.sqrt(x_prev ** 2 + y_prev ** 2 + z_prev ** 2)
            impact_lon = np.arctan2(y_prev, x_prev)
            impact_lat = np.arcsin(np.clip(z_prev / impact_radius, -1.0, 1.0))
            temp_at_impact = calculate_surface_temperature(impact_lon, impact_lat, AU, subsolar_lon_rad)

            if np.random.random() < calculate_sticking_probability(temp_at_impact):
                final_pos = np.array([x_prev, y_prev, z_prev])
                lon_idx, lat_idx = xyz_to_lonlat_idx(final_pos[0], final_pos[1], final_pos[2], N_LON, N_LAT)
                final_state = {'pos': final_pos, 'weight': weight, 'lon_idx': lon_idx, 'lat_idx': lat_idx}
                return {'status': 'stuck', 'final_state': final_state}
            else:
                v_in_sq = vx_ms_prev ** 2 + vy_ms_prev ** 2 + vz_ms_prev ** 2
                E_in = 0.5 * MASS_NA * v_in_sq
                E_T = K_BOLTZMANN * temp_at_impact
                E_out = BETA * E_T + (1.0 - BETA) * E_in
                v_out_speed = np.sqrt(E_out / (0.5 * MASS_NA)) if E_out > 0 else 0.0
                impact_normal = np.array([x_prev, y_prev, z_prev])
                rebound_direction = sample_cosine_direction(impact_normal)
                vx_ms, vy_ms, vz_ms = v_out_speed * rebound_direction

                impact_pos_norm = np.linalg.norm(impact_normal)
                if impact_pos_norm > 0:
                    x, y, z = RM * impact_normal / impact_pos_norm
                continue

    final_state = {'pos': np.array([x, y, z]), 'vel': np.array([vx_ms, vy_ms, vz_ms]), 'weight': weight}
    return {'status': 'alive', 'final_state': final_state}


# --- ★★★ メイン制御関数 (粒子生成ロジック修正版) ★★★ ---
def main_snapshot_simulation():
    start_time = time.time()

    # --- シミュレーション設定 ---
    OUTPUT_DIRECTORY = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult3D_test"

    N_LON, N_LAT = 24, 24
    INITIAL_CNA_PER_M2 = 1.5e17
    TIME_STEP_SEC = 3600
    TOTAL_SIM_YEARS = 1.0

    SUPERPARTICLES_PER_CELL = 100

    TARGET_TAA_DEGREES = np.arange(0, 360, 1)

    settings = {
        'BETA': 0.5, 'T1AU': 61728.4, 'DT': 10.0,
        'N_LON': N_LON, 'N_LAT': N_LAT,
        'GRID_RADIUS_RM': 5.0
    }

    constants = {
        'C': 299792458.0, 'PI': np.pi, 'H': 6.62607015e-34,
        'MASS_NA': 22.98976928 * 1.66054e-27, 'RM': 2439.7e3,
        'GM_MERCURY': 2.2032e13, 'K_BOLTZMANN': 1.380649e-23
    }

    run_name = f"snapshot_SPperCell{SUPERPARTICLES_PER_CELL}_Q2.0"
    target_output_dir = os.path.join(OUTPUT_DIRECTORY, run_name)
    os.makedirs(target_output_dir, exist_ok=True)
    print(f"結果は '{target_output_dir}' に保存されます。")

    MERCURY_YEAR_SEC = 87.97 * 24 * 3600
    TOTAL_SIM_SEC = TOTAL_SIM_YEARS * MERCURY_YEAR_SEC
    time_steps = np.arange(0, TOTAL_SIM_SEC, TIME_STEP_SEC)

    lon_edges = np.linspace(-np.pi, np.pi, N_LON + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    cell_areas_m2 = (constants['RM'] ** 2) * (lon_edges[1] - lon_edges[0]) * (
                np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))
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
    spec_data_dict = {'wl': wl, 'gamma': gamma, 'sigma0_perdnu2': sigma_const * 0.641,
                      'sigma0_perdnu1': sigma_const * 0.320, 'JL': 5.18e14}

    # --- メインループ ---
    active_particles = []
    previous_taa = -1
    target_taa_idx = 0

    with tqdm(total=len(time_steps), desc="Time Evolution") as pbar:
        for step_idx, t_sec in enumerate(time_steps):
            TAA, AU, Vms_ms, subsolar_lon_rad = get_orbital_params(t_sec, orbit_data, MERCURY_YEAR_SEC)
            pbar.set_description(f"TAA={TAA:.1f} | N_particles={len(active_particles)}")

            desorption_map = np.zeros_like(surface_grid_density)
            newly_launched_particles = []

            for i_lon in range(N_LON):
                for i_lat in range(N_LAT):
                    lon_center_rad = (lon_edges[i_lon] + lon_edges[i_lon + 1]) / 2
                    lat_center_rad = (lat_edges[i_lat] + lat_edges[i_lat + 1]) / 2
                    cos_Z = np.cos(lat_center_rad) * np.cos(lon_center_rad - subsolar_lon_rad)
                    if cos_Z <= 0: continue

                    F_UV_current_per_m2 = (1.5e14 * 1e4) / (AU ** 2)
                    Q_PSD_m2 = 2.0e-20 / 1e4
                    current_density = surface_grid_density[i_lon, i_lat]
                    desorption_rate_per_m2 = F_UV_current_per_m2 * Q_PSD_m2 * cos_Z * current_density
                    n_atoms_to_desorb = desorption_rate_per_m2 * cell_areas_m2[i_lat] * TIME_STEP_SEC

                    if n_atoms_to_desorb <= 0:
                        continue

                    desorption_map[i_lon, i_lat] = n_atoms_to_desorb

                    weight_for_this_cell = n_atoms_to_desorb / SUPERPARTICLES_PER_CELL

                    for _ in range(SUPERPARTICLES_PER_CELL):
                        random_lon_rad = np.random.uniform(lon_edges[i_lon], lon_edges[i_lon + 1])
                        sin_lat_min, sin_lat_max = np.sin(lat_edges[i_lat]), np.sin(lat_edges[i_lat + 1])
                        random_lat_rad = np.arcsin(np.random.uniform(sin_lat_min, sin_lat_max))

                        initial_pos = lonlat_to_xyz(random_lon_rad, random_lat_rad, constants['RM'])
                        normal_vec = initial_pos / np.linalg.norm(initial_pos)
                        speed = sample_maxwellian_speed(constants['MASS_NA'], 1500.0)
                        initial_vel = speed * sample_cosine_direction(normal_vec)

                        # 計算したセルごとの重みを使用
                        newly_launched_particles.append(
                            {'pos': initial_pos, 'vel': initial_vel, 'weight': weight_for_this_cell})

            # 地表密度を更新
            surface_grid_density -= desorption_map / cell_areas_m2[np.newaxis, :]
            surface_grid_density[surface_grid_density < 0] = 0

            active_particles.extend(newly_launched_particles)

            # --- 2. 既存の粒子を1ステップ進める ---
            tasks = [{'constants': constants, 'settings': settings, 'spec': spec_data_dict,
                      'particle_state': p, 'orbit': (TAA, AU, Vms_ms, subsolar_lon_rad),
                      'duration': TIME_STEP_SEC} for p in active_particles]

            next_active_particles = []
            if tasks:
                with Pool(processes=cpu_count()) as pool:
                    results = list(pool.imap(simulate_particle_for_one_step, tasks, chunksize=100))

                for res in results:
                    if res['status'] == 'alive':
                        next_active_particles.append(res['final_state'])
                    elif res['status'] == 'stuck':
                        state = res['final_state']
                        surface_grid_density[state['lon_idx'], state['lat_idx']] += state['weight'] / cell_areas_m2[
                            state['lat_idx']]
            active_particles = next_active_particles

            # --- 3. スナップショットを保存 ---
            if TAA < previous_taa: target_taa_idx = 0
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
            print(f"エラー: 必須ファイル '{f}' が見つかりません。");
            sys.exit()
    main_snapshot_simulation()