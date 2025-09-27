import numpy as np
import sys
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- 物理定数 ---
PHYSICAL_CONSTANTS = {
    'C': 299792458.0,  # 光速 [m/s]
    'PI': np.pi,
    'H': 6.62607015e-34,  # プランク定数 [J・s]
    'MASS_NA': 22.98976928 * 1.66054e-27,  # Na原子の質量 [kg]
    'RM': 2439.7e3,  # 水星の半径 [m]
    'GM_MERCURY': 2.2032e13,  # 万有引力定数 * 水星の質量 [m^3/s^2]
    'K_BOLTZMANN': 1.380649e-23,  # ボルツマン定数 [J/K]
    'E_CHARGE': 1.602176634e-19,  # 素電荷 [C]
    'ME': 9.1093897e-31,  # 電子の質量 [kg]
    'EPSILON_0': 8.854187817e-12  # 真空の誘電率 [F/m]
}


# --- 物理モデルに基づく関数 (変更なし) ---

def calculate_surface_temperature(x, y, z, AU):
    T0 = 100.0
    T1 = 600.0
    if x <= 0:
        return T0
    cos_theta = x / np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if cos_theta < 0:
        return T0
    temp = T0 + T1 * (cos_theta ** 0.25) * ((0.306 / AU) ** 2)
    return temp


def calculate_sticking_probability(surface_temp_K):
    A = 0.08
    B = 458.0
    porosity = 0.8
    p_stick = A * np.exp(B / surface_temp_K)
    p_stick_eff = p_stick / (1.0 - (1.0 - p_stick) * porosity)
    return p_stick_eff


# --- サンプリング関数 (変更なし) ---

def sample_maxwellian_speed(mass_kg, temp_k):
    sigma = np.sqrt(PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_k / mass_kg)
    vx = np.random.normal(0, sigma)
    vy = np.random.normal(0, sigma)
    vz = np.random.normal(0, sigma)
    speed = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    return speed


def sample_weibull_speed(mass_kg, U_ev=0.05, beta_shape=0.7):
    p = np.random.random()
    E_ev = U_ev * (p ** (-1.0 / (beta_shape + 1.0)) - 1.0)
    E_joule = E_ev * PHYSICAL_CONSTANTS['E_CHARGE']
    v_ms = np.sqrt(2 * E_joule / mass_kg)
    return v_ms


def sample_cosine_direction(normal_vector):
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    if np.abs(normal_vector[0]) > np.abs(normal_vector[1]):
        inv_len = 1.0 / np.sqrt(normal_vector[0] ** 2 + normal_vector[2] ** 2)
        t1 = np.array([-normal_vector[2] * inv_len, 0, normal_vector[0] * inv_len])
    else:
        inv_len = 1.0 / np.sqrt(normal_vector[1] ** 2 + normal_vector[2] ** 2)
        t1 = np.array([0, normal_vector[2] * inv_len, -normal_vector[1] * inv_len])
    t2 = np.cross(normal_vector, t1)
    p1, p2 = np.random.random(), np.random.random()
    sin_theta_sq = p1
    cos_theta = np.sqrt(1.0 - sin_theta_sq)
    sin_theta = np.sqrt(sin_theta_sq)
    phi = 2 * np.pi * p2
    direction = t1 * sin_theta * np.cos(phi) + t2 * sin_theta * np.sin(phi) + normal_vector * cos_theta
    return direction


def sample_isotropic_direction(normal_vector):
    vec = np.random.randn(3)
    vec /= np.linalg.norm(vec)
    if np.dot(vec, normal_vector) < 0:
        vec = -vec
    return vec


def _calculate_acceleration(pos, vel, Vms_ms, AU, spec_data, settings):
    x, y, z = pos
    vx, vy, vz = vel
    velocity_for_doppler = vx + Vms_ms
    w_na_d2 = 589.1582e-9 * (1.0 - velocity_for_doppler / PHYSICAL_CONSTANTS['C'])
    w_na_d1 = 589.7558e-9 * (1.0 - velocity_for_doppler / PHYSICAL_CONSTANTS['C'])
    wl, gamma, sigma0_perdnu1, sigma0_perdnu2, JL = spec_data.values()
    if not (wl[0] * 1e-9 <= w_na_d2 < wl[-1] * 1e-9 and wl[0] * 1e-9 <= w_na_d1 < wl[-1] * 1e-9):
        b = 0.0
    else:
        gamma2 = np.interp(w_na_d2 * 1e9, wl, gamma)
        gamma1 = np.interp(w_na_d1 * 1e9, wl, gamma)
        F_lambda_1AU_m = JL * 1e9
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
    accel_g = np.array([0.0, 0.0, 0.0])
    if settings['GRAVITY_ENABLED']:
        r_sq_grav = x ** 2 + y ** 2 + z ** 2
        if r_sq_grav > 0:
            r_grav = np.sqrt(r_sq_grav)
            grav_accel_total = -PHYSICAL_CONSTANTS['GM_MERCURY'] / r_sq_grav
            accel_g = grav_accel_total * (pos / r_grav)
    return accel_srp + accel_g


# --- シミュレーションの本体 ---
def simulate_single_particle_for_density(args):
    """
    一個のナトリウム原子の挙動を追跡し、グリッドへの滞在時間と最終的な消滅要因を返す。
    """
    # --- 引数の展開 ---
    settings, spec_data = args['settings'], args['spec']
    TAA, AU, lon, lat, Vms_ms = args['orbit']
    grid_params = args['grid_params']

    # --- 設定の展開 ---
    DT, IONIZATION_MODEL = settings['DT'], settings['ionization_model']
    RM = PHYSICAL_CONSTANTS['RM']
    MASS_NA = PHYSICAL_CONSTANTS['MASS_NA']
    K_BOLTZMANN = PHYSICAL_CONSTANTS['K_BOLTZMANN']

    # --- グリッドパラメータ ---
    N_R, N_THETA, N_PHI = grid_params['n_r'], grid_params['n_theta'], grid_params['n_phi']
    R_MAX = grid_params['max_r']
    DR = R_MAX / N_R
    D_THETA = PHYSICAL_CONSTANTS['PI'] / N_THETA
    D_PHI = 2 * PHYSICAL_CONSTANTS['PI'] / N_PHI
    local_density_grid = np.zeros((N_R, N_THETA, N_PHI), dtype=np.float32)

    # ▼▼▼ 変更・追加箇所 ▼▼▼
    # --- 粒子の初期化 ---
    # 1. 放出位置の決定 (棄却サンプリング法)
    #    設定に応じて、指定された太陽天頂角(cosZ)の範囲に粒子を生成します。
    max_iter = 1000  # 無限ループ防止
    for _ in range(max_iter):
        # 全球表面で一様な点をランダムに生成
        u, v = np.random.random(), np.random.random()
        phi_surf = 2 * np.pi * u
        cos_theta_surf = 2 * v - 1
        sin_theta_surf = np.sqrt(1 - cos_theta_surf ** 2)

        # 座標を計算 (シミュレーション座標系: xが太陽方向)
        px = sin_theta_surf * np.cos(phi_surf)
        py = sin_theta_surf * np.sin(phi_surf)
        pz = cos_theta_surf

        # 太陽天頂角のコサインを計算 (cosZ = x/R)
        cos_Z = px

        # 設定に基づいて放出位置をフィルタリング
        loc_model = settings.get('ejection_location_model', 'dayside_uniform')
        if loc_model == 'dayside_uniform':
            if cos_Z > 0:  # 昼側なら採用
                break
        elif loc_model == 'terminator':
            min_cos_z, max_cos_z = settings['ejection_cos_z_range']
            if min_cos_z <= cos_Z <= max_cos_z:  # 指定範囲内なら採用
                break
        else:
            raise ValueError(f"Unknown ejection location model: {loc_model}")
    else:
        # サンプリングに失敗した場合のエラーハンドリング
        raise RuntimeError(f"Failed to sample ejection position in {max_iter} iterations for model '{loc_model}'.")

    # 採用された位置ベクトルを決定
    pos = RM * np.array([px, py, pz])

    # 2. 角度依存の重みを計算
    #    PSDの放出率は太陽光の入射角(cosZ)に比例するため、この重みをかけます。
    weight_cos_z = cos_Z

    # 3. 初期速度の決定
    #    設定に応じて、分布(maxwellian/weibull)または固定値(constant)から速さを決定します。
    speed_model = settings['ejection_speed_model']
    if speed_model == 'maxwellian':
        ejection_speed = sample_maxwellian_speed(MASS_NA, 1500.0)
    elif speed_model == 'weibull':
        ejection_speed = sample_weibull_speed(MASS_NA)
    elif speed_model == 'constant':
        ejection_speed = settings['constant_speed_ms']
    else:
        raise ValueError(f"Unknown speed model: {speed_model}")

    # 4. 放出方向を決定し、速度ベクトルを計算
    surface_normal = pos / np.linalg.norm(pos)
    if settings['ejection_direction_model'] == 'isotropic':
        ejection_direction = sample_isotropic_direction(surface_normal)
    else:  # 'cosine'
        ejection_direction = sample_cosine_direction(surface_normal)
    vel = ejection_speed * ejection_direction
    # ▲▲▲ 変更・追加ここまで ▲▲▲

    # --- 時間発展ループ (以降は変更なし) ---
    tau = settings['T1AU'] * AU ** 2
    itmax = int(tau * 5.0 / DT + 0.5)
    death_reason = 'ionized'

    for it in range(itmax):
        ionization_prob_per_step = 1.0 - np.exp(-DT / tau)
        if IONIZATION_MODEL == 'particle_death':
            if np.random.random() < ionization_prob_per_step:
                death_reason = 'ionized'
                break

        Nad = 1.0
        if IONIZATION_MODEL == 'weight_decay':
            Nad = np.exp(-DT * it / tau)

        k1_vel = DT * _calculate_acceleration(pos, vel, Vms_ms, AU, spec_data, settings)
        k1_pos = DT * vel
        k2_vel = DT * _calculate_acceleration(pos + 0.5 * k1_pos, vel + 0.5 * k1_vel, Vms_ms, AU, spec_data, settings)
        k2_pos = DT * (vel + 0.5 * k1_vel)
        k3_vel = DT * _calculate_acceleration(pos + 0.5 * k2_pos, vel + 0.5 * k2_vel, Vms_ms, AU, spec_data, settings)
        k3_pos = DT * (vel + 0.5 * k2_vel)
        k4_vel = DT * _calculate_acceleration(pos + k3_pos, vel + k3_vel, Vms_ms, AU, spec_data, settings)
        k4_pos = DT * (vel + k3_vel)

        pos_prev = pos.copy()
        pos += (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos) / 6.0
        vel += (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel) / 6.0

        r = np.linalg.norm(pos)
        if r == 0: continue
        if r >= R_MAX:
            death_reason = 'escaped'
            break

        theta = np.arccos(pos[2] / r)
        phi = np.arctan2(pos[1], pos[0])
        ir = int(r / DR)
        itheta = int(theta / D_THETA)
        iphi = int((phi + PHYSICAL_CONSTANTS['PI']) / D_PHI)

        if 0 <= ir < N_R and 0 <= itheta < N_THETA and 0 <= iphi < N_PHI:
            final_weight = weight_cos_z * Nad if IONIZATION_MODEL == 'weight_decay' else weight_cos_z
            #local_density_grid[ir, itheta, iphi] += final_weight * DT
            local_density_grid[ir, itheta, iphi] += 1 * DT

        r_current = np.linalg.norm(pos)
        if r_current <= RM:
            temp_at_impact = calculate_surface_temperature(pos_prev[0], pos_prev[1], pos_prev[2], AU)
            stick_prob = calculate_sticking_probability(temp_at_impact)

            if np.random.random() < stick_prob:
                death_reason = 'stuck'
                break

            E_in = 0.5 * MASS_NA * np.sum(vel ** 2)
            E_T = K_BOLTZMANN * temp_at_impact
            E_out = settings['BETA'] * E_T + (1.0 - settings['BETA']) * E_in
            v_out_sq = E_out / (0.5 * MASS_NA)
            v_out_speed = np.sqrt(v_out_sq) if v_out_sq > 0 else 0.0

            impact_normal = pos_prev / np.linalg.norm(pos_prev)
            rebound_direction = sample_isotropic_direction(impact_normal)
            vel = v_out_speed * rebound_direction

            pos = RM * impact_normal

    return (local_density_grid, death_reason)


def main():
    """
    水星ナトリウム大気のモンテカルロシミュレーションを実行するメイン関数。
    """
    # --- シミュレーション設定 ---
    OUTPUT_DIRECTORY = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult3D"
    N_R = 100
    N_THETA = 24
    N_PHI = 24
    GRID_RADIUS_RM = 5.0
    N_PARTICLES = 10000

    # ▼▼▼ 変更・追加箇所 ▼▼▼
    # シミュレーションの挙動を制御する設定
    settings = {
        'GRAVITY_ENABLED': True,
        'BETA': 0.0,
        'T1AU': 1.9 * 1e5,
        'DT': 10.0,
        'ejection_direction_model': 'isotropic',  # 'cosine' or 'isotropic'
        'ionization_model': 'particle_death',  # 'weight_decay' or 'particle_death'

        # --- 初速モデル ---
        # 'maxwellian': 温度(1500K)に応じたマクスウェル分布
        # 'weibull'   : ワイブル分布
        # 'constant'  : 以下で指定する固定値
        'ejection_speed_model': 'constant',
        'constant_speed_ms': 0.0,  # 'constant'モデル使用時の初速 [m/s]

        # --- 放出位置モデル ---
        # 'dayside_uniform': 昼側表面から均一に放出
        # 'terminator'     : 昼夜境界線付近から放出
        'ejection_location_model': 'terminator',
        # 'terminator'モデル使用時、放出する太陽天頂角のコサイン(cosZ)の範囲 [最小値, 最大値]
        # (cosZ=0 が境界線、cosZ=1 が太陽直下点)
        'ejection_cos_z_range': [0.0, 0.1],
    }
    # ▲▲▲ 変更・追加ここまで ▲▲▲

    # --- ファイル名とディレクトリ設定 ---
    dist_tag = "CO" if settings['ejection_direction_model'] == 'cosine' else "ISO"
    # ▼▼▼ 変更・追加箇所 ▼▼▼
    # 新しい設定をファイル名に反映
    speed_tag = settings['ejection_speed_model'][:2].upper()
    loc_tag = settings['ejection_location_model']
    ion_tag = "WD" if settings['ionization_model'] == 'weight_decay' else "PD"
    base_name_template = f"density3d_test"
    # ▲▲▲ 変更・追加ここまで ▲▲▲

    sub_folder_name = base_name_template
    target_output_dir = os.path.join(OUTPUT_DIRECTORY, sub_folder_name)
    os.makedirs(target_output_dir, exist_ok=True)
    print(f"結果は '{target_output_dir}' に保存されます。")

    log_file_path = os.path.join(target_output_dir, "death_statistics.csv")
    with open(log_file_path, 'w', newline='') as f:
        f.write("TAA,Ionized_Count,Ionized_Percent,Stuck_Count,Stuck_Percent,Escaped_Count,Escaped_Percent\n")
    print(f"死因統計は {log_file_path} に記録されます。")

    # --- グリッドパラメータの準備 ---
    grid_params = {
        'n_r': N_R, 'n_theta': N_THETA, 'n_phi': N_PHI,
        'max_r': PHYSICAL_CONSTANTS['RM'] * GRID_RADIUS_RM
    }

    # --- 外部データの読み込み ---
    try:
        spec_data_np = np.loadtxt('SolarSpectrum_Na0.txt', usecols=(0, 3))
        wl, gamma = spec_data_np[:, 0], spec_data_np[:, 1]
        orbit_lines = open('orbit360.txt', 'r').readlines()
    except FileNotFoundError as e:
        print(f"エラー: データファイルが見つかりません - {e}", file=sys.stderr)
        sys.exit(1)

    if not np.all(np.diff(wl) > 0):
        sort_indices = np.argsort(wl)
        wl, gamma = wl[sort_indices], gamma[sort_indices]

    sigma_const = PHYSICAL_CONSTANTS['E_CHARGE'] ** 2 / (
            4 * PHYSICAL_CONSTANTS['ME'] * PHYSICAL_CONSTANTS['C'] * PHYSICAL_CONSTANTS['EPSILON_0'])
    spec_data_dict = {
        'wl': wl, 'gamma': gamma,
        'sigma0_perdnu1': sigma_const * 0.320,
        'sigma0_perdnu2': sigma_const * 0.641,
        'JL': 5.18e14 * 1e4
    }

    # --- TAAごとのシミュレーションループ (以降は変更なし) ---
    for line in orbit_lines:
        TAA, AU, lon, lat, Vms_ms = map(float, line.split())
        print(f"\n--- TAA = {TAA:.1f}度のシミュレーションを開始 ---")

        F_UV_at_1AU_per_m2 = 1.5e14 * 1e4
        Q_PSD_m2 = 2.0e-20 / 1e4
        cNa_per_m2 = 1.5e13 * 1e4

        F_UV_current_per_m2 = F_UV_at_1AU_per_m2 / (AU ** 2)
        R_PSD_peak_per_m2 = F_UV_current_per_m2 * Q_PSD_m2 * cNa_per_m2

        effective_area_m2 = np.pi * (PHYSICAL_CONSTANTS['RM'] ** 2)
        total_flux_for_this_taa = R_PSD_peak_per_m2 * effective_area_m2

        task_args = {
            'settings': settings, 'spec': spec_data_dict,
            'orbit': (TAA, AU, lon, lat, Vms_ms), 'grid_params': grid_params
        }
        tasks = [task_args] * N_PARTICLES

        with Pool(processes=cpu_count()) as pool:
            results = list(tqdm(pool.imap(simulate_single_particle_for_density, tasks), total=N_PARTICLES,
                                desc=f"TAA={TAA:.1f}"))

        print("結果を集計・保存中...")
        master_density_grid = np.zeros((N_R, N_THETA, N_PHI), dtype=np.float32)
        death_counts = {'ionized': 0, 'stuck': 0, 'escaped': 0}

        for grid, reason in results:
            master_density_grid += grid
            if reason in death_counts:
                death_counts[reason] += 1

        print("\n" + "=" * 40)
        print(f" TAA = {TAA:.1f} における粒子の消滅要因")
        print("-" * 40)
        total_simulated = sum(death_counts.values())
        if total_simulated > 0:
            ionized_percent = (death_counts['ionized'] / total_simulated) * 100
            stuck_percent = (death_counts['stuck'] / total_simulated) * 100
            escaped_percent = (death_counts['escaped'] / total_simulated) * 100
            print(f"  寿命 (電離): {death_counts['ionized']:>6d} particles ({ionized_percent:6.2f} %)")
            print(f"  表面に吸着: {death_counts['stuck']:>6d} particles ({stuck_percent:6.2f} %)")
            print(f"  系外へ脱出: {death_counts['escaped']:>6d} particles ({escaped_percent:6.2f} %)")
        print("=" * 40)

        with open(log_file_path, 'a', newline='') as f:
            log_line = (f"{TAA:.1f},{death_counts['ionized']},{ionized_percent:.2f},"
                        f"{death_counts['stuck']},{stuck_percent:.2f},"
                        f"{death_counts['escaped']},{escaped_percent:.2f}\n")
            f.write(log_line)
        print(f"死因を {log_file_path} に追記しました。")

        r_edges = np.linspace(0, grid_params['max_r'], N_R + 1)
        theta_edges = np.linspace(0, PHYSICAL_CONSTANTS['PI'], N_THETA + 1)
        delta_r3_3 = (r_edges[1:] ** 3 - r_edges[:-1] ** 3) / 3.0
        delta_cos_theta = np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:])
        D_PHI_rad = 2 * PHYSICAL_CONSTANTS['PI'] / N_PHI
        cell_volumes_m3 = delta_r3_3[:, np.newaxis, np.newaxis] * \
                          delta_cos_theta[np.newaxis, :, np.newaxis] * \
                          D_PHI_rad
        cell_volumes_m3[cell_volumes_m3 == 0] = 1e-30

        number_density_m3 = (total_flux_for_this_taa / N_PARTICLES) * (master_density_grid / cell_volumes_m3)
        number_density_cm3 = number_density_m3 / 1e6

        base_filename = f"density3d_taa{TAA:.0f}_{base_name_template.replace('density3d_', '')}"
        full_path_npy = os.path.join(target_output_dir, f"{base_filename}.npy")
        np.save(full_path_npy, number_density_cm3)
        print(f"結果を {full_path_npy} に保存しました。")

    print("\nすべてのシミュレーションが完了しました。")


if __name__ == '__main__':
    main()