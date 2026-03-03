import numpy as np
import sys
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv

# --- 物理定数 ---
# (変更なし、元のコードと同じ)
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


# --- 物理モデルに基づく関数 ---
# (変更なし、元のコードと同じ)
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


# --- サンプリング関数 ---
# (変更なし、元のコードと同じ)
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


# --- 加速度計算関数 ---
# (変更なし、元のコードと同じ)
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


# --- 元のシミュレーション本体 ---
# (変更なし、元のコードと同じ)
def simulate_single_particle_for_density(args):
    settings, spec_data = args['settings'], args['spec']
    TAA, AU, lon, lat, Vms_ms = args['orbit']
    grid_params = args['grid_params']
    DT, IONIZATION_MODEL = settings['DT'], settings['ionization_model']
    RM = PHYSICAL_CONSTANTS['RM']
    MASS_NA = PHYSICAL_CONSTANTS['MASS_NA']
    K_BOLTZMANN = PHYSICAL_CONSTANTS['K_BOLTZMANN']
    N_R, N_THETA, N_PHI = grid_params['n_r'], grid_params['n_theta'], grid_params['n_phi']
    R_MAX = grid_params['max_r']
    DR = R_MAX / N_R
    D_THETA = PHYSICAL_CONSTANTS['PI'] / N_THETA
    D_PHI = 2 * PHYSICAL_CONSTANTS['PI'] / N_PHI
    local_density_grid = np.zeros((N_R, N_THETA, N_PHI), dtype=np.float32)
    particle_weight = 1.0
    phi_source = PHYSICAL_CONSTANTS['PI'] * np.random.random() - (PHYSICAL_CONSTANTS['PI'] / 2.0)
    cos_theta_source = 2 * np.random.random() - 1.0
    sin_theta_source = np.sqrt(1.0 - cos_theta_source ** 2)
    pos = np.array(
        [RM * sin_theta_source * np.cos(phi_source), RM * sin_theta_source * np.sin(phi_source), RM * cos_theta_source])
    weight_cos_z = pos[0] / RM
    if settings['speed_distribution'] == 'maxwellian':
        ejection_speed = sample_maxwellian_speed(MASS_NA, 1500.0)
    else:
        ejection_speed = sample_weibull_speed(MASS_NA)
    surface_normal = pos.copy()
    if settings['ejection_direction_model'] == 'isotropic':
        ejection_direction = sample_isotropic_direction(surface_normal)
    else:
        ejection_direction = sample_cosine_direction(surface_normal)
    vel = ejection_speed * ejection_direction
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
            local_density_grid[ir, itheta, iphi] += final_weight * DT
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


# --- ★★★ 新しい関数 (tm計算用) ★★★ ---
def simulate_for_migration_time(args):
    """
    Smyth論文の手法に基づき、原子がターミネーターに到達するまでの時間を計算する。
    この関数内では、電離と表面への吸着は起こらない。
    """
    # --- 引数の展開 ---
    settings, spec_data = args['settings'], args['spec']
    TAA, AU, lon, lat, Vms_ms = args['orbit']
    start_phi_rad = args['start_phi']  # 粒子放出開始点(太陽直下点からの角度[rad])

    # --- 定数 ---
    DT = settings['DT']
    RM = PHYSICAL_CONSTANTS['RM']
    MASS_NA = PHYSICAL_CONSTANTS['MASS_NA']
    K_BOLTZMANN = PHYSICAL_CONSTANTS['K_BOLTZMANN']

    # --- 粒子の初期化 ---
    # 特定の地点(start_phi_rad)から粒子を放出する
    # 議論を簡単にするため、ここでは赤道上(theta=pi/2)から放出する
    pos = np.array([
        RM * np.cos(start_phi_rad),
        RM * np.sin(start_phi_rad),
        0.0
    ])

    # 初期速度 (Smyth論文のtm計算では1 km/sの固定値を使用)
    ejection_speed = 1000.0  # [m/s]
    surface_normal = pos / np.linalg.norm(pos)
    ejection_direction = sample_isotropic_direction(surface_normal)  # 等方的に放出
    vel = ejection_speed * ejection_direction

    # --- 時間発展ループ ---
    # tm計算では非常に長い時間追跡する必要がある場合がある
    itmax = int(300 * 3600 / DT)  # 最大300時間追跡

    for it in range(itmax):
        # --- 軌道計算 (4次ルンゲ＝クッタ法) ---
        # (元のコードと同じロジック)
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

        # --- ターミネーター到達判定 ---
        # ターミネーターはX=0の平面。X座標が正から負に変わったら到達とみなす
        if pos_prev[0] > 0 and pos[0] <= 0:
            # 到達時間を返す
            return it * DT

        # --- 表面との衝突判定と処理 ---
        r_current = np.linalg.norm(pos)
        if r_current <= RM:
            # tm計算では吸着しない (stick_prob = 0)
            # 熱的に accomodate して再放出される
            temp_at_impact = calculate_surface_temperature(pos_prev[0], pos_prev[1], pos_prev[2], AU)
            E_in = 0.5 * MASS_NA * np.sum(vel ** 2)
            E_T = K_BOLTZMANN * temp_at_impact
            E_out = settings['BETA'] * E_T + (1.0 - settings['BETA']) * E_in
            v_out_sq = E_out / (0.5 * MASS_NA)
            v_out_speed = np.sqrt(v_out_sq) if v_out_sq > 0 else 0.0

            impact_normal = pos_prev / np.linalg.norm(pos_prev)
            rebound_direction = sample_isotropic_direction(impact_normal)
            vel = v_out_speed * rebound_direction
            pos = RM * impact_normal

    # 最大時間を超えてもターミネーターに到達しなかった場合は無効値(None)を返す
    return None


def main_smyth_method():
    """
    Smyth論文の手法にのっとり、移動時間tmを計算し、柱密度を計算。
    結果をCSVファイルに保存し、グラフをプロットする。
    """
    # --- シミュレーション設定 ---
    N_PARTICLES_FOR_TM = 1000
    #START_ANGLE_DEG = 0.0  # 太陽直下点から放出
    START_ANGLE_DEG = 60.0  # 太陽直下点から60°離れた地点から放出
    #BETA_VALUE = 0.0  # 弾性衝突のケース
    BETA_VALUE = 0.5  # 現実的なケース

    settings = {
        'GRAVITY_ENABLED': True,
        'BETA': BETA_VALUE,
        'T1AU': 1.9e5,
        'DT': 20.0,
    }

    # ★ ファイル名を設定
    output_dir = "smyth_method_results"
    os.makedirs(output_dir, exist_ok=True)
    filename_base = f"smyth_results_beta_{settings['BETA']:.1f}_start_{START_ANGLE_DEG:.0f}deg"
    csv_filepath = os.path.join(output_dir, f"{filename_base}.csv")
    plot_filepath = os.path.join(output_dir, f"{filename_base}.png")

    # --- 外部データの読み込み ---
    # (変更なし)
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

    print("--- Smyth (1995) の手法に基づく柱密度計算 ---")
    print(f"結果をCSVファイルに保存します: {csv_filepath}")
    print(f"粒子放出点: 太陽直下点から {START_ANGLE_DEG}° の地点")
    print(f"エネルギー調整係数 (beta): {settings['BETA']}")
    print("-" * 50)
    print(f"{'TAA':>6s} {'t_M (hr)':>10s} {'tau (hr)':>10s} {'N_bar (cm-2)':>15s}")
    print("-" * 50)

    # ★ プロット用とファイル保存用のリストを準備
    results_data = []

    # --- TAAごとのシミュレーションループ ---
    for line in orbit_lines:
        TAA, AU, lon, lat, Vms_ms = map(float, line.split())

        task_args = {
            'settings': settings, 'spec': spec_data_dict,
            'orbit': (TAA, AU, lon, lat, Vms_ms),
            'start_phi': np.deg2rad(START_ANGLE_DEG)
        }
        tasks = [task_args] * N_PARTICLES_FOR_TM

        with Pool(processes=cpu_count()) as pool:
            results = list(tqdm(pool.imap(simulate_for_migration_time, tasks), total=N_PARTICLES_FOR_TM,
                                desc=f"Calculating t_M for TAA={TAA:.1f}", leave=False))

        valid_times = [t for t in results if t is not None]
        t_M_sec = np.mean(valid_times) if valid_times else float('inf')

        tau_sec = settings['T1AU'] * AU ** 2
        phi_0_cm2 = 4.6e7
        R_p = 0.307
        phi_cm2 = phi_0_cm2 * (R_p / AU) ** 2
        phi_m2 = phi_cm2 * 1e4

        exponent = -t_M_sec / tau_sec if tau_sec > 0 else -float('inf')
        N_bar_m2 = phi_m2 * tau_sec * (1.0 - np.exp(exponent))
        N_bar_cm2 = N_bar_m2 / 1e4

        # 結果を画面に出力
        t_M_hr = t_M_sec / 3600
        tau_hr = tau_sec / 3600
        print(f"{TAA:6.1f} {t_M_hr:10.2f} {tau_hr:10.2f} {N_bar_cm2:15.2e}")

        # ★ 結果をリストに追加
        results_data.append([TAA, t_M_hr, tau_hr, N_bar_cm2])

    # ★ ループ終了後、ファイルに全結果を書き込み
    with open(csv_filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['TAA_deg', 't_M_hr', 'tau_hr', 'N_bar_cm-2'])  # ヘッダー
        writer.writerows(results_data)  # データ
    print("-" * 50)
    print(f"計算結果を {csv_filepath} に保存しました。")

    # ★ ループ終了後、グラフをプロット
    taa_values = [row[0] for row in results_data]
    n_bar_values = [row[3] for row in results_data]

    plt.figure(figsize=(12, 7))
    plt.plot(taa_values, n_bar_values, marker='.', linestyle='-', label=f'beta={settings["BETA"]}')

    plt.xlabel("True Anomaly Angle (deg)", fontsize=14)
    plt.ylabel("Average Column Density (atoms cm⁻²)", fontsize=14)
    plt.title(f"Calculated Column Density vs. TAA (Smyth Method, start={START_ANGLE_DEG}°)", fontsize=16)
    plt.yscale('log')  # 縦軸を対数スケールに
    plt.xticks(np.arange(0, 361, 30))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    # ★ グラフをファイルに保存
    plt.savefig(plot_filepath)
    print(f"グラフを {plot_filepath} に保存しました。")

    # ★ グラフを画面に表示
    plt.show()


if __name__ == '__main__':
    # (元のmain関数はコメントアウトしたまま)
    main_smyth_method()