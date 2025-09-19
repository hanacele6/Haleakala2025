import numpy as np
import sys
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import matplotlib.pyplot as plt


# --- 物理モデルに基づく関数 ---

def calculate_surface_temperature(x, y, z, AU):
    """
    水星表面の座標(x, y, z)と太陽距離(AU)から局所的な表面温度を計算する。
    太陽直下点を(x>0, y=0, z=0)とする。
    """
    T0 = 100.0  # 夜側の最低温度 [K]
    T1 = 600.0  # 太陽直下点での最大温度上昇 [K]
    if x <= 0:
        return T0
    # 太陽方向ベクトル(1,0,0)と位置ベクトルのなす角の余弦を計算
    cos_theta = x / np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if cos_theta < 0:
        return T0
    # 表面温度の経験式
    temp = T0 + T1 * (cos_theta ** 0.25) * ((0.306 / AU) ** 2)
    return temp


def calculate_sticking_probability(surface_temp_K):
    """
    表面温度に基づいて、ナトリウム原子の表面への吸着確率を計算する。
    """
    # 論文で使用されている定数 Johnson 2002
    A = 0.08
    B = 458.0
    porosity = 0.8  # 表面の多孔性
    # 吸着確率の基本式
    p_stick = A * np.exp(B / surface_temp_K)
    # 多孔性を考慮した実効的な吸着確率
    p_stick_eff = p_stick / (1.0 - (1.0 - p_stick) * porosity)
    return p_stick_eff


# --- 物理モデルに基づくサンプリング関数 ---
def sample_maxwellian_speed(mass_kg, temp_k):
    """
    指定された温度のマクスウェル分布に従う速さをサンプリングする。
    """
    K_BOLTZMANN = 1.380649e-23  # ボルツマン定数 [J/K]

    # 速度の各成分の標準偏差を計算
    sigma = np.sqrt(K_BOLTZMANN * temp_k / mass_kg)

    # 3つの独立した速度成分を正規分布からサンプリング
    vx = np.random.normal(0, sigma)
    vy = np.random.normal(0, sigma)
    vz = np.random.normal(0, sigma)

    # 3次元速度ベクトルの大きさ（速さ）を計算して返す
    speed = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    return speed


def sample_weibull_speed(mass_kg,
                         U_ev=0.05,  # Leblanc et al., 2022
                         beta_shape=0.7):
    """
    ワイブル分布に従う放出エネルギーから速さをサンプリングする。
    """
    E_CHARGE = 1.602176634e-19  # 電子の電荷 [C]
    p = np.random.random()
    # 逆関数法を用いてエネルギーをサンプリング
    E_ev = U_ev * (p ** (-1.0 / (beta_shape + 1.0)) - 1.0)
    E_joule = E_ev * E_CHARGE
    # エネルギーを速度に変換
    v_ms = np.sqrt(2 * E_joule / mass_kg)
    return v_ms


def sample_cosine_direction(normal_vector):
    """
    指定された法線ベクトル(normal_vector)に対してコサイン則(Lambert則)に従う3D方向ベクトルをサンプリングする。
    """
    # 法線ベクトルを正規化
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # 法線ベクトルと直交する接ベクトルt1を生成 (簡易的な方法)
    if np.abs(normal_vector[0]) > np.abs(normal_vector[1]):
        inv_len = 1.0 / np.sqrt(normal_vector[0] ** 2 + normal_vector[2] ** 2)
        t1 = np.array([-normal_vector[2] * inv_len, 0, normal_vector[0] * inv_len])
    else:
        inv_len = 1.0 / np.sqrt(normal_vector[1] ** 2 + normal_vector[2] ** 2)
        t1 = np.array([0, normal_vector[2] * inv_len, -normal_vector[1] * inv_len])
    # もう一つの接ベクトルt2を外積で生成
    t2 = np.cross(normal_vector, t1)

    # コサイン則に従うローカル座標での方向をサンプリング
    p1, p2 = np.random.random(), np.random.random()
    sin_theta_sq = p1
    cos_theta = np.sqrt(1.0 - sin_theta_sq)
    sin_theta = np.sqrt(sin_theta_sq)
    phi = 2 * np.pi * p2

    # ローカル座標系でのベクトルをワールド座標系に変換
    direction = t1 * sin_theta * np.cos(phi) + t2 * sin_theta * np.sin(phi) + normal_vector * cos_theta
    return direction


def sample_isotropic_direction(normal_vector):
    """
    指定された法線ベクトルの半球方向へ等方的に(Isotropic)な3D方向ベクトルをサンプリングする。
    """
    # 空間でランダムな方向ベクトルを生成
    vec = np.random.randn(3)
    vec /= np.linalg.norm(vec)

    # 法線ベクトルとの内積をチェックし、向きを調整
    # 内積が負ならベクトルが地表を向いているので反転させる
    if np.dot(vec, normal_vector) < 0:
        vec = -vec

    return vec


# --- シミュレーションのコア関数 (3D改造版) ---
def simulate_single_particle_for_density(args):
    """
    一個の粒子を追跡し、その軌跡を球座標密度グリッドに記録する。
    """
    # 引数をアンパック
    constants, settings, spec_data = args['consts'], args['settings'], args['spec']
    TAA, AU, lon, lat, Vms_ms = args['orbit']
    grid_params = args['grid_params']

    # 物理定数と設定をアンパック
    C, PI, H, MASS_NA, RM, GM_MERCURY, K_BOLTZMANN = constants.values()
    # <<< 修正: ionization_modelをアンパック >>>
    GRAVITY_ENABLED, BETA, T1AU, DT, SPEED_DISTRIBUTION, EJECTION_DIRECTION_MODEL, IONIZATION_MODEL = settings.values()
    wl, gamma, sigma0_perdnu2, sigma0_perdnu1, JL = spec_data.values()

    # 球座標グリッドのパラメータをアンパック
    N_R, N_THETA, N_PHI, R_MAX = grid_params['n_r'], grid_params['n_theta'], grid_params['n_phi'], grid_params['max_r']
    DR = R_MAX / N_R
    D_THETA = PI / N_THETA
    D_PHI = 2 * PI / N_PHI

    local_density_grid = np.zeros((N_R, N_THETA, N_PHI), dtype=np.float32)

    # <<< 追加: 粒子消滅法の場合、重みは常に1 >>>
    particle_weight = 1.0

    # 太陽光フラックスに比例した放出(cos則)をシミュレート
    p1, p2 = np.random.random(), np.random.random()
    cos_theta_source = np.sqrt(p1)
    sin_theta_source = np.sqrt(1.0 - cos_theta_source ** 2)
    phi_source = 2 * PI * p2

    x = RM * cos_theta_source
    y = RM * sin_theta_source * np.cos(phi_source)
    z = RM * sin_theta_source * np.sin(phi_source)

    if SPEED_DISTRIBUTION == 'maxwellian':
        ejection_speed = sample_maxwellian_speed(MASS_NA, 1500.0)
    else:
        ejection_speed = sample_weibull_speed(MASS_NA)

    surface_normal = np.array([x, y, z])
    if EJECTION_DIRECTION_MODEL == 'isotropic':
        ejection_direction = sample_isotropic_direction(surface_normal)
    else:
        ejection_direction = sample_cosine_direction(surface_normal)

    vx_ms = ejection_speed * ejection_direction[0]
    vy_ms = ejection_speed * ejection_direction[1]
    vz_ms = ejection_speed * ejection_direction[2]

    Vms = Vms_ms
    tau = T1AU * AU ** 2
    itmax = int(tau * 5.0 / DT + 0.5)

    death_reason = 'ionized'
    for it in range(itmax):


        ionization_prob_per_step = 1.0 - np.exp(-DT / tau)

        if IONIZATION_MODEL == 'particle_death':
            if np.random.random() < ionization_prob_per_step:
                death_reason = 'ionized'
                break  # ループを抜けて粒子を消滅させる

        Nad = 1.0  # デフォルト値
        if IONIZATION_MODEL == 'weight_decay':
            Nad = np.exp(-DT * it / tau)


        velocity_for_doppler = vx_ms + Vms
        w_na_d2 = 589.1582e-9 * (1.0 - velocity_for_doppler / C)
        w_na_d1 = 589.7558e-9 * (1.0 - velocity_for_doppler / C)

        if not (wl[0] * 1e-9 <= w_na_d2 < wl[-1] * 1e-9 and wl[0] * 1e-9 <= w_na_d1 < wl[-1] * 1e-9):
            death_reason = 'escaped';
            break

        gamma2 = np.interp(w_na_d2 * 1e9, wl, gamma)
        gamma1 = np.interp(w_na_d1 * 1e9, wl, gamma)
        F_lambda_1AU_m = JL * 1e9

        F_lambda_d1_at_Mercury = F_lambda_1AU_m / (AU ** 2) * gamma1
        F_nu_d1 = F_lambda_d1_at_Mercury * (w_na_d1 ** 2) / C
        J1 = sigma0_perdnu1 * F_nu_d1
        F_lambda_d2_at_Mercury = F_lambda_1AU_m / (AU ** 2) * gamma2
        F_nu_d2 = F_lambda_d2_at_Mercury * (w_na_d2 ** 2) / C
        J2 = sigma0_perdnu2 * F_nu_d2
        b = 1 / MASS_NA * ((H / w_na_d1) * J1 + (H / w_na_d2) * J2)

        if x < 0 and np.sqrt(y ** 2 + z ** 2) < RM:
            b = 0.0

        accel_gx, accel_gy, accel_gz = 0.0, 0.0, 0.0
        if GRAVITY_ENABLED:
            r_sq_grav = x ** 2 + y ** 2 + z ** 2
            if r_sq_grav > 0:
                r_grav = np.sqrt(r_sq_grav)
                grav_accel_total = GM_MERCURY / r_sq_grav
                accel_gx = -grav_accel_total * (x / r_grav)
                accel_gy = -grav_accel_total * (y / r_grav)
                accel_gz = -grav_accel_total * (z / r_grav)

        vx_ms_prev, vy_ms_prev, vz_ms_prev = vx_ms, vy_ms, vz_ms
        accel_srp_x = -b
        total_accel_x, total_accel_y, total_accel_z = accel_srp_x + accel_gx, accel_gy, accel_gz
        vx_ms += total_accel_x * DT
        vy_ms += total_accel_y * DT
        vz_ms += total_accel_z * DT

        x_prev, y_prev, z_prev = x, y, z
        x += ((vx_ms_prev + vx_ms) / 2.0) * DT
        y += ((vy_ms_prev + vy_ms) / 2.0) * DT
        z += ((vz_ms_prev + vz_ms) / 2.0) * DT

        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        if r == 0: continue
        if r >= R_MAX: death_reason = 'escaped'; break

        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        ir = int(r / DR)
        itheta = int(theta / D_THETA)
        iphi = int((phi + PI) / D_PHI)

        if 0 <= ir < N_R and 0 <= itheta < N_THETA and 0 <= iphi < N_PHI:

            if IONIZATION_MODEL == 'weight_decay':
                local_density_grid[ir, itheta, iphi] += Nad * DT
            else:  # particle_death
                local_density_grid[ir, itheta, iphi] += particle_weight * DT

        r_current = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        if r_current <= RM:
            temp_at_impact = calculate_surface_temperature(x_prev, y_prev, z_prev, AU)
            stick_prob = calculate_sticking_probability(temp_at_impact)

            if np.random.random() < stick_prob:
                death_reason = 'stuck';
                break

            v_in_sq = vx_ms_prev ** 2 + vy_ms_prev ** 2 + vz_ms_prev ** 2
            E_in = 0.5 * MASS_NA * (v_in_sq)
            E_T = K_BOLTZMANN * temp_at_impact
            E_out = BETA * E_T + (1.0 - BETA) * E_in
            v_out_sq = E_out / (0.5 * MASS_NA)
            v_out_speed = np.sqrt(v_out_sq) if v_out_sq > 0 else 0.0

            impact_normal = np.array([x_prev, y_prev, z_prev])
            rebound_direction = sample_cosine_direction(impact_normal)

            vx_ms = v_out_speed * rebound_direction[0]
            vy_ms = v_out_speed * rebound_direction[1]
            vz_ms = v_out_speed * rebound_direction[2]

            impact_pos_norm = np.linalg.norm(impact_normal)
            x = RM * impact_normal[0] / impact_pos_norm
            y = RM * impact_normal[1] / impact_pos_norm
            z = RM * impact_normal[2] / impact_pos_norm

    return (local_density_grid, death_reason)


# --- メインの制御関数 ---
def main():
    # --- ★★★ 設定項目 ★★★ ---
    OUTPUT_DIRECTORY = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult3D"

    N_R = 100
    N_THETA = 24
    N_PHI = 24
    GRID_RADIUS_RM = 5.0
    N_PARTICLES = 10000

    settings = {
        'GRAVITY_ENABLED': True,
        'BETA': 0.5,  # 水星表面との衝突での係数 0で弾性衝突、1で完全にエネルギーを失う　
        # 理想的な石英表面において、ナトリウムではβ≈0.62、カリウムではβ≈0.26
        'T1AU': 61728.4,  # 電離寿命　実験値 [s]
        # 'T1AU': 168918.0, # 電離寿命 理論値[s]
        'DT': 10.0,  # 時間ステップ [s]
        'speed_distribution': 'maxwellian',  # 'maxwellian' または 'weibull'
        'ejection_direction_model': 'isotropic',  # 'cosine'(cos則) または 'isotropic'(等方)
        'ionization_model': 'particle_death'  # 'weight_decay' または 'particle_death'
    }


    dist_tag = "CO" if settings['ejection_direction_model'] == 'cosine' else "ISO"
    speed_tag = "MW" if settings['speed_distribution'] == 'maxwellian' else "WB"
    ion_tag = "WD" if settings['ionization_model'] == 'weight_decay' else "PD"
    base_name_template = f"density3d_beta{settings['BETA']:.2f}_Q2.0_{speed_tag}_{dist_tag}_{ion_tag}_pl{N_THETA}x{N_PHI}"

    sub_folder_name = base_name_template
    target_output_dir = os.path.join(OUTPUT_DIRECTORY, sub_folder_name)
    os.makedirs(target_output_dir, exist_ok=True)
    print(f"結果は '{target_output_dir}' に保存されます。")

    log_file_path = os.path.join(target_output_dir, "death_statistics_Q2.0.csv")
    with open(log_file_path, 'w', newline='') as f:
        f.write("TAA,Ionized_Count,Ionized_Percent,Stuck_Count,Stuck_Percent,Escaped_Count,Escaped_Percent\n")
    print(f"統計情報は {log_file_path} に記録されます。")

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
        'n_r': N_R, 'n_theta': N_THETA, 'n_phi': N_PHI,
        'max_r': constants['RM'] * GRID_RADIUS_RM
    }

    try:
        spec_data_np = np.loadtxt('SolarSpectrum_Na0.txt', usecols=(0, 3))
        wl, gamma = spec_data_np[:, 0], spec_data_np[:, 1]
        orbit_lines = open('orbit360.txt', 'r').readlines()
    except FileNotFoundError as e:
        print(f"エラー: データファイルが見つかりません - {e}")
        sys.exit()

    if not np.all(np.diff(wl) > 0):
        sort_indices = np.argsort(wl)
        wl, gamma = wl[sort_indices], gamma[sort_indices]

    ME, E_CHARGE = 9.1093897e-31, 1.60217733e-19  # 電子の質量 [kg] # 電子の電荷 [C]
    epsilon_0 = 8.854187817e-12  # 真空の誘電率
    sigma_const = E_CHARGE ** 2 / (4 * ME * constants['C'] * epsilon_0)  # SI [m^2/s]
    spec_data_dict = {
        'wl': wl, 'gamma': gamma,
        'sigma0_perdnu2': sigma_const * 0.641,  # 0.641 = D2線の振動子強度
        'sigma0_perdnu1': sigma_const * 0.320,  # 0.320 = D1線の振動子強度
        'JL': 5.18e14 * 1e4  # 1AUでの太陽フラックス [phs/s/m2/nm]
    }

    for line in orbit_lines:
        TAA, AU, lon, lat, Vms_ms = map(float, line.split())
        print(f"\n--- TAA = {TAA:.1f}度のシミュレーションを開始 ---")

        F_UV_at_1AU_per_m2 = 1.5e14 * 1e4  # 1天文単位での紫外線光子フラックス [photons/m^2/s]
        #Q_PSD_cm2 = 1.0e-20 / 1e4  # 光脱離断面積 [m^2]
        Q_PSD_cm2 = 2.0e-20 / 1e4 # suzukiが使ってたやつ
        #Q_PSD_cm2 = 3.0e-20 / 1e4  # YakshinskiyとMadey（1999）
        # Q_PSD_cm2 = 1.4e-21 / 1e4 # Killenら（2004）
        RM_m = constants['RM']  # 水星半径 [m]
        # cNa = 0.053 * 7.5e14 * 1e4 # Moroni (2023) 水星表面のナトリウム原子の割合
        cNa = 1.5e13 * 1e4  # suzuki (2019)
        F_UV_current_per_m2 = F_UV_at_1AU_per_m2 / (AU ** 2)
        R_PSD_peak_per_m2 = F_UV_current_per_m2 * Q_PSD_cm2 * cNa

        effective_area_m2 = np.pi * (RM_m ** 2)
        total_flux_for_this_taa = R_PSD_peak_per_m2 * effective_area_m2

        task_args = {
            'consts': constants, 'settings': settings, 'spec': spec_data_dict,
            'orbit': (TAA, AU, lon, lat, Vms_ms), 'grid_params': grid_params
        }
        tasks = [task_args] * N_PARTICLES

        with Pool(processes=cpu_count()) as pool:
            results = list(tqdm(pool.imap(simulate_single_particle_for_density, tasks), total=N_PARTICLES,
                                desc=f"TAA={TAA:.1f}"))

        print("結果を集計・保存しています...")
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
        print(f"統計情報を {log_file_path} に追記しました。")

        R_MAX = grid_params['max_r']
        D_THETA = constants['PI'] / N_THETA
        D_PHI = 2 * constants['PI'] / N_PHI
        r_edges = np.linspace(0, R_MAX, N_R + 1)
        theta_edges = np.linspace(0, constants['PI'], N_THETA + 1)

        # セルの体積 V = (1/3)(r_out^3 - r_in^3) * (cos(θ_in) - cos(θ_out)) * Δφ
        delta_r3_3 = (r_edges[1:] ** 3 - r_edges[:-1] ** 3) / 3.0
        delta_cos_theta = np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:])

        cell_volumes_m3 = delta_r3_3[:, np.newaxis, np.newaxis] * \
                          delta_cos_theta[np.newaxis, :, np.newaxis] * \
                          D_PHI
        cell_volumes_m3[cell_volumes_m3 == 0] = 1e-30

        # 数密度 [atoms/m^3] を計算
        # (総放出率 / テスト粒子数) * (グリッド上の延べ時間 / セルの体積)
        number_density_m3 = (total_flux_for_this_taa / N_PARTICLES) * (master_density_grid / cell_volumes_m3)

        # 最終出力のために [atoms/cm^3] に変換
        number_density_cm3 = number_density_m3 / 1e6

        parameter_part = base_name_template.replace("density3d_", "")
        base_filename = f"density3d_taa{TAA:.0f}_{parameter_part}"
        full_path_npy = os.path.join(target_output_dir, f"{base_filename}.npy")
        np.save(full_path_npy, number_density_cm3)
        print(f"結果を {full_path_npy} に保存しました。")

    print("\n★★★ すべてのシミュレーションが完了しました ★★★")


if __name__ == '__main__':
    main()