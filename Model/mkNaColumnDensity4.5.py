import numpy as np
import sys
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# """
# 水星のナトリウム(Na)大気（エキソスフィア）の3次元モンテカルロシミュレーションコード
# ===================================================================================
#
# 概要:
# このスクリプトは、水星の希薄な大気であるエキソスフィアの構造をシミュレートします。
# 特に、ナトリウム原子が水星表面から放出され、太陽放射圧と水星重力の影響を受けて
# 飛行し、最終的に電離・表面への再吸着・系外への脱出によって失われるまでを追跡します。
#
# 主な物理プロセス:
# 1. 表面からのNa原子放出:
#    - 光刺激脱離 (PSD; Photon-Stimulated Desorption): 太陽からの紫外線によって原子が叩き出される過程。
#    - 熱脱離 (TD; Thermal Desorption): 表面温度の上昇により原子が蒸発する過程。
# 2. 軌道運動:
#    - 水星の重力
#    - 太陽光による放射圧 (ドップラー効果を考慮)
# 3. 損失過程:
#    - 光電離 (Photoionization): 太陽光により原子がイオン化されて失われる。
#    - 表面への再衝突と吸着
#    - 水星重力圏からの脱出
#
# 参考文献:
# Suzuki, Y., et al. (2020). Seasonal Variability of Mercury’s Sodium Exosphere
# Deduced From MESSENGER Data and Numerical Simulation. JGR: Planets.
# """

# --- 物理定数 ---
PHYSICAL_CONSTANTS = {
    'C': 299792458.0,  # 光速 [m/s]
    'PI': np.pi,  # 円周率
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
def calculate_surface_temperature(x, y, z, AU):
    """水星表面の局所的な温度を計算する。(論文 式1準拠)

    Args:
        x, y, z (float): 水星中心座標系での位置 [m] (x軸が太陽方向)
        AU (float): 太陽と水星の距離 [天文単位]

    Returns:
        float: 表面温度 [K]
    """
    T0, T1 = 100.0, 600.0
    if x <= 0: return T0
    cos_theta = x / np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return T0 + T1 * (cos_theta ** 0.25) * ((0.306 / AU) ** 2)


def calculate_sticking_probability(surface_temp_K):
    """表面に再衝突したNa原子が吸着される確率を計算する。(論文 式12, 13)

    Args:
        surface_temp_K (float): 衝突地点の表面温度 [K]

    Returns:
        float: 吸着確率 (0-1)
    """
    A, B, porosity = 0.08, 458.0, 0.8
    p_stick = A * np.exp(B / surface_temp_K)
    # 多孔性を考慮した実効的な吸着確率
    return p_stick / (1.0 - (1.0 - p_stick) * porosity)


def calculate_td_rate(surface_temp_K, U_ev=1.85, nu_hz=1e13):
    """【内部関数】表面温度から熱脱離(TD)の1原子あたりの脱離率 [/s] を計算する。(論文 式2)

    Args:
        surface_temp_K (float): 局所的な表面温度 [K]
        U_ev (float): 結合エネルギー [eV]
        nu_hz (float): 表面での原子の振動周波数 [Hz]

    Returns:
        float: 1原子あたりの脱離率 [/sec]
    """
    if surface_temp_K <= 0: return 0.0
    U_joule = U_ev * PHYSICAL_CONSTANTS['E_CHARGE']
    k_B = PHYSICAL_CONSTANTS['K_BOLTZMANN']
    exponent = -U_joule / (k_B * surface_temp_K)
    if exponent < -700: return 0.0  # expのアンダーフロー防止
    return nu_hz * np.exp(exponent)


def calculate_td_flux(surface_temp_K, cNa_m2):
    """熱脱離(TD)による局所的な放出フラックス [atoms/m^2/s] を計算する。"""
    td_rate_per_atom = calculate_td_rate(surface_temp_K)
    return td_rate_per_atom * cNa_m2


def calculate_psd_flux(AU, cos_Z, F_UV_1AU, Q_PSD_m2, cNa_m2):
    """光刺激脱離(PSD)による局所的な放出フラックス [atoms/m^2/s] を計算する。(論文 式4)"""
    if cos_Z <= 0:
        return 0.0

    # 1AUでの太陽直下点でのフラックスを計算
    flux_peak_at_1au = F_UV_1AU * Q_PSD_m2 * cNa_m2
    # 実際の軌道距離と天頂角でのフラックスを計算
    local_flux = (flux_peak_at_1au / AU ** 2) * cos_Z
    return local_flux


def calculate_total_global_flux_td(AU, cNa_m2, n_samples=10000):
    """全球での総TDフラックス [atoms/s] をモンテカルロ積分で近似計算する。"""
    RM = PHYSICAL_CONSTANTS['RM']
    total_flux = 0.0
    # 日照側表面全体で平均のフラックスを計算
    for _ in range(n_samples):
        phi = np.pi * (np.random.random() - 0.5)
        cos_theta = 2 * np.random.random() - 1.0
        sin_theta = np.sqrt(1.0 - cos_theta ** 2)
        x = RM * sin_theta * np.cos(phi)
        y, z = RM * sin_theta * np.sin(phi), RM * cos_theta
        temp = calculate_surface_temperature(x, y, z, AU)
        total_flux += calculate_td_flux(temp, cNa_m2)

    average_flux = total_flux / n_samples
    dayside_area = 2 * PHYSICAL_CONSTANTS['PI'] * RM ** 2
    return average_flux * dayside_area


# --- 粒子生成のためのサンプリング関数 ---
def sample_maxwellian_speed(mass_kg, temp_k):
    """マクスウェル分布に従う速さ(スカラー)をサンプリングする。"""
    sigma = np.sqrt(PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_k / mass_kg)
    v_components = np.random.normal(0, sigma, 3)
    return np.linalg.norm(v_components)


def sample_cosine_direction(normal_vector):
    """法線ベクトル周りにコサイン則に従う方向ベクトルを生成する。"""
    if np.abs(normal_vector[0]) > np.abs(normal_vector[1]):
        inv_len = 1.0 / np.sqrt(normal_vector[0] ** 2 + normal_vector[2] ** 2)
        t1 = np.array([-normal_vector[2] * inv_len, 0, normal_vector[0] * inv_len])
    else:
        inv_len = 1.0 / np.sqrt(normal_vector[1] ** 2 + normal_vector[2] ** 2)
        t1 = np.array([0, normal_vector[2] * inv_len, -normal_vector[1] * inv_len])
    t2 = np.cross(normal_vector, t1)
    p1, p2 = np.random.random(), np.random.random()
    cos_theta, sin_theta = np.sqrt(1.0 - p1), np.sqrt(p1)
    phi = 2 * np.pi * p2
    return t1 * sin_theta * np.cos(phi) + t2 * sin_theta * np.sin(phi) + normal_vector * cos_theta


def sample_isotropic_direction(normal_vector):
    """法線ベクトルの半球上で等方的に分布する方向ベクトルを生成する。"""
    vec = np.random.randn(3)
    vec /= np.linalg.norm(vec)
    if np.dot(vec, normal_vector) < 0: vec = -vec
    return vec


def sample_td_velocity(mass_kg, temp_k, normal_vector, direction_model='cosine'):
    """熱脱離(TD)における速度ベクトルを計算する。(論文 式3)"""
    speed = sample_maxwellian_speed(mass_kg, temp_k)
    if direction_model == 'cosine':
        direction = sample_cosine_direction(normal_vector)
    elif direction_model == 'isotropic':
        direction = sample_isotropic_direction(normal_vector)
    else:
        raise ValueError(f"Unknown TD direction model: {direction_model}")
    return speed * direction


# --- コア計算モジュール ---
def _calculate_acceleration(pos, vel, Vms_ms, AU, spec_data, settings):
    """【RK4内部用】指定された位置と速度における粒子の加速度を計算する。"""
    x, y, z = pos;
    vx, vy, vz = vel
    # 太陽放射圧による加速度
    velocity_for_doppler = vx + Vms_ms
    w_na_d2 = 589.1582e-9 * (1.0 - velocity_for_doppler / PHYSICAL_CONSTANTS['C'])
    w_na_d1 = 589.7558e-9 * (1.0 - velocity_for_doppler / PHYSICAL_CONSTANTS['C'])
    wl, gamma, sigma0_perdnu1, sigma0_perdnu2, JL = spec_data.values()
    if not (wl[0] * 1e-9 <= w_na_d2 < wl[-1] * 1e-9 and wl[0] * 1e-9 <= w_na_d1 < wl[-1] * 1e-9):
        b = 0.0
    else:
        gamma2 = np.interp(w_na_d2 * 1e9, wl, gamma);
        gamma1 = np.interp(w_na_d1 * 1e9, wl, gamma)
        F_lambda_1AU_m = JL * 1e9;
        F_lambda_d1_at_Mercury = F_lambda_1AU_m / (AU ** 2) * gamma1
        F_nu_d1 = F_lambda_d1_at_Mercury * (w_na_d1 ** 2) / PHYSICAL_CONSTANTS['C'];
        J1 = sigma0_perdnu1 * F_nu_d1
        F_lambda_d2_at_Mercury = F_lambda_1AU_m / (AU ** 2) * gamma2
        F_nu_d2 = F_lambda_d2_at_Mercury * (w_na_d2 ** 2) / PHYSICAL_CONSTANTS['C'];
        J2 = sigma0_perdnu2 * F_nu_d2
        b = 1 / PHYSICAL_CONSTANTS['MASS_NA'] * (
                    (PHYSICAL_CONSTANTS['H'] / w_na_d1) * J1 + (PHYSICAL_CONSTANTS['H'] / w_na_d2) * J2)
    if x < 0 and np.sqrt(y ** 2 + z ** 2) < PHYSICAL_CONSTANTS['RM']: b = 0.0
    accel_srp = np.array([-b, 0.0, 0.0])
    # 水星重力による加速度
    accel_g = np.array([0.0, 0.0, 0.0])
    if settings['GRAVITY_ENABLED']:
        r_sq_grav = np.sum(pos ** 2)
        if r_sq_grav > 0:
            grav_accel_total = -PHYSICAL_CONSTANTS['GM_MERCURY'] / r_sq_grav
            accel_g = grav_accel_total * (pos / np.linalg.norm(pos))
    return accel_srp + accel_g


def simulate_single_particle_for_density(args):
    """一個のNa原子の挙動を追跡し、グリッドへの滞在時間と最終的な消滅要因を返す。"""
    # --- 1. 引数と設定の展開 ---
    settings, spec_data = args['settings'], args['spec']
    TAA, AU, lon, lat, Vms_ms = args['orbit']
    grid_params = args['grid_params']
    ejection_process = args['ejection_process']
    psd_params = args['psd_params']
    cNa_m2 = psd_params['cNa_m2']

    DT, IONIZATION_MODEL = settings['DT'], settings['ionization_model']
    RM, MASS_NA = PHYSICAL_CONSTANTS['RM'], PHYSICAL_CONSTANTS['MASS_NA']
    N_R, N_THETA, N_PHI = grid_params['n_r'], grid_params['n_theta'], grid_params['n_phi']
    R_MAX = grid_params['max_r']
    local_density_grid = np.zeros((N_R, N_THETA, N_PHI), dtype=np.float32)

    # --- 2. 粒子の初期化 (放出過程に応じて) ---
    # 昼側表面で均一な位置に粒子を生成する (TDとPSDで共通のロジック)
    phi_source = np.pi * (np.random.random() - 0.5)
    cos_theta_source = 2 * np.random.random() - 1.0
    sin_theta_source = np.sqrt(1.0 - cos_theta_source ** 2)
    pos = RM * np.array(
        [sin_theta_source * np.cos(phi_source), sin_theta_source * np.sin(phi_source), cos_theta_source])
    cos_Z = pos[0] / RM

    if ejection_process == 'PSD':
        # 局所的な放出フラックスを計算し、それを粒子の重みとする
        weight = calculate_psd_flux(AU, cos_Z, **psd_params)

        # 放出速度を計算
        ejection_speed = sample_maxwellian_speed(MASS_NA, 1500.0)
        surface_normal = pos / RM
        direction = sample_isotropic_direction(surface_normal) if settings[
                                                                      'psd_direction_model'] == 'isotropic' else sample_cosine_direction(
            surface_normal)
        vel = ejection_speed * direction

    elif ejection_process == 'TD':
        # 局所的な表面温度を計算
        local_temp = calculate_surface_temperature(pos[0], pos[1], pos[2], AU)
        # 局所的な放出フラックスを計算し、それを粒子の重みとする
        weight = calculate_td_flux(local_temp, cNa_m2)

        # 放出速度を計算
        surface_normal = pos / RM
        vel = sample_td_velocity(MASS_NA, local_temp, surface_normal, settings['td_direction_model'])

    else:
        raise ValueError(f"Unknown ejection process: {ejection_process}")

    # --- 3. 時間発展ループ ---
    tau = settings['T1AU'] * AU ** 2
    itmax = int(tau * 5.0 / DT + 0.5)
    death_reason = 'ionized'
    for it in range(itmax):
        # 3a. 損失プロセス (光電離)
        if IONIZATION_MODEL == 'particle_death' and np.random.random() < (1.0 - np.exp(-DT / tau)):
            death_reason = 'ionized';
            break
        Nad = np.exp(-DT * it / tau) if IONIZATION_MODEL == 'weight_decay' else 1.0

        # 3b. 軌道計算 (4次ルンゲ＝クッタ法)
        pos_prev, vel_prev = pos.copy(), vel.copy()
        k1_v = DT * _calculate_acceleration(pos, vel, Vms_ms, AU, spec_data, settings);
        k1_p = DT * vel
        k2_v = DT * _calculate_acceleration(pos + 0.5 * k1_p, vel + 0.5 * k1_v, Vms_ms, AU, spec_data, settings);
        k2_p = DT * (vel + 0.5 * k1_v)
        k3_v = DT * _calculate_acceleration(pos + 0.5 * k2_p, vel + 0.5 * k2_v, Vms_ms, AU, spec_data, settings);
        k3_p = DT * (vel + 0.5 * k2_v)
        k4_v = DT * _calculate_acceleration(pos + k3_p, vel + k3_v, Vms_ms, AU, spec_data, settings);
        k4_p = DT * (vel + k3_v)
        pos += (k1_p + 2 * k2_p + 2 * k3_p + k4_p) / 6.0
        vel += (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6.0

        # 3c. グリッドへの滞在時間記録と脱出判定
        r = np.linalg.norm(pos)
        if r >= R_MAX: death_reason = 'escaped'; break
        ir = int(r / (R_MAX / N_R))
        itheta = int(np.arccos(pos[2] / r) / (np.pi / N_THETA))
        iphi = int((np.arctan2(pos[1], pos[0]) + np.pi) / (2 * np.pi / N_PHI))
        if 0 <= ir < N_R and 0 <= itheta < N_THETA and 0 <= iphi < N_PHI:
            local_density_grid[ir, itheta, iphi] += weight * Nad * DT

        # 3d. 表面との衝突判定と処理
        if r <= RM:
            temp_impact = calculate_surface_temperature(pos_prev[0], pos_prev[1], pos_prev[2], AU)
            if np.random.random() < calculate_sticking_probability(temp_impact):
                death_reason = 'stuck';
                break
            E_in = 0.5 * MASS_NA * np.sum(vel_prev ** 2)
            E_T = PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_impact
            E_out = settings['BETA'] * E_T + (1.0 - settings['BETA']) * E_in
            v_out = np.sqrt(2 * E_out / MASS_NA) if E_out > 0 else 0.0
            impact_normal = pos_prev / np.linalg.norm(pos_prev)
            vel = v_out * sample_isotropic_direction(impact_normal)
            pos = RM * impact_normal

    return local_density_grid, death_reason


def main():
    """シミュレーションを実行するメイン関数"""
    # --- 1. シミュレーション設定 ---
    OUTPUT_DIRECTORY = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult3D"
    N_R, N_THETA, N_PHI = 100, 24, 24  # 半径, 天頂角, 方位角のグリッド数
    GRID_RADIUS_RM = 5.0  # シミュレーション空間の半径 (水星半径単位)
    N_PARTICLES = 10000  # 各TAAでシミュレートする粒子数

    settings = {
        'GRAVITY_ENABLED': True,  # 重力計算のON/OFF
        'BETA': 0.5,  # 表面衝突時の熱緩和係数
        'T1AU': 1.9e5,  # 1AUでの光電離寿命 [s]
        'DT': 10.0,  # 時間ステップ [s]
        'ionization_model': 'particle_death',  # 'particle_death' or 'weight_decay'
        'psd_direction_model': 'isotropic',  # PSDの放出角度モデル 'isotropic' or 'cosine'
        'td_direction_model': 'cosine',  # TDの放出角度モデル
    }

    # --- 2. ディレクトリとログファイルの設定 ---
    dist_tag = f"PSD_{settings['psd_direction_model'][0:3].upper()}_TD_{settings['td_direction_model'][0:3].upper()}"
    base_name_template = f"density3d_{dist_tag}_beta{settings['BETA']:.2f}_pl{N_THETA}x{N_PHI}"
    target_output_dir = os.path.join(OUTPUT_DIRECTORY, base_name_template)
    os.makedirs(target_output_dir, exist_ok=True)
    print(f"結果は '{target_output_dir}' に保存されます。")
    log_file_path = os.path.join(target_output_dir, "death_statistics.csv")
    with open(log_file_path, 'w', newline='') as f:
        f.write("TAA,Prob_TD,Ionized_Count,Ionized_Percent,Stuck_Count,Stuck_Percent,Escaped_Count,Escaped_Percent\n")

    # --- 3. 物理定数・外部データの準備 ---
    grid_params = {'n_r': N_R, 'n_theta': N_THETA, 'n_phi': N_PHI, 'max_r': PHYSICAL_CONSTANTS['RM'] * GRID_RADIUS_RM}
    spec_data_np = np.loadtxt('SolarSpectrum_Na0.txt', usecols=(0, 3));
    wl, gamma = spec_data_np[:, 0], spec_data_np[:, 1]
    orbit_lines = open('orbit360.txt', 'r').readlines()
    sigma_const = PHYSICAL_CONSTANTS['E_CHARGE'] ** 2 / (
                4 * PHYSICAL_CONSTANTS['ME'] * PHYSICAL_CONSTANTS['C'] * PHYSICAL_CONSTANTS['EPSILON_0'])
    spec_data_dict = {'wl': wl, 'gamma': gamma, 'sigma0_perdnu1': sigma_const * 0.320,
                      'sigma0_perdnu2': sigma_const * 0.641, 'JL': 5.18e14 * 1e4}

    # PSD計算に必要な物理定数を辞書にまとめる
    psd_params = {
        'F_UV_1AU': 1.5e18,  # 1AUでの紫外線光子フラックス [photons/s/m^2]
        'Q_PSD_m2': 2.0e-24,  # 光刺激脱離の断面積 [m^2]
        'cNa_m2': 1.5e17  # 表面のNa原子密度 [atoms/m^2]
    }

    # --- 4. TAAごとのシミュレーションループ ---
    for line in orbit_lines:
        TAA, AU, lon, lat, Vms_ms = map(float, line.split())
        print(f"\n--- TAA = {TAA:.1f}度のシミュレーションを開始 ---")

        # 4a. このTAAにおける総放出率の計算
        # 太陽直下点でのPSD放出率を計算
        R_PSD_peak_m2 = calculate_psd_flux(AU, 1.0, **psd_params)
        # 全球からの総放出フラックスを計算
        total_flux_psd = R_PSD_peak_m2 * np.pi * (PHYSICAL_CONSTANTS['RM'] ** 2)
        total_flux_td = calculate_total_global_flux_td(AU, psd_params['cNa_m2'])
        total_flux_combined = total_flux_psd + total_flux_td
        # TDプロセスである確率を計算
        prob_is_td = total_flux_td / total_flux_combined if total_flux_combined > 0 else 0.0
        print(
            f"PSD Flux: {total_flux_psd:.2e} atoms/s | TD Flux: {total_flux_td:.2e} atoms/s | P(TD): {prob_is_td:.4f}")

        # 4b. 並列処理タスクの準備
        tasks = [{'settings': settings, 'spec': spec_data_dict, 'orbit': (TAA, AU, lon, lat, Vms_ms),
                  'grid_params': grid_params,
                  'ejection_process': 'TD' if np.random.random() < prob_is_td else 'PSD',
                  'psd_params': psd_params}
                 for _ in range(N_PARTICLES)]

        # 4c. 並列計算の実行と結果集計
        with Pool(processes=cpu_count()) as pool:
            results = list(
                tqdm(pool.imap(simulate_single_particle_for_density, tasks), total=N_PARTICLES, desc=f"TAA={TAA:.1f}"))
        death_counts = {'ionized': 0, 'stuck': 0, 'escaped': 0}
        master_density_grid = np.zeros((N_R, N_THETA, N_PHI), dtype=np.float32)
        for grid, reason in results:
            master_density_grid += grid
            death_counts[reason] += 1

        # 4d. 死因統計の記録
        total_simulated = sum(death_counts.values())
        ionized_p = (death_counts['ionized'] / total_simulated) * 100 if total_simulated > 0 else 0
        stuck_p = (death_counts['stuck'] / total_simulated) * 100 if total_simulated > 0 else 0
        escaped_p = (death_counts['escaped'] / total_simulated) * 100 if total_simulated > 0 else 0
        with open(log_file_path, 'a', newline='') as f:
            f.write(
                f"{TAA:.1f},{prob_is_td:.4f},{death_counts['ionized']},{ionized_p:.2f},{death_counts['stuck']},{stuck_p:.2f},{death_counts['escaped']},{escaped_p:.2f}\n")

        # 4e. 数密度の計算と保存
        r_edges = np.linspace(0, grid_params['max_r'], N_R + 1)
        theta_edges = np.linspace(0, np.pi, N_THETA + 1)
        delta_r3_3 = (r_edges[1:] ** 3 - r_edges[:-1] ** 3) / 3.0
        delta_cos_theta = np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:])
        D_PHI_rad = 2 * np.pi / N_PHI
        cell_volumes = delta_r3_3[:, np.newaxis, np.newaxis] * delta_cos_theta[np.newaxis, :, np.newaxis] * D_PHI_rad
        cell_volumes[cell_volumes == 0] = 1e-30

        # 規格化係数を計算して数密度に変換
        dayside_area = 2 * PHYSICAL_CONSTANTS['PI'] * PHYSICAL_CONSTANTS['RM'] ** 2
        normalization_factor = dayside_area / N_PARTICLES
        number_density_m3 = normalization_factor * (master_density_grid / cell_volumes)
        # 単位を [atoms/m^3] から [atoms/cm^3] に変換して保存
        np.save(os.path.join(target_output_dir, f"density3d_taa{TAA:.0f}.npy"), number_density_m3 / 1e6)

    print("\nすべてのシミュレーションが完了しました。")


if __name__ == '__main__':
    main()