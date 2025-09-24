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


# --- 物理モデルに基づく関数 ---

def calculate_surface_temperature(x, y, z, AU):
    """
    水星表面の座標(x, y, z)と太陽距離(AU)から局所的な表面温度を計算する。

    太陽直下点を(x>0, y=0, z=0)と仮定した経験式に基づいています。

    Args:
        x (float): 粒子のX座標 [m]
        y (float): 粒子のY座標 [m]
        z (float): 粒子のZ座標 [m]
        AU (float): 水星と太陽の距離 [天文単位]

    Returns:
        float: 表面温度 [K]
    """
    T0 = 100.0  # 夜側の最低温度 [K]
    T1 = 600.0  # 太陽直下点での最大温度上昇 [K]
    if x <= 0:
        return T0
    cos_theta = x / np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if cos_theta < 0:
        return T0
    # 表面温度の経験式
    temp = T0 + T1 * (cos_theta ** 0.25) * ((0.306 / AU) ** 2)
    return temp


def calculate_sticking_probability(surface_temp_K):
    """
    表面温度に基づいて、ナトリウム原子の表面への吸着確率を計算する。

    Johnson (2002) の論文で用いられている定数と式を参考にしています。
    表面の多孔性を考慮に入れています。

    Args:
        surface_temp_K (float): 表面温度 [K]

    Returns:
        float: 実効的な吸着確率 (0から1)
    """
    A = 0.08
    B = 458.0
    porosity = 0.8  # 表面の多孔性
    # 吸着確率の基本式
    p_stick = A * np.exp(B / surface_temp_K)
    # 多孔性を考慮した実効的な吸着確率
    p_stick_eff = p_stick / (1.0 - (1.0 - p_stick) * porosity)
    return p_stick_eff


# --- サンプリング関数 ---

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
    vx = np.random.normal(0, sigma)
    vy = np.random.normal(0, sigma)
    vz = np.random.normal(0, sigma)
    speed = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    return speed


def sample_weibull_speed(mass_kg, U_ev=0.05, beta_shape=0.7):
    """
    ワイブル分布に従う放出エネルギーから速さをサンプリングする。
    Leblanc et al. (2022) の研究を参考にしています。

    Args:
        mass_kg (float): 粒子の質量 [kg]
        U_ev (float, optional): 分布のスケールパラメータ [eV]. Defaults to 0.05.
        beta_shape (float, optional): 分布の形状パラメータ. Defaults to 0.7.

    Returns:
        float: サンプリングされた速さ [m/s]
    """
    p = np.random.random()
    E_ev = U_ev * (p ** (-1.0 / (beta_shape + 1.0)) - 1.0)
    E_joule = E_ev * PHYSICAL_CONSTANTS['E_CHARGE']
    v_ms = np.sqrt(2 * E_joule / mass_kg)
    return v_ms


def sample_cosine_direction(normal_vector):
    """
    法線ベクトル周りにコサイン分布（ランバート反射）に従う方向ベクトルを生成する。

    Args:
        normal_vector (np.ndarray): 表面の法線ベクトル (3次元)

    Returns:
        np.ndarray: サンプリングされた方向ベクトル (3次元、正規化済み)
    """
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # 法線ベクトルに直交する2つの基底ベクトルを計算
    if np.abs(normal_vector[0]) > np.abs(normal_vector[1]):
        inv_len = 1.0 / np.sqrt(normal_vector[0] ** 2 + normal_vector[2] ** 2)
        t1 = np.array([-normal_vector[2] * inv_len, 0, normal_vector[0] * inv_len])
    else:
        inv_len = 1.0 / np.sqrt(normal_vector[1] ** 2 + normal_vector[2] ** 2)
        t1 = np.array([0, normal_vector[2] * inv_len, -normal_vector[1] * inv_len])
    t2 = np.cross(normal_vector, t1)

    # 球面座標系でランダムサンプリング
    p1, p2 = np.random.random(), np.random.random()
    sin_theta_sq = p1
    cos_theta = np.sqrt(1.0 - sin_theta_sq)
    sin_theta = np.sqrt(sin_theta_sq)
    phi = 2 * np.pi * p2

    # 直交座標系に変換
    direction = t1 * sin_theta * np.cos(phi) + t2 * sin_theta * np.sin(phi) + normal_vector * cos_theta
    return direction


def sample_isotropic_direction(normal_vector):
    """
    法線ベクトルの半球上で等方的に分布する方向ベクトルを生成する。

    Args:
        normal_vector (np.ndarray): 表面の法線ベクトル (3次元)

    Returns:
        np.ndarray: サンプリングされた方向ベクトル (3次元、正規化済み)
    """
    vec = np.random.randn(3)
    vec /= np.linalg.norm(vec)

    # 法線ベクトルと同じ向き（半球内）になるように調整
    if np.dot(vec, normal_vector) < 0:
        vec = -vec
    return vec


def _calculate_acceleration(pos, vel, Vms_ms, AU, spec_data, settings):
    """
    【RK4用】指定された位置と速度における粒子の加速度を計算する。

    重力と太陽放射圧を考慮します。放射圧はドップラー効果により速度に依存します。
    この関数はルンゲ＝クッタ法の内部で繰り返し呼ばれます。

    Args:
        pos (np.ndarray): 粒子の位置ベクトル [x, y, z] [m]
        vel (np.ndarray): 粒子の速度ベクトル [vx, vy, vz] [m/s]
        Vms_ms (float): 水星の公転速度 [m/s]
        AU (float): 太陽距離 [天文単位]
        spec_data (dict): 太陽スペクトル関連データ
        settings (dict): シミュレーション設定

    Returns:
        np.ndarray: 加速度ベクトル [ax, ay, az] [m^2/s]
    """
    x, y, z = pos
    vx, vy, vz = vel

    # --- 1. 放射圧の計算 ---
    # ドップラー効果を考慮した波長を計算
    velocity_for_doppler = vx + Vms_ms
    w_na_d2 = 589.1582e-9 * (1.0 - velocity_for_doppler / PHYSICAL_CONSTANTS['C'])
    w_na_d1 = 589.7558e-9 * (1.0 - velocity_for_doppler / PHYSICAL_CONSTANTS['C'])

    # スペクトルデータから該当波長のフラックスを内挿
    wl, gamma, sigma0_perdnu1, sigma0_perdnu2, JL = spec_data.values()
    if not (wl[0] * 1e-9 <= w_na_d2 < wl[-1] * 1e-9 and wl[0] * 1e-9 <= w_na_d1 < wl[-1] * 1e-9):
        b = 0.0  # スペクトル範囲外なら放射圧は0
    else:
        gamma2 = np.interp(w_na_d2 * 1e9, wl, gamma)
        gamma1 = np.interp(w_na_d1 * 1e9, wl, gamma)
        F_lambda_1AU_m = JL * 1e9

        # D1, D2線それぞれの散乱率 (J) を計算
        F_lambda_d1_at_Mercury = F_lambda_1AU_m / (AU ** 2) * gamma1
        F_nu_d1 = F_lambda_d1_at_Mercury * (w_na_d1 ** 2) / PHYSICAL_CONSTANTS['C']
        J1 = sigma0_perdnu1 * F_nu_d1
        F_lambda_d2_at_Mercury = F_lambda_1AU_m / (AU ** 2) * gamma2
        F_nu_d2 = F_lambda_d2_at_Mercury * (w_na_d2 ** 2) / PHYSICAL_CONSTANTS['C']
        J2 = sigma0_perdnu2 * F_nu_d2

        # 散乱率から放射圧による加速度係数 (b) を計算
        b = 1 / PHYSICAL_CONSTANTS['MASS_NA'] * (
                    (PHYSICAL_CONSTANTS['H'] / w_na_d1) * J1 + (PHYSICAL_CONSTANTS['H'] / w_na_d2) * J2)

    # 水星の影に入っている場合は放射圧を0にする
    if x < 0 and np.sqrt(y ** 2 + z ** 2) < PHYSICAL_CONSTANTS['RM']:
        b = 0.0

    accel_srp = np.array([-b, 0.0, 0.0])  # 放射圧は太陽と反対方向 (-X方向)

    # --- 2. 重力加速度の計算 ---
    accel_g = np.array([0.0, 0.0, 0.0])
    if settings['GRAVITY_ENABLED']:
        r_sq_grav = x ** 2 + y ** 2 + z ** 2
        if r_sq_grav > 0:
            r_grav = np.sqrt(r_sq_grav)
            grav_accel_total = -PHYSICAL_CONSTANTS['GM_MERCURY'] / r_sq_grav
            accel_g = grav_accel_total * (pos / r_grav)

    # --- 3. 合成加速度 ---
    return accel_srp + accel_g


# --- シミュレーションの本体 ---
def simulate_single_particle_for_density(args):
    """
    一個のナトリウム原子の挙動を追跡し、グリッドへの滞在時間と最終的な消滅要因を返す。

    Args:
        args (dict): シミュレーションに必要な全てのパラメータを含む辞書。
            'settings', 'spec', 'orbit', 'grid_params' のキーを持つ。

    Returns:
        tuple: (local_density_grid, death_reason)
            - local_density_grid (np.ndarray): この粒子が各グリッドセルに滞在した時間の合計。
            - death_reason (str): 粒子の消滅要因 ('ionized', 'stuck', 'escaped')。
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

    # --- 粒子の初期化 ---
    # 粒子消滅法の場合、重みは常に1
    particle_weight = 1.0

    # 経度方向(phi)を日照側(-pi/2 から +pi/2)でサンプリング
    phi_source = PHYSICAL_CONSTANTS['PI'] * np.random.random() - (PHYSICAL_CONSTANTS['PI'] / 2.0)
    # 緯度方向を示すcos(theta)は全球(-1から1)でサンプリング
    cos_theta_source = 2 * np.random.random() - 1.0
    sin_theta_source = np.sqrt(1.0 - cos_theta_source ** 2)

    # 初期位置 (水星表面)
    pos = np.array([
        RM * sin_theta_source * np.cos(phi_source),
        RM * sin_theta_source * np.sin(phi_source),
        RM * cos_theta_source
    ])

    # 角度依存の重み (放出点が太陽方向を向いている度合い)
    weight_cos_z = pos[0] / RM

    # 初期速度
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

    # --- 時間発展ループ ---
    tau = settings['T1AU'] * AU ** 2  # 電離寿命
    itmax = int(tau * 5.0 / DT + 0.5)  # シミュレーション最大時間 (電離寿命の5倍)
    death_reason = 'ionized'  # デフォルトの死因

    for it in range(itmax):

        # --- 電離プロセス ---
        ionization_prob_per_step = 1.0 - np.exp(-DT / tau)
        if IONIZATION_MODEL == 'particle_death':
            if np.random.random() < ionization_prob_per_step:
                death_reason = 'ionized'
                break  # ループを抜けて粒子を消滅させる

        Nad = 1.0  # 重み
        if IONIZATION_MODEL == 'weight_decay':
            Nad = np.exp(-DT * it / tau)  # 時間経過で重みを減衰させる

        # --- 軌道計算 (4次ルンゲ＝クッタ法) ---
        # k1
        k1_vel = DT * _calculate_acceleration(pos, vel, Vms_ms, AU, spec_data, settings)
        k1_pos = DT * vel
        # k2
        k2_vel = DT * _calculate_acceleration(pos + 0.5 * k1_pos, vel + 0.5 * k1_vel, Vms_ms, AU, spec_data, settings)
        k2_pos = DT * (vel + 0.5 * k1_vel)
        # k3
        k3_vel = DT * _calculate_acceleration(pos + 0.5 * k2_pos, vel + 0.5 * k2_vel, Vms_ms, AU, spec_data, settings)
        k3_pos = DT * (vel + 0.5 * k2_vel)
        # k4
        k4_vel = DT * _calculate_acceleration(pos + k3_pos, vel + k3_vel, Vms_ms, AU, spec_data, settings)
        k4_pos = DT * (vel + k3_vel)

        # 位置と速度の更新
        pos_prev = pos.copy()  # 衝突判定用に更新前の位置を保持
        pos += (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos) / 6.0
        vel += (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel) / 6.0

        # --- グリッドへの滞在時間記録 ---
        r = np.linalg.norm(pos)
        if r == 0: continue
        if r >= R_MAX:
            death_reason = 'escaped'
            break

        # 球面座標系に変換してグリッドインデックスを計算
        theta = np.arccos(pos[2] / r)
        phi = np.arctan2(pos[1], pos[0])
        ir = int(r / DR)
        itheta = int(theta / D_THETA)
        iphi = int((phi + PHYSICAL_CONSTANTS['PI']) / D_PHI)

        # グリッド内にいれば滞在時間を加算
        if 0 <= ir < N_R and 0 <= itheta < N_THETA and 0 <= iphi < N_PHI:
            final_weight = weight_cos_z * Nad if IONIZATION_MODEL == 'weight_decay' else weight_cos_z
            local_density_grid[ir, itheta, iphi] += final_weight * DT

        # --- 表面との衝突判定と処理 ---
        r_current = np.linalg.norm(pos)
        if r_current <= RM:
            # 衝突点の表面温度と吸着確率を計算
            temp_at_impact = calculate_surface_temperature(pos_prev[0], pos_prev[1], pos_prev[2], AU)
            stick_prob = calculate_sticking_probability(temp_at_impact)

            if np.random.random() < stick_prob:
                death_reason = 'stuck'
                break  # 吸着して消滅

            # 吸着しない場合、熱的に accomodate して再放出される
            E_in = 0.5 * MASS_NA * np.sum(vel ** 2)
            E_T = K_BOLTZMANN * temp_at_impact
            E_out = settings['BETA'] * E_T + (1.0 - settings['BETA']) * E_in
            v_out_sq = E_out / (0.5 * MASS_NA)
            v_out_speed = np.sqrt(v_out_sq) if v_out_sq > 0 else 0.0

            # 等方的に再放出
            impact_normal = pos_prev / np.linalg.norm(pos_prev)
            rebound_direction = sample_isotropic_direction(impact_normal)
            vel = v_out_speed * rebound_direction

            # 粒子を表面に正確に戻す
            pos = RM * impact_normal

    return (local_density_grid, death_reason)


def main():
    """
    水星ナトリウム大気のモンテカルロシミュレーションを実行するメイン関数。
    """
    # --- シミュレーション設定 ---
    OUTPUT_DIRECTORY = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult3D"
    N_R = 100  # 半径方向のグリッド数
    N_THETA = 24  # 天頂角方向のグリッド数
    N_PHI = 24  # 方位角方向のグリッド数
    GRID_RADIUS_RM = 5.0  # シミュレーション空間の半径 (水星半径単位)
    N_PARTICLES = 10000  # 各TAAでシミュレートする粒子数

    # シミュレーションの挙動を制御する設定
    settings = {
        'GRAVITY_ENABLED': True,
        'BETA': 0.5,  # 表面衝突時の熱 accomodation 係数 (0:弾性衝突, 1:完全な熱化)
        #'T1AU': 61728.4,  # 1AUでの電離寿命 [s] (実験値)
        'T1AU': 1.9*1e5,  # 1AUでの電離寿命 [s] (理論値) Suzuki(2019)
        'DT': 10.0,  # 時間ステップ [s]
        'speed_distribution': 'maxwellian',  # 'maxwellian' or 'weibull'
        'ejection_direction_model': 'isotropic',  # 'cosine' or 'isotropic'
        'ionization_model': 'particle_death'  # 'weight_decay' or 'particle_death'
    }

    # --- ファイル名とディレクトリ設定 ---
    dist_tag = "CO" if settings['ejection_direction_model'] == 'cosine' else "ISO"
    speed_tag = "MW" if settings['speed_distribution'] == 'maxwellian' else "WB"
    ion_tag = "WD" if settings['ionization_model'] == 'weight_decay' else "PD"
    base_name_template = f"density3d_beta{settings['BETA']:.2f}_Q3.0_{speed_tag}_{dist_tag}_{ion_tag}_pl{N_THETA}x{N_PHI}"

    sub_folder_name = base_name_template
    target_output_dir = os.path.join(OUTPUT_DIRECTORY, sub_folder_name)
    os.makedirs(target_output_dir, exist_ok=True)
    print(f"結果は '{target_output_dir}' に保存されます。")

    log_file_path = os.path.join(target_output_dir, "death_statistics_Q3.0.csv")
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

    # 波長データがソートされていることを保証 (np.interp のため)
    if not np.all(np.diff(wl) > 0):
        sort_indices = np.argsort(wl)
        wl, gamma = wl[sort_indices], gamma[sort_indices]

    # 散乱断面積の計算に必要な定数を準備
    sigma_const = PHYSICAL_CONSTANTS['E_CHARGE'] ** 2 / (
                4 * PHYSICAL_CONSTANTS['ME'] * PHYSICAL_CONSTANTS['C'] * PHYSICAL_CONSTANTS['EPSILON_0'])
    spec_data_dict = {
        'wl': wl, 'gamma': gamma,
        'sigma0_perdnu1': sigma_const * 0.320,  # D1線の振動子強度を乗算
        'sigma0_perdnu2': sigma_const * 0.641,  # D2線の振動子強度を乗算
        'JL': 5.18e14 * 1e4  # 1AUでの太陽フラックス [phs/s/m2/nm]
    }

    # --- TAAごとのシミュレーションループ ---
    for line in orbit_lines:
        TAA, AU, lon, lat, Vms_ms = map(float, line.split())
        print(f"\n--- TAA = {TAA:.1f}度のシミュレーションを開始 ---")

        # --- このTAAにおける総放出率の計算 ---
        F_UV_at_1AU_per_m2 = 1.5e14 * 1e4  # 1AUでの紫外線光子フラックス [photons/m^2/s]
        Q_PSD_m2 = 3.0e-20 / 1e4  # 光脱離断面積 [m^2]
        # Q_PSD_cm2 = 1.0e-20 / 1e4  # 光脱離断面積 [m^2]
        # Q_PSD_cm2 = 2.0e-20 / 1e4 # suzukiが使ってたやつ
        # Q_PSD_cm2 = 3.0e-20 / 1e4  # YakshinskiyとMadey（1999）
        # Q_PSD_cm2 = 1.4e-21 / 1e4 # Killenら（2004）
        # cNa = 0.053 * 7.5e14 * 1e4 # Moroni (2023) 水星表面のナトリウム原子の割合
        cNa_per_m2 = 1.5e13 * 1e4  # 表面のナトリウム原子のカラム密度 [atoms/m^2]

        F_UV_current_per_m2 = F_UV_at_1AU_per_m2 / (AU ** 2)
        R_PSD_peak_per_m2 = F_UV_current_per_m2 * Q_PSD_m2 * cNa_per_m2  # ピーク放出率 [atoms/m^2/s]

        # 太陽に照らされている半球の面積を考慮
        effective_area_m2 = np.pi * (PHYSICAL_CONSTANTS['RM'] ** 2)
        total_flux_for_this_taa = R_PSD_peak_per_m2 * effective_area_m2  # 総放出率 [atoms/s]

        # --- 並列処理の準備 ---
        task_args = {
            'settings': settings, 'spec': spec_data_dict,
            'orbit': (TAA, AU, lon, lat, Vms_ms), 'grid_params': grid_params
        }
        tasks = [task_args] * N_PARTICLES

        # --- 並列計算の実行 ---
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

        # --- 死因統計の表示と記録 ---
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

        # --- 数密度の計算と保存 ---
        # 1. 各グリッドセルの体積を計算
        r_edges = np.linspace(0, grid_params['max_r'], N_R + 1)
        theta_edges = np.linspace(0, PHYSICAL_CONSTANTS['PI'], N_THETA + 1)

        delta_r3_3 = (r_edges[1:] ** 3 - r_edges[:-1] ** 3) / 3.0
        delta_cos_theta = np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:])
        D_PHI_rad = 2 * PHYSICAL_CONSTANTS['PI'] / N_PHI

        # ブロードキャストを利用して全セルの体積を一度に計算
        cell_volumes_m3 = delta_r3_3[:, np.newaxis, np.newaxis] * \
                          delta_cos_theta[np.newaxis, :, np.newaxis] * \
                          D_PHI_rad
        cell_volumes_m3[cell_volumes_m3 == 0] = 1e-30  # ゼロ除算を防止

        # 2. 数密度 [atoms/m^3] を計算
        # (総放出率 / テスト粒子数) は、テスト粒子1個が持つ「本物の粒子数」の重みを表す
        # (グリッド上の滞在時間 / セルの体積) は、時間平均された数密度への寄与を表す
        number_density_m3 = (total_flux_for_this_taa / N_PARTICLES) * (master_density_grid / cell_volumes_m3)

        # 3. 最終出力のために [atoms/cm^3] に変換して保存
        number_density_cm3 = number_density_m3 / 1e6

        base_filename = f"density3d_taa{TAA:.0f}_{base_name_template.replace('density3d_', '')}"
        full_path_npy = os.path.join(target_output_dir, f"{base_filename}.npy")
        np.save(full_path_npy, number_density_cm3)
        print(f"結果を {full_path_npy} に保存しました。")

    print("\nすべてのシミュレーションが完了しました。")


if __name__ == '__main__':
    main()