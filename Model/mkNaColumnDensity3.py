import numpy as np
import sys
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import matplotlib.pyplot as plt


# --- 物理モデルに基づく関数 ---

def calculate_surface_temperature(x, y, AU):
    """
    水星表面の座標(x, y)と太陽距離(AU)から局所的な表面温度を計算する。
    太陽直下点を(x>0, y=0)とする。
    """
    T0 = 100.0  # 夜側の最低温度 [K]
    T1 = 600.0  # 太陽直下点での最大温度上昇 [K]
    if x <= 0:
        return T0
    # 太陽方向ベクトルと位置ベクトルのなす角の余弦を計算
    cos_theta = x / np.sqrt(x ** 2 + y ** 2)
    if cos_theta < 0:
        return T0
    # 表面温度の経験式
    temp = T0 + T1 * (cos_theta ** 0.25) * ((0.306 / AU) ** 2)
    return temp


def calculate_sticking_probability(surface_temp_K):
    """
    表面温度に基づいて、ナトリウム原子の表面への吸着確率を計算する。
    """
    # 論文で使用されている定数
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
                         # U_ev = 0.0098, #Killen et al., 2007
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


def sample_cosine_angle():
    """
    コサイン則に従う放出角度をサンプリングする（ランバート反射）。
    """
    p = np.random.random()
    # 逆関数法
    return np.arcsin(2 * p - 1)


# --- シミュレーションのコア関数 (改造版) ---
def simulate_single_particle_for_density(args):
    """
    一個の粒子を追跡し、その軌跡を極座標密度グリッドに記録する。
    """
    # 引数をアンパック
    constants, settings, spec_data = args['consts'], args['settings'], args['spec']
    TAA, AU, lon, lat, Vms_ms = args['orbit']
    grid_params = args['grid_params']

    # 物理定数と設定をアンパック
    C, PI, H, MASS_NA, RM, GM_MERCURY, K_BOLTZMANN = constants.values()
    GRAVITY_ENABLED, BETA, T1AU, DT, SPEED_DISTRIBUTION = settings.values()
    wl, gamma, sigma0_perdnu2, sigma0_perdnu1, JL = spec_data.values()

    # <--- 極座標グリッドのパラメータをアンパック ---
    N_R, N_THETA, R_MAX = grid_params['n_r'], grid_params['n_theta'], grid_params['max_r']
    DR = R_MAX / N_R  # 半径方向のセルサイズ
    D_THETA = 2 * PI / N_THETA  # 角度方向のセルサイズ (ラジアン)

    # <--- グリッドの形状を(半径方向, 角度方向)に変更 ---
    local_density_grid = np.zeros((N_R, N_THETA), dtype=np.float32)

    # 放出位置
    p = np.random.random()

    # 太陽直下点からの角度θを、cos(θ)分布に従ってサンプリングする
    # CDF: p = sin^2(θ)  =>  θ = arcsin(sqrt(p))
    source_angle_rad = np.arcsin(np.sqrt(p))

    # Y軸の正負をランダムに割り振る (半球の上下に均等に分布させるため)
    if np.random.random() < 0.5:
        source_angle_rad *= -1.0

    x = RM * np.cos(source_angle_rad)
    y = RM * np.sin(source_angle_rad)

    # 設定に応じて放出速度の分布を切り替える
    if SPEED_DISTRIBUTION == 'maxwellian':
        # 論文記載の1500Kを使用
        ejection_speed = sample_maxwellian_speed(MASS_NA, 1500.0)  # 温度[K]
    elif SPEED_DISTRIBUTION == 'weibull':
        ejection_speed = sample_weibull_speed(MASS_NA)
    else:
        # デフォルトとしてWeibullを使用
        ejection_speed = sample_weibull_speed(MASS_NA)

    # 表面の法線方向に対してコサイン則で放出
    surface_normal_angle = np.arctan2(y, x)
    random_offset_angle = sample_cosine_angle()
    ejection_angle_rad = surface_normal_angle + random_offset_angle
    vx_ms = ejection_speed * np.cos(ejection_angle_rad)
    vy_ms = ejection_speed * np.sin(ejection_angle_rad)

    Vms = Vms_ms
    # 現在の太陽距離(AU)での電離寿命を計算
    tau = T1AU * AU ** 2
    # シミュレーションの最大時間ステップ数を設定 (電離寿命の5倍)
    itmax = int(tau * 5.0 / DT + 0.5)

    for it in range(itmax):
        # 視線速度（水星の公転速度 + ナトリウム原子の速度）を計算
        velocity_for_doppler = vx_ms + Vms
        # ドップラーシフトを考慮した波長を計算
        w_na_d2 = 589.1582e-9 * (1.0 - velocity_for_doppler / C)
        w_na_d1 = 589.7558e-9 * (1.0 - velocity_for_doppler / C)

        # スペクトルデータの範囲外になったらループを抜ける
        if not (wl[0] * 1e-9 <= w_na_d2 < wl[-1] * 1e-9 and wl[0] * 1e-9 <= w_na_d1 < wl[-1] * 1e-9):
            break

        # 波長に対応する太陽スペクトルの値を線形補間で取得
        gamma2 = np.interp(w_na_d2 * 1e9, wl, gamma)
        gamma1 = np.interp(w_na_d1 * 1e9, wl, gamma)

        # 1AUでの波長あたりの太陽光子フラックス [photons/s/m^3]
        F_lambda_1AU_m = JL * 1e9

        # --- 放射圧の計算 ---
        # --- D1線に対する計算 ---
        # 水星位置での波長あたりの光子フラックス [photons/s/m^3]
        F_lambda_d1_at_Mercury = F_lambda_1AU_m / (AU ** 2) * gamma1
        # 水星位置での周波数あたりの光子フラックス [photons/s/m^2/Hz]
        F_nu_d1 = F_lambda_d1_at_Mercury * (w_na_d1 ** 2) / C
        # D1線の散乱率 J1 [1/s]
        J1 = sigma0_perdnu1 * F_nu_d1

        # --- D2線に対する計算 ---
        # 水星位置での波長あたりの光子フラックス [photons/s/m^3]
        F_lambda_d2_at_Mercury = F_lambda_1AU_m / (AU ** 2) * gamma2
        # 水星位置での周波数あたりの光子フラックス [photons/s/m^2/Hz]
        F_nu_d2 = F_lambda_d2_at_Mercury * (w_na_d2 ** 2) / C
        # D2線の散乱率 J2 [1/s]
        J2 = sigma0_perdnu2 * F_nu_d2

        # 2. 散乱率 J から加速度 b [m/s^2] を計算
        #    加速度 b = (原子が吸収した光子の運動量 p) * (散乱率 J) / (原子の質量 m)
        #    運動量 p = h/λ = h*ν/c
        b = 1 / MASS_NA * ((H / w_na_d1) * J1 + (H / w_na_d2) * J2)

        # 水星の影に入った場合は放射圧を0にする
        if x < 0 and np.sqrt(y ** 2) < RM:
            b = 0.0

        # 電離による原子数の減衰を考慮
        Nad = np.exp(-DT * it / tau)

        # --- 重力加速度の計算 ---
        accel_gx, accel_gy = 0.0, 0.0
        if GRAVITY_ENABLED:
            r_sq_grav = x ** 2 + y ** 2
            if r_sq_grav > 0:
                r_grav = np.sqrt(r_sq_grav)
                grav_accel_total = GM_MERCURY / r_sq_grav
                accel_gx = -grav_accel_total * (x / r_grav)
                accel_gy = -grav_accel_total * (y / r_grav)

        vx_ms_prev, vy_ms_prev = vx_ms, vy_ms

        # 放射圧の方向は常に太陽と反対方向 (x軸の負の方向)
        accel_srp_x = -b

        # --- 運動方程式を解く (リープ・フロッグ法) ---
        total_accel_x, total_accel_y = accel_srp_x + accel_gx, accel_gy
        vx_ms += total_accel_x * DT
        vy_ms += total_accel_y * DT
        x_prev, y_prev = x, y
        x += ((vx_ms_prev + vx_ms) / 2.0) * DT
        y += ((vy_ms_prev + vy_ms) / 2.0) * DT

        # <--- デカルト座標から極座標への変換とインデックス計算 ---
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)  # -pi から +pi の範囲

        # 座標をグリッドのインデックスに変換
        ir = int(r / DR)
        # thetaを[0, 2*pi]の範囲に変換し、インデックスを計算
        itheta = int((theta + PI) / D_THETA)

        # グリッドの範囲内にあるかチェックして加算
        # Nad * DT は、この時間ステップでこのセルに粒子が存在した「延べ時間」に相当
        if 0 <= ir < N_R and 0 <= itheta < N_THETA:
            local_density_grid[ir, itheta] += Nad * DT
        # <--- 変更点ここまで ---

        # --- 地表衝突時の処理  ---
        r_current = np.sqrt(x ** 2 + y ** 2)
        if r_current <= RM:
            # 1. 衝突地点の局所的な表面温度を計算
            temp_at_impact = calculate_surface_temperature(x_prev, y_prev, AU)
            # 2. その温度における吸着確率を計算
            stick_prob = calculate_sticking_probability(temp_at_impact)
            # 3. 吸着判定
            if np.random.random() < stick_prob:
                break  # 吸着したら追跡終了
            # 4. 反射する場合: 局所温度を使って反射エネルギーを計算
            v_in_sq = vx_ms_prev ** 2 + vy_ms_prev ** 2
            E_in = 0.5 * MASS_NA * (v_in_sq)
            E_T = K_BOLTZMANN * temp_at_impact  # ★ 局所温度を使用
            # BETA係数に応じてエネルギーを部分的に失う
            E_out = BETA * E_T + (1.0 - BETA) * E_in
            v_out_sq = E_out / (0.5 * MASS_NA)
            v_out_speed = np.sqrt(v_out_sq) if v_out_sq > 0 else 0.0
            # 反射角はランダム（等方的な反射を仮定）
            surface_angle = np.arctan2(y_prev, x_prev)
            rebound_angle_rad = surface_angle + np.random.uniform(-PI / 2.0, PI / 2.0)
            vx_ms = v_out_speed * np.cos(rebound_angle_rad)
            vy_ms = v_out_speed * np.sin(rebound_angle_rad)
            # 衝突位置を表面に補正
            x = RM * np.cos(surface_angle)
            y = RM * np.sin(surface_angle)

    return local_density_grid


# --- メインの制御関数 ---
def main():
    # --- ★★★ 設定項目 ★★★ ---
    OUTPUT_DIRECTORY = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult"

    # <--- グリッド定義を極座標用に変更 ---
    N_R = 100  # 半径方向の分割数
    N_THETA = 24  # 角度方向の分割数
    GRID_RADIUS_RM = 5.0  # グリッドの最大半径（水星半径の何倍か）
    # <--- 変更点ここまで ---

    N_PARTICLES = 10000  # 粒子数

    settings = {
        'GRAVITY_ENABLED': True,
        'BETA': 0.5,  # 水星表面との衝突での係数 0で弾性衝突、1で完全にエネルギーを失う　
        # 理想的な石英表面において、ナトリウムではβ≈0.62、カリウムではβ≈0.26
        'T1AU': 61728.4,  # 電離寿命　実験値 [s]
        'DT': 10.0,  # 時間ステップ [s]
        'speed_distribution': 'maxwellian',  # 'maxwellian' または 'weibull'
    }
    # --- 設定はここまで ---

    # --- ファイル名と保存先フォルダの設定 ---
    base_name_template = f"density_map_beta{settings['BETA']:.2f}_Q1.0_MW_pl{N_THETA}"
    sub_folder_name = base_name_template
    target_output_dir = os.path.join(OUTPUT_DIRECTORY, sub_folder_name)
    os.makedirs(target_output_dir, exist_ok=True)
    print(f"結果は '{target_output_dir}' に保存されます。")

    constants = {
        'C': 299792458.0,  # 光速 [m/s]
        'PI': np.pi,
        'H': 6.62607015e-34,  # プランク定数 [kg・m^2/s] (J・s)
        'MASS_NA': 22.98976928 * 1.66054e-27,  # Na原子の質量 [kg]
        'RM': 2439.7e3,  # 水星の半径 [m]
        'GM_MERCURY': 2.2032e13,  # G*M_mercury [m^3/s^2]  (万有引力定数 * 水星の質量)
        'K_BOLTZMANN': 1.380649e-23  # ボルツマン定数[J/K]
    }

    # <--- grid_params辞書を極座標用に更新 ---
    grid_params = {
        'n_r': N_R,
        'n_theta': N_THETA,
        'max_r': constants['RM'] * GRID_RADIUS_RM
    }
    # <--- 変更点ここまで ---

    # --- 外部データファイルの読み込み ---
    try:
        spec_data_np = np.loadtxt('SolarSpectrum_Na0.txt', usecols=(0, 3))
        wl, gamma = spec_data_np[:, 0], spec_data_np[:, 1]
        orbit_lines = open('orbit360.txt', 'r').readlines()
    except FileNotFoundError as e:
        print(f"エラー: データファイルが見つかりません - {e}")
        sys.exit()

    # 波長データが昇順でない場合、ソートする
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

    # --- TAAごとのループ処理 ---
    for line in orbit_lines:
        TAA, AU, lon, lat, Vms_ms = map(float, line.split())
        print(f"\n--- TAA = {TAA:.1f}度のシミュレーションを開始 ---")

        # --- このTAAでの総放出率を計算 ---
        # 1. 論文で使われている物理定数を定義
        F_UV_at_1AU_per_cm2 = 1.5e14 * 1e4  # 1天文単位での紫外線光子フラックス [photons/m^2/s]
        Q_PSD_cm2 = 1.0e-20 / 1e4  # 光脱離断面積 [m^2]
        RM_m = constants['RM']  # 水星半径 [m]
        cNa = 1.5e13 * 1e4  # 表面ナトリウム原子数密度 [atoms/m^2]

        # 2. 現在の太陽距離(AU)における太陽直下点での最大放出率を計算
        F_UV_current_per_cm2 = F_UV_at_1AU_per_cm2 / (AU ** 2)
        R_PSD_peak_per_cm2 = F_UV_current_per_cm2 * Q_PSD_cm2 * cNa  # [atoms/m^2/s]

        # 3. 太陽に照らされた半球全体で積分し、総放出量を計算
        # cos分布を半球で積分した際の有効面積は π * R^2 となる
        effective_area_cm2 = np.pi * (RM_m ** 2)
        total_flux_for_this_taa = R_PSD_peak_per_cm2 * effective_area_cm2  # [atoms/sec]

        # --- 並列計算の準備 ---
        task_args = {
            'consts': constants, 'settings': settings, 'spec': spec_data_dict,
            'orbit': (TAA, AU, lon, lat, Vms_ms), 'grid_params': grid_params
        }
        tasks = [task_args] * N_PARTICLES

        # --- 並列計算の実行 ---
        with Pool(processes=cpu_count()) as pool:
            results = list(tqdm(pool.imap(simulate_single_particle_for_density, tasks), total=N_PARTICLES,
                                desc=f"TAA={TAA:.1f}"))

        print("結果を集計・保存しています...")
        # 全てのプロセスからの結果（グリッド）を合計する
        master_density_grid = np.sum(results, axis=0)

        # <--- 極座標のセル面積を計算し、柱密度を算出 ---
        # 各セルの内側と外側の半径を計算
        R_MAX = grid_params['max_r']
        DR = R_MAX / N_R
        D_THETA = 2 * np.pi / N_THETA

        # 半径インデックス ir に対応するセルの面積を計算
        # Area = (θ/2) * (r_outer^2 - r_inner^2)
        # ここでθは D_THETA, r_inner = ir*DR, r_outer = (ir+1)*DR
        r_indices = np.arange(N_R)
        r_inner = r_indices * DR
        r_outer = (r_indices + 1) * DR
        cell_areas_1d = (D_THETA / 2.0) * (r_outer ** 2 - r_inner ** 2)

        # ゼロ除算を避けるため、面積が0の場合は小さな値に置き換える
        cell_areas_1d[cell_areas_1d == 0] = 1e-10

        # 1Dの面積配列をブロードキャストして2Dグリッドに適用
        cell_areas_m2 = np.tile(cell_areas_1d[:, np.newaxis], (1, N_THETA))

        # 柱密度 [atoms/m^2] を計算
        # (総放出率 / テスト粒子数) * (グリッド上の延べ時間 / セルの面積)
        column_density_m2 = (total_flux_for_this_taa / N_PARTICLES) * (master_density_grid / cell_areas_m2)

        # 最終出力のために [atoms/cm^2] に変換
        column_density_cm2 = column_density_m2 / 1e4
        # <--- 変更点ここまで ---

        # --- 結果の保存 ---
        parameter_part = base_name_template.replace("density_map_", "")
        base_filename = f"density_map_taa{TAA:.0f}_{parameter_part}"
        full_path_npy = os.path.join(target_output_dir, f"{base_filename}.npy")
        np.save(full_path_npy, column_density_cm2)
        print(f"結果を {full_path_npy} に保存しました。")

    print("\n★★★ すべてのシミュレーションが完了しました ★★★")


if __name__ == '__main__':
    main()