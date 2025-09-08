import numpy as np
import sys
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import matplotlib.pyplot as plt


# --- 物理モデルに基づく関数---

def calculate_surface_temperature(x, y, AU):
    T0 = 100.0
    T1 = 600.0
    if x <= 0:
        return T0
    cos_theta = x / np.sqrt(x ** 2 + y ** 2)
    if cos_theta < 0:
        return T0
    temp = T0 + T1 * (cos_theta ** 0.25) * ((0.306 / AU) ** 2)
    return temp


def calculate_sticking_probability(surface_temp_K):
    # 論文で使用されている定数
    A = 0.08
    B = 458.0
    porosity = 0.8
    p_stick = A * np.exp(B / surface_temp_K)
    p_stick_eff = p_stick / (1.0 - (1.0 - p_stick) * porosity)
    return p_stick_eff


# --- 物理モデルに基づくサンプリング関数 ---
def sample_weibull_speed(mass_kg,
                         U_ev=0.05,  # Leblanc et al., 2022
                         #U_ev = 0.0098, #Killen et al., 2007
                         beta_shape=0.7):

    E_CHARGE = 1.602176634e-19 # 電子の電荷 [C]
    p = np.random.random()
    E_ev = U_ev * (p ** (-1.0 / (beta_shape + 1.0)) - 1.0)
    E_joule = E_ev * E_CHARGE
    v_ms = np.sqrt(2 * E_joule / mass_kg)
    return v_ms


def sample_cosine_angle():
    p = np.random.random()
    return np.arcsin(2 * p - 1)


# --- シミュレーションのコア関数 (改造版) ---
def simulate_single_particle_for_density(args):
    constants, settings, spec_data = args['consts'], args['settings'], args['spec']
    TAA, AU, lon, lat, Vms_ms = args['orbit']
    grid_params = args['grid_params']
    C, PI, H, MASS_NA, RM, GM_MERCURY, K_BOLTZMANN = constants.values()
    GRAVITY_ENABLED, BETA, T1AU, DT = settings.values()
    wl, gamma, sigma0_perdnu2, sigma0_perdnu1, JL = spec_data.values()
    GRID_SIZE, GRID_MAX_R = grid_params['size'], grid_params['max_r']
    CELL_SIZE = 2 * GRID_MAX_R / GRID_SIZE
    local_density_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    #放出位置
    p = np.random.random()

    # 太陽直下点からの角度θを、cos(θ)分布に従ってサンプリングする
    # CDF: p = sin^2(θ)  =>  θ = arcsin(sqrt(p))
    source_angle_rad = np.arcsin(np.sqrt(p))

    # Y軸の正負をランダムに割り振る (半球の上下に均等に分布させるため)
    if np.random.random() < 0.5:
        source_angle_rad *= -1.0

    x = RM * np.cos(source_angle_rad)
    y = RM * np.sin(source_angle_rad)
    ejection_speed = sample_weibull_speed(MASS_NA)
    surface_normal_angle = np.arctan2(y, x)
    random_offset_angle = sample_cosine_angle()
    ejection_angle_rad = surface_normal_angle + random_offset_angle
    vx_ms = ejection_speed * np.cos(ejection_angle_rad)
    vy_ms = ejection_speed * np.sin(ejection_angle_rad)
    Vms = Vms_ms
    tau = T1AU * AU ** 2
    itmax = int(tau * 5.0 / DT + 0.5)
    for it in range(itmax):
        velocity_for_doppler = vx_ms + Vms
        w_na_d2 = 589.1582e-9 * (1.0 - velocity_for_doppler / C)
        w_na_d1 = 589.7558e-9 * (1.0 - velocity_for_doppler / C)
        if not (wl[0]*1e-9 <= w_na_d2 < wl[-1]*1e-9 and wl[0]*1e-9 <= w_na_d1 < wl[-1]*1e-9):
            break
        gamma2 = np.interp(w_na_d2*1e9, wl, gamma)
        gamma1 = np.interp(w_na_d1*1e9, wl, gamma)
        m_na_wl = (w_na_d2 + w_na_d1) / 2.0
        jl_nu = JL * 1e9 * ((m_na_wl) ** 2 / (C)) #[photons/m^2]
        J2 = sigma0_perdnu2 * jl_nu / AU ** 2 * gamma2 #[photons/s]
        J1 = sigma0_perdnu1 * jl_nu / AU ** 2 * gamma1 #[photons/s]
        b = (H / MASS_NA) * (J1 / (w_na_d1) + J2 / (w_na_d2)) #[m/s^2]
        if x < 0 and np.sqrt(y ** 2) < RM:
            b = 0.0
        Nad = np.exp(-DT * it / tau)
        accel_gx, accel_gy = 0.0, 0.0
        if GRAVITY_ENABLED:
            r_sq_grav = x ** 2 + y ** 2
            if r_sq_grav > 0:
                r_grav = np.sqrt(r_sq_grav)
                grav_accel_total = GM_MERCURY / r_sq_grav
                accel_gx = -grav_accel_total * (x / r_grav)
                accel_gy = -grav_accel_total * (y / r_grav)
        vx_ms_prev, vy_ms_prev = vx_ms, vy_ms
        accel_srp_x = b
        total_accel_x, total_accel_y = accel_srp_x + accel_gx, accel_gy
        vx_ms += total_accel_x * DT
        vy_ms += total_accel_y * DT
        x_prev, y_prev = x, y
        x += ((vx_ms_prev + vx_ms) / 2.0) * DT
        y += ((vy_ms_prev + vy_ms) / 2.0) * DT
        ix = int((x + GRID_MAX_R) / CELL_SIZE)
        iy = int((y + GRID_MAX_R) / CELL_SIZE)
        if 0 <= ix < GRID_SIZE and 0 <= iy < GRID_SIZE:
            local_density_grid[iy, ix] += Nad * DT

        # --- 地表衝突時の処理  ---
        r_current = np.sqrt(x ** 2 + y ** 2)
        if r_current <= RM:
            # 1. 衝突地点の局所的な表面温度を計算
            temp_at_impact = calculate_surface_temperature(x_prev, y_prev, AU)
            # 2. その温度における吸着確率を計算
            stick_prob = calculate_sticking_probability(temp_at_impact)
            # 3. 吸着判定
            if np.random.random() < stick_prob:
                break
            # 4. 反射する場合: 局所温度を使って反射エネルギーを計算
            v_in_sq = vx_ms_prev ** 2 + vy_ms_prev ** 2
            E_in = 0.5 * MASS_NA * (v_in_sq)
            E_T = K_BOLTZMANN * temp_at_impact  # ★ 局所温度を使用
            E_out = BETA * E_T + (1.0 - BETA) * E_in
            v_out_sq = E_out / (0.5 * MASS_NA)
            v_out_speed = np.sqrt(v_out_sq) if v_out_sq > 0 else 0.0
            surface_angle = np.arctan2(y_prev, x_prev)
            rebound_angle_rad = surface_angle + np.random.uniform(-PI / 2.0, PI / 2.0)
            vx_ms = v_out_speed * np.cos(rebound_angle_rad)
            vy_ms = v_out_speed * np.sin(rebound_angle_rad)
            x = RM * np.cos(surface_angle)
            y = RM * np.sin(surface_angle)

    return local_density_grid


# --- メインの制御関数 ---
def main():
    # --- ★★★ 設定項目 ★★★ ---
    OUTPUT_DIRECTORY = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult"
    GRID_SIZE = 51
    GRID_RADIUS_RM = 5.0
    #TOTAL_SOURCE_FLUX = 1.0e25
    N_PARTICLES = 10000 # 粒子数

    settings = {
        'GRAVITY_ENABLED': True,
        'BETA': 0.5, #水星表面との衝突での係数 0で弾性衝突、1で完全にエネルギーを失う　
                     #理想的な石英表面において、ナトリウムではβ≈0.62、カリウムではβ≈0.26
        #'T1AU': 168918.0, #電離寿命　理論値
        'T1AU': 61728.4, #電離寿命　実験値
        'DT': 10.0, #時間ステップ
        # 'STICKING_COEFFICIENT': 0.15
    }
    # --- 設定はここまで ---

    # 1. 保存先となるサブフォルダの名前をパラメータから決定
    #    ユーザーの希望に合わせて beta は小数点以下1桁で表示
    sub_folder_name = f"density_map_beta{settings['BETA']:.2f}_Q0.14"

    # 2. サブフォルダのフルパスを作成
    target_output_dir = os.path.join(OUTPUT_DIRECTORY, sub_folder_name)

    # 3. サブフォルダをループの「前」に一度だけ作成
    os.makedirs(target_output_dir, exist_ok=True)
    print(f"結果は '{target_output_dir}' に保存されます。")

    constants = {
        'C': 299792458.0,  # 光速 [m/s]
        'PI': np.pi,
        'H': 6.62607015e-34, # プランク定数 [kg・m^2/s] (J・s)
        'MASS_NA': 22.98976928 * 1.66054e-27, # Na原子の質量 [kg]
        'RM': 2439.7e3,  # 水星の半径 [m]
        'GM_MERCURY': 2.2032e13, # G*M_mercury [m^3/s^2]  (万有引力定数 * 水星の質量)
        'K_BOLTZMANN': 1.380649e-23 #ボルツマン定数[J/K]
    }

    grid_params = {'size': GRID_SIZE, 'max_r': constants['RM'] * GRID_RADIUS_RM}

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

    ME, E_CHARGE = 9.1093897e-31, 1.60217733e-19 #電子の質量 [kg] # 電子の電荷 [C]
    epsilon_0 = 8.854187817e-12 #真空の誘電率
    #sigma_const = constants['PI'] * E_CHARGE ** 2 / (ME * constants['C']) #CGS
    sigma_const = E_CHARGE ** 2 / (4 * ME * constants['C'] * epsilon_0) #SI [m^2/s]
    spec_data_dict = {
        'wl': wl, 'gamma': gamma,
        'sigma0_perdnu2': sigma_const * 0.641,# 0.641 = D2線の振動子強度
        'sigma0_perdnu1': sigma_const * 0.320,# 0.320 = D1線の振動子強度
        'JL': 5.18e14* 1e4 # 1AUでの太陽フラックス [phs/s/m2/nm]
    }

    # --- TAAごとのループ処理 ---
    for line in orbit_lines:
        TAA, AU, lon, lat, Vms_ms = map(float, line.split())

        print(f"\n--- TAA = {TAA:.1f}度のシミュレーションを開始 ---")

        # 1. 論文で使われている物理定数を定義
        F_UV_at_1AU_per_cm2 = 1.5e14 * 1e4  # 1天文単位での紫外線光子フラックス [photons/m^2/s] (論文 Source [223] より)
        #Q_PSD_cm2 = 2.0e-20 / 1e4 # 光脱離断面積 [m^2] (論文 より)
        #Q_PSD_cm2 = 3.0e-20 / 1e4 #YakshinskiyとMadey（1999）
        Q_PSD_cm2 = 1.4e-21 / 1e4 #Killenら（2004）
        RM_m = constants['RM']  # 水星半径 [m]
        cNa = 1.5e13 * 1e4  # 表面ナトリウム原子数密度 [atoms/m^2] Leblanc and Johnson (2003)

        # 2. 現在の太陽距離(AU)における太陽直下点での最大放出率を計算
        F_UV_current_per_cm2 = F_UV_at_1AU_per_cm2 / (AU ** 2)
        R_PSD_peak_per_cm2 = F_UV_current_per_cm2 * Q_PSD_cm2 * cNa

        # 3. 太陽に照らされた半球全体で積分し、総放出量を計算
        # cos分布を半球で積分した際の有効面積は π * R^2 となる
        effective_area_cm2 = np.pi * (RM_m ** 2)
        total_flux_for_this_taa = R_PSD_peak_per_cm2 * effective_area_cm2 # [particles/sec]

        task_args = {
            'consts': constants, 'settings': settings, 'spec': spec_data_dict,
            'orbit': (TAA, AU, lon, lat, Vms_ms), 'grid_params': grid_params
        }
        tasks = [task_args] * N_PARTICLES

        with Pool(processes=cpu_count()) as pool:
            results = list(tqdm(pool.imap(simulate_single_particle_for_density, tasks), total=N_PARTICLES,
                                desc=f"TAA={TAA:.1f}"))

        print("結果を集計・保存しています...")
        master_density_grid = np.sum(results, axis=0)
        cell_area_m2 = (2 * grid_params['max_r'] / GRID_SIZE) ** 2
        #column_density_grid = (TOTAL_SOURCE_FLUX / N_PARTICLES) * (master_density_grid / cell_area_cm2)

        # 柱密度をまず [atoms/m^2] で計算
        column_density_m2 = (total_flux_for_this_taa / N_PARTICLES) * (master_density_grid / cell_area_m2)

        # 最終出力のために [atoms/cm^2] に変換
        column_density_cm2 = column_density_m2 / 1e4

        # 1. TAAごとのサブフォルダ名を決定
        base_filename = f"density_map_taa{TAA:.0f}_beta{settings['BETA']:.2f}_Q0.14"

        # 2. 保存先のパスを、ループの外で作成したフォルダに指定
        full_path_npy = os.path.join(target_output_dir, f"{base_filename}.npy")
        np.save(full_path_npy, column_density_cm2)

        print(f"結果を {full_path_npy} に保存しました。")

    print("\n★★★ すべてのシミュレーションが完了しました ★★★")


if __name__ == '__main__':
    main()