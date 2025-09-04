import numpy as np
import sys
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import matplotlib.pyplot as plt


# --- 物理モデルに基づくサンプリング関数 (変更なし) ---
def sample_weibull_speed(mass_kg, U_ev=0.05, beta_shape=0.7):
    E_CHARGE_SI = 1.602176634e-19
    p = np.random.random()
    E_ev = U_ev * (p ** (-1.0 / (beta_shape + 1.0)) - 1.0)
    E_joule = E_ev * E_CHARGE_SI
    v_ms = np.sqrt(2 * E_joule / mass_kg)
    return v_ms / 1000.0


def sample_cosine_angle():
    p = np.random.random()
    return np.arcsin(2 * p - 1)


# --- シミュレーションのコア関数 (変更なし) ---
def simulate_single_particle_for_density(args):
    constants, settings, spec_data = args['consts'], args['settings'], args['spec']
    TAA, AU, lon, lat, Vms_ms = args['orbit']
    grid_params = args['grid_params']
    C, PI, H, MASS_NA, RM, GM_MERCURY, K_BOLTZMANN = constants.values()
    GRAVITY_ENABLED, BETA, T_SURFACE, T1AU, DT, STICKING_COEFFICIENT = settings.values()
    wl, gamma, sigma0_perdnu2, sigma0_perdnu1, JL = spec_data.values()
    GRID_SIZE, GRID_MAX_R = grid_params['size'], grid_params['max_r']
    CELL_SIZE = 2 * GRID_MAX_R / GRID_SIZE
    local_density_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    p = np.random.random()
    source_angle_rad = np.arcsin(2 * p - 1)
    x = RM * np.cos(source_angle_rad)
    y = RM * np.sin(source_angle_rad)
    ejection_speed = sample_weibull_speed(MASS_NA)
    surface_normal_angle = np.arctan2(y, x)
    random_offset_angle = sample_cosine_angle()
    ejection_angle_rad = surface_normal_angle + random_offset_angle
    vx_eject = ejection_speed * np.cos(ejection_angle_rad)
    vy_eject = ejection_speed * np.sin(ejection_angle_rad)
    vx_ms, vy_ms = vx_eject, vy_eject
    Vms = Vms_ms / 1000.0
    tau = T1AU * AU ** 2
    itmax = int(tau * 5.0 / DT + 0.5)
    for it in range(itmax):
        velocity_for_doppler = vx_ms + Vms
        w_na_d2 = 589.1582 * (1.0 - velocity_for_doppler / C)
        w_na_d1 = 589.7558 * (1.0 - velocity_for_doppler / C)
        if not (wl[0] <= w_na_d2 < wl[-1] and wl[0] <= w_na_d1 < wl[-1]):
            break
        gamma2 = np.interp(w_na_d2, wl, gamma)
        gamma1 = np.interp(w_na_d1, wl, gamma)
        m_na_wl = (w_na_d2 + w_na_d1) / 2.0
        jl_nu = JL * 1e9 * ((m_na_wl * 1e-9) ** 2 / (C * 1e3))
        J2 = sigma0_perdnu2 * jl_nu / AU ** 2 * gamma2
        J1 = sigma0_perdnu1 * jl_nu / AU ** 2 * gamma1
        b = (H / MASS_NA) * (J1 / (w_na_d1 * 1e-7) + J2 / (w_na_d2 * 1e-7))
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
        accel_srp_x = b / 100.0 / 1000.0
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
        r_current = np.sqrt(x ** 2 + y ** 2)
        if r_current <= RM:
            if np.random.random() < STICKING_COEFFICIENT:
                break
            v_in_sq = vx_ms_prev ** 2 + vy_ms_prev ** 2
            E_in = 0.5 * MASS_NA * (v_in_sq * 1e6)
            E_T = K_BOLTZMANN * T_SURFACE
            E_out = BETA * E_T + (1.0 - BETA) * E_in
            v_out_sq = E_out / (0.5 * MASS_NA) / 1e6
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
    # 1. 出力先フォルダの指定 (Windowsのパスは'\'を'/'に置き換えるか、r""で囲む)
    OUTPUT_DIRECTORY = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult"

    # 2. シミュレーションの基本設定
    GRID_SIZE = 51
    GRID_RADIUS_RM = 5.0
    TOTAL_SOURCE_FLUX = 10.0e24  # [atoms/s]
    N_PARTICLES = 10000  # 粒子数（テスト時は10000などに減らすと速いです）

    settings = {
        'GRAVITY_ENABLED': True,
        'BETA': 0.5,
        #'T1AU': 61728.4,
        'T1AU':168918.0,
        'DT': 10.0,
        'STICKING_COEFFICIENT': 0.15
    }
    # --- 設定はここまで ---

    # 出力フォルダが存在しない場合は作成
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    print(f"結果は '{OUTPUT_DIRECTORY}' に保存されます。")

    # --- 物理定数とグリッドパラメータの準備 ---
    grid_params = {'size': GRID_SIZE, 'max_r': 2439.7 * GRID_RADIUS_RM}
    constants = {
        'C': 299792.458, 'PI': np.pi, 'H': 6.626068e-34 * 1e7,
        'MASS_NA': 22.98976928 * 1.66054e-27, 'RM': 2439.7,
        'GM_MERCURY': 2.2032e4, 'K_BOLTZMANN': 1.380649e-23
    }

    # --- 太陽光スペクトルと軌道データの読み込み ---
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

    ME, E_CHARGE = 9.1093897e-31, 1.60217733e-19
    sigma_const = constants['PI'] * E_CHARGE ** 2 / (ME * constants['C'] * 1e3)
    spec_data_dict = {
        'wl': wl, 'gamma': gamma, 'sigma0_perdnu2': sigma_const * 0.641,
        'sigma0_perdnu1': sigma_const * 0.320, 'JL': 5.18e14
    }

    AU_REF, ALBEDO = 0.387, 0.142
    S_1AU_SI, SIGMA_SB_SI = 1366.0, 5.67e-8
    S_REF = S_1AU_SI / (AU_REF ** 2)
    T_SURFACE_REF = ((S_REF * (1 - ALBEDO)) / (4 * SIGMA_SB_SI)) ** 0.25

    # --- TAAごとのループ処理 ---
    # テスト実行の場合は orbit_lines[::30] のように間引いてください
    for line in orbit_lines:
        TAA, AU, lon, lat, Vms_ms = map(float, line.split())

        print(f"\n--- TAA = {TAA:.1f}度のシミュレーションを開始 ---")

        current_settings = settings.copy()
        current_settings['T_SURFACE'] = T_SURFACE_REF * np.sqrt(AU_REF / AU)

        task_args = {
            'consts': constants, 'settings': current_settings, 'spec': spec_data_dict,
            'orbit': (TAA, AU, lon, lat, Vms_ms), 'grid_params': grid_params
        }
        tasks = [task_args] * N_PARTICLES

        with Pool(processes=cpu_count()) as pool:
            results = list(tqdm(pool.imap(simulate_single_particle_for_density, tasks), total=N_PARTICLES,
                                desc=f"TAA={TAA:.1f}"))

        print("結果を集計・保存しています...")
        master_density_grid = np.sum(results, axis=0)
        cell_area_cm2 = (2 * grid_params['max_r'] * 1e5 / GRID_SIZE) ** 2
        column_density_grid = (TOTAL_SOURCE_FLUX / N_PARTICLES) * (master_density_grid / cell_area_cm2)

        # --- ファイルへの保存 ---
        # ファイル名にTAAとBETAの値を含める
        base_filename = f"density_map_taa{TAA:.0f}_beta{settings['BETA']:.2f}"

        # 方法1: Numpyバイナリ形式 (.npy) - 高速・小容量（推奨）
        full_path_npy = os.path.join(OUTPUT_DIRECTORY, f"{base_filename}.npy")
        np.save(full_path_npy, column_density_grid)

        # 方法2: CSV形式 - Excelなどで開けるが、低速・大容量
        # こちらを使いたい場合は、上のnp.saveをコメントアウトし、下の2行を有効化してください
        # full_path_csv = os.path.join(OUTPUT_DIRECTORY, f"{base_filename}.csv")
        # np.savetxt(full_path_csv, column_density_grid, fmt='%.4e', delimiter=',')

        print(f"結果を {full_path_npy} に保存しました。")

    print("\n★★★ すべてのシミュレーションが完了しました ★★★")


if __name__ == '__main__':
    main()