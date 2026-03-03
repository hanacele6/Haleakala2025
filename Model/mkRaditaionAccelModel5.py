import numpy as np
import sys
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


# --- 関数 ---
def simulate_single_particle(args):
    """1個の原子の軌道をシミュレートし、結果を返す"""
    # (中身はご提示のコードのままで完璧です)
    # --- 引数を受け取る ---
    constants, settings, spec_data = args['consts'], args['settings'], args['spec']
    TAA, AU, lon, lat, Vms_ms = args['orbit']

    # --- 定数と設定を展開 ---
    C, PI, H, MASS_NA, RM, GM_MERCURY, K_BOLTZMANN = constants.values()
    GRAVITY_ENABLED, V_EJECT_SPEED, BETA, T_SURFACE, T1AU, DT, STICKING_COEFFICIENT = settings.values()
    wl, gamma, sigma0_perdnu2, sigma0_perdnu1, JL = spec_data.values()

    # --- シミュレーション初期化 ---
    # 供給源の角度を設定 (ラジアン)
    # 0度 = 太陽直下点, 90度 = ターミネーター
    source_angle_rad = np.deg2rad(0.0)

    # 新しい初期位置
    x = -RM * np.cos(source_angle_rad)
    y = RM * np.sin(source_angle_rad)

    SRPt, tot_x, tot_V, tot_Nad = 0.0, 0.0, 0.0, 0.0
    b0 = 0.0
    time_to_terminator = -1.0

    Vms = Vms_ms / 1000.0
    # 現在地の表面の法線角度を計算
    surface_normal_angle = np.arctan2(y, x)

    # 法線方向を基準としたランダムな放出角度を決定
    # （法線方向から +/- 90度の範囲）
    random_offset_angle = np.random.uniform(-PI / 2.0, PI / 2.0)
    ejection_angle_rad = surface_normal_angle + random_offset_angle

    # 正しい放出速度を計算
    vx_eject = V_EJECT_SPEED * np.cos(ejection_angle_rad)
    vy_eject = V_EJECT_SPEED * np.sin(ejection_angle_rad)

    vx_ms0 = Vms
    vx_ms = vx_eject +vx_ms0
    vy_ms = vy_eject
    tau = T1AU * AU ** 2
    itmax = int(tau * 5.0 / DT + 0.5)

    for it in range(itmax):
        velocity_for_doppler = vx_ms
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

        if it == 0:
            b0 = b

        if x > 0 and np.sqrt(y ** 2) < RM:
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
        total_accel_x = accel_srp_x + accel_gx
        total_accel_y = accel_gy
        vx_ms += total_accel_x * DT
        vy_ms += total_accel_y * DT
        x += ((vx_ms_prev + vx_ms) / 2.0) * DT
        y += (vy_ms_prev + vy_ms) / 2.0 * DT

        if x >= 0 and time_to_terminator < 0:
            time_to_terminator = it * DT
            break

        r_current = np.sqrt(x ** 2 + y ** 2)
        if r_current <= RM:
            if np.random.random() < settings['STICKING_COEFFICIENT']:
                time_to_terminator = -1.0
                break

            v_in_sq = (vx_ms - vx_ms0) ** 2 + vy_ms ** 2
            E_in = 0.5 * MASS_NA * (v_in_sq * 1e10)
            E_T = K_BOLTZMANN * T_SURFACE
            E_out = BETA * E_T + (1.0 - BETA) * E_in
            v_out_sq = E_out / (0.5 * MASS_NA) / 1e10
            v_out_speed = np.sqrt(v_out_sq) if v_out_sq > 0 else 0.0
            surface_angle = np.arctan2(y, x)
            rebound_angle_rad = surface_angle + np.random.uniform(-PI / 2.0, PI / 2.0)
            vx_ms = vx_ms0 + v_out_speed * np.cos(rebound_angle_rad)
            vy_ms = v_out_speed * np.sin(rebound_angle_rad)
            x = RM * np.cos(surface_angle)
            y = RM * np.sin(surface_angle)


        SRPt += b * DT * Nad
        tot_x += (x + RM) * DT * Nad  # 高度に変換
        tot_V += np.sqrt((vx_ms - vx_ms0) ** 2 + vy_ms ** 2) * DT * Nad
        tot_Nad += Nad * DT

    if tot_Nad > 0:
        return {
            'b0': b0,
            'srpt_avg': SRPt / tot_Nad,
            'v_ave': tot_V / tot_Nad,
            'x_ave': tot_x / tot_Nad,
            'tm': time_to_terminator
        }
    return None


def run_main_simulation():
    """メインの制御関数"""
    # --- シミュレーション設定 ---
    settings = {
        'GRAVITY_ENABLED': True,
        'V_EJECT_SPEED': 0.9,
        'BETA': 0.5,
        'T_SURFACE': 600.0,
        #'T1AU': 168918.0,
        'T1AU': 61728.4,
        'DT': 10.0,
        'STICKING_COEFFICIENT': 0.0
    }
    N_PARTICLES = 1000

    # --- 物理定数 ---
    constants = {
        'C': 299792.458, 'PI': np.pi, 'H': 6.626068e-34 * 1e4 * 1e3,
        'MASS_NA': 22.98976928 * 1.6605402e-27 * 1e3, 'RM': 2439.7,
        'GM_MERCURY': 2.2032e4, 'K_BOLTZMANN': 1.380649e-16
    }

    # --- データファイルの読み込み ---
    spectrum_file = 'SolarSpectrum_Na0.txt'
    orbit_file = 'orbit360.txt'
    output_file = f"MC_FullResults_BETA{settings['BETA']}_0.txt"
    print(f"モード: 詳細結果出力, β={settings['BETA']}, N={N_PARTICLES}, 並列処理有効")

    # ... (ファイル読み込み) ...
    try:
        spec_data_np = np.loadtxt(spectrum_file, usecols=(0, 3))
        wl, gamma = spec_data_np[:, 0], spec_data_np[:, 1]
    except FileNotFoundError:
        print(f"エラー: スペクトルファイル '{spectrum_file}' が見つかりません。")
        sys.exit()
    if not np.all(np.diff(wl) > 0):
        sort_indices = np.argsort(wl)
        wl, gamma = wl[sort_indices], gamma[sort_indices]
    ME = 9.1093897e-31 * 1e3
    E_CHARGE = 1.60217733e-19 * 2.99792458e8 * 10.0
    sigma_const = constants['PI'] * E_CHARGE ** 2 / ME / (constants['C'] * 1e5)
    spec_data_dict = {
        'wl': wl, 'gamma': gamma, 'sigma0_perdnu2': sigma_const * 0.641,
        'sigma0_perdnu1': sigma_const * 0.320, 'JL': 5.18e14
    }

    print("シミュレーションを開始します...")

    try:
        with open(orbit_file, 'r') as f_orbit, open(output_file, 'w') as f_out:
            header = f"{'TAA':>12s} {'b0_avg[cm/s^2]':>20s} {'SRPt_avg[cm/s^2]':>20s} {'V_ave[km/s]':>20s} {'x_ave[km]':>20s} {'tm_avg[s]':>20s}\n"
            f_out.write(header)
            orbit_lines = f_orbit.readlines()

            for line in tqdm(orbit_lines, desc="Total Progress"):
                TAA, AU, lon, lat, Vms_ms = map(float, line.split())
                task_args = {
                    'consts': constants, 'settings': settings, 'spec': spec_data_dict,
                    'orbit': (TAA, AU, lon, lat, Vms_ms)
                }
                tasks = [task_args] * N_PARTICLES

                valid_results_list = []
                with Pool(processes=cpu_count()) as pool:
                    results = pool.map(simulate_single_particle, tasks)
                    valid_results_list = [r for r in results if r is not None and r['tm'] >= 0]

                if len(valid_results_list) > 0:
                    tm_avg = np.mean([r['tm'] for r in valid_results_list])
                    b0_avg = np.mean([r['b0'] for r in valid_results_list])
                    srpt_avg_mean = np.mean([r['srpt_avg'] for r in valid_results_list])
                    v_ave_mean = np.mean([r['v_ave'] for r in valid_results_list])
                    x_ave_mean = np.mean([r['x_ave'] for r in valid_results_list])
                else:
                    tm_avg, b0_avg, srpt_avg_mean, v_ave_mean, x_ave_mean = -1.0, -1.0, -1.0, -1.0, -1.0

                f_out.write(
                    f"{TAA:12.4f} {b0_avg:20.8f} {srpt_avg_mean:20.8f} {v_ave_mean:20.8f} {x_ave_mean:20.8f} {tm_avg:20.4f}\n"
                )

            print("\nシミュレーションが完了しました。")
    except Exception as e:
        print(f"エラー: シミュレーション中に問題が発生しました: {e}")


if __name__ == '__main__':
    run_main_simulation()