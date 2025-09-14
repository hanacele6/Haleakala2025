import numpy as np
import sys
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


# --- 関数 ---
def simulate_single_particle(args):
    """[SI版] 1個の原子の軌道をシミュレートし、結果を返す"""
    # --- 引数を受け取る ---
    constants, settings, spec_data = args['consts'], args['settings'], args['spec']
    TAA, AU, lon, lat, Vms_ms = args['orbit']

    # --- 定数と設定を展開 (すべてSI単位) ---
    C, PI, H, MASS_NA, RM, GM_MERCURY, K_BOLTZMANN = constants.values()
    GRAVITY_ENABLED, V_EJECT_SPEED, BETA, T_SURFACE, T1AU, DT, STICKING_COEFFICIENT = settings.values()
    wl, gamma, sigma0_perdnu2, sigma0_perdnu1, JL = spec_data.values()

    # --- シミュレーション初期化 (単位はメートル) ---
    source_angle_rad = np.deg2rad(60.0)
    x = -RM * np.cos(source_angle_rad)
    y = RM * np.sin(source_angle_rad)

    SRPt, tot_x, tot_V, tot_Nad = 0.0, 0.0, 0.0, 0.0
    b0 = 0.0
    time_to_terminator = -1.0

    Vms = Vms_ms  # [m/s] CGS版の /1000 は不要
    surface_normal_angle = np.arctan2(y, x)
    random_offset_angle = np.random.uniform(-PI / 2.0, PI / 2.0)
    ejection_angle_rad = surface_normal_angle + random_offset_angle

    # 放出速度 (m/s)
    vx_eject = V_EJECT_SPEED * np.cos(ejection_angle_rad)
    vy_eject = V_EJECT_SPEED * np.sin(ejection_angle_rad)

    # 水星から見た相対速度 (m/s)
    vx_ms = vx_eject
    vy_ms = vy_eject

    tau = T1AU * AU ** 2
    itmax = int(tau * 5.0 / DT + 0.5)

    for it in range(itmax):
        # ドップラー効果の計算 (速度は m/s)
        velocity_for_doppler = vx_ms + Vms

        # 波長は計算上nmのままでOK (v/cが比率のため)
        w_na_d2_nm = 589.1582 * (1.0 - velocity_for_doppler / C)
        w_na_d1_nm = 589.7558 * (1.0 - velocity_for_doppler / C)
        if not (wl[0] <= w_na_d2_nm < wl[-1] and wl[0] <= w_na_d1_nm < wl[-1]):
            break

        gamma2 = np.interp(w_na_d2_nm, wl, gamma)
        gamma1 = np.interp(w_na_d1_nm, wl, gamma)

        # [SI] 波長をメートルに変換してFνを計算
        w_na_d2_m = w_na_d2_nm * 1e-9
        w_na_d1_m = w_na_d1_nm * 1e-9
        JL_per_meter = JL * 1e9  # ph/s/m^2/nm -> ph/s/m^3
        jl_nu = JL_per_meter * (w_na_d1_m ** 2) / C  # Fν = Fλ * λ^2/c

        J2 = sigma0_perdnu2 * jl_nu / AU ** 2 * gamma2
        J1 = sigma0_perdnu1 * jl_nu / AU ** 2 * gamma1

        # [SI] 加速度 b の計算 (結果は m/s^2)
        b = (H / MASS_NA) * (J1 / w_na_d1_m + J2 / w_na_d2_m)

        if it == 0:
            b0 = b

        if x > 0 and np.sqrt(y ** 2) < RM:
            b = 0.0

        Nad = np.exp(-DT * it / tau)

        # [SI] 加速度と位置更新 (すべてメートル基準)
        accel_gx, accel_gy = 0.0, 0.0
        if GRAVITY_ENABLED:
            r_sq_grav = x ** 2 + y ** 2
            if r_sq_grav > 0:
                r_grav = np.sqrt(r_sq_grav)
                grav_accel_total = GM_MERCURY / r_sq_grav  # [m/s^2]
                accel_gx = -grav_accel_total * (x / r_grav)
                accel_gy = -grav_accel_total * (y / r_grav)

        vx_ms_prev, vy_ms_prev = vx_ms, vy_ms
        accel_srp_x = b  # [m/s^2] 単位変換は不要
        total_accel_x = accel_srp_x + accel_gx
        total_accel_y = accel_gy
        vx_ms += total_accel_x * DT
        vy_ms += total_accel_y * DT
        x += ((vx_ms_prev + vx_ms) / 2.0) * DT  # [m]
        y += ((vy_ms_prev + vy_ms) / 2.0) * DT  # [m]

        if x >= 0 and time_to_terminator < 0:
            time_to_terminator = it * DT
            break

        r_current = np.sqrt(x ** 2 + y ** 2)
        if r_current <= RM:
            if np.random.random() < settings['STICKING_COEFFICIENT']:
                time_to_terminator = -1.0
                break

            # [SI] 衝突計算 (エネルギーはジュール)
            v_in_sq = vx_ms ** 2 + vy_ms ** 2  # (m/s)^2
            E_in = 0.5 * MASS_NA * v_in_sq  # [J]
            E_T = K_BOLTZMANN * T_SURFACE  # [J]
            E_out = BETA * E_T + (1.0 - BETA) * E_in  # [J]
            v_out_sq = E_out / (0.5 * MASS_NA)  # (m/s)^2
            v_out_speed = np.sqrt(v_out_sq) if v_out_sq > 0 else 0.0  # [m/s]

            surface_angle = np.arctan2(y, x)
            rebound_angle_rad = surface_angle + np.random.uniform(-PI / 2.0, PI / 2.0)

            vx_ms = v_out_speed * np.cos(rebound_angle_rad)
            vy_ms = v_out_speed * np.sin(rebound_angle_rad)
            x = RM * np.cos(surface_angle)
            y = RM * np.sin(surface_angle)

        # 統計量もメートル基準で計算
        SRPt += b * DT * Nad
        tot_x += (x + RM) * DT * Nad  # xはメートル
        tot_V += np.sqrt(vx_ms ** 2 + vy_ms ** 2) * DT * Nad  # vは m/s
        tot_Nad += Nad * DT

    if tot_Nad > 0:
        return {
            'b0': b0,  # m/s^2
            'srpt_avg': SRPt / tot_Nad,  # m/s^2
            'v_ave': tot_V / tot_Nad,  # m/s
            'x_ave': tot_x / tot_Nad,  # m
            'tm': time_to_terminator
        }
    return None


def run_main_simulation():
    """[SI版] メインの制御関数"""
    # 表面温度の計算部分は元からSI単位なので変更なし
    AU_REF = 0.387
    ALBEDO = 0.142
    S_1AU_SI = 1366.0
    SIGMA_SB_SI = 5.67e-8
    S_REF = S_1AU_SI / (AU_REF ** 2)
    T_SURFACE_REF = ((S_REF * (1 - ALBEDO)) / (4 * SIGMA_SB_SI)) ** 0.25

    # --- [SI] シミュレーション設定 ---
    settings = {
        'GRAVITY_ENABLED': True,
        'V_EJECT_SPEED': 900.0,  # [m/s]
        'BETA': 0.5,
        'T_SURFACE': T_SURFACE_REF,
        'T1AU': 61728.4,
        'DT': 5.0,
        'STICKING_COEFFICIENT': 0.0
    }
    N_PARTICLES = 1000

    # --- [SI] 物理定数 ---
    constants = {
        'C': 299792458.0,  # [m/s]
        'PI': np.pi,
        'H': 6.62607015e-34,  # [J·s or kg·m^2/s]
        'MASS_NA': 22.98976928 * 1.66054e-27,  # [kg]
        'RM': 2439.7e3,  # [m]
        'GM_MERCURY': 2.2032e13,  # [m^3/s^2]
        'K_BOLTZMANN': 1.380649e-23  # [J/K]
    }

    # --- データファイルの読み込み ---
    spectrum_file = 'SolarSpectrum_Na0.txt'
    orbit_file = 'orbit360.txt'
    output_file = f"MC_FullResults_BETA{settings['BETA']}_60_DT_SI.txt"
    print(f"モード: 詳細結果出力 (SI単位系), β={settings['BETA']}, N={N_PARTICLES}")
    print(f"表面温度: 距離に応じて変動 (基準値: {T_SURFACE_REF:.2f} K @ {AU_REF} AU)")

    try:
        spec_data_np = np.loadtxt(spectrum_file, usecols=(0, 3))
        wl, gamma = spec_data_np[:, 0], spec_data_np[:, 1]
    except FileNotFoundError:
        sys.exit(f"エラー: スペクトルファイル '{spectrum_file}' が見つかりません。")
    if not np.all(np.diff(wl) > 0):
        sort_indices = np.argsort(wl)
        wl, gamma = wl[sort_indices], gamma[sort_indices]

    # --- [SI] 断面積の計算 ---
    ME = 9.1093837e-31  # 電子の質量 [kg]
    E_CHARGE = 1.602176634e-19  # 電荷 [C]
    EPSILON_0 = 8.8541878128e-12  # 真空の誘電率 [F/m]
    sigma_const = E_CHARGE ** 2 / (4 * ME * constants['C'] * EPSILON_0)  # [m^2·Hz]

    spec_data_dict = {
        'wl': wl, 'gamma': gamma,
        'sigma0_perdnu2': sigma_const * 0.641,
        'sigma0_perdnu1': sigma_const * 0.320,
        'JL': 5.18e18  # [phs/s/m^2/nm]
    }

    print("シミュレーションを開始します...")

    try:
        with open(orbit_file, 'r') as f_orbit, open(output_file, 'w') as f_out:
            # 出力ヘッダーの単位表記を更新
            header = f"{'TAA':>12s} {'b0_avg[m/s^2]':>20s} {'SRPt_avg[m/s^2]':>20s} {'V_ave[m/s]':>20s} {'x_ave[m]':>20s} {'tm_avg[s]':>20s}\n"
            f_out.write(header)
            orbit_lines = f_orbit.readlines()

            for line in tqdm(orbit_lines, desc="Total Progress"):
                TAA, AU, lon, lat, Vms_ms = map(float, line.split())
                current_T_surface = T_SURFACE_REF * np.sqrt(AU_REF / AU)
                current_settings = settings.copy()
                current_settings['T_SURFACE'] = current_T_surface
                task_args = {'consts': constants, 'settings': current_settings, 'spec': spec_data_dict,
                             'orbit': (TAA, AU, lon, lat, Vms_ms)}
                tasks = [task_args] * N_PARTICLES

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