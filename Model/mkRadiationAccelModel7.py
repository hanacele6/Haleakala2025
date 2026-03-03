import numpy as np
import sys
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


# --- 関数 ---
def simulate_single_particle(args):
    """1個の原子の軌道をシミュレートし、結果を返す"""
    # --- 引数を受け取る ---
    constants, settings, spec_data = args['consts'], args['settings'], args['spec']
    TAA, AU, lon, lat, Vms_ms = args['orbit']

    # --- 定数と設定を展開 ---
    C, PI, H, MASS_NA, RM, GM_MERCURY, K_BOLTZMANN = constants.values()
    # T_SURFACEはsettingsから動的に受け取る
    GRAVITY_ENABLED, V_EJECT_SPEED, BETA, T_SURFACE, T1AU, DT, STICKING_COEFFICIENT = settings.values()
    wl, gamma, sigma0_perdnu2, sigma0_perdnu1, JL = spec_data.values()

    # --- シミュレーション初期化 ---
    # このシミュレーションは「水星固定座標系」で計算する
    # 粒子の位置(x, y)と速度(vx_ms, vy_ms)は、常に水星から見た相対的な値となる
    # 0度 = 太陽直下点, 90度 = ターミネーター
    source_angle_rad = np.deg2rad(60.0)
    x = -RM * np.cos(source_angle_rad)
    y = RM * np.sin(source_angle_rad)

    SRPt, tot_x, tot_V, tot_Nad = 0.0, 0.0, 0.0, 0.0
    b0 = 0.0
    time_to_terminator = -1.0

    Vms = Vms_ms / 1000.0
    surface_normal_angle = np.arctan2(y, x)

    random_offset_angle = np.random.uniform(-PI / 2.0, PI / 2.0)
    ejection_angle_rad = surface_normal_angle + random_offset_angle

    # vx_eject, vy_eject は水星表面からの放出速度（相対速度）
    vx_eject = V_EJECT_SPEED * np.cos(ejection_angle_rad)
    vy_eject = V_EJECT_SPEED * np.sin(ejection_angle_rad)

    #vx_msは水星から見た相対速度なので、初速度は放出速度そのもの
    vx_ms = vx_eject
    vy_ms = vy_eject

    # 参考: 水星の公転速度は vx_ms0 として保持しておく
    vx_ms0 = Vms

    tau = T1AU * AU ** 2
    itmax = int(tau * 5.0 / DT + 0.5)

    for it in range(itmax):
        #ドップラー効果の計算の時だけ、太陽から見た絶対速度を計算する
        # (粒子の対水星速度 + 水星の対太陽速度)
        velocity_for_doppler = vx_ms + Vms
        w_na_d2 = 589.1582 * (1.0 - velocity_for_doppler / C) #ドップラーシフトの計算
        w_na_d1 = 589.7558 * (1.0 - velocity_for_doppler / C) #ドップラーシフトの計算
        if not (wl[0] <= w_na_d2 < wl[-1] and wl[0] <= w_na_d1 < wl[-1]):
            break
        gamma2 = np.interp(w_na_d2, wl, gamma)
        gamma1 = np.interp(w_na_d1, wl, gamma)
        m_na_wl = (w_na_d2 + w_na_d1) / 2.0
        jl_nu = JL * 1e9 * ((m_na_wl * 1e-9) ** 2 / (C * 1e3))
        J2 = sigma0_perdnu2 * jl_nu / AU ** 2 * gamma2
        J1 = sigma0_perdnu1 * jl_nu / AU ** 2 * gamma1
        b = (H / MASS_NA) * (J1 / (w_na_d1 * 1e-7) + J2 / (w_na_d2 * 1e-7)) #b = h/m*(J/λ)

        if it == 0:
            b0 = b

        if x > 0 and np.sqrt(y ** 2) < RM:
            b = 0.0

        Nad = np.exp(-DT * it / tau)

        # 加速度と位置更新は、すべて水星固定座標系で行う
        accel_gx, accel_gy = 0.0, 0.0
        if GRAVITY_ENABLED:
            r_sq_grav = x ** 2 + y ** 2
            if r_sq_grav > 0:
                r_grav = np.sqrt(r_sq_grav)
                grav_accel_total = GM_MERCURY / r_sq_grav
                accel_gx = -grav_accel_total * (x / r_grav) #x方向の重力加速度
                accel_gy = -grav_accel_total * (y / r_grav) #y方向の重力加速度

        vx_ms_prev, vy_ms_prev = vx_ms, vy_ms
        accel_srp_x = b / 100.0 / 1000.0
        total_accel_x = accel_srp_x + accel_gx
        total_accel_y = accel_gy
        vx_ms += total_accel_x * DT #v(t+Δt)=v(t)+a(t)Δt
        vy_ms += total_accel_y * DT #v(t+Δt)=v(t)+a(t)Δt
        x += ((vx_ms_prev + vx_ms) / 2.0) * DT #x(t+Δt)=x(t)+ (v(t)+v(t+Δt))/2*Δt
        y += (vy_ms_prev + vy_ms) / 2.0 * DT

        if x >= 0 and time_to_terminator < 0:
            time_to_terminator = it * DT
            break

        r_current = np.sqrt(x ** 2 + y ** 2)
        if r_current <= RM:
            if np.random.random() < settings['STICKING_COEFFICIENT']:
                time_to_terminator = -1.0
                break

            # 【重要】衝突計算も水星固定座標系で行う
            # 地面は静止している(速度0)と見なせるため、衝突速度はvx_ms, vy_msそのもの
            v_in_sq = vx_ms ** 2 + vy_ms ** 2
            E_in = 0.5 * MASS_NA * (v_in_sq * 1e10) #1/2*mv^2
            E_T = K_BOLTZMANN * T_SURFACE
            E_out = BETA * E_T + (1.0 - BETA) * E_in #β=((E_out - E_in)/(E_T - E_in)) → E_out = βE_T + (1-β)E_in
            v_out_sq = E_out / (0.5 * MASS_NA) / 1e10
            v_out_speed = np.sqrt(v_out_sq) if v_out_sq > 0 else 0.0
            surface_angle = np.arctan2(y, x) #法線の角度を計算
            rebound_angle_rad = surface_angle + np.random.uniform(-PI / 2.0, PI / 2.0) #法線を基準にランダムな角度を計算　反射角とする

            # 反発後の速度も、もちろん相対速度
            vx_ms = v_out_speed * np.cos(rebound_angle_rad)
            vy_ms = v_out_speed * np.sin(rebound_angle_rad)
            x = RM * np.cos(surface_angle)
            y = RM * np.sin(surface_angle)

        # 統計量の平均速度は、水星から見た相対速度で計算する
        SRPt += b * DT * Nad
        tot_x += (x + RM) * DT * Nad
        tot_V += np.sqrt(vx_ms ** 2 + vy_ms ** 2) * DT * Nad
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
    # --- ★★★ 変更点 ★★★ ---
    # 物理定数に基づいて表面の基準温度を計算する
    AU_REF = 0.387         # 水星の平均軌道半径 [AU]
    ALBEDO = 0.142          # 水星のアルベド（反射能）
    S_1AU_SI = 1366.0      # 太陽定数 (1AUでのフラックス) [W/m^2]
    SIGMA_SB_SI = 5.67e-8  # シュテファン・ボルツマン定数 [W m^-2 K^-4]

    # 平均距離における太陽フラックスを計算
    S_REF = S_1AU_SI / (AU_REF**2)
    # 放射平衡から有効温度を計算 (T = [S*(1-A)/(4*sigma)]^0.25)
    T_SURFACE_REF = ((S_REF * (1 - ALBEDO)) / (4 * SIGMA_SB_SI))**0.25
    # -------------------------

    # --- シミュレーション設定 ---
    settings = {
        'GRAVITY_ENABLED': True,
        'V_EJECT_SPEED': 0.9, #放出速度
        'BETA': 0.5,
        'T_SURFACE': T_SURFACE_REF, # 計算された基準温度を初期値とする
        'T1AU': 61728.4, #電離寿命 [s]
        'DT': 5.0, # 時間ステップ [s]
        'STICKING_COEFFICIENT': 0.0
    }
    N_PARTICLES = 1000

    # --- 物理定数 ---
    constants = {
        'C': 299792.458, # 光速 [km/s]
        'PI': np.pi,
        'H': 6.626068e-34 * 1e4 * 1e3, # プランク定数 [cm2*g/s]
        'MASS_NA': 22.98976928 * 1.6605402e-27 * 1e3,  # Na原子の質量 [g]
        'RM': 2439.7, # 水星の半径 [km]
        'GM_MERCURY': 2.2032e4,  # G * M_mercury [km^3/s^2] (万有引力定数 * 水星の質量)
        'K_BOLTZMANN': 1.380649e-16 #ボルツマン定数[erg/K]
    }

    # --- データファイルの読み込み ---
    spectrum_file = 'SolarSpectrum_Na0.txt'
    orbit_file = 'orbit360.txt'
    output_file = f"MC_FullResults_BETA{settings['BETA']}_60_DT.txt"
    print(f"モード: 詳細結果出力, β={settings['BETA']}, N={N_PARTICLES}, 並列処理有効")
    # ★★★ 変更点 ★★★
    print(f"表面温度: 距離に応じて変動 (計算された基準値: {T_SURFACE_REF:.2f} K @ {AU_REF} AU)")


    try:
        spec_data_np = np.loadtxt(spectrum_file, usecols=(0, 3))
        wl, gamma = spec_data_np[:, 0], spec_data_np[:, 1]
    except FileNotFoundError:
        print(f"エラー: スペクトルファイル '{spectrum_file}' が見つかりません。")
        sys.exit()
    if not np.all(np.diff(wl) > 0):
        sort_indices = np.argsort(wl)
        wl, gamma = wl[sort_indices], gamma[sort_indices]
    ME = 9.1093897e-31 * 1e3 ## 電子の質量 [g]
    E_CHARGE = 1.60217733e-19 * 2.99792458e8 * 10.0  # 電子の電荷 [esu]
    sigma_const = constants['PI'] * E_CHARGE ** 2 / ME / (constants['C'] * 1e5)
    spec_data_dict = {
        'wl': wl,
        'gamma': gamma,
        'sigma0_perdnu2': sigma_const * 0.641,
        'sigma0_perdnu1': sigma_const * 0.320,
        'JL': 5.18e14 # 1AUでの太陽フラックス [phs/s/cm2/nm]
    }

    print("シミュレーションを開始します...")

    try:
        with open(orbit_file, 'r') as f_orbit, open(output_file, 'w') as f_out:
            header = f"{'TAA':>12s} {'b0_avg[cm/s^2]':>20s} {'SRPt_avg[cm/s^2]':>20s} {'V_ave[km/s]':>20s} {'x_ave[km]':>20s} {'tm_avg[s]':>20s}\n"
            f_out.write(header)
            orbit_lines = f_orbit.readlines()

            for line in tqdm(orbit_lines, desc="Total Progress"):
                TAA, AU, lon, lat, Vms_ms = map(float, line.split())

                # 太陽からの距離(AU)に基づいて表面温度を計算する
                # 温度は距離の平方根に反比例する (T ∝ 1/√AU)
                current_T_surface = T_SURFACE_REF * np.sqrt(AU_REF / AU)

                # 現在のループ用の設定を作成
                current_settings = settings.copy()
                current_settings['T_SURFACE'] = current_T_surface

                task_args = {
                    'consts': constants,
                    'settings': current_settings, # 更新した設定を使用
                    'spec': spec_data_dict,
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

