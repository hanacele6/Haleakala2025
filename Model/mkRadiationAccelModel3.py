import numpy as np
import sys


def run_monte_carlo_simulation():
    """
    モンテカルロ法を用いて、水星から放出されたNa原子の運動をシミュレートする。
    太陽放射圧、水星の重力、地表との衝突（バウンド）を考慮する。
    """
    # --- ▼▼▼ シミュレーション設定 ▼▼▼ ---
    GRAVITY_ENABLED = True  # True: 重力あり, False: 重力なし
    N_PARTICLES = 1000  # 各軌道位置(TAA)でシミュレートする原子の数
    V_EJECT_SPEED = 0.9  # 原子の初期放出速度 [km/s] (例: 熱的な速度)
    BETA = 0.5  # エネルギー順応係数 (論文の結論に近い値)
    T_SURFACE = 400.0  # 水星の地表温度 [K] (バウンド計算用)
    # -----------------------------------

    # --- 1. 物理定数 ---
    C = 299792.458
    PI = np.pi
    D2R = PI / 180.0
    JL = 5.18e14
    ME = 9.1093897e-31 * 1e3
    E_CHARGE = 1.60217733e-19 * 2.99792458e8 * 10.0
    H = 6.626068e-34 * 1e4 * 1e3
    MASS_NA = 22.98976928 * 1.6605402e-27 * 1e3
    RM = 2439.7
    GM_MERCURY = 2.2032e4
    K_BOLTZMANN = 1.380649e-16  # ボルツマン定数 [erg/K]

    # --- 2. シミュレーションパラメータ ---
    T1AU = 168918.0
    DT = 10.0

    # --- 3. データファイルの読み込み ---
    spectrum_file = 'SolarSpectrum_Na0.txt'
    orbit_file = 'orbit360.txt'
    if GRAVITY_ENABLED:
        output_file = f'MC_BETA{BETA}_grav.txt'
        print(f"モード: 重力あり, β={BETA}, N={N_PARTICLES}")
    else:
        output_file = f'MC_BETA{BETA}_nograv.txt'
        print(f"モード: 重力なし, β={BETA}, N={N_PARTICLES}")

    try:
        spec_data = np.loadtxt(spectrum_file, usecols=(0, 3))
        wl, gamma = spec_data[:, 0], spec_data[:, 1]
    except FileNotFoundError:
        print(f"エラー: スペクトルファイル '{spectrum_file}' が見つかりません。")
        sys.exit()
    except Exception as e:
        print(f"エラー: スペクトルファイルの読み込み中に問題が発生しました: {e}")
        sys.exit()

    if not np.all(np.diff(wl) > 0):
        print("警告: スペクトルデータの波長がソートされていません。ソートします。")
        sort_indices = np.argsort(wl)
        wl, gamma = wl[sort_indices], gamma[sort_indices]

    # --- 4. 散乱断面積に関連する定数の計算 ---
    sigma_const = PI * E_CHARGE ** 2 / ME / (C * 1e5)
    sigma0_perdnu2 = sigma_const * 0.641
    sigma0_perdnu1 = sigma_const * 0.320

    print("シミュレーションを開始します...")

    # --- 5. メインループ ---
    try:
        with open(orbit_file, 'r') as f_orbit, open(output_file, 'w') as f_out:
            header = f"{'TAA':>19s} {'b0_avg[cm/s^2]':>20s} {'SRPt_avg[cm/s^2]':>20s} {'V_ave[km/s]':>20s} {'x_ave[km]':>20s} {'z_M[km]':>20s}\n"
            f_out.write(header)

            total_sims = 360
            for iTAA in range(total_sims):
                line = f_orbit.readline()
                if not line:
                    break
                TAA, AU, lon, lat, Vms_ms = map(float, line.split())

                # --- ▼▼▼ モンテカルロ法：各TAAの結果を保存するリストを初期化 ▼▼▼ ---
                b0_results = []
                srpt_avg_results = []
                v_ave_results = []
                x_ave_results = []
                # ----------------------------------------------------------------

                for i_particle in range(N_PARTICLES):
                    # --- ▼▼▼ 1個の原子（パーティクル）のシミュレーション ▼▼▼ ---

                    # --- 初期位置と速度をリセット ---
                    x, y = 0.0, RM  # 太陽直下点から放出
                    SRPt, tot_x, tot_V, tot_Nad = 0.0, 0.0, 0.0, 0.0
                    b0 = 0.0
                    Vms = Vms_ms / 1000.0

                    # --- ランダムな初期放出速度を設定 ---
                    ejection_angle_rad = np.random.uniform(0, PI)  # 上向き半球 (0-180度)
                    vx_eject = V_EJECT_SPEED * np.cos(ejection_angle_rad)
                    vy_eject = V_EJECT_SPEED * np.sin(ejection_angle_rad)

                    vx_ms0 = Vms
                    vx_ms = vx_ms0 + vx_eject
                    vy_ms = vy_eject

                    tau = T1AU * AU ** 2
                    itmax = int(tau * 5.0 / DT + 0.5)

                    for it in range(itmax):
                        # ... (放射圧 b の計算部分は変更なし) ...
                        w_na_d2 = 589.1582 * (1.0 - vx_ms / C)
                        w_na_d1 = 589.7558 * (1.0 - vx_ms / C)
                        if not (wl[0] <= w_na_d2 < wl[-1] and wl[0] <= w_na_d1 < wl[-1]):
                            break
                        gamma2 = np.interp(w_na_d2, wl, gamma)
                        gamma1 = np.interp(w_na_d1, wl, gamma)
                        m_na_wl = (w_na_d2 + w_na_d1) / 2.0
                        jl_nu = JL * 1e9 * ((m_na_wl * 1e-9) ** 2 / (C * 1e3))
                        J2 = sigma0_perdnu2 * jl_nu / AU ** 2 * gamma2
                        J1 = sigma0_perdnu1 * jl_nu / AU ** 2 * gamma1
                        b = (H / MASS_NA) * (J1 / (w_na_d1 * 1e-7) + J2 / (w_na_d2 * 1e-7))

                        # 惑星の影の処理（論文ではより複雑だが、ここでは単純化）
                        if x < 0 and np.sqrt(y ** 2) < RM:
                            b = 0.0

                        if it == 0:
                            b0 = b

                        Nad = np.exp(-DT * it / tau)

                        # ... (重力計算と速度・位置更新は変更なし) ...
                        accel_gx, accel_gy = 0.0, 0.0
                        if GRAVITY_ENABLED:
                            r_sq_grav = x ** 2 + y ** 2
                            if r_sq_grav > 0:  # ゼロ除算を避ける
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
                        x += ((vx_ms_prev + vx_ms) / 2.0 - vx_ms0) * DT
                        y += (vy_ms_prev + vy_ms) / 2.0 * DT

                        # --- ▼▼▼ 地表との衝突判定とバウンド処理 ▼▼▼ ---
                        r_current = np.sqrt(x ** 2 + y ** 2)
                        if r_current <= RM:
                            v_in_sq = vx_ms ** 2 + vy_ms ** 2
                            E_in = 0.5 * MASS_NA * (v_in_sq * 1e10)  # [erg]
                            E_T = K_BOLTZMANN * T_SURFACE
                            E_out = BETA * E_T + (1.0 - BETA) * E_in

                            v_out_sq = E_out / (0.5 * MASS_NA) / 1e10  # [km/s]^2
                            if v_out_sq > 0:
                                v_out_speed = np.sqrt(v_out_sq)
                            else:  # エネルギーが負になった場合など
                                v_out_speed = 0.0

                            surface_angle = np.arctan2(y, x)
                            rebound_angle_rad = surface_angle + np.random.uniform(-PI / 2.0, PI / 2.0)

                            vx_ms = v_out_speed * np.cos(rebound_angle_rad)
                            vy_ms = v_out_speed * np.sin(rebound_angle_rad)

                            x = RM * np.cos(surface_angle)
                            y = RM * np.sin(surface_angle)
                        # -----------------------------------------------

                        SRPt += b * DT * Nad
                        tot_x += x * DT * Nad
                        tot_V += vx_ms * DT * Nad
                        tot_Nad += Nad * DT

                    # --- ▲▲▲ 1個の原子のシミュレーション終了 ▲▲▲ ---
                    if tot_Nad > 0:
                        b0_results.append(b0)
                        srpt_avg_results.append(SRPt / tot_Nad)
                        v_ave_results.append(tot_V / tot_Nad - vx_ms0)
                        x_ave_results.append(tot_x / tot_Nad)

                # --- ▼▼▼ 全パーティクルの結果を平均化 ▼▼▼ ---
                if len(b0_results) > 0:
                    b0_avg = np.mean(b0_results)
                    srpt_avg_mean = np.mean(srpt_avg_results)
                    v_ave_mean = np.mean(v_ave_results)
                    x_ave_mean = np.mean(x_ave_results)
                    z_M = AU * np.sin(lat * D2R)
                    f_out.write(
                        f"{TAA:20.8f} {b0_avg:20.8f} {srpt_avg_mean:20.8f} {v_ave_mean:20.8f} {x_ave_mean:20.8f} {z_M:20.8f}\n")
                # ---------------------------------------------

                progress = (iTAA + 1) / total_sims
                bar_length = 40
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                print(f'  進捗: |{bar}| {iTAA + 1}/{total_sims} ({progress:.1%})完了', end='\r')

            print("\nシミュレーションが完了しました。")

    except FileNotFoundError:
        print(f"エラー: 軌道ファイル '{orbit_file}' が見つかりません。")
    except Exception as e:
        print(f"エラー: シミュレーション中に問題が発生しました: {e}")


if __name__ == '__main__':
    run_monte_carlo_simulation()