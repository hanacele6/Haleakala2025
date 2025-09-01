import numpy as np
import sys


def run_sra_simulation():
    """
    水星から放出されたNa原子の運動をシミュレートする。
    太陽放射圧に加え、水星の重力の影響をON/OFFで切り替え可能。
    """
    # --- シミュレーション設定 ---
    GRAVITY_ENABLED = True  # True: 重力ありの計算, False: 重力なしの計算

    # --- 1. 物理定数 ---
    C = 299792.458  # 光速 [km/s]
    PI = np.pi
    D2R = PI / 180.0
    JL = 5.18e14  # 1AUでの太陽フラックス [phs/s/cm2/nm]
    ME = 9.1093897e-31 * 1e3  # 電子の質量 [g]
    E_CHARGE = 1.60217733e-19 * 2.99792458e8 * 10.0  # 電子の電荷 [esu]
    H = 6.626068e-34 * 1e4 * 1e3  # プランク定数 [cm2*g/s]
    MASS_NA = 22.98976928 * 1.6605402e-27 * 1e3  # Na原子の質量 [g]
    RM = 2439.7  # 水星の半径 [km]

    # G * M_mercury [km^3/s^2] (万有引力定数 * 水星の質量)
    GM_MERCURY = 2.2032e4

    # --- 2. シミュレーションパラメータ ---
    T1AU = 168918.0  # Na原子の寿命（Huebnerの理論値）[s]
    DT = 10.0  # 時間ステップ [s]

    # --- 3. データファイルの読み込み ---
    spectrum_file = 'SolarSpectrum_Na0.txt'
    orbit_file = 'orbit360.txt'
    if GRAVITY_ENABLED:
        output_file = 'TAA_SRP_Alt360_grav.txt'
        print("モード: 重力あり")
    else:
        output_file = 'TAA_SRP_Alt360_nograv.txt'
        print("モード: 重力なし")

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
            header = f"{'TAA':>19s} {'b0[cm/s^2]':>20s} {'SRPt_avg[cm/s^2]':>20s} {'V_ave[km/s]':>20s} {'x_ave[km]':>20s} {'z_M[km]':>20s}\n"
            f_out.write(header)

            total_sims = 360
            for iTAA in range(total_sims):
                line = f_orbit.readline()
                if not line:
                    print(f"\n警告: 軌道ファイルが{total_sims}行未満で終了しました (行: {iTAA + 1})。")
                    break

                TAA, AU, lon, lat, Vms_ms = map(float, line.split())

                x, y = 0.0, RM
                x, y = -RM, 0.0
                SRPt, tot_x, tot_V, tot_Nad = 0.0, 0.0, 0.0, 0.0
                b0 = 0.0
                Vms = Vms_ms / 1000.0
                vx_eject = 0.0
                #vy_eject = 3.58
                vy_eject = 0.0
                vx_ms0 = Vms
                vx_ms = vx_ms0 + vx_eject
                vy_ms = vy_eject

                tau = T1AU * AU ** 2
                itmax = int(tau * 5.0 / DT + 0.5)

                for it in range(itmax):
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

                    if abs(y) < RM and x >= 0.0:
                        b = 0.0

                    if it == 0:
                        b0 = b

                    Nad = np.exp(-DT * it / tau)

                    # --- ▼▼▼ 変更点: ここから重力計算と速度・位置更新 ▼▼▼ ---

                    # 1. 重力加速度の計算 (有効な場合)
                    accel_gx, accel_gy = 0.0, 0.0
                    if GRAVITY_ENABLED:
                        r_sq = x ** 2 + y ** 2
                        # ゼロ除算を避けるため、また水星半径より内側は考えない
                        if r_sq > RM ** 2:
                            r = np.sqrt(r_sq)
                            # 水星中心に向かう重力加速度 g = GM/r^2 [km/s^2]
                            grav_accel_total = GM_MERCURY / r_sq
                            # 加速度をx, y成分に分解
                            accel_gx = -grav_accel_total * (x / r)
                            accel_gy = -grav_accel_total * (y / r)

                    # 2. 速度と位置の更新
                    vx_ms_prev = vx_ms
                    vy_ms_prev = vy_ms  # y方向の速度も保存

                    # 放射圧の加速度bを[cm/s^2] -> [km/s^2]に変換
                    accel_srp_x = b / 100.0 / 1000.0

                    # 合計の加速度 [km/s^2]
                    total_accel_x = accel_srp_x + accel_gx
                    total_accel_y = accel_gy  # y方向の放射圧は0と仮定

                    # 速度を更新
                    vx_ms += total_accel_x * DT
                    vy_ms += total_accel_y * DT

                    # 位置を更新 (平均速度を用いる)
                    x += ((vx_ms_prev + vx_ms) / 2.0 - vx_ms0) * DT
                    y += (vy_ms_prev + vy_ms) / 2.0 * DT

                    SRPt += b * DT * Nad
                    tot_x += x * DT * Nad
                    tot_V += vx_ms * DT * Nad
                    tot_Nad += Nad * DT

                if tot_Nad > 0:
                    x_ave = tot_x / tot_Nad
                    V_ave = tot_V / tot_Nad - vx_ms0
                    SRPt_avg = SRPt / tot_Nad
                    z_M = AU * np.sin(lat * D2R)
                    f_out.write(f"{TAA:20.8f} {b0:20.8f} {SRPt_avg:20.8f} {V_ave:20.8f} {x_ave:20.8f} {z_M:20.8f}\n")

                progress = (iTAA + 1) / total_sims
                bar_length = 40
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                print(f'  進捗: |{bar}| {iTAA + 1}/{total_sims} ({progress:.1%})完了', end='\r')

            print()
            print(f"シミュレーションが完了しました。結果は'{output_file}'に出力されました。")

    except FileNotFoundError:
        print(f"エラー: 軌道ファイル '{orbit_file}' が見つかりません。")
    except Exception as e:
        print(f"エラー: シミュレーション中に問題が発生しました: {e}")


if __name__ == '__main__':
    run_sra_simulation()
