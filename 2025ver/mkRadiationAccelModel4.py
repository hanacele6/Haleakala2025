import numpy as np
import sys
from multiprocessing import Pool, cpu_count


# --- ▼▼▼ 1. 1個の原子のシミュレーションを行う「ワーカー関数」を定義 ▼▼▼ ---
# この関数が各CPUコアで並列に実行される
def simulate_single_particle(args):
    """1個の原子の軌道をシミュレートし、結果を返す"""
    # --- 引数を受け取る ---
    # 共通の定数や設定
    constants, settings, spec_data = args['consts'], args['settings'], args['spec']
    # このTAAに固有の軌道パラメータ
    TAA, AU, lon, lat, Vms_ms = args['orbit']

    # --- 定数と設定を展開 ---
    C, PI, H, MASS_NA, RM, GM_MERCURY, K_BOLTZMANN = constants.values()
    GRAVITY_ENABLED, V_EJECT_SPEED, BETA, T_SURFACE, T1AU, DT = settings.values()
    wl, gamma, sigma0_perdnu2, sigma0_perdnu1, JL = spec_data.values()

    # --- シミュレーション初期化 ---
    #x, y = 0.0, RM
    x, y = -RM, 0.0
    SRPt, tot_x, tot_V, tot_Nad = 0.0, 0.0, 0.0, 0.0
    b0 = 0.0
    Vms = Vms_ms / 1000.0

    # --- ランダムな初期放出速度を設定 ---
    ejection_angle_rad = np.random.uniform(0, PI)
    vx_eject = V_EJECT_SPEED * np.cos(ejection_angle_rad)
    vy_eject = V_EJECT_SPEED * np.sin(ejection_angle_rad)

    vx_ms0 = Vms
    vx_ms = vx_ms0 + vx_eject
    vy_ms = vy_eject

    tau = T1AU * AU ** 2
    itmax = int(tau * 5.0 / DT + 0.5)
    time_to_terminator = -1.0

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

        if x < 0 and np.sqrt(y ** 2) < RM:
            b = 0.0

        if it == 0:
            b0 = b

        Nad = np.exp(-DT * it / tau)

        # ... (重力計算と速度・位置更新は変更なし) ...
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
        x += ((vx_ms_prev + vx_ms) / 2.0 - vx_ms0) * DT
        y += (vy_ms_prev + vy_ms) / 2.0 * DT

        # ▼▼▼ 追加：昼夜境界線（x=0）への到達を判定 ▼▼▼
        #if x <= 0 and time_to_terminator < 0:
        #    time_to_terminator = it * DT
        #    break
        if x >= 0 and time_to_terminator < 0:  # xが正になった瞬間を捉える
            time_to_terminator = it * DT
            break

        # ... (地表との衝突判定とバウンド処理は変更なし) ...
        r_current = np.sqrt(x ** 2 + y ** 2)
        if r_current <= RM:
            v_in_sq = vx_ms ** 2 + vy_ms ** 2
            E_in = 0.5 * MASS_NA * (v_in_sq * 1e10)
            E_T = K_BOLTZMANN * T_SURFACE
            E_out = BETA * E_T + (1.0 - BETA) * E_in
            v_out_sq = E_out / (0.5 * MASS_NA) / 1e10
            v_out_speed = np.sqrt(v_out_sq) if v_out_sq > 0 else 0.0
            surface_angle = np.arctan2(y, x)
            rebound_angle_rad = surface_angle + np.random.uniform(-PI / 2.0, PI / 2.0)
            vx_ms = v_out_speed * np.cos(rebound_angle_rad)
            vy_ms = v_out_speed * np.sin(rebound_angle_rad)
            x = RM * np.cos(surface_angle)
            y = RM * np.sin(surface_angle)

        SRPt += b * DT * Nad
        tot_x += x * DT * Nad
        tot_V += vx_ms * DT * Nad
        tot_Nad += Nad * DT

    if tot_Nad > 0:
        return {
            'b0': b0,
            'srpt_avg': SRPt / tot_Nad,
            'v_ave': tot_V / tot_Nad - vx_ms0,
            'x_ave': tot_x / tot_Nad,
            'tm': time_to_terminator
        }
    return None


def run_main_simulation():
    """メインの制御関数"""
    # --- シミュレーション設定 ---
    settings = {
        'GRAVITY_ENABLED': True,# True: 重力あり, False: 重力なし
        'V_EJECT_SPEED': 0.9,# 原子の初期放出速度 [km/s] (例: 熱的な速度)
        'BETA': 0.5,# エネルギー順応係数 (論文の結論に近い値)
        'T_SURFACE': 400.0,# 水星の地表温度 [K] (バウンド計算用)
        'T1AU': 168918.0,# Na原子の寿命（Huebnerの理論値）[s]
        'DT': 10.0,# 時間ステップ [s]
    }
    N_PARTICLES = 1000# 各軌道位置(TAA)でシミュレートする原子の数

    # --- 物理定数 ---
    constants = {
        'C': 299792.458, 'PI': np.pi, 'H': 6.626068e-34 * 1e4 * 1e3,
        'MASS_NA': 22.98976928 * 1.6605402e-27 * 1e3, 'RM': 2439.7,
        'GM_MERCURY': 2.2032e4, 'K_BOLTZMANN': 1.380649e-16
    }

    # --- データファイルの読み込み ---
    spectrum_file = 'SolarSpectrum_Na0.txt'
    orbit_file = 'orbit360.txt'
    output_file = f"TAA_MigrationTime_BETA{settings['BETA']}.txt"
    print(f"モード: 移動時間計算, β={settings['BETA']}, N={N_PARTICLES}, 並列処理有効")

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
            header = f"{'TAA':>10s},{'MigrationTime_avg[s]':>25s}\n"
            f_out.write(header)

            orbit_lines = f_orbit.readlines()
            total_sims = len(orbit_lines)

            for iTAA, line in enumerate(orbit_lines):
                TAA, AU, lon, lat, Vms_ms = map(float, line.split())

                task_args = {
                    'consts': constants, 'settings': settings, 'spec': spec_data_dict,
                    'orbit': (TAA, AU, lon, lat, Vms_ms)
                }
                tasks = [task_args] * N_PARTICLES

                with Pool(processes=cpu_count()) as pool:
                    results = pool.map(simulate_single_particle, tasks)

                # ターミネーターに到達したパーティクル（tmが0以上）だけをフィルタリング
                valid_results = [r['tm'] for r in results if r is not None and r['tm'] >= 0]

                if len(valid_results) > 0:
                    # 到達したパーティクルの平均移動時間を計算
                    tm_avg = np.mean(valid_results)
                else:
                    # どのパーティクルも到達しなかった場合（非常に長い時間かかる場合など）
                    tm_avg = -1.0

                    # TAAと計算した平均移動時間のみをファイルに書き出す
                f_out.write(f"{TAA:10.4f},{tm_avg:25.4f}\n")
                # --------------------------------------------------------

                progress = (iTAA + 1) / total_sims
                bar_length = 40
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                print(f'  進捗: |{bar}| {iTAA + 1}/{total_sims} ({progress:.1%})完了', end='\r')

            print("\nシミュレーションが完了しました。")
    except Exception as e:
        print(f"エラー: シミュレーション中に問題が発生しました: {e}")


if __name__ == '__main__':
    run_main_simulation()