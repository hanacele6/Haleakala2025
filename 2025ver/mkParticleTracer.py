import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def visualize_successful_trajectory():
    """
    指定した単一のTAAにおいて、成功する軌道が見つかるまでシミュレーションを繰り返し、
    その最初の成功例の軌道と放射圧の変化を可視化する。
    """
    # --- ▼▼▼ 設定項目 ▼▼▼ ---
    TARGET_TAA = 150.0
    settings = {
        'V_EJECT_SPEED': 0.9, 'BETA': 0.5, 'T_SURFACE': 600.0,
        'T1AU': 61728.4, 'DT': 10.0,
    }
    # --- ▲▲▲ 設定はここまで ▲▲▲ ---

    # --- 物理定数とデータ読み込み (変更なし) ---
    constants = {
        'C': 299792.458, 'PI': np.pi, 'H': 6.626068e-34 * 1e7,
        'MASS_NA': 22.98976928 * 1.6605402e-24, 'RM': 2439.7,
        'GM_MERCURY': 2.2032e4, 'K_BOLTZMANN': 1.380649e-16
    }
    try:
        spec_data_np = np.loadtxt('SolarSpectrum_Na0.txt', usecols=(0, 3))
        orbit_data = np.loadtxt('orbit360.txt')
    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません。 {e}")
        sys.exit()

    # (スペクトル・定数準備は変更なし)
    wl, gamma = spec_data_np[:, 0], spec_data_np[:, 1]
    ME = 9.1093897e-28;
    E_CHARGE = 1.60217733e-19 * 2.99792458e9
    sigma_const = constants['PI'] * E_CHARGE ** 2 / ME / (constants['C'] * 1e5)
    sigma0_perdnu2 = sigma_const * 0.641;
    sigma0_perdnu1 = sigma_const * 0.320;
    JL = 5.18e14

    # 指定TAAのデータ取得 (変更なし)
    orbit_taa_col = orbit_data[:, 0]
    idx = np.abs(orbit_taa_col - TARGET_TAA).argmin()
    TAA, AU, lon, lat, Vms_ms = orbit_data[idx]
    print(f"指定されたTAA {TARGET_TAA}° に最も近い軌道データ (TAA={TAA:.2f}°) を使用します。")

    C, PI, H, MASS_NA, RM, GM_MERCURY, K_BOLTZMANN = constants.values()
    V_EJECT_SPEED, BETA, T_SURFACE, T1AU, DT = settings.values()
    Vms = Vms_ms / 1000.0
    tau = T1AU * AU ** 2
    itmax = int(tau * 3.0 / DT)

    # ▼▼▼ 修正点: 成功するまで試行を繰り返すループ ▼▼▼
    attempt_count = 0
    while True:
        attempt_count += 1
        print(f"\n--- 試行回数: {attempt_count} ---")

        # --- 各試行の前にシミュレーション状態をリセット ---
        x, y = -RM, 0.0
        # 地表面の法線角度を計算 (コード1と同じロジック)
        surface_normal_angle = np.arctan2(y, x)
        # 法線方向を基準としたランダムな放出角度を決定
        random_offset_angle = np.random.uniform(-PI / 2.0, PI / 2.0)
        ejection_angle_rad = surface_normal_angle + random_offset_angle

        vx_eject = V_EJECT_SPEED * np.cos(ejection_angle_rad)
        vy_eject = V_EJECT_SPEED * np.sin(ejection_angle_rad)
        vx_ms, vy_ms = vx_eject, vy_eject

        time_history, x_history, y_history, b_history = [], [], [], []
        bounce_points = []
        success = False  # 成功フラグ

        # --- メインループ (変更なし) ---
        for it in range(itmax):
            velocity_for_doppler = vx_ms + Vms
            # ... (放射圧 b の計算は変更なし) ...
            w_na_d2 = 589.1582 * (1.0 - velocity_for_doppler / C);
            w_na_d1 = 589.7558 * (1.0 - velocity_for_doppler / C)
            if not (wl[0] <= w_na_d2 < wl[-1] and wl[0] <= w_na_d1 < wl[-1]): break
            gamma2 = np.interp(w_na_d2, wl, gamma);
            gamma1 = np.interp(w_na_d1, wl, gamma)
            m_na_wl = (w_na_d2 + w_na_d1) / 2.0;
            jl_nu = JL * 1e9 * ((m_na_wl * 1e-9) ** 2 / (C * 1e3))
            J2 = sigma0_perdnu2 * jl_nu / AU ** 2 * gamma2;
            J1 = sigma0_perdnu1 * jl_nu / AU ** 2 * gamma1
            b = (H / MASS_NA) * (J1 / (w_na_d1 * 1e-7) + J2 / (w_na_d2 * 1e-7))

            # 履歴を追加
            time_history.append(it * DT);
            x_history.append(x);
            y_history.append(y);
            b_history.append(b)

            # ... (速度・位置更新は変更なし) ...
            r_sq_grav = x ** 2 + y ** 2;
            accel_gx, accel_gy = 0.0, 0.0
            if r_sq_grav > 0:
                r_grav = np.sqrt(r_sq_grav);
                grav_accel_total = GM_MERCURY / r_sq_grav
                accel_gx = -grav_accel_total * (x / r_grav);
                accel_gy = -grav_accel_total * (y / r_grav)
            accel_srp_x = b / 100000.0;
            total_accel_x = accel_srp_x + accel_gx;
            total_accel_y = accel_gy
            vx_ms += total_accel_x * DT;
            vy_ms += total_accel_y * DT;
            x += vx_ms * DT;
            y += vy_ms * DT

            # ▼▼▼ 修正点: ゴール判定時にフラグを立てる ▼▼▼
            if x >= 0:
                print(f"ゴールに到達しました！ 到達時間: {it * DT:.1f} 秒")
                # 最後の地点を記録
                time_history.append((it + 1) * DT);
                x_history.append(x);
                y_history.append(y);
                b_history.append(b)
                success = True
                break

            # ... (衝突判定は変更なし) ...
            r_current = np.sqrt(x ** 2 + y ** 2)
            if r_current <= RM:
                bounce_points.append((x, y))
                v_in_sq = vx_ms ** 2 + vy_ms ** 2;
                E_in = 0.5 * MASS_NA * (v_in_sq * 1e10)
                E_T = K_BOLTZMANN * T_SURFACE;
                E_out = BETA * E_T + (1.0 - BETA) * E_in
                v_out_sq = E_out / (0.5 * MASS_NA) / 1e10;
                v_out_speed = np.sqrt(v_out_sq) if v_out_sq > 0 else 0.0
                surface_angle = np.arctan2(y, x);
                ejection_angle_rad = surface_angle + np.random.uniform(-PI / 2.0, PI / 2.0)
                vx_ms = v_out_speed * np.cos(ejection_angle_rad);
                vy_ms = v_out_speed * np.sin(ejection_angle_rad)
                x = RM * np.cos(surface_angle);
                y = RM * np.sin(surface_angle)

        # ▼▼▼ 修正点: 試行の成否を判定 ▼▼▼
        if success:
            break  # 成功したので while ループを抜ける
        else:
            print("タイムアウトしました。新しい乱数で再試行します。")

    print(f"\n成功した軌道が見つかりました ({attempt_count}回目)。グラフを描画します。")
    # --- グラフ描画 (変更なし) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    mercury_circle = Circle((0, 0), RM, color='gray', alpha=0.5, label='Mercury')
    ax1.add_patch(mercury_circle)
    ax1.plot(x_history, y_history, '-', label='Trajectory', lw=1)
    ax1.plot(x_history[0], y_history[0], 'go', label='Start')
    ax1.plot(x_history[-1], y_history[-1], 'ro', label='End')
    if bounce_points:
        bx, by = zip(*bounce_points);
        ax1.plot(bx, by, 'rx', markersize=8, label='Bounce')
    ax1.axvline(0, color='k', linestyle='--', lw=1, label='Terminator')
    ax1.set_title(f'Successful Particle Trajectory (TAA = {TAA:.1f} deg)')
    ax1.set_xlabel('X [km] (Sun is to the left)');
    ax1.set_ylabel('Y [km]')
    ax1.set_aspect('equal');
    ax1.legend();
    ax1.grid(True)
    ax2.plot(time_history, b_history, '-')
    ax2.set_title(f'Solar Radiation Pressure vs. Time (TAA = {TAA:.1f} deg)')
    ax2.set_xlabel('Time [s]');
    ax2.set_ylabel('Radiation Pressure b [cm/s^2]')
    ax2.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    visualize_successful_trajectory()