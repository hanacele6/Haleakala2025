import numpy as np
import sys
import matplotlib.pyplot as plt


def calculate_and_plot_initial_srp():
    """
    Na原子が最初に受ける太陽放射圧(b0)を計算し、
    ファイルに保存すると同時にグラフでプロットする。
    """
    # --- 1. 物理定数 (省略) ---
    C = 299792.458;
    PI = np.pi;
    JL = 5.18e14;
    ME = 9.1093897e-28
    E_CHARGE = 1.60217733e-19 * 2.99792458e9;
    H = 6.626068e-34 * 1e7;
    MASS_NA = 22.98976928 * 1.6605402e-24

    # --- 2. データファイル ---
    spectrum_file = 'SolarSpectrum_Na0.txt'
    orbit_file = 'orbit360.txt'
    output_file = 'TAA_SRP_b0_only.txt'

    try:
        spec_data = np.loadtxt(spectrum_file, usecols=(0, 3))
        wl, gamma = spec_data[:, 0], spec_data[:, 1]
    except FileNotFoundError:
        print(f"エラー: スペクトルファイル '{spectrum_file}' が見つかりません。")
        sys.exit()

    # (スペクトルソート、定数計算は省略)
    if not np.all(np.diff(wl) > 0):
        sort_indices = np.argsort(wl);
        wl, gamma = wl[sort_indices], gamma[sort_indices]
    sigma_const = PI * E_CHARGE ** 2 / ME / (C * 1e5);
    sigma0_perdnu2 = sigma_const * 0.641;
    sigma0_perdnu1 = sigma_const * 0.320

    print("初期放射圧(b0)の計算を開始します...")

    # 結果を保存するためのリストを準備
    taa_results = []
    b0_results = []

    try:
        with open(orbit_file, 'r') as f_orbit, open(output_file, 'w') as f_out:
            header = f"{'TAA':>19s} {'b0[cm/s^2]':>20s}\n"
            f_out.write(header)

            orbit_lines = f_orbit.readlines()
            total_sims = len(orbit_lines)

            for iTAA, line in enumerate(orbit_lines):
                TAA, AU, lon, lat, Vms_ms = map(float, line.split())
                Vms = Vms_ms / 1000.0
                vx_ms = Vms
                w_na_d2 = 589.1582 * (1.0 - vx_ms / C);
                w_na_d1 = 589.7558 * (1.0 - vx_ms / C)
                gamma2 = np.interp(w_na_d2, wl, gamma);
                gamma1 = np.interp(w_na_d1, wl, gamma)
                m_na_wl = (w_na_d2 + w_na_d1) / 2.0;
                jl_nu = JL * 1e9 * ((m_na_wl * 1e-9) ** 2 / (C * 1e3))
                J2 = sigma0_perdnu2 * jl_nu / AU ** 2 * gamma2;
                J1 = sigma0_perdnu1 * jl_nu / AU ** 2 * gamma1
                b0 = (H / MASS_NA) * (J1 / (w_na_d1 * 1e-7) + J2 / (w_na_d2 * 1e-7))

                f_out.write(f"{TAA:20.8f} {b0:20.8f}\n")

                # グラフ描画用に結果をリストに追加
                taa_results.append(TAA)
                b0_results.append(b0)

                print(f'  進捗: {iTAA + 1}/{total_sims} 完了', end='\r')

        print(f"\n計算が完了しました。結果は'{output_file}'に出力されました。")

        # ▼▼▼ グラフ描画機能 ▼▼▼
        print("グラフをプロットします...")
        plt.figure(figsize=(10, 6))
        plt.plot(taa_results, b0_results, marker='.', linestyle='-', markersize=4)
        plt.title('Initial Solar Radiation Pressure (b0) vs. TAA')
        plt.xlabel('True Anomaly Angle (TAA) [degrees]')
        plt.ylabel('Initial SRP (b0) [cm/s^2]')
        plt.grid(True)
        plt.xticks(np.arange(0, 361, 30))  # X軸の目盛りを30度ごとに設定
        plt.show()
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    except Exception as e:
        print(f"\nエラー: {e}")


if __name__ == '__main__':
    calculate_and_plot_initial_srp()