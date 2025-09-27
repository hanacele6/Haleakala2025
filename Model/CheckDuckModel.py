import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sys

def calculate_dusk_model_combined():
    """
    水星の軌道データと太陽スペクトルから、各軌道位置での初期太陽放射圧(b0)を直接計算し、
    それを用いてナトリウム大気の「duskモデル」を評価する。
    粒子追跡シミュレーションを省略し、2つのプロセスを1つに統合したコード。
    """
    print("シミュレーションを開始します...")

    # --- 1. 物理定数 ---
    # SRAシミュレーションから持ってきた定数
    C = 299792.458          # 光速 [km/s]
    PI = np.pi
    JL = 5.18e14            # 1AUでの太陽フラックス [phs/s/cm2/nm]
    ME = 9.1093897e-31 * 1e3 # 電子の質量 [g]
    E_CHARGE = 1.60217733e-19 * 2.99792458e8 * 10.0 # 電子の電荷 [esu]
    H = 6.626068e-34 * 1e4 * 1e3 # プランク定数 [cm2*g/s]
    MASS_NA = 22.98976928 * 1.6605402e-27 * 1e3 # Na原子の質量 [g]

    # Duskモデルから持ってきた定数
    BASE_LIFETIME = 169200.0  # 光脱離の基準タイムスケール τ [s]
    BASE_PHOTON_FLUX = 4.6e7  # 基準光子フラックス Φ₀ [atoms/cm^2/s]
    MERCURY_RADIUS = 2.44e8   # 水星の半径 Rm [cm]

    # --- 2. 入力ファイル ---
    spectrum_file = 'SolarSpectrum_Na0.txt'
    orbit_file = 'orbit360.txt'

    # --- 3. データの読み込み ---
    # 太陽スペクトルデータの読み込み
    try:
        spec_data = np.loadtxt(spectrum_file, usecols=(0, 3))
        wl, gamma = spec_data[:, 0], spec_data[:, 1]
    except FileNotFoundError:
        print(f"エラー: スペクトルファイル '{spectrum_file}' が見つかりません。")
        sys.exit()
    except Exception as e:
        print(f"エラー: スペクトルファイルの読み込み中に問題が発生しました: {e}")
        sys.exit()

    # 波長データがソートされているか確認
    if not np.all(np.diff(wl) > 0):
        print("警告: スペクトルデータの波長がソートされていません。ソートします。")
        sort_indices = np.argsort(wl)
        wl, gamma = wl[sort_indices], gamma[sort_indices]

    # --- 4. 散乱断面積に関連する定数の計算 ---
    sigma_const = PI * E_CHARGE ** 2 / ME / (C * 1e5)
    sigma0_perdnu2 = sigma_const * 0.641
    sigma0_perdnu1 = sigma_const * 0.320

    # --- 5. b0の計算とデータ格納 ---
    # 結果を格納するためのリストを初期化
    taa_degrees_list = []
    b0_values_list = []
    sun_distance_au_list = []

    print(f"軌道ファイル '{orbit_file}' を読み込み、b0 を計算しています...")
    try:
        with open(orbit_file, 'r') as f_orbit:
            for line in f_orbit:
                try:
                    TAA, AU, _, _, Vms_ms = map(float, line.split())
                except ValueError:
                    continue # 空行などをスキップ

                # 初期速度条件 (粒子追跡はしないので、放出直後の速度のみ考慮)
                Vms = Vms_ms / 1000.0 # [km/s]
                vx_eject = 0.0
                vx_ms = Vms + vx_eject

                # ドップラーシフトした波長の計算
                w_na_d2 = 589.1582 * (1.0 - vx_ms / C)
                w_na_d1 = 589.7558 * (1.0 - vx_ms / C)

                # 波長がスペクトル範囲内にあるかチェック
                if not (wl[0] <= w_na_d2 < wl[-1] and wl[0] <= w_na_d1 < wl[-1]):
                    # 範囲外の場合はb0=0としておくか、警告を出す
                    b0 = 0.0
                else:
                    # スペクトルデータからガンマ値を内挿
                    gamma2 = np.interp(w_na_d2, wl, gamma)
                    gamma1 = np.interp(w_na_d1, wl, gamma)

                    # 放射圧による加速度 b (b0) の計算
                    m_na_wl = (w_na_d2 + w_na_d1) / 2.0
                    jl_nu = JL * 1e9 * ((m_na_wl * 1e-9) ** 2 / (C * 1e3))
                    J2 = sigma0_perdnu2 * jl_nu / AU ** 2 * gamma2
                    J1 = sigma0_perdnu1 * jl_nu / AU ** 2 * gamma1

                    b0 = (H / MASS_NA) * (J1 / (w_na_d1 * 1e-7) + J2 / (w_na_d2 * 1e-7))

                # 計算結果をリストに追加
                taa_degrees_list.append(TAA)
                b0_values_list.append(b0)
                sun_distance_au_list.append(AU)

    except FileNotFoundError:
        print(f"エラー: 軌道ファイル '{orbit_file}' が見つかりません。")
        sys.exit()

    # リストをNumPy配列に変換
    taa_degrees = np.array(taa_degrees_list)
    b0_values = np.array(b0_values_list)
    sun_distance_au = np.array(sun_distance_au_list)

    print("b0 の計算が完了しました。Duskモデルを計算します...")

    # --- 6. Duskモデルの計算 (ベクトル化) ---
    # 太陽光フラックス(Φ)を計算
    photon_flux = BASE_PHOTON_FLUX * (0.306 / sun_distance_au) ** 2
    #photon_flux = BASE_PHOTON_FLUX  /( sun_distance_au ** 2 )

    # 距離に応じた寿命(τ1)を計算
    distance_adjusted_lifetime = BASE_LIFETIME * (sun_distance_au ** 2)

    # 粒子が水星表面から脱出するまでの時間(tm)を計算
    # b0_values (cm/s^2) と MERCURY_RADIUS (cm) で単位が揃っている
    # ゼロ除算を避けるために微小値を追加
    migration_time = np.sqrt(2 * MERCURY_RADIUS / (b0_values + 1e-9))

    # メインのモデル計算
    dusk_model = photon_flux * distance_adjusted_lifetime * (1 - np.exp(-migration_time / distance_adjusted_lifetime))


    # --- 7. 結果の可視化 ---
    print("計算結果をグラフで表示します...")
    try:
        font_path = 'C:/Windows/Fonts/meiryo.ttc'  # Windowsの例
        jp_font = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = jp_font.get_name()
    except FileNotFoundError:
        print("日本語フォントが見つかりません。グラフのラベルが文字化けする可能性があります。")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(taa_degrees, dusk_model, color='dodgerblue', lw=2.5, label='Dusk Model')
    ax.set_title('水星ナトリウム大気 Duskモデル (b0 直接計算版)', fontsize=16)
    ax.set_xlabel('真近点離角 (TAA) [度]', fontsize=12)
    ax.set_ylabel('ナトリウム原子柱密度 [atoms/cm$^2$]', fontsize=12)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # b0の値も同時にプロットして相関を確認する（右軸）
    ax2 = ax.twinx()
    ax2.plot(taa_degrees, b0_values, color='coral', linestyle='--', lw=2, label='初期放射圧 (b0)')
    ax2.set_ylabel('初期放射圧 (b0) [cm/s$^2$]', fontsize=12,)
    ax2.tick_params(axis='y',)

    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
    plt.tight_layout()
    plt.show()

    # --- 8. 結果のファイル出力 ---
    # --- 8. 結果のファイル出力 ---
    # output_data = np.column_stack((taa_degrees, sun_distance_au, b0_values, dusk_model))
    # output_filename = 'dusk_model_combined_output.csv'
    # np.savetxt( # np.savxt というタイポがあったため修正
    #     output_filename,
    #     output_data,
    #     fmt='%.6f',
    #     delimiter=',',
    #     header='TAA_deg,Sun_Distance_AU,b0_cm_s2,Dusk_Model_atoms_cm2'
    # )
    # print(f"計算が完了し、'{output_filename}' に結果を出力しました。")
    print("計算とグラフの表示が完了しました。")

if __name__ == '__main__':
    calculate_dusk_model_combined()
