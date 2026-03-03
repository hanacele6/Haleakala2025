import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def run_dusk_model_final():
    """
    モンテカルロ法で計算したリアルな「移動時間」を使い、
    最終的なナトリウム原子の柱密度を計算・可視化する。
    """

    # BETAの値は、シミュレーションで使った値と同じにする
    BETA = 0.5
    migration_time_file = f"MC_FullResults_BETA{BETA}_60_DT.txt"
    #migration_time_file = f"MC_FullResults_BETA0.5.txt"
    # --------------------------------------------------------------------------

    # --- 定数と物理パラメータ ---
    # 論文やシミュレーションの設定と値を合わせる
    #LIFETIME_AT_1AU = 168918.0  # Na原子の寿命 @1AU [s] (シミュレーションの値)
    LIFETIME_AT_1AU = 61728.4 #実験値
    # 論文で示唆された範囲内の値 (2-5e7)
    FLUX_AT_PERIHELION = 4.6e7  # 近日点(0.307AU)での基準光子フラックス Φ₀ [atoms/cm^2/s]

    ECCENTRICITY = 0.2056
    SEMI_MAJOR_AXIS = 0.3871

    SOURCE_POWER_LAW = 1.5  # 2.0ならR⁻², 1.5ならR⁻¹⁵ モデル
    # --------------------------------------------------------------------------

    try:
        # TAAとMigrationTimeをファイルから読み込む
        taa_degrees, migration_time = np.loadtxt(
            migration_time_file,
            #delimiter=',',
            unpack=True,
            usecols = (0, 5),
            skiprows=1
        )
        # 単位を時間に変換
        migration_time_hr = migration_time / 3600.0
        print(f"'{migration_time_file}' から移動時間データを読み込みました。")
    except (IOError, ValueError):
        print(f"エラー: '{migration_time_file}' に問題があります。")
        return  # プログラムを終了
    # ----------------------------------------------------

    # --- モデル計算 (NumPyによるベクトル化) ---
    taa_radians = np.deg2rad(taa_degrees)

    # 1. 太陽からの距離Rを各TAAに対して計算 [AU]
    l = SEMI_MAJOR_AXIS * (1 - ECCENTRICITY ** 2)
    sun_distance_au = l / (1 + ECCENTRICITY * np.cos(taa_radians))

    # 2. 太陽光フラックス(Φ)を計算
    # 0.307 AUは水星の近日点距離
    photon_flux = FLUX_AT_PERIHELION * (0.307 / sun_distance_au) ** SOURCE_POWER_LAW

    # 3. 距離に応じた光電離寿命(τ)を計算 [s]
    distance_adjusted_lifetime = (LIFETIME_AT_1AU * sun_distance_au ** 2)

    # 4. メインのモデル計算 (論文の式2に基づく修正版)
    # まず、全ての点で輸送損失がないと仮定して柱密度を計算 (N = Φ * τ)
    dusk_model = photon_flux * distance_adjusted_lifetime

    # ターミネーターに到達した原子 (tm >= 0) が存在するインデックスを取得
    valid_tm_mask = migration_time >= 0

    # 該当するインデックスのデータに対してのみ、輸送損失係数を適用
    # N' = (Φ * τ) * (1 - exp(-tm/τ))
    loss_factor = (1 - np.exp(-migration_time[valid_tm_mask] / distance_adjusted_lifetime[valid_tm_mask]))
    dusk_model[valid_tm_mask] = dusk_model[valid_tm_mask] * loss_factor



    # --- 結果の可視化 (変更なし) ---
    print("計算結果をグラフで表示します...")
    try:
        font_path = 'C:/Windows/Fonts/meiryo.ttc'
        jp_font = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = jp_font.get_name()
    except FileNotFoundError:
        print("日本語フォントが見つかりません。")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(taa_degrees, dusk_model, color='dodgerblue', lw=2.5)
    ax.set_title('水星ナトリウム大気 最終モデル (詳細シミュレーション使用)', fontsize=16)
    ax.set_xlabel('真近点角 (TAA) [度]', fontsize=12)
    ax.set_ylabel('ナトリウム原子柱密度 [atoms/cm$^2$]', fontsize=12)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.tight_layout()
    plt.show()

    # --- 結果のファイル出力 (変更なし) ---
    output_data = np.column_stack((taa_degrees, dusk_model))
    output_filename = 'dusk_model_output2.csv'
    np.savetxt(output_filename, output_data, fmt='%.4f', delimiter=',', header='TAA,Column_Density')
    print(f"計算が完了し、'{output_filename}' に結果を出力しました。")


if __name__ == '__main__':
    run_dusk_model_final()