import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def run_dusk_model_improved():
    """
    水星の公転軌道上の各位置における「duskモデル」の値を計算し、
    結果をグラフで可視化するとともに、ファイルに出力する。

    このモデルは、主に太陽光による表面からの光脱離によって生成される
    ナトリウム大気の平衡状態をシミュレートする。
    """
    # --- 定数と物理パラメータ ---
    # 定数名を分かりやすくし、単位や意味をコメントで明記
    BASE_LIFETIME = 169200.0  # 光脱離の基準タイムスケール τ [s]
    BASE_PHOTON_FLUX = 4.6e7  # 基準光子フラックス Φ₀ [atoms/cm^2/s]
    MERCURY_RADIUS = 2.44e8  # 水星の半径 Rm [cm]
    ECCENTRICITY = 0.2056  # 軌道の離心率 e
    SEMI_MAJOR_AXIS = 0.3871  # 軌道長半径 a [AU]

    # --- 入力ファイルの読み込み ---
    # TAA_SRP.txtから太陽放射圧(SRP)のデータを読み込む
    try:
        # 1列目のTAAはここでは使わないので、srp_valuesのみ取得
        _, srp_values = np.loadtxt('TAA_SRP3.txt', unpack=True)
    except (IOError, ValueError):
        print("エラー: 'TAA_SRP.txt' が見つからないか、形式が正しくありません。")
        print("処理を続けるためにダミーデータを生成します。")
        # 360日分の仮のSRP値 (加速度 cm/s^2) を生成
        srp_values = np.full(360, 20.0 + 5.0 * np.cos(np.deg2rad(np.arange(360))))

    # --- モデル計算 (NumPyによるベクトル化) ---
    # ベクトル化により、forループなしで全360点の計算を一括実行できる

    # 1. 角度の配列 (0°から359°) を準備
    taa_degrees = np.arange(360)
    taa_radians = np.deg2rad(taa_degrees)

    # 2. 太陽からの距離Rを各TAAに対して計算 [AU]
    # 楕円軌道の極座標表示式 r = l / (1 + e*cosθ) を使用
    l = SEMI_MAJOR_AXIS * (1 - ECCENTRICITY ** 2)
    sun_distance_au = l / (1 + ECCENTRICITY * np.cos(taa_radians))

    # 3. 太陽光フラックス(Φ)を計算
    # フラックスは距離の2乗に反比例する
    # 0.306 AUは水星の近日点距離に由来する基準距離
    photon_flux = BASE_PHOTON_FLUX * (0.306 / sun_distance_au) ** 2

    # 4. 距離に応じた寿命(τ1)を計算
    # 寿命は距離の2乗に比例すると仮定
    distance_adjusted_lifetime = BASE_LIFETIME * sun_distance_au ** 2

    # 5. 粒子が水星表面から脱出するまでの時間(tm)を計算
    migration_time = np.sqrt(2 * MERCURY_RADIUS / srp_values)

    # 6. メインのモデル計算
    # N = Φ * τ * (1 - exp(-tm/τ))
    dusk_model = photon_flux * distance_adjusted_lifetime * (1 - np.exp(-migration_time / distance_adjusted_lifetime))

    # --- 結果の可視化 ---
    print("計算結果をグラフで表示します...")

    # 日本語フォントの設定
    try:
        font_path = 'C:/Windows/Fonts/meiryo.ttc'  # Windowsの例
        jp_font = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = jp_font.get_name()
    except FileNotFoundError:
        print("日本語フォントが見つかりません。グラフのラベルが文字化けする可能性があります。")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(taa_degrees, dusk_model, color='dodgerblue', lw=2.5)
    ax.set_title('水星ナトリウム大気 Duskモデル', fontsize=16)
    ax.set_xlabel('真近点角 (TAA) [度]', fontsize=12)
    ax.set_ylabel('ナトリウム原子柱密度 [atoms/cm$^2$]', fontsize=12)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=10)
    # y軸を科学記数法で見やすくする
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.tight_layout()
    plt.show()

    # --- 結果のファイル出力 ---
    output_data = np.column_stack((taa_degrees, dusk_model))
    output_filename = 'dusk_model_output.csv'
    np.savetxt(
        output_filename,
        output_data,
        fmt='%.4f',
        delimiter=',',
        header='TAA,Dusk_Model_Value'
    )
    print(f"計算が完了し、'{output_filename}' に結果を出力しました。")


if __name__ == '__main__':
    run_dusk_model_improved()