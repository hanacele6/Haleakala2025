import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
# 1. ユーザーコードから抽出した物理モデル関数
# ==============================================================================
def calculate_surface_temperature_leblanc_subsolar(AU):
    """
    ユーザーコードの calculate_surface_temperature_leblanc を
    太陽直下点 (cos_theta = 1) 用に簡略化したもの。

    Parameters:
        AU (float or np.array): 太陽からの距離 [au]
    Returns:
        float or np.array: 表面温度 [K]
    """
    # コード内の定数
    T0 = 100.0
    T1 = 600.0
    REF_AU = 0.306  # 近日点付近の基準距離

    # スケーリング係数 (ユーザーコード準拠: scaling = np.sqrt(0.306 / AU))
    scaling = np.sqrt(REF_AU / AU)

    # 太陽直下点では cos_theta = 1 なので、(cos_theta ** 0.25) は 1
    # T = T0 + T1 * 1.0 * scaling
    return T0 + T1 * scaling


# ==============================================================================
# 2. 水星の軌道計算 (ファイルがない場合のための理論式)
# ==============================================================================
def get_mercury_distance_theoretical(taa_deg):
    """
    TAAから水星の太陽距離(AU)を計算する
    (軌道長半径 a と 離心率 e を使用)
    """
    # 水星の軌道要素 (近似値)
    a = 0.387098  # 軌道長半径 [AU]
    e = 0.205630  # 離心率

    # TAAをラジアンに変換
    nu = np.deg2rad(taa_deg)

    # 楕円軌道の極方程式: r = a(1 - e^2) / (1 + e cos(ν))
    r_au = a * (1 - e ** 2) / (1 + e * np.cos(nu))
    return r_au


# ==============================================================================
# 3. メイン処理
# ==============================================================================
def main():
    # TAAの範囲 (0度から360度)
    taa_deg = np.linspace(0, 360, 361)

    # 1. 各TAAにおける距離(AU)を計算
    r_au = get_mercury_distance_theoretical(taa_deg)

    # 2. 温度計算 (ユーザーコードのロジックを使用)
    temps = calculate_surface_temperature_leblanc_subsolar(r_au)

    # --- グラフ描画 ---
    plt.figure(figsize=(10, 6))
    plt.plot(taa_deg, temps, label='Subsolar Temperature (Leblanc Model)', color='orange', linewidth=2)

    # 特定のポイントをハイライト
    # 近日点 (TAA = 0)
    peri_temp = temps[0]
    plt.scatter([0, 360], [peri_temp, peri_temp], color='red', zorder=5)
    plt.text(0, peri_temp + 5, f'Perihelion\n{peri_temp:.1f} K', ha='center', color='red', fontweight='bold')

    # 遠日点 (TAA = 180)
    apo_idx = 180
    apo_temp = temps[apo_idx]
    plt.scatter(180, apo_temp, color='blue', zorder=5)
    plt.text(180, apo_temp - 25, f'Aphelion\n{apo_temp:.1f} K', ha='center', color='blue', fontweight='bold')

    # グラフの装飾
    plt.title("Mercury Subsolar Surface Temperature vs TAA", fontsize=14)
    plt.xlabel("True Anomaly Angle (TAA) [deg]", fontsize=12)
    plt.ylabel("Temperature [K]", fontsize=12)
    plt.xlim(0, 360)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(np.arange(0, 361, 30))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()