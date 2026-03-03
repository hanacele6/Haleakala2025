import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. 物理定数と設定
# ==============================================================================
KB = 1.380649e-23  # ボルツマン定数 (J/K)
EV_TO_JOULE = 1.60217663e-19  # eV から Joule への変換係数


# ==============================================================================
# 2. 計算関数
# ==============================================================================

def calculate_prob_single(T_array, U_eV, v_Hz, dt=1.0):
    """
    単一の結合エネルギー U を持つ場合の放出確率を計算

    Parameters:
    - T_array: 温度の配列 (K)
    - U_eV: 結合エネルギー (eV)
    - v_Hz: 振動数 (Hz)
    - dt: 単位時間 (sec, Figure 3は /sec なので通常 1.0)

    Returns:
    - prob: 放出確率 (0.0 - 1.0)
    """
    T = np.array(T_array)
    U_J = U_eV * EV_TO_JOULE

    # アレニウスの式: Rate = v * exp(-U / kT)
    rate = v_Hz * np.exp(-U_J / (KB * T))

    # 確率への変換: Prob = 1 - exp(-Rate * dt)
    # Rateが巨大な場合、確率は1.0に収束する
    prob = 1.0 - np.exp(-rate * dt)
    return prob


def calculate_prob_distributed(T_array, U_mean, U_min, U_max, sigma, v_Hz, dt=1.0):
    """
    結合エネルギー U がガウス分布する場合の放出確率を計算 (数値積分)

    ユーザー指定パラメータ:
    U_MEAN=1.85, U_MIN=1.40, U_MAX=2.70, SIGMA=0.20
    """
    # Uの積分グリッド作成 (minからmaxまで)
    u_grid_ev = np.linspace(U_min, U_max, 200)
    u_grid_j = u_grid_ev * EV_TO_JOULE

    # ガウス分布 (PDF) の計算
    pdf = np.exp(- (u_grid_ev - U_mean) ** 2 / (2 * sigma ** 2))

    # 確率密度の正規化 (合計が1になるように)
    pdf_sum = np.sum(pdf)
    if pdf_sum == 0:
        return np.zeros_like(T_array)
    normalized_weights = pdf / pdf_sum

    probs = []

    # 各温度ごとに「平均放出率」を計算してから確率に変換
    for T in T_array:
        # この温度における各Uの放出率を計算
        rates = v_Hz * np.exp(-u_grid_j / (KB * T))

        # 重み付き平均放出率
        avg_rate = np.sum(rates * normalized_weights)

        # 確率に変換
        prob = 1.0 - np.exp(-avg_rate * dt)
        probs.append(prob)

    return np.array(probs)


# ==============================================================================
# 3. メイン処理とグラフ描画
# ==============================================================================
def main():
    # X軸: 温度範囲 (論文のFigure 3に合わせて400Kから800K)
    temps = np.linspace(400, 800, 200)

    # --- A. 論文 Figure 3 の再現データ (凡例に基づく) ---
    # 1. Red Line (Hunten & Sprague, 2002)
    #    U=1.40 eV, v=1e13 Hz [cite: 199, 204]
    y_red = calculate_prob_single(temps, U_eV=1.40, v_Hz=1e13)

    # 2. Blue Line (Leblanc & Johnson, 2010)
    #    U=1.85 eV, v=1e11 Hz [cite: 199, 204]
    y_blue = calculate_prob_single(temps, U_eV=1.85, v_Hz=1e11)

    # 3. Green Line (This Study / Suzuki et al., 2020)
    #    U=1.85 eV, v=1e13 Hz [cite: 200, 204]
    y_green = calculate_prob_single(temps, U_eV=1.85, v_Hz=1e13)

    # --- B. ユーザー指定の分布モデル ---
    #    U_MEAN=1.85, U_MIN=1.40, U_MAX=2.70, SIGMA=0.20
    #    振動数 v は、比較のため「This Study」と同じ 1e13 Hz と仮定します
    y_user = calculate_prob_distributed(
        temps,
        U_mean=1.85,
        U_min=1.40,
        U_max=2.70,
        sigma=0.20,
        v_Hz=1e13
    )

    # --- C. プロット設定 ---
    plt.figure(figsize=(10, 7))

    # 論文データの描画
    plt.plot(temps, y_red, color='red', label='Hunten (2002): 1.40eV, 1e13Hz', linewidth=2)
    plt.plot(temps, y_blue, color='blue', label='Leblanc (2010): 1.85eV, 1e11Hz', linewidth=2)
    plt.plot(temps, y_green, color='green', label='Suzuki (2020): 1.85eV, 1e13Hz', linewidth=2)

    # ユーザーモデルの描画 (黒の破線)
    plt.plot(temps, y_user, color='black', linestyle='--', linewidth=2.5,
             label='Leblanc (2003): (Mean=1.85eV, $\sigma$=0.20), 1e13Hz')

    # グラフ装飾 (論文の見た目に近づける)
    plt.xlim(400, 800)
    plt.ylim(0, 1.05)
    plt.xlabel('Temperature (K)', fontsize=12)
    plt.ylabel('Desorption Probability (/sec)', fontsize=12)
    #plt.title('Reproduction of Suzuki et al. (2020) Fig 3 with User Model', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)

    # 表示
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()