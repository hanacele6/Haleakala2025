import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. あなたのコードから物理定数を転記
# ==============================================================================
PHYSICAL_CONSTANTS = {
    'K_BOLTZMANN': 1.380649e-23,
    'EV_TO_JOULE': 1.602e-19,
}


# ==============================================================================
# 2. 「固定値モデル」 (汎用化)
# ==============================================================================
def calculate_rate_FIXED(surface_temp_K: float, u_eff_ev: float = 1.85) -> float:
    # --- あなたのコード 358行目付近 ---
    if surface_temp_K < 10.0: return 0.0

    VIB_FREQ = 1e13
    KB = PHYSICAL_CONSTANTS['K_BOLTZMANN']
    EV_J = PHYSICAL_CONSTANTS['EV_TO_JOULE']

    # 固定値 (引数を使用)
    U_JOULE = u_eff_ev * EV_J

    exponent = -U_JOULE / (KB * surface_temp_K)

    # ガード処理
    if exponent < -700:
        return 0.0

    rate = VIB_FREQ * np.exp(exponent)
    return rate


# ==============================================================================
# 3. 「分布モデル」
# ==============================================================================
def calculate_rate_DISTRIBUTED(surface_temp_K: float) -> float:
    # --- あなたのコード 333行目付近 (コメントアウト部分) ---
    if surface_temp_K < 10.0: return 0.0

    VIB_FREQ = 1e13
    KB = PHYSICAL_CONSTANTS['K_BOLTZMANN']
    EV_J = PHYSICAL_CONSTANTS['EV_TO_JOULE']

    # ガウス分布パラメータ
    U_MEAN = 1.85
    U_MIN = 1.40
    U_MAX = 2.70
    SIGMA = 0.20

    # グリッド計算
    u_ev_grid = np.linspace(U_MIN, U_MAX, 50)
    u_joule_grid = u_ev_grid * EV_J

    # 確率密度関数 (PDF)
    pdf = np.exp(- (u_ev_grid - U_MEAN) ** 2 / (2 * SIGMA ** 2))
    pdf_sum = np.sum(pdf)
    if pdf_sum == 0: return 0.0
    norm_pdf = pdf / pdf_sum

    # 各エネルギーでのRate
    exponent = -u_joule_grid / (KB * surface_temp_K)
    rates = np.zeros_like(u_ev_grid)

    # ガード処理
    mask = exponent > -700
    rates[mask] = VIB_FREQ * np.exp(exponent[mask])

    # 積分 (合計)
    effective_rate = np.sum(rates * norm_pdf)
    return effective_rate


# ==============================================================================
# 4. 比較実行 (修正版)
# ==============================================================================
def main():
    print("=== Your Code Logic Comparison ===")

    # 水星の温度範囲 (100K 〜 720K)
    temps = np.linspace(100, 725, 100)

    # --- 追加設定: 他に比較したい固定Uの値 (コメントアウト) ---
    # additional_u_values = [1.50, 2.20]

    # 結果を格納する辞書を初期化
    results = {
        'Fixed (U=1.85eV)': [],
        'Distributed': []
    }

    # 不要な初期化をコメントアウト
    # for u_val in additional_u_values:
    #     results[f'Fixed (U={u_val:.2f}eV)'] = []

    # チェックポイント設定 (今回は表示しないためコメントアウトのまま、あるいは未使用)
    # check_points = [400, 500, 600, 700]
    # check_points_to_print = list(check_points)

    # ヘッダー表示 (シンプル化)
    # header = f"{'Temp [K]':^10} | {'U=1.85 (Fix)':^14} | {'Distributed':^14}"
    # for u_val in additional_u_values:
    #     header += f" | {f'U={u_val:.2f} (Fix)':^14}"
    # print(header)
    # print("-" * len(header))

    # --- 計算ループ ---
    for T in temps:
        # 1. 元の固定値モデル (U=1.85)
        r_fix_base = calculate_rate_FIXED(T, u_eff_ev=1.85)
        results['Fixed (U=1.85eV)'].append(r_fix_base)

        # 2. 分布モデル
        r_dist = calculate_rate_DISTRIBUTED(T)
        results['Distributed'].append(r_dist)

        # 3. 追加の固定値モデル (コメントアウト)
        # current_additional_rates = []
        # for u_val in additional_u_values:
        #    r_add = calculate_rate_FIXED(T, u_eff_ev=u_val)
        #    results[f'Fixed (U={u_val:.2f}eV)'].append(r_add)
        #    current_additional_rates.append(r_add)

    # --- グラフ描画 ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 元のプロット (青点線)
    ax1.plot(temps, results['Fixed (U=1.85eV)'], 'b--', label='Fixed U=1.85eV', zorder=5)

    # 分布モデル (赤実線)
    ax1.plot(temps, results['Distributed'], 'r-', label='Distributed', linewidth=2.5, zorder=10)

    # 追加のプロット (コメントアウト)
    # colors = ['green', 'purple', 'cyan']
    # linestyles = ['-.', ':']
    # for i, u_val in enumerate(additional_u_values):
    #     key = f'Fixed (U={u_val:.2f}eV)'
    #     color = colors[i % len(colors)]
    #     ls = linestyles[i % len(linestyles)]
    #     ax1.plot(temps, results[key], color=color, linestyle=ls, label=key)

    ax1.set_yscale('log')
    ax1.set_xlabel('Temperature [K]' ,fontsize = "15")
    ax1.set_ylabel('Desorption Rate [1/s]',fontsize = "15")

    # タイトル不要
    # ax1.set_title('Comparison of Desorption Rates')

    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # 凡例不要
    # ax1.legend(loc='best')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()