import numpy as np
import matplotlib.pyplot as plt


def plot_td_activation_energy_distribution():
    # ----------------------------------------------------------------
    # 1. シミュレーションコードからパラメータを抽出
    # ----------------------------------------------------------------
    U_MEAN = 1.85  # 平均 [eV]
    U_MIN = 1.40  # 最小値 [eV]
    U_MAX = 2.70  # 最大値 [eV]
    SIGMA = 0.20  # 標準偏差 [eV]

    # ----------------------------------------------------------------
    # 2. データ生成
    # ----------------------------------------------------------------
    # グラフ描画用に、少し広い範囲でx軸をとる
    x_range = np.linspace(1.0, 3.2, 500)

    # ガウス分布の計算 (正規化前)
    gaussian = np.exp(- (x_range - U_MEAN) ** 2 / (2 * SIGMA ** 2))

    # シミュレーションで使用される有効範囲 (1.40 - 2.70 eV) のマスク
    mask_active = (x_range >= U_MIN) & (x_range <= U_MAX)

    # 実際に計算に使われる離散グリッド (コード内の50分割を再現)
    u_ev_grid_sim = np.linspace(U_MIN, U_MAX, 50)
    pdf_sim = np.exp(- (u_ev_grid_sim - U_MEAN) ** 2 / (2 * SIGMA ** 2))

    # ----------------------------------------------------------------
    # 3. プロット
    # ----------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    # 全体のガウス分布（破線：カットオフ外の形状を示すため）
    plt.plot(x_range, gaussian, 'k--', alpha=0.3, label='Gaussian (Full)')

    # シミュレーションで採用されている範囲（実線と塗りつぶし）
    plt.plot(x_range[mask_active], gaussian[mask_active],
             color='red', linewidth=2, label='Active Range (Used in Sim)')
    plt.fill_between(x_range, gaussian, 0, where=mask_active,
                     color='red', alpha=0.2)

    # シミュレーションの離散グリッド点（散布図）
    plt.scatter(u_ev_grid_sim, pdf_sim, color='darkred', s=20, zorder=5,
                label='Simulation Grid (50 points)')

    # パラメータの垂直線
    plt.axvline(U_MEAN, color='blue', linestyle=':', label=f'Mean: {U_MEAN} eV')
    plt.axvline(U_MIN, color='green', linestyle='-', alpha=0.5, label=f'Min: {U_MIN} eV')
    plt.axvline(U_MAX, color='green', linestyle='-', alpha=0.5, label=f'Max: {U_MAX} eV')

    # ----------------------------------------------------------------
    # 4. 装飾
    # ----------------------------------------------------------------
    plt.title('Distribution of Activation Energy $U$ for Thermal Desorption', fontsize=14)
    plt.xlabel('Activation Energy $U$ [eV]', fontsize=12)
    plt.ylabel('Relative Probability (unnormalized)', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim(1.2, 3.0)
    plt.ylim(0, 1.1)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_td_activation_energy_distribution()