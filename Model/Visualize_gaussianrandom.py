import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# パラメータ設定（元の SIMULATION_SETTINGS から抽出）
# ==============================================================================
N_U_BINS = 10               # ビンの分割数
U_MIN = 1.4                 # 最小束縛エネルギー [eV]
U_MAX = 2.7                 # 最大束縛エネルギー [eV]
U_GAUSSIAN_MU = 1.85        # ガウス分布の中心 [eV]
U_GAUSSIAN_SIGMA = 0.15     # ガウス分布の広がり（標準偏差）

# ==============================================================================
# ビンと確率重みの計算 (setup_binding_energy_bins のロジック)
# ==============================================================================
# 1. 最小から最大までを N分割してビン(U_bins)を作成
U_bins = np.linspace(U_MIN, U_MAX, N_U_BINS)
bin_width = (U_MAX - U_MIN) / (N_U_BINS - 1) if N_U_BINS > 1 else 1.0

# 2. ガウス分布に基づく確率の重みを計算
V_weights = np.exp(-0.5 * ((U_bins - U_GAUSSIAN_MU) / U_GAUSSIAN_SIGMA)**2)

# 3. 確率の規格化 (合計を1.0にする)
weight_sum = np.sum(V_weights)
if weight_sum > 1e-30:
    V_weights /= weight_sum
else:
    V_weights = np.ones(N_U_BINS) / N_U_BINS

# ==============================================================================
# プロット用の連続曲線（理論値）の計算
# ==============================================================================
# なめらかな曲線を引くための細かいx座標（描画範囲を少し広めに取る）
u_continuous = np.linspace(U_MIN - 0.2, U_MAX + 0.2, 200)

# 連続的な確率密度関数 (PDF) を計算し、離散化したビンのスケール(確率)に合わせるためにビン幅を掛ける
pdf_continuous = (1.0 / (np.sqrt(2 * np.pi) * U_GAUSSIAN_SIGMA)) * \
                 np.exp(-0.5 * ((u_continuous - U_GAUSSIAN_MU) / U_GAUSSIAN_SIGMA)**2)
prob_continuous = pdf_continuous * bin_width

# ==============================================================================
# グラフの描画
# ==============================================================================
plt.figure(figsize=(10, 6))

# ビンの分布を棒グラフで表示 (幅はビン間隔の80%にして見やすくする)
plt.bar(U_bins, V_weights, width=bin_width*0.8, color='skyblue', edgecolor='black', 
        alpha=0.7, label='Bin Probabilities (V_weights)')

# 各ビンの中心座標（実際のU_binsの値）を赤い点でプロット
plt.plot(U_bins, V_weights, 'ro', markersize=6, label='Bin Centers (U_bins)')

# 理論的な連続ガウス分布を点線で表示
plt.plot(u_continuous, prob_continuous, 'b--', linewidth=2, label='Continuous Gaussian Curve')

# 中心値（μ）に縦線を引く
plt.axvline(U_GAUSSIAN_MU, color='green', linestyle=':', linewidth=2, 
            label=f'Mean (μ = {U_GAUSSIAN_MU} eV)')

# グラフの装飾
plt.title(f'Gaussian Random Model Distribution\n(Bins={N_U_BINS}, μ={U_GAUSSIAN_MU}, σ={U_GAUSSIAN_SIGMA})', fontsize=14)
plt.xlabel('Binding Energy U [eV]', fontsize=12)
plt.ylabel('Probability Weight (Normalized)', fontsize=12)

# X軸の目盛りを実際のビンの値に合わせる
plt.xticks(U_bins, [f"{val:.2f}" for val in U_bins], rotation=45)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# 表示
plt.show()