import numpy as np
import matplotlib.pyplot as plt

# --- シミュレーションのパラメータ ---
N_SAMPLES = 500000
N_BINS = 50
RADIUS = 1.0

# --- ランバート分布に従うサンプルを生成 ---
u2 = np.random.random(N_SAMPLES)
sin_theta = np.sqrt(u2)
cos_theta = np.sqrt(1 - sin_theta**2)
theta_rad = np.arccos(cos_theta)

# ==============================================================================
# グラフ: 手動計算の結果を「確率密度」として表示
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title('Normalized to Probability Density P(θ) = cos(θ)', fontsize=16)

# --- ステップ1: 実際に数える ---
counts, bin_edges_rad = np.histogram(theta_rad, bins=N_BINS, range=(0, np.pi/2))
bin_centers_rad = (bin_edges_rad[:-1] + bin_edges_rad[1:]) / 2
bin_width_rad = bin_edges_rad[1] - bin_edges_rad[0]

# --- ステップ2: 表面積を計算する ---
areas = 2 * np.pi * RADIUS**2 * np.abs(np.cos(bin_edges_rad[:-1]) - np.cos(bin_edges_rad[1:]))

# --- ステップ3: 割り算して強度を求める ---
intensity = np.divide(counts, areas, where=areas > 1e-9)

# --- ★★★ ここからが追加した規格化のコード ★★★ ---
# 棒グラフの全面積を計算 (高さ×幅の合計)
total_area = np.sum(intensity * bin_width_rad)
# 全面積で割って、確率密度に変換
prob_density = intensity / total_area

# --- ステップ4: 結果をプロットする ---
# 規格化された確率密度を棒グラフで表示
ax.bar(bin_centers_rad, prob_density, width=bin_width_rad, label='Calculated Probability Density')

# 理論曲線は P(θ) = cos(θ)
# ∫cos(θ)dθ [0, π/2] = 1 なので、この式がそのまま正規化されたPDFとなる
x_theory_rad = np.linspace(0, np.pi/2, 200)
y_theory = np.cos(x_theory_rad)
ax.plot(x_theory_rad, y_theory, 'b-', lw=2, label='Theory: P(θ) = cos(θ)')

ax.set_xlabel('Zenith Angle θ [radians]', fontsize="16")
ax.set_ylabel('Probability Density', fontsize="16")
ax.grid(True)
ax.legend()
plt.show()