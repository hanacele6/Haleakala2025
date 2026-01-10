import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 設定
# ==============================================================================
KB = 1.380649e-23
EV_J = 1.602e-19
VIB_FREQ = 1e13
TEMP = 700.0  # 水星昼側の温度 [K]

# 分布パラメータ
U_MEAN = 1.85
SIGMA = 0.20
U_MIN = 1.40
U_MAX = 2.70

# シミュレーション時間設定
DT = 0.1  # 時間ステップ [s]
MAX_TIME = 10.0  # シミュレーション時間 [s] (短時間で勝負がつきます)
TIME_STEPS = int(MAX_TIME / DT)

# グリッド設定
N_BINS = 200
u_ev_grid = np.linspace(U_MIN, U_MAX, N_BINS)
u_joule_grid = u_ev_grid * EV_J

# ==============================================================================
# 準備計算
# ==============================================================================
# 1. 各エネルギーごとの放出率 k(U) [1/s]
exponent = -u_joule_grid / (KB * TEMP)
rate_per_u = VIB_FREQ * np.exp(exponent)

# 2. 初期の分布形状 (正規化された確率)
pdf = np.exp(- (u_ev_grid - U_MEAN) ** 2 / (2 * SIGMA ** 2))
pdf /= np.sum(pdf)

# ==============================================================================
# 時間発展シミュレーション
# ==============================================================================
times = np.arange(0, MAX_TIME, DT)

# --- Case A: 最初に割り振って固定 (Depletion Model) ---
# inventory_A[i] は、エネルギー U[i] を持つ原子の「現在の量」
inventory_A = np.copy(pdf)  # 初期状態はガウス分布 (合計1.0)
flux_history_A = []

# --- Case B: 毎回くじ引き (Resampling / Mixing Model) ---
# total_inventory_B は原子の「総量」だけ管理。分布は常にpdfと同じ形と仮定。
total_inventory_B = 1.0
flux_history_B = []

# Case Bの実効レート（常に一定）
# 平均放出率 = sum( k(U) * P(U) )
effective_rate_B = np.sum(rate_per_u * pdf)

print(f"Simulation Start: T={TEMP}K, Time={MAX_TIME}s")
print(f"Initial Effective Rate (Case B): {effective_rate_B:.2f} [1/s]")

for t in times:
    # === Case A の計算 ===
    # 各ビンごとに独立して減衰する
    # Loss[i] = N[i] * (1 - exp(-k[i]*dt))
    decay_factor_A = np.exp(-rate_per_u * DT)
    loss_A = inventory_A * (1.0 - decay_factor_A)

    # 放出総量
    total_flux_A = np.sum(loss_A) / DT
    flux_history_A.append(total_flux_A)

    # 在庫更新 (減った分を引く)
    inventory_A -= loss_A

    # === Case B の計算 ===
    # 総量に対して、常に「高い実効レート」がかかる
    # Loss = Total * (1 - exp(-Rate_eff * dt))
    decay_factor_B = np.exp(-effective_rate_B * DT)
    loss_B = total_inventory_B * (1.0 - decay_factor_B)

    # 放出総量
    total_flux_B = loss_B / DT
    flux_history_B.append(total_flux_B)

    # 在庫更新
    total_inventory_B -= loss_B

# ==============================================================================
# 結果の可視化
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1. 放出フラックスの時間変化
ax1 = axes[0]
ax1.plot(times, flux_history_A, 'b-', linewidth=2, label='Case A: Fixed Identity (Depletion)')
ax1.plot(times, flux_history_B, 'r--', linewidth=2, label='Case B: Resampling (Infinite Mixing)')
ax1.set_yscale('log')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Total Desorption Flux [relative]')
ax1.set_title(f'Flux Evolution at {TEMP}K\n(donnamon)')
ax1.grid(True, which='both', alpha=0.5)
ax1.legend()

# 2. Case A の在庫分布の変化 (最初と最後)
ax2 = axes[1]
ax2.plot(u_ev_grid, pdf, 'k--', label='Initial Distribution (t=0)')
ax2.plot(u_ev_grid, inventory_A, 'b-', linewidth=2, label=f'Remaining Inventory (t={MAX_TIME}s)')
ax2.fill_between(u_ev_grid, inventory_A, color='blue', alpha=0.1)

# どこが削れたか？
ax2.set_xlabel('Binding Energy U [eV]')
ax2.set_ylabel('Population')
ax2.set_title('Surface Inventory Change (Case A)\nLow energy atoms vanish!')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()