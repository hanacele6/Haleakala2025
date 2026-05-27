import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

# ==============================================================================
# 水星ナトリウム外気圏：正味の損失量（Net Loss）評価用 0Dモデル
# ==============================================================================

# --- 共通の物理パラメータ ---
T_SURF = 600.0             # 表面温度 [K]
KB_EV = 8.617e-5           # ボルツマン定数 [eV/K]
HOP_TAU_THRESHOLD = 30.0   # 即時バウンドの閾値 [秒]
NU = 1e13                  # 熱振動の頻度因子 [1/s]
PSD_ESCAPE_RATE = 4.0e-7   # 宇宙へのエスケープ率 (PSD等による) [1/s]

# シミュレーション時間設定
SIM_DAYS = 88.0            # 1水星年
DT = 1000.0                # タイムステップ [秒]
STEPS = int((SIM_DAYS * 86400) / DT)

# ------------------------------------------------------------------------------
# 限界束縛エネルギー U_crit の計算
# ------------------------------------------------------------------------------
# tau_td = (1/NU) * exp(U / kT) <= HOP_TAU_THRESHOLD となる限界の U
# この U_crit より深ければ「吸着」、浅ければ「バウンド」となる
U_crit = KB_EV * T_SURF * np.log(NU * HOP_TAU_THRESHOLD)
print(f"温度 {T_SURF} K におけるバウンド限界 U_crit: {U_crit:.4f} eV")

# ==============================================================================
# メインシミュレーション関数
# ==============================================================================
def run_net_loss_simulation(mu, sigma=0.25):
    # 1. ガウス分布から「吸着する割合 (P_stick)」を計算
    # U_crit より大きい領域の面積 (Survival Function = 1 - CDF)
    p_stick = norm.sf(U_crit, loc=mu, scale=sigma)
    
    # 初期状態 (初期質量 = 1.0)
    airborne = 1.0
    stuck = 0.0
    escaped = 0.0
    
    # 記録用リスト
    history_remaining = np.zeros(STEPS)
    history_escaped = np.zeros(STEPS)
    
    for step in range(STEPS):
        # A. 着地と吸着
        # 空中にいる粒子が着地し、P_stick の割合だけが地中に囚われる
        new_stuck = airborne * p_stick
        airborne -= new_stuck
        stuck += new_stuck
        
        # B. 宇宙へのエスケープ (PSD等)
        # 地中に吸着した粒子のみが、PSD大砲によって一定のRateで宇宙へ撃ち出される
        # (解析的減衰: dt間の減少率)
        loss_fraction = 1.0 - np.exp(-PSD_ESCAPE_RATE * DT)
        escape_loss = stuck * loss_fraction
        
        stuck -= escape_loss
        escaped += escape_loss
        
        # C. 記録 (系内の残量 = airborne + stuck)
        history_remaining[step] = airborne + stuck
        history_escaped[step] = escaped
        
    return history_remaining, history_escaped, p_stick

# ==============================================================================
# 実行と結果のプロット
# ==============================================================================
time_days = np.arange(STEPS) * DT / 86400

# 比較する2つのモデル設定
# 1. 質量が消滅した現状のモデル
rem_bug, esc_bug, p_stick_bug = run_net_loss_simulation(mu=1.85)
# 2. 質量が残るように修正したモデル
rem_fix, esc_fix, p_stick_fix = run_net_loss_simulation(mu=1.65)

print(f"\n[Gaussian mu=1.85] 1回着地時の吸着確率: {p_stick_bug*100:.1f} %")
print(f"[Gaussian mu=1.65] 1回着地時の吸着確率: {p_stick_fix*100:.1f} %")

# プロット作成
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(time_days, rem_bug, color='blue', linestyle='-', linewidth=2, 
        label=f'Remaining (mu=1.85, Stuck {p_stick_bug*100:.0f}%)')
ax.plot(time_days, esc_bug, color='red', linestyle='--', linewidth=2, 
        label=f'Escaped (mu=1.85)')

ax.plot(time_days, rem_fix, color='cyan', linestyle='-', linewidth=2, 
        label=f'Remaining (mu=1.65, Stuck {p_stick_fix*100:.0f}%)')
ax.plot(time_days, esc_fix, color='orange', linestyle='--', linewidth=2, 
        label=f'Escaped (mu=1.65)')

ax.set_title(f'Net Loss of Sodium (T={T_SURF} K, U_crit={U_crit:.2f} eV)', fontsize=14)
ax.set_xlabel('Time [Earth Days]', fontsize=12)
ax.set_ylabel('Mass Fraction (1.0 = Initial Mass)', fontsize=12)
ax.set_xlim(0, SIM_DAYS)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, linestyle=':', alpha=0.7)
ax.legend(loc='center right', fontsize=11)

plt.tight_layout()
plt.show()