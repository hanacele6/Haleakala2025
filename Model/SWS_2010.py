import numpy as np
import matplotlib.pyplot as plt

def generate_imf_timeseries(total_hours=40, dt_min=1):
    """
    Leblanc 2010に基づく疑似的なIMF時系列データを生成する関数
    - 変化の周期は平均1時間（正規分布）
    """
    num_steps = int(total_hours * 60 / dt_min)
    time_array = np.linspace(0, total_hours, num_steps)
    
    # 配列の初期化
    bx = np.zeros(num_steps)
    by = np.zeros(num_steps)
    bz = np.zeros(num_steps)
    
    current_step = 0
    while current_step < num_steps:
        # 磁場の状態が継続する時間をランダムに決定 (平均60分、標準偏差15分)
        duration_min = int(np.random.normal(loc=60, scale=15))
        duration_min = max(10, duration_min) # 最低10分は継続
        
        next_step = min(current_step + int(duration_min / dt_min), num_steps)
        
        # Leblanc 2010に基づく磁場強度のランダム設定 (nT)
        bx[current_step:next_step] = np.random.uniform(-27, 27)
        by[current_step:next_step] = np.random.uniform(-15, 15)
        bz[current_step:next_step] = np.random.uniform(-15, 15)
        
        current_step = next_step
        
    return time_array, bx, by, bz

def calculate_sws_area(bx, bz, nominal_area=100):
    """
    IMFからSWSの有効面積（ノミナル値に対するパーセンテージ）を計算する関数
    """
    # 基本の開口面積（Bzが負のときにスパッタリング発生、正のときは完全ゼロ）
    # ※Bzの強さに応じて面積がスケールすると仮定
    base_area = np.where(bz < 0, np.abs(bz) / 15.0 * nominal_area * 2, 0.0)
    
    # Bxによる南北の非対称性の計算
    # Bx > 0 なら北半球に偏る、Bx < 0 なら南半球に偏る
    north_ratio = np.where(bz < 0, 0.5 + (bx / 27.0) * 0.4, 0.0)
    north_ratio = np.clip(north_ratio, 0.1, 0.9) # 極端な偏りを防ぐ
    
    # 最終的な面積の算出（ゼロガードを徹底）
    total_area = np.where(bz < 0, base_area, 0.0)
    north_area = np.where(bz < 0, total_area * north_ratio, 0.0)
    south_area = np.where(bz < 0, total_area * (1.0 - north_ratio), 0.0)
    
    return total_area, north_area, south_area

# --- メイン処理 ---
np.random.seed(42) # 再現性のためにシードを固定

# 1. IMFデータの生成
time_hrs, bx, by, bz = generate_imf_timeseries(total_hours=40, dt_min=1)

# 2. SWS有効面積の計算
total_area, north_area, south_area = calculate_sws_area(bx, bz, nominal_area=100)

# 3. プロット (Leblanc 2010 Fig.1 風)
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
plt.subplots_adjust(hspace=0.1)

# Bx, By, Bzのプロット
axes[0].step(time_hrs, bx, color='black')
axes[0].set_ylabel('Bx IMF (nT)')
axes[0].axhline(0, color='gray', linestyle='--', linewidth=1)
axes[0].set_ylim(-30, 30)

axes[1].step(time_hrs, by, color='black')
axes[1].set_ylabel('By IMF (nT)')
axes[1].axhline(0, color='gray', linestyle='--', linewidth=1)
axes[1].set_ylim(-20, 20)

axes[2].step(time_hrs, bz, color='black')
axes[2].set_ylabel('Bz IMF (nT)')
axes[2].axhline(0, color='gray', linestyle='--', linewidth=1)
axes[2].set_ylim(-20, 20)

# SWS面積のプロット
axes[3].step(time_hrs, total_area, color='black', label='Total Area')
axes[3].step(time_hrs, north_area, color='black', linestyle='--', label='North Area')
axes[3].set_ylabel('Bomb. Surface\n(% of nominal)')
axes[3].set_xlabel('Time (h)')
axes[3].set_xlim(0, 40)
axes[3].set_ylim(0, 220)
axes[3].legend(loc='upper right')

plt.show()