# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib
matplotlib.rcParams['font.family'] = 'MS Gothic'

# ==========================================
# 1. 設定
# ==========================================
file_paths = {
    'dt = 1s (Ref)': r'./SimulationResult_202606/ParabolicHop_72x36_NoEq_DT1_0616_Multi_BD0.5_UG_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)/budget_statistics_per_taa.csv',
    'dt = 100s':     r'./SimulationResult_202606/ParabolicHop_72x36_NoEq_DT100_0609_Multi_BD0.5_UG_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)_1e24/budget_statistics_per_taa.csv'
}

REF_LABEL = 'dt = 1s (Ref)'
TGT_LABEL = 'dt = 100s'

processes_groups = {
    'Generation': ['Gen_PSD', 'Gen_TD', 'Gen_SWS'],
    'Loss': ['Loss_Stuck', 'Loss_Ionized', 'Loss_Escaped'],
    'Diffusion': ['Supply_Internal', 'Trans_Inward']
}

# 共通TAA軸（きれいに揃える）
common_taa = np.linspace(0, 360, 360)

processed_data = {}
raw_data = {}  # 累積計算用に生データも保持

print("=== statistics_per_taa 読み込み開始 ===")

# ==========================================
# 2. データ読み込み & 補間
# ==========================================
for label, path in file_paths.items():
    print(f"\n--- {label} ---")
    print(f"path: {path}")

    if not os.path.exists(path):
        print("[ERROR] ファイルなし")
        continue

    df = pd.read_csv(path)

    print("columns:", list(df.columns))
    print("shape:", df.shape)

    if 'TAA_Bin' not in df.columns:
        print("[ERROR] TAA_Bin がない")
        continue

    df = df.sort_values('TAA_Bin')
    raw_data[label] = df  # 生データを保存

    interp_dict = {}

    for group, cols in processes_groups.items():
        for col in cols:
            if col not in df.columns:
                continue

            df_sub = df[['TAA_Bin', col]].dropna()

            if len(df_sub) < 2:
                print(f"[WARNING] {col} データ不足")
                continue

            y_first = df_sub[col].iloc[0]
            y_last  = df_sub[col].iloc[-1]

            f_interp = interp1d(
                df_sub['TAA_Bin'],
                df_sub[col],
                kind='linear',
                bounds_error=False,
                fill_value=(y_first, y_last)
            )

            interp_dict[col] = f_interp(common_taa)
            print(f"{col}: OK")

    processed_data[label] = interp_dict
    print(f"{label}: series数 = {len(interp_dict)}")

# ==========================================
# 3. プロット（既存）
# ==========================================
print("\n=== プロット処理 ===")

if REF_LABEL not in processed_data or TGT_LABEL not in processed_data:
    print("[ERROR] 比較対象不足")
else:
    ref_dict = processed_data[REF_LABEL]
    tgt_dict = processed_data[TGT_LABEL]

    for group_name, cols in processes_groups.items():
        n_cols = len(cols)
        fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 7))

        if n_cols == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        fig.suptitle(f"{group_name}: {TGT_LABEL} vs {REF_LABEL}", fontsize=16)

        for i, col in enumerate(cols):
            if col not in ref_dict or col not in tgt_dict:
                print(f"[SKIP] {col}")
                continue

            ref_vals  = ref_dict[col]
            tgt_vals  = tgt_dict[col]
            diff_vals = tgt_vals - ref_vals

            ax1 = axes[0, i]
            ax1.plot(common_taa, ref_vals, label=REF_LABEL, color='red')
            ax1.plot(common_taa, tgt_vals, label=TGT_LABEL, color='blue')
            ax1.set_title(col)
            ax1.set_xlim(0, 360)
            ax1.grid(True)
            ax1.legend()

            ax2 = axes[1, i]
            ax2.plot(common_taa, diff_vals, color='purple')
            ax2.axhline(0, color='black', linestyle='--')
            ax2.set_xlim(0, 360)
            ax2.set_xlabel("TAA [deg]")
            ax2.grid(True)

            max_diff = np.nanmax(np.abs(diff_vals))
            if max_diff == 0 or np.isnan(max_diff):
                max_diff = 1e-10
            ax2.set_ylim(-max_diff * 1.1, max_diff * 1.1)

        plt.tight_layout()

    # ==========================================
    # 4. 吸着損失割合の計算 & プロット（既存）
    # ==========================================
    print("\n=== 吸着損失割合の比較 ===")

    required_gen = ['Gen_PSD', 'Gen_TD', 'Gen_SWS']
    loss_col = 'Loss_Stuck'

    for col in required_gen + [loss_col]:
        if col not in ref_dict or col not in tgt_dict:
            print(f"[ERROR] 必須データ不足: {col}")
            exit()

    ref_gen_total = sum(ref_dict[col] for col in required_gen)
    tgt_gen_total = sum(tgt_dict[col] for col in required_gen)

    ref_gen_total = np.where(ref_gen_total == 0, np.nan, ref_gen_total)
    tgt_gen_total = np.where(tgt_gen_total == 0, np.nan, tgt_gen_total)

    ref_ratio = ref_dict[loss_col] / ref_gen_total
    tgt_ratio = tgt_dict[loss_col] / tgt_gen_total
    diff_ratio = tgt_ratio - ref_ratio

    fig, axes = plt.subplots(2, 1, figsize=(8, 7))
    fig.suptitle("Adsorption Loss Ratio (Loss_Stuck / Generation)", fontsize=16)

    ax1 = axes[0]
    ax1.plot(common_taa, ref_ratio, label=REF_LABEL, color='red')
    ax1.plot(common_taa, tgt_ratio, label=TGT_LABEL, color='blue')
    ax1.set_xlim(0, 360)
    ax1.set_ylabel("Ratio")
    ax1.grid(True)
    ax1.legend()

    ax2 = axes[1]
    ax2.plot(common_taa, diff_ratio, color='purple')
    ax2.axhline(0, color='black', linestyle='--')
    ax2.set_xlim(0, 360)
    ax2.set_xlabel("TAA [deg]")
    ax2.set_ylabel("Δ Ratio")
    ax2.grid(True)

    max_diff = np.nanmax(np.abs(diff_ratio))
    if max_diff == 0 or np.isnan(max_diff):
        max_diff = 1e-10
    ax2.set_ylim(-max_diff * 1.1, max_diff * 1.1)

    plt.tight_layout()

    # ==========================================
    # 5. 【新機能】累積合計の比較
    # ==========================================
    print("\n=== 累積合計の比較 ===")

    # 比較するプロセス
    cumsum_cols = ['Gen_TD', 'Gen_PSD', 'Loss_Stuck']

    # --- 数値出力 ---
    print(f"\n{'プロセス':<15} {'dt=1累積':>15} {'dt=100累積':>15} {'差(100-1)':>15} {'差(%)':>10}")
    print("-" * 65)

    cumsum_results = {}
    for col in cumsum_cols:
        if col not in ref_dict or col not in tgt_dict:
            print(f"[SKIP] {col}")
            continue

        ref_total = np.sum(ref_dict[col])
        tgt_total = np.sum(tgt_dict[col])
        diff      = tgt_total - ref_total
        pct       = diff / ref_total * 100 if ref_total != 0 else np.nan

        cumsum_results[col] = {
            'ref': ref_total,
            'tgt': tgt_total,
            'diff': diff,
            'pct': pct
        }
        print(f"{col:<15} {ref_total:>15.3e} {tgt_total:>15.3e} {diff:>15.3e} {pct:>9.2f}%")

    # TDとStuckの差の比較（どちらの効果が大きいか）
    if 'Gen_TD' in cumsum_results and 'Loss_Stuck' in cumsum_results:
        delta_td    = cumsum_results['Gen_TD']['diff']
        delta_stuck = cumsum_results['Loss_Stuck']['diff']
        print(f"\n[診断]")
        print(f"  ΔGen_TD    = {delta_td:.3e}  (dt100でTDがこれだけ少ない/多い)")
        print(f"  ΔLoss_Stuck = {delta_stuck:.3e}  (dt100でStuckがこれだけ少ない/多い)")
        print(f"  |ΔTD| / |ΔStuck| = {abs(delta_td)/abs(delta_stuck):.3f}")
        if abs(delta_td) > abs(delta_stuck):
            print("  → TDの変化のほうが大きい：TDの過小/過大評価が表面密度の主因")
        else:
            print("  → Stuckの変化のほうが大きい：吸着率の変化が表面密度の主因")

    # --- TAA別累積プロット ---
    fig, axes = plt.subplots(len(cumsum_cols), 2, figsize=(14, 4 * len(cumsum_cols)))
    fig.suptitle("累積合計の比較 (TAA別)", fontsize=16)

    for i, col in enumerate(cumsum_cols):
        if col not in ref_dict or col not in tgt_dict:
            continue

        ref_vals  = ref_dict[col]
        tgt_vals  = tgt_dict[col]

        ref_cumsum = np.cumsum(ref_vals)
        tgt_cumsum = np.cumsum(tgt_vals)
        diff_cumsum = tgt_cumsum - ref_cumsum

        # 左：累積値の比較
        ax_l = axes[i, 0]
        ax_l.plot(common_taa, ref_cumsum, label=REF_LABEL, color='red')
        ax_l.plot(common_taa, tgt_cumsum, label=TGT_LABEL, color='blue')
        ax_l.set_title(f"{col} 累積合計")
        ax_l.set_xlim(0, 360)
        ax_l.set_xlabel("TAA [deg]")
        ax_l.grid(True)
        ax_l.legend()

        # 右：累積差分
        ax_r = axes[i, 1]
        ax_r.plot(common_taa, diff_cumsum, color='purple')
        ax_r.axhline(0, color='black', linestyle='--')
        ax_r.set_title(f"Δ{col} 累積差分 (dt100 - dt1)")
        ax_r.set_xlim(0, 360)
        ax_r.set_xlabel("TAA [deg]")
        ax_r.grid(True)

        # 最終値をアノテーション
        ax_r.annotate(
            f"最終差: {diff_cumsum[-1]:.2e}",
            xy=(350, diff_cumsum[-1]),
            fontsize=9,
            color='purple'
        )

    # ==========================================
    # 表面インベントリ（密度）の純収支（Net Balance）の累積を比較
    # ==========================================
    print("\n=== 表面密度の純収支（Net Balance）の比較 ===")
    
    # dt=1s (Ref) の表面純収支フラックス
    ref_net_flux = (ref_dict['Loss_Stuck'] + ref_dict['Supply_Internal']) - \
                   (ref_dict['Gen_TD'] + ref_dict['Gen_PSD'] + ref_dict['Gen_SWS'])
    
    # dt=100s の表面純収支フラックス
    tgt_net_flux = (tgt_dict['Loss_Stuck'] + tgt_dict['Supply_Internal']) - \
                   (tgt_dict['Gen_TD'] + tgt_dict['Gen_PSD'] + tgt_dict['Gen_SWS'])
    
    # 純収支を累積することで「表面の総原子数の相対的な変動」を復元
    ref_inventory_change = np.cumsum(ref_net_flux)
    tgt_inventory_change = np.cumsum(tgt_net_flux)
    diff_inventory = tgt_inventory_change - ref_inventory_change
    
    # --- プロット ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Cumulative Net Surface Balance (Stock Proxy)", fontsize=16)
    
    # 左：純収支の累積値（これが全表面原子数のカーブの形状と一致するはず）
    ax_l = axes[0]
    ax_l.plot(common_taa, ref_inventory_change, label=REF_LABEL, color='red')
    ax_l.plot(common_taa, tgt_inventory_change, label=TGT_LABEL, color='blue')
    ax_l.set_title("累積 表面純収支 (Inflow - Outflow)")
    ax_l.set_xlim(0, 360)
    ax_l.set_xlabel("TAA [deg]")
    ax_l.grid(True)
    ax_l.legend()
    
    # 右：累積値の差分（これが「密度の差」の定量的な原因）
    ax_r = axes[1]
    ax_r.plot(common_taa, diff_inventory, color='purple')
    ax_r.axhline(0, color='black', linestyle='--')
    ax_r.set_title("Δ 累積 表面純収支 (dt100 - dt1)")
    ax_r.set_xlim(0, 360)
    ax_r.set_xlabel("TAA [deg]")
    ax_r.grid(True)
    
    plt.tight_layout()
    plt.show()
