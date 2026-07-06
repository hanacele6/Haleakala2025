# -*- coding: utf-8 -*-
import numpy as np
import os
import re
import matplotlib.pyplot as plt

# --- ユーザー設定 ---
BASE_RESULT_DIR = r"./SimulationResult_202606"

# 比較したいフォルダ名
TARGET_MODEL_NAMES = [
    "ParabolicHop_72x36_NoEq_DT100_0622_Multi_BD0.5_UG_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)_1res_season2",
    #"ParabolicHop_72x36_NoEq_DT100_0623_Multi_BD0.5_UG_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)_3res_season2",
    "ParabolicHop_72x36_NoEq_DT100_0624_Multi_BD0.5_UG_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)_5res_season3"
]

# --- 処理とプロット ---
plt.figure(figsize=(10, 6))

print("=== 全空間の総原子数 比較 ===")

for folder in TARGET_MODEL_NAMES:
    target_dir = os.path.join(BASE_RESULT_DIR, folder)
    
    if not os.path.exists(target_dir):
        print(f"ディレクトリが見つかりません: {folder}")
        continue

    # atom_count_gridファイルだけを取得
    files = [f for f in os.listdir(target_dir) if f.startswith("atom_count_grid_") and f.endswith(".npy")]
    
    if not files:
        print(f"データがありません: {folder}")
        continue
        
    taa_list = []
    total_atoms_list = []

    for f in files:
        match = re.search(r'_taa(\d+)\.npy$', f)
        if match:
            taa = int(match.group(1))
            filepath = os.path.join(target_dir, f)
            
            # グリッドをロード
            grid = np.load(filepath)
            
            # ★ 領域や体積を一切無視して、単純に配列の全要素を足し合わせる
            total_atoms = np.sum(grid) 
            
            taa_list.append(taa)
            total_atoms_list.append(total_atoms)

    # TAA順にソート
    taa_list = np.array(taa_list)
    total_atoms_list = np.array(total_atoms_list)
    sorted_idx = np.argsort(taa_list)
    taa_list = taa_list[sorted_idx]
    total_atoms_list = total_atoms_list[sorted_idx]
    
    # 凡例用のラベル作成 (例: 1res, 3res)
    res_label = "Unknown"
    if "1res" in folder: res_label = "101³"
    elif "3res" in folder: res_label = "301³"
    elif "5res" in folder: res_label = "501³"

    # 平均値を計算してコンソールに出力
    mean_atoms = np.mean(total_atoms_list)
    print(f"モデル: {res_label:5s} | 空間内の平均総原子数: {mean_atoms:.4e} [atoms]")

    # プロット
    plt.plot(taa_list, total_atoms_list, marker='o', markersize=4, label=f"Resolution: {res_label}")

# グラフの装飾
plt.xlabel("True Anomaly Angle (deg)", fontsize=14)
plt.ylabel("Total Number of Atoms in Space", fontsize=14)
plt.title("Total Simulated Atoms vs Resolution (No Regional Slicing)", fontsize=16)
plt.grid(True, linestyle='--')
plt.xlim(0, 360)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()