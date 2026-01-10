# -*- coding: utf-8 -*-
"""
solve_inverse.py (Robust Version)
==============================================================================
逆解析ソルバー
概要:
    生成されたカーネル(kernel_taaXXX.npy)と観測データ(csv)を用いて、
    観測を再現する最適な表面密度分布を推定します。
    CSV読み込み機能強化版。
==============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import glob
import sys

# ==============================================================================
# 設定
# ==============================================================================
KERNEL_DIR = "./Inverse_Kernels_Full_2"
CSV_DAWN = "./dawn.csv"
CSV_DUSK = "./dusk.csv"

# グリッド定義 (生成コードと合わせる)
N_LON = 18
N_LAT = 9

# 正則化パラメータ
LAMBDA_SMOOTH = 0.1


def read_observation_csv(filepath, label):
    """
    CSVファイルを堅牢に読み込む関数
    """
    print(f"[{label}] Reading {filepath} ...")

    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return None, None

    # エンコーディングをいくつか試す
    encodings = ['utf-8', 'shift_jis', 'cp932']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            print(f"  -> Successfully read with encoding: {enc}")
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        print(f"Error: Failed to decode {filepath}. Check encoding.")
        return None, None

    # データ構造の確認
    print(f"  -> Columns found: {list(df.columns)}")
    if len(df.columns) < 5:
        print("Error: CSV must have at least 5 columns.")
        return None, None

    # 4列目(index 3)がTAA、5列目(index 4)が密度
    # 強制的に数値に変換 (エラーはNaNになる)
    taa_col = pd.to_numeric(df.iloc[:, 3], errors='coerce')
    den_col = pd.to_numeric(df.iloc[:, 4], errors='coerce')

    # NaNを含む行（数値変換できなかった行）を削除
    valid_mask = taa_col.notna() & den_col.notna()
    taa_clean = taa_col[valid_mask].values
    den_clean = den_col[valid_mask].values

    dropped_count = len(df) - len(taa_clean)
    if dropped_count > 0:
        print(f"  -> Dropped {dropped_count} rows due to non-numeric data.")

    # ★重要: TAAでソートする (np.interpのため必須)
    sort_idx = np.argsort(taa_clean)
    taa_sorted = taa_clean[sort_idx]
    den_sorted = den_clean[sort_idx]

    print(f"  -> Loaded {len(taa_sorted)} data points.")
    print(f"  -> TAA Range: {taa_sorted.min():.1f} ~ {taa_sorted.max():.1f}")

    return taa_sorted, den_sorted


def load_kernels_and_obs():
    print("\n--- Loading Kernels ---")
    kernel_files = glob.glob(os.path.join(KERNEL_DIR, "kernel_taa*.npy"))
    if not kernel_files:
        print("Error: Kernel files not found. Run simulation_kernel_gen first.")
        return None, None, None

    kernels = {}
    for f in kernel_files:
        try:
            base = os.path.basename(f)
            # ファイル名からTAA抽出 "kernel_taa045.npy" -> 45
            taa_str = base.replace("kernel_taa", "").replace(".npy", "")
            taa = int(taa_str)
            kernels[taa] = np.load(f)
        except:
            continue

    sorted_taas = sorted(kernels.keys())
    print(f"Loaded {len(sorted_taas)} kernels.")

    print("\n--- Loading Observations ---")
    obs_taa_dawn, obs_val_dawn = read_observation_csv(CSV_DAWN, "Dawn")
    obs_taa_dusk, obs_val_dusk = read_observation_csv(CSV_DUSK, "Dusk")

    if obs_taa_dawn is None or obs_taa_dusk is None:
        return None, None, None

    # カーネルのTAAに対応する観測値を補間してターゲットデータを作成
    obs_targets = []
    valid_taas = []
    valid_kernels = []

    print("\n--- Matching TAA ---")
    count = 0
    for t in sorted_taas:
        # 観測データの範囲内にあるTAAのみ使用
        # (外挿を防ぐため、dawn/dusk両方の範囲内にあるものだけ使う)
        in_dawn = (t >= obs_taa_dawn.min()) and (t <= obs_taa_dawn.max())
        in_dusk = (t >= obs_taa_dusk.min()) and (t <= obs_taa_dusk.max())

        if in_dawn and in_dusk:
            val_dawn = np.interp(t, obs_taa_dawn, obs_val_dawn)
            val_dusk = np.interp(t, obs_taa_dusk, obs_val_dusk)

            obs_targets.append([val_dawn, val_dusk])
            valid_taas.append(t)
            valid_kernels.append(kernels[t])
            count += 1

    print(f"Matched {count} data points for optimization.")

    return np.array(valid_taas), np.array(valid_kernels), np.array(obs_targets)


def predict(surface_vector, kernels_array):
    surface_map = surface_vector.reshape((N_LON, N_LAT))
    # pred[t, region] = sum( surface * kernel[t] )
    pred = np.tensordot(kernels_array, surface_map, axes=([1, 2], [0, 1]))
    return pred


def objective_function(x, kernels_array, obs_data):
    # 予測
    preds = predict(x, kernels_array)

    # 誤差 (Obsが0に近い場合の除算エラーを防ぐため単純二乗和推奨)
    diff = preds - obs_data
    loss_fit = np.sum(diff ** 2)

    # 正則化 (Smoothing)
    map_2d = x.reshape((N_LON, N_LAT))
    diff_lon = map_2d - np.roll(map_2d, 1, axis=0)  # 経度方向は周期的
    diff_lat = map_2d[:, 1:] - map_2d[:, :-1]  # 緯度方向は端がある

    loss_smooth = np.sum(diff_lon ** 2) + np.sum(diff_lat ** 2)

    # スケール調整: 観測値の平均二乗などで正規化するとλの設定が楽になるが、
    # ここではシンプルに係数で調整
    obs_scale = np.mean(obs_data ** 2) if np.mean(obs_data ** 2) > 0 else 1.0

    return loss_fit + LAMBDA_SMOOTH * obs_scale * 0.1 * loss_smooth


def main():
    taas, kernels, obs_data = load_kernels_and_obs()
    if taas is None or len(taas) == 0:
        print("No valid data for optimization. Aborting.")
        return

    print("\n--- Starting Optimization ---")

    # カーネルを配列化
    kernels_array = kernels

    # 初期値推定
    # 平均的に観測値を満たすような一様分布の値を計算
    mean_obs = np.mean(obs_data)
    mean_kernel = np.mean(kernels_array)
    init_val = mean_obs / (mean_kernel * N_LON * N_LAT + 1e-30)

    x0 = np.full(N_LON * N_LAT, init_val)

    # 密度は非負
    bounds = [(0, None) for _ in range(len(x0))]

    res = minimize(objective_function, x0, args=(kernels_array, obs_data),
                   method='L-BFGS-B', bounds=bounds,
                   options={'disp': True, 'maxiter': 2000})

    print(f"\nOptimization Finished. Success: {res.success}")

    best_map = res.x.reshape((N_LON, N_LAT))

    # --- 結果プロット ---
    pred_final = predict(res.x, kernels_array)

    # 1. フィッティング結果
    plt.figure(figsize=(12, 6))

    # 元の観測データをロードしなおしてプロット（補間前も含めて表示）
    # (ここでは簡易的にマッチングに使った点だけ表示)
    plt.plot(taas, obs_data[:, 0], 'ro', label='Observed Dawn', alpha=0.6)
    plt.plot(taas, obs_data[:, 1], 'bo', label='Observed Dusk', alpha=0.6)

    plt.plot(taas, pred_final[:, 0], 'r-', label='Model Prediction (Dawn)', linewidth=2)
    plt.plot(taas, pred_final[:, 1], 'b-', label='Model Prediction (Dusk)', linewidth=2)

    plt.title("Inverse Analysis Result: Column Density Fitting")
    plt.xlabel("True Anomaly Angle (deg)")
    plt.ylabel("Column Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2. 推定された表面密度マップ
    plt.figure(figsize=(10, 6))
    # 経度は -180~180 (Subsolar=0)
    # 緯度は -90~90
    extent = [-180, 180, -90, 90]

    # 転置して表示 (行=経度, 列=緯度 なので imshow では転置が必要)
    im = plt.imshow(best_map.T, origin='lower', extent=extent, aspect='auto', cmap='inferno')
    cbar = plt.colorbar(im, label='Relative Surface Source Strength')

    plt.xlabel("Relative Longitude from Subsolar (deg)")
    plt.ylabel("Latitude (deg)")
    plt.title("Estimated Surface Source Distribution")
    plt.tight_layout()
    plt.show()

    # 保存
    np.savetxt("estimated_source_map.csv", best_map, delimiter=",")
    print("\nResult saved to 'estimated_source_map.csv'")


if __name__ == "__main__":
    main()