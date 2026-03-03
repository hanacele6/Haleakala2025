# -*- coding: utf-8 -*-
"""
==============================================================================
水星ナトリウム外気圏シミュレーション比較解析ツール (Final Fixed)
(Mercury Na Exosphere Simulation Comparison Tool)

修正内容:
    1. ユーザーコードの正規化ロジックに完全準拠。
       - 総原子数(Total Atoms)を集計。
       - Dawn/Duskそれぞれ「半球の半分(1/4球)の面積」で割って柱密度を算出。
    2. Dawn/Duskの領域分割をユーザーコードのインデックス操作に一致。
       - Y < Center : Dawn
       - Y > Center : Dusk

作成者: Assistant / Koki Masaki
日付: 2025/12/31
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# --- 設定: 解析対象のディレクトリ ---
RESULT_DIRS = {
    'Leblanc Model': './SimulationResult_202512/ParabolicHop_72x36_EqMode_DT500_PLeblanc_DLeblanc',
    'Uniform Model': './SimulationResult_202512/ParabolicHop_72x36_EqMode_DT500_PLeblanc_DKillen',
    'Suzuki Model': './SimulationResult_202512/ParabolicHop_72x36_EqMode_DT500_PLeblanc_Dsuzuki'
}

# --- 物理定数・グリッド設定 (ユーザーコードと同一) ---
RM_M = 2440e3
GRID_RESOLUTION = 101
GRID_MAX_RM = 5.0

# ユーザーコードの正規化面積定数
NORMALIZATION_AREA_CM2 = 3.7408e17  # 半球 (2pi R^2)
NORMALIZATION_AREA_HALF_CM2 = NORMALIZATION_AREA_CM2 / 2.0  # 1/4球 (pi R^2)

# グリッドの物理サイズ
GRID_WIDTH_M = GRID_MAX_RM * RM_M * 2.0
CELL_LENGTH_M = GRID_WIDTH_M / GRID_RESOLUTION
CELL_VOLUME_M3 = CELL_LENGTH_M ** 3


def load_simulation_data_exact_logic(result_dir):
    """
    ユーザーコードのロジックを再現して Dawn/Dusk の柱密度を計算する
    """
    data = {}
    files = glob.glob(os.path.join(result_dir, 'density_grid_*.npy'))
    print(f"Loading {len(files)} files from {result_dir}...")

    for f in sorted(files):
        try:
            # TAA抽出
            taa_str = f.split('taa')[-1].split('.')[0]
            taa = int(taa_str)

            # 3Dグリッド読み込み [atoms/m^3]
            density_grid_m3 = np.load(f)

            # --- ユーザーコードの処理を再現 ---

            # 1. 昼側 (X >= 0) のみを切り出し
            # グリッド中心インデックス
            mid_index_x = (GRID_RESOLUTION - 1) // 2
            mid_index_y = (GRID_RESOLUTION - 1) // 2

            dayside_grid = density_grid_m3[mid_index_x:, :, :]

            # 2. 原子の総数に変換
            atoms_grid = dayside_grid * CELL_VOLUME_M3

            # 3. 境界処理 (X=0 Terminator面は半分)
            atoms_grid[0, :, :] *= 0.5

            # 4. Y軸中心(正午-真夜中線)の原子数 (分配用)
            sum_mid = np.sum(atoms_grid[:, mid_index_y, :])

            # --- DAWN 計算 ---
            # Yインデックス 0 ～ mid-1
            sum_dawn = np.sum(atoms_grid[:, :mid_index_y, :])
            total_atoms_dawn = sum_dawn + (0.5 * sum_mid)
            # 面積で割る (1/4球面積)
            dens_dawn = total_atoms_dawn / NORMALIZATION_AREA_HALF_CM2

            # --- DUSK 計算 ---
            # Yインデックス mid+1 ～ end
            sum_dusk = np.sum(atoms_grid[:, mid_index_y + 1:, :])
            total_atoms_dusk = sum_dusk + (0.5 * sum_mid)
            # 面積で割る (1/4球面積)
            dens_dusk = total_atoms_dusk / NORMALIZATION_AREA_HALF_CM2

            # 空間マップ用 (原子数積算マップ) - 軸修正用
            # X軸方向には積分せず、Z軸方向に積分して XYマップを作る
            # grid_atoms (Global) = density_grid_m3 * CELL_VOLUME_M3
            grid_atoms_global = density_grid_m3 * CELL_VOLUME_M3
            col_map_atoms = np.sum(grid_atoms_global, axis=2)

            data[taa] = {
                'dens_dawn': dens_dawn,
                'dens_dusk': dens_dusk,
                'col_map_atoms': col_map_atoms
            }
        except Exception as e:
            # print(f"Error {f}: {e}")
            pass

    return data


def main_analysis():
    results = {}
    for model_name, path in RESULT_DIRS.items():
        if os.path.exists(path):
            results[model_name] = load_simulation_data_exact_logic(path)
        else:
            print(f"Warning: Directory not found {path}")

    if not results:
        return

    # プロット用データ整理
    plot_data = {name: {'taa': [], 'dawn': [], 'dusk': []} for name in results}

    for model_name, data in results.items():
        taas = sorted(data.keys())
        for t in taas:
            plot_data[model_name]['taa'].append(t)
            plot_data[model_name]['dawn'].append(data[t]['dens_dawn'])
            plot_data[model_name]['dusk'].append(data[t]['dens_dusk'])

    # --- Plot 1: Dawn Side Comparison ---
    plt.figure(figsize=(10, 6))
    for name, pdata in plot_data.items():
        if pdata['taa']:
            idx = np.argsort(pdata['taa'])
            t_sorted = np.array(pdata['taa'])[idx]
            d_sorted = np.array(pdata['dawn'])[idx]
            plt.plot(t_sorted, d_sorted, marker='o',linestyle='', label=name)

        plt.xticks(np.arange(0, 361, 60))  # 0, 60, 120, ..., 360
        plt.xlim(0, 360)  # 表示範囲を0～360に固定

    #plt.title("Dawn Side Average Column Density (Normalized)")
    plt.xlabel("True Anomaly Angle (deg)", fontsize=18)  # 軸ラベル
    plt.ylabel("Column Density (atoms/cm²)", fontsize=18)  # 軸ラベル
    plt.tick_params(axis='both', which='major', labelsize=14)  # 目盛りの数字
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Analysis_Dawn_Density_Exact.png")
    plt.show()

    # --- Plot 2: Dusk Side Comparison ---
    plt.figure(figsize=(10, 6))
    for name, pdata in plot_data.items():
        if pdata['taa']:
            idx = np.argsort(pdata['taa'])
            t_sorted = np.array(pdata['taa'])[idx]
            d_sorted = np.array(pdata['dusk'])[idx]
            plt.plot(t_sorted, d_sorted, marker='o', linestyle='', label=name)

    plt.xticks(np.arange(0, 361, 60))  # 0, 60, 120, ..., 360
    plt.xlim(0, 360)  # 表示範囲を0～360に固定

    #plt.title("Dusk Side Average Column Density (Normalized)")
    plt.xlabel("True Anomaly Angle (deg)", fontsize=18)  # 軸ラベル
    plt.ylabel("Column Density (atoms/cm²)", fontsize=18)  # 軸ラベル
    plt.tick_params(axis='both', which='major', labelsize=14)  # 目盛りの数字
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Analysis_Dusk_Density_Exact.png")
    plt.show()

    # --- Plot 3: Spatial Map (Axis Corrected & Units Converted) ---
    target_taa = 300
    all_taas = sorted(list(set(t for r in results.values() for t in r.keys())))
    if not all_taas: return
    nearest_taa = min(all_taas, key=lambda x: abs(x - target_taa))
    print(f"Generating Spatial Map for TAA ~ {nearest_taa}")

    # 1セルの面積 [cm^2] (マップの値を密度にするため)
    # ユーザーコードは「総原子数/全表面積」で平均密度を出しているが、
    # マップは「その場所の柱密度」を出したいので、セルの断面積で割るのが物理的に自然。
    # しかし「ユーザーコードと同じ値」と比較するなら、ここも合わせる必要があるが、
    # マップは相対分布を見るものなので、ここでは物理的に正しい [atoms/cm^2] (local) で表示する。
    cell_area_cm2 = (CELL_LENGTH_M * 100) ** 2

    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
    if len(results) == 1: axes = [axes]

    for ax, (model_name, data) in zip(axes, results.items()):
        if nearest_taa in data:
            # 原子数マップ (X, Y)
            col_atoms = data[nearest_taa]['col_map_atoms']

            # 局所的な柱密度に変換
            col_dens_map = col_atoms / cell_area_cm2

            # 軸の転置 (+X=Sun, +Y=Dusk)
            # imshow は (Row, Col) = (Y, X) なので、転置すると 横軸がX(Sun)になる
            to_plot = col_dens_map.T

            extent = [-GRID_MAX_RM, GRID_MAX_RM, -GRID_MAX_RM, GRID_MAX_RM]

            im = ax.imshow(np.log10(to_plot + 1e5), origin='lower', cmap='inferno', extent=extent)

            ax.set_title(f"{model_name}\n(TAA={nearest_taa})")
            ax.set_xlabel("Sunward Axis (+X) [RM]")
            ax.set_ylabel("Dawn(-) - Dusk(+) Axis [RM]")

            # ガイド
            ax.text(3.5, 0, 'Sun', color='white', ha='center', fontweight='bold')
            ax.text(-3.5, 0, 'Anti', color='white', ha='center')
            ax.text(0, 3.5, 'Dusk', color='white', ha='center', fontweight='bold')
            ax.text(0, -3.5, 'Dawn', color='white', ha='center', fontweight='bold')

            plt.colorbar(im, ax=ax, label="Log Column Density (cm⁻²)")
        else:
            ax.text(0.5, 0.5, "No Data", ha='center')

    plt.tight_layout()
    plt.savefig("Analysis_Spatial_Map_Exact.png")
    plt.show()


if __name__ == "__main__":
    main_analysis()