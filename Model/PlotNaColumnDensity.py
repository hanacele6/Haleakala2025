import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
from tqdm import tqdm

# ==============================================================================
# 設定: シミュレーションスクリプトの実行設定と合わせてください
# ==============================================================================

# ★★★ 1. シミュレーションの出力ディレクトリを指定
OUTPUT_DIRECTORY = r"./SimulationResult_202510"

# ★★★ 2. シミュレーションで指定した RUN_NAME を指定
# (例: "Grid101_Range5RM_SP1e25")
RUN_NAME = "Grid101_Range5RM_SP1e+24_2"

# ★★★ 3. シミュレーションで使用したグリッド設定
GRID_RESOLUTION = 101  # グリッドの解像度 (例: 101)
GRID_MAX_RM = 5.0  # グリッドの最大範囲 (例: 5.0)

# ★★★ 4. 視線方向の指定 (0: X軸, 1: Y軸, 2: Z軸)
# Z軸 (軌道面の上) からの見下ろしを推奨
INTEGRATION_AXIS = 2

# --- 物理定数 (シミュレーションコードから) ---
RM_MERCURY = 2.440e6  # 水星の半径 [m]


# ==============================================================================
# メイン処理: データの読み込みとプロット
# ==============================================================================

def analyze_and_plot():
    """
    シミュレーション結果を分析し、TAA vs 柱密度のグラフをプロットする。
    """

    # --- 1. パスの設定とファイルの検索 ---
    data_path = os.path.join(OUTPUT_DIRECTORY, RUN_NAME)
    file_pattern = os.path.join(data_path, "density_grid_t*_taa*.npy")
    npy_files = glob.glob(file_pattern)

    if not npy_files:
        print(f"エラー: {data_path} に .npy ファイルが見つかりません。")
        print("シミュレーションが完了しているか、RUN_NAMEが正しいか確認してください。")
        return

    print(f"{len(npy_files)}個の .npy ファイルを処理します...")

    # --- 2. グリッドのセルサイズを計算 ---
    grid_min = -GRID_MAX_RM * RM_MERCURY
    grid_max = GRID_MAX_RM * RM_MERCURY
    cell_size_m = (grid_max - grid_min) / GRID_RESOLUTION

    # --- 3. TAAをファイル名から抽出する正規表現 ---
    taa_pattern = re.compile(r"taa(\d{1,3})\.npy")

    results = []  # (taa, max_column_density) を格納するリスト

    # --- 4. 各ファイルをループ処理 ---
    for f_path in tqdm(npy_files, desc="Processing files"):
        filename = os.path.basename(f_path)

        match = taa_pattern.search(filename)
        if not match:
            continue
        taa = int(match.group(1))

        density_grid = np.load(f_path)

        column_density_map_2d = np.sum(density_grid, axis=INTEGRATION_AXIS) * cell_size_m

        max_column_density = np.max(column_density_map_2d)

        if max_column_density > 0:
            results.append((taa, max_column_density))

    if not results:
        print("エラー: 有効なデータがありませんでした。")
        return

    # --- 5. TAAでソート ---
    results.sort(key=lambda x: x[0])

    taas = [r[0] for r in results]
    col_densities = [r[1] for r in results]

    # --- 6. プロットの作成 (ご要望に基づき修正) ---
    print("プロットを作成中...")
    plt.figure(figsize=(12, 7))
    plt.plot(taas, col_densities, marker='o', linestyle='-', markersize=4)

    # ▼▼▼ 修正点 1: logスケールの設定を削除 ▼▼▼
    # plt.yscale('log')

    plt.xlabel("True Anomaly Anomaly (TAA) [degrees]", fontsize=14)

    # ▼▼▼ 修正点 1b: Y軸ラベルから (Log Scale) を削除 ▼▼▼
    plt.ylabel("Maximum Column Density [atoms/m^2]", fontsize=14)
    plt.title(f"Na Column Density vs. TAA\n(Run: {RUN_NAME}, LOS: {['X', 'Y', 'Z'][INTEGRATION_AXIS]}-axis)",
              fontsize=16)

    plt.grid(True, linestyle='--', alpha=0.6)

    plt.xticks(np.arange(0, 361, 45))
    plt.xlim(0, 360)

    # 柱密度の最小値が0に近い場合があるため、Y軸の下限を0に設定
    plt.ylim(bottom=0)

    # ▼▼▼ 修正点 2: savefig を plt.show() に変更 ▼▼▼
    # plot_filename = os.path.join(data_path, f"PLOT_TAA_vs_MaxColumnDensity_({RUN_NAME}).png")
    # plt.savefig(plot_filename)
    # print(f"プロットを {plot_filename} に保存しました。")

    print("プロットを表示します。")
    plt.show()


if __name__ == '__main__':
    analyze_and_plot()