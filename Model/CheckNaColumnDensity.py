import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re

# --- 設定項目 ---
# シミュレーション結果が保存されているフォルダを指定
RESULTS_DIR = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult"

# 読み込むファイルの命名規則（ワイルドカード * を使用）
# .npyファイルを読み込む場合: "density_map_taa*_beta0.50.npy"
# .csvファイルを読み込む場合: "density_map_taa*_beta0.50_PSD.csv"
FILE_PATTERN = "density_map_taa*_beta0.50.npy"

# 密度を監視したい特定の座標を水星半径(RM)単位で指定
# (0, 0)   : 水星の中心
# (1, 0)   : 太陽直下点 (Subsolar point)
# (0, 1)   : 北極方向
# (-1, 0)  : 反太陽点 (Anti-solar point)
TARGET_COORD_RM = (0.0, 1.0)

# シミュレーションで設定したグリッドの半径（水星半径単位）
GRID_RADIUS_RM = 5.0


# -----------------

def plot_taa_vs_density(results_dir, file_pattern, target_coord_rm, grid_radius_rm):
    """
    複数のCSVまたはNPYファイルを読み込み、特定座標の密度とTAAの関係をプロットする。
    """
    # ファイルの検索
    search_path = os.path.join(results_dir, file_pattern)
    file_list = glob.glob(search_path)

    if not file_list:
        print(f"エラー: '{search_path}' に一致するファイルが見つかりません。")
        print("RESULTS_DIR と FILE_PATTERN の設定を確認してください。")
        return

    print(f"{len(file_list)}個のファイルを検出しました。")

    taas = []
    densities = []

    # 各ファイルをループ処理
    for filepath in file_list:
        try:
            # --- ファイル名からTAAを抽出 ---
            filename = os.path.basename(filepath)
            # TAAの後のアンダースコアやベータも考慮した正規表現に修正
            match = re.search(r'taa(\d+)', filename)
            if not match:
                print(f"警告: ファイル名 '{filename}' からTAAを抽出できませんでした。スキップします。")
                continue
            taa = int(match.group(1))

            # --- ★★★ ファイル形式に応じて読み込み方法を変更 ★★★ ---
            file_extension = os.path.splitext(filename)[1]

            if file_extension == '.npy':
                density_grid = np.load(filepath)
            elif file_extension == '.csv':
                density_grid = np.loadtxt(filepath, delimiter=',')
            else:
                print(f"警告: サポートされていないファイル形式です ('{filename}')。スキップします。")
                continue
            # --- データの抽出 ---
            GRID_SIZE = density_grid.shape[0]

            # 物理座標 (RM) をグリッドのインデックスに変換
            target_x_rm, target_y_rm = target_coord_rm

            col_index = int((target_x_rm + grid_radius_rm) / (2 * grid_radius_rm) * GRID_SIZE)
            row_index = int((target_y_rm + grid_radius_rm) / (2 * grid_radius_rm) * GRID_SIZE)

            if not (0 <= row_index < GRID_SIZE and 0 <= col_index < GRID_SIZE):
                print(f"警告: 座標 {target_coord_rm} はグリッド範囲外です。スキップします。")
                continue

            density = density_grid[row_index, col_index]

            taas.append(taa)
            densities.append(density)

        except Exception as e:
            print(f"ファイル '{filepath}' の処理中にエラーが発生しました: {e}")

    if not taas:
        print("プロットできるデータがありませんでした。")
        return

    # TAAでデータをソートする
    sorted_pairs = sorted(zip(taas, densities))
    taas_sorted, densities_sorted = zip(*sorted_pairs)

    # --- 描画処理 ---
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(taas_sorted, densities_sorted, marker='o', linestyle='-', color='teal')

    ax.set_title(f"Column Density Variation at ({target_coord_rm[0]}$R_M$, {target_coord_rm[1]}$R_M$) vs. TAA",
                 fontsize=16)
    ax.set_xlabel("True Anomaly Angle (TAA) [degrees]", fontsize=12)
    ax.set_ylabel("Column Density [atoms/cm$^2$]", fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, which="both", linestyle='--', alpha=0.7)

    plt.tight_layout()

    # ファイルに保存
    output_filename = f"taa_vs_density_at_{target_coord_rm[0]}_{target_coord_rm[1]}.png"
    plt.savefig(output_filename)
    print(f"\nグラフを '{output_filename}' に保存しました。")

    # プロットを表示
    plt.show()


if __name__ == '__main__':
    # ファイルパスを結合
    full_path = os.path.join(RESULTS_DIR, FILE_PATTERN)
    plot_taa_vs_density(RESULTS_DIR, FILE_PATTERN, TARGET_COORD_RM, GRID_RADIUS_RM)