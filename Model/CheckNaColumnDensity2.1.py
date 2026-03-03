import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re

# --- ★★★ 設定項目 ★★★ ---
# シミュレーション結果が保存されている親フォルダを指定
RESULTS_DIR = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult"

# 読み込むファイルの命名規則（ワイルドカード * を使用）
FILE_PATTERN = "density_map_taa*_beta0.50_Q1.0_MW_pl24_RA.npy"

# ★ 平均を取りたい「角度」を度で指定 (0度は太陽方向)
AVERAGING_ANGLE_DEG = 60

# ★ 平均を取る「半径の範囲」を水星半径単位で指定
RADIAL_RANGE_RM_MIN = 1.0  # 平均を開始する半径
RADIAL_RANGE_RM_MAX = 1.7  # 平均を終了する半径

# シミュレーションで設定したグリッドの最大半径（水星半径単位）
GRID_RADIUS_RM = 5.0


# -----------------

# ★ 関数名を分かりやすく変更
def plot_and_save_radial_average_vs_taa(results_dir, file_pattern, avg_angle_deg, radial_range_rm, grid_radius_rm):
    """
    極座標のシミュレーション結果(NPY)を読み込み、
    指定された角度に沿った半径方向の平均密度とTAAの関係をプロットし、
    結果をCSVファイルとして保存する。
    """
    search_path = os.path.join(results_dir, '**', file_pattern)
    file_list = glob.glob(search_path, recursive=True)

    if not file_list:
        print(f"エラー: '{search_path}' に一致するファイルが見つかりません。")
        return

    print(f"{len(file_list)}個のファイルを検出しました。")

    taas = []
    avg_densities = []  # ★ 平均密度を格納するリスト

    for filepath in file_list:
        try:
            filename = os.path.basename(filepath)
            match = re.search(r'taa(\d+)', filename)
            if not match:
                print(f"警告: ファイル名 '{filename}' からTAAを抽出できませんでした。スキップします。")
                continue
            taa = int(match.group(1))

            if not filename.endswith('.npy'):
                continue

            density_grid = np.load(filepath)
            N_R, N_THETA = density_grid.shape

            # --- ★ ここから計算ロジックを変更 ---

            # 1. 角度方向のインデックス(itheta)を計算
            target_theta_rad = np.deg2rad((avg_angle_deg + 180) % 360 - 180)
            itheta = int(((target_theta_rad + np.pi) / (2 * np.pi)) * N_THETA)
            itheta = np.clip(itheta, 0, N_THETA - 1)  # 範囲内に収める

            # 2. 半径方向のインデックス範囲(ir_start, ir_end)を計算
            r_min, r_max = radial_range_rm
            ir_start = int((r_min / grid_radius_rm) * N_R)
            ir_end = int((r_max / grid_radius_rm) * N_R)

            # インデックスが範囲内かチェック
            if not (0 <= ir_start < ir_end <= N_R):
                print(f"警告: 半径範囲 {radial_range_rm} RM はグリッド範囲外です。スキップします。")
                continue

            # 3. 指定した角度の列について、指定した半径範囲でデータをスライス
            radial_slice = density_grid[ir_start:ir_end, itheta]

            # 4. スライスしたデータの平均値を計算
            if radial_slice.size > 0:
                average_density = np.mean(radial_slice)
            else:
                average_density = 0  # スライスが空の場合は0とする

            # --- ★ 変更ここまで ---

            taas.append(taa)
            avg_densities.append(average_density)

        except Exception as e:
            print(f"ファイル '{filepath}' の処理中にエラーが発生しました: {e}")

    if not taas:
        print("処理できるデータがありませんでした。")
        return

    # TAAでソート
    sorted_pairs = sorted(zip(taas, avg_densities))
    taas_sorted, densities_sorted = zip(*sorted_pairs)

    # --- ★ 出力ファイル名とグラフタイトルを更新 ---
    r_min, r_max = radial_range_rm
    output_filename_base = f"avg_density_at_{avg_angle_deg}deg_from_{r_min:.1f}-{r_max:.1f}RM"
    output_csv_filename = f"{output_filename_base}.csv"

    output_data = np.column_stack((taas_sorted, densities_sorted))
    np.savetxt(output_csv_filename, output_data, fmt='%.6e', delimiter=',', header='TAA,Average_Column_Density',
               comments='')
    print(f"\n結果を '{output_csv_filename}' に保存しました。")

    # --- グラフ描画 ---
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(taas_sorted, densities_sorted, marker='o', linestyle='-', color='darkgreen')

    # ★ タイトルとラベルを更新
    title = (f"Average Column Density at {avg_angle_deg}° (from {r_min}$R_M$ to {r_max}$R_M$) vs. TAA")
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("True Anomaly Angle (TAA) [degrees]", fontsize=12)
    ax.set_ylabel("Average Column Density [atoms/cm$^2$]", fontsize=12)  # ラベルをAverageに変更
    ax.set_yscale('log')
    ax.grid(True, which="both", linestyle='--', alpha=0.7)

    plt.tight_layout()

    output_png_filename = f"{output_filename_base}.png"
    plt.savefig(output_png_filename)
    print(f"グラフを '{output_png_filename}' に保存しました。")
    plt.show()


if __name__ == '__main__':
    # ★ 新しい設定をタプルとして渡す
    radial_range = (RADIAL_RANGE_RM_MIN, RADIAL_RANGE_RM_MAX)

    # ★ 変更した関数を呼び出す
    plot_and_save_radial_average_vs_taa(RESULTS_DIR, FILE_PATTERN, AVERAGING_ANGLE_DEG, radial_range, GRID_RADIUS_RM)