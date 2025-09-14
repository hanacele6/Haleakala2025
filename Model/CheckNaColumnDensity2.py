import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re

# --- 設定項目 ---
# シミュレーション結果が保存されているフォルダを指定
RESULTS_DIR = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult"

# 読み込むファイルの命名規則（ワイルドカード * を使用）
# <--- 変更点: 極座標のファイルパターンを指定 ---
FILE_PATTERN = "density_map_taa*_beta0.50_Q1.0_MW_pl24_RA.npy"

# <--- 変更点: 密度を監視したい特定の座標を「極座標」で指定 ---
# (半径[RM単位], 角度[度])で指定します
# (1.0, 0)   : 太陽直下点 (Subsolar point)
# (1.0, 90)  : 北極方向
# (1.0, 180) : 反太陽点 (Anti-solar point)
TARGET_COORD_POLAR = (1.2, 0)  # (半径, 角度[度])

# シミュレーションで設定したグリッドの最大半径（水星半径単位）
GRID_RADIUS_RM = 5.0


# -----------------

def plot_and_save_taa_vs_density_polar(results_dir, file_pattern, target_coord_polar, grid_radius_rm):
    """
    極座標のシミュレーション結果(NPY)を読み込み、特定座標の密度とTAAの関係をプロットし、
    結果をCSVファイルとして保存する。
    """
    search_path = os.path.join(results_dir, '**', file_pattern)
    file_list = glob.glob(search_path, recursive=True)

    if not file_list:
        print(f"エラー: '{search_path}' に一致するファイルが見つかりません。")
        print("RESULTS_DIR と FILE_PATTERN の設定を確認してください。")
        return

    print(f"{len(file_list)}個のファイルを検出しました。")

    taas = []
    densities = []

    for filepath in file_list:
        try:
            filename = os.path.basename(filepath)
            match = re.search(r'taa(\d+)', filename)
            if not match:
                print(f"警告: ファイル名 '{filename}' からTAAを抽出できませんでした。スキップします。")
                continue
            taa = int(match.group(1))

            # .npyファイルのみを対象とする
            if not filename.endswith('.npy'):
                print(f"警告: サポートされていないファイル形式です ('{filename}')。スキップします。")
                continue

            density_grid = np.load(filepath)

            # <--- 変更点: ここからインデックス計算ロジックを全面変更 ---

            # グリッドの形状から半径・角度の分割数を取得
            N_R, N_THETA = density_grid.shape

            # 指定された物理座標
            target_r_rm, target_theta_deg = target_coord_polar

            # 角度をラジアンに変換 (-180~180度の範囲にしておく)
            target_theta_rad = np.deg2rad((target_theta_deg + 180) % 360 - 180)

            # 1. 半径方向のインデックス(ir)を計算
            # (半径 / 最大半径) * 半径方向の分割数
            ir = int((target_r_rm / grid_radius_rm) * N_R)

            # 2. 角度方向のインデックス(itheta)を計算
            # シミュレーションでは角度を-pi~+piで扱っているので、それに合わせる
            # (-pi~pi -> 0~2pi -> 0~1 -> 0~N_THETA)
            itheta = int(((target_theta_rad + np.pi) / (2 * np.pi)) * N_THETA)

            # インデックスが範囲内かチェック
            if not (0 <= ir < N_R and 0 <= itheta < N_THETA):
                print(f"警告: 座標 {target_coord_polar} はグリッド範囲外です。スキップします。")
                continue

            # 密度データを取得 (グリッドは [半径, 角度] の順)
            density = density_grid[ir, itheta]
            # <--- 変更点ここまで ---

            taas.append(taa)
            densities.append(density)

        except Exception as e:
            print(f"ファイル '{filepath}' の処理中にエラーが発生しました: {e}")

    if not taas:
        print("処理できるデータがありませんでした。")
        return

    sorted_pairs = sorted(zip(taas, densities))
    taas_sorted, densities_sorted = zip(*sorted_pairs)

    # --- 出力ファイル名とグラフタイトルを更新 ---
    output_filename_base = f"taa_vs_density_at_{target_coord_polar[0]:.1f}RM_{target_coord_polar[1]:.0f}deg"
    output_csv_filename = f"{output_filename_base}.csv"

    output_data = np.column_stack((taas_sorted, densities_sorted))
    np.savetxt(output_csv_filename, output_data, fmt='%.6e', delimiter=',', header='TAA,Column_Density', comments='')
    print(f"\n結果を '{output_csv_filename}' に保存しました。")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(taas_sorted, densities_sorted, marker='o', linestyle='-', color='darkblue')

    # タイトルを更新
    title = (f"Column Density Variation at (r={target_coord_polar[0]}$R_M$, "
             f"$\\theta$={target_coord_polar[1]}°) vs. TAA")
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("True Anomaly Angle (TAA) [degrees]", fontsize=12)
    ax.set_ylabel("Column Density [atoms/cm$^2$]", fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, which="both", linestyle='--', alpha=0.7)

    plt.tight_layout()

    output_png_filename = f"{output_filename_base}.png"
    plt.savefig(output_png_filename)
    print(f"グラフを '{output_png_filename}' に保存しました。")
    plt.show()


if __name__ == '__main__':
    plot_and_save_taa_vs_density_polar(RESULTS_DIR, FILE_PATTERN, TARGET_COORD_POLAR, GRID_RADIUS_RM)