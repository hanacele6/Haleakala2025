import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import sys

# --- ★★★ 設定項目 ★★★ ---

# 1. シミュレーション結果が保存されている親フォルダを指定
RESULTS_DIR = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult"

# 2. プロットしたいTAAの値を指定
TAA_TO_PLOT = 90

# 3. シミュレーションで設定したグリッドの半径（水星半径単位）
GRID_RADIUS_RM = 5.0


# --- 設定はここまで ---

# <--- 変更点: 極座標データをプロットするための新しい関数 ---
def visualize_density_map_polar(filepath, grid_radius_rm, taa):
    """
    指定された.npyファイル（極座標データ）を元に、2次元の密度マップを正しく可視化する。
    """
    print(f"ファイルを読み込んでいます: {filepath}")
    density_grid = np.load(filepath)

    # グリッドの形状から半径・角度の分割数を取得
    N_R, N_THETA = density_grid.shape

    # 1. pcolormesh用の座標グリッドを生成
    #    各セルの「角」の座標が必要なため、分割数+1の大きさの配列を作る

    # 半径方向の区切り (0 から最大半径まで)
    r_edges = np.linspace(0, grid_radius_rm, N_R + 1)

    # 角度方向の区切り (-pi から +pi まで。円を正しく描画するため)
    theta_edges = np.linspace(-np.pi, np.pi, N_THETA + 1)

    # 2. メッシュグリッドを作成
    #    1次元の半径・角度配列から、2次元のグリッド座標を生成
    theta_grid, r_grid = np.meshgrid(theta_edges, r_edges)

    # 3. 極座標グリッドをデカルト座標(x, y)に変換
    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)

    # --- 描画処理 ---
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('black')

    # 0や負の値をマスクして対数スケールエラーを回避
    masked_density = np.ma.masked_where(density_grid <= 0, density_grid)

    # 4. pcolormeshでプロット
    #    imshowの代わりにpcolormeshを使い、X, Y座標と色(密度)を指定する
    im = ax.pcolormesh(x_grid, y_grid, masked_density,
                       cmap='inferno',
                       norm=LogNorm(vmin=masked_density.min(), vmax=masked_density.max()),
                       shading='auto')

    # グラフの装飾
    mercury_circle = plt.Circle((0, 0), 1, color='white', fill=False, linestyle='--', linewidth=1.2,
                                label='Mercury ($R_M=1$)')
    ax.add_artist(mercury_circle)

    cbar = fig.colorbar(im, ax=ax, extend='min')
    cbar.set_label('Column Density [atoms/cm$^2$]', fontsize=12)

    ax.set_title(f'Sodium Column Density Map at TAA = {taa}° (Polar)', fontsize=16)
    # <--- 変更点: X軸ラベルの修正
    ax.set_xlabel('X-axis [$R_M$] (Sun is at +X)', fontsize=12)
    ax.set_ylabel('Y-axis [$R_M$]', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', color='white', alpha=0.5)
    ax.legend(handles=[mercury_circle], loc='upper right')

    # プロット範囲をグリッド半径に合わせる
    ax.set_xlim(-grid_radius_rm, grid_radius_rm)
    ax.set_ylim(-grid_radius_rm, grid_radius_rm)

    plt.tight_layout()
    plt.show()


def find_and_plot_results(results_dir, taa_to_plot, grid_radius_rm):
    """
    結果フォルダを自動で検索し、ユーザーが選択したフォルダから
    指定されたTAAのファイルをプロットする。
    """
    try:
        sub_folders = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    except FileNotFoundError:
        print(f"エラー: 親フォルダが見つかりません: {results_dir}")
        sys.exit()

    if not sub_folders:
        print(f"エラー: '{results_dir}' 内にシミュレーション結果のフォルダが見つかりません。")
        return

    target_sub_folder = ""
    if len(sub_folders) == 1:
        target_sub_folder = sub_folders[0]
        print(f"フォルダが一つ見つかりました。これを使用します: {target_sub_folder}")
    else:
        print("複数のシミュレーションフォルダが見つかりました。どちらをプロットしますか？")
        for i, folder in enumerate(sub_folders):
            print(f"  [{i + 1}] {folder}")
        try:
            choice = int(input("プロットしたいフォルダの番号を入力してください: ")) - 1
            if 0 <= choice < len(sub_folders):
                target_sub_folder = sub_folders[choice]
            else:
                print("エラー: 無効な番号です。")
                return
        except ValueError:
            print("エラー: 数値を入力してください。")
            return

    full_folder_path = os.path.join(results_dir, target_sub_folder)
    target_filepath = None
    search_pattern = f"taa{taa_to_plot}_"

    for filename in os.listdir(full_folder_path):
        if search_pattern in filename and filename.endswith('.npy'):
            target_filepath = os.path.join(full_folder_path, filename)
            break

    if target_filepath:
        # <--- 変更点: 新しい極座標プロット関数を呼び出す ---
        visualize_density_map_polar(target_filepath, grid_radius_rm, taa_to_plot)
    else:
        print(f"エラー: フォルダ '{target_sub_folder}' 内に TAA = {taa_to_plot} の .npy ファイルが見つかりません。")
        print("ファイル名が `...taa{TAA}_...` の形式になっているか確認してください。")


if __name__ == '__main__':
    find_and_plot_results(RESULTS_DIR, TAA_TO_PLOT, GRID_RADIUS_RM)