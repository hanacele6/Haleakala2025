import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import sys

# --- ★★★ 設定項目 ★★★ ---

# 1. シミュレーション結果が保存されている親フォルダを指定
RESULTS_DIR = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult"

# 2. プロットしたいTAAの値を指定
TAA_TO_PLOT = 180

# 3. シミュレーションで設定したグリッドの半径（水星半径単位）
GRID_RADIUS_RM = 5.0


# --- 設定はここまで ---


def visualize_density_map(filepath, grid_radius_rm, taa):
    """
    指定された.npyファイルのパスを元に、2次元の密度マップを可視化する。
    """
    # .npyファイルを読み込む
    print(f"ファイルを読み込んでいます: {filepath}")
    density_grid = np.load(filepath)

    # 描画処理
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('black')

    # データの最小値が0の場合、対数スケールでエラーになるため、0より大きい最小値をクリップ値として使用
    positive_min = np.min(density_grid[density_grid > 0]) if np.any(density_grid > 0) else 1e-10

    # imshowで2次元配列を画像として表示
    im = ax.imshow(density_grid,
                   cmap='inferno',
                   norm=LogNorm(vmin=positive_min, vmax=np.max(density_grid)),
                   extent=[-grid_radius_rm, grid_radius_rm, -grid_radius_rm, grid_radius_rm],
                   origin='lower'
                   )

    # グラフの装飾
    mercury_circle = plt.Circle((0, 0), 1, color='white', fill=False, linestyle='--', linewidth=1.2,
                                label='Mercury ($R_M=1$)')
    ax.add_artist(mercury_circle)

    cbar = fig.colorbar(im, ax=ax, extend='min')
    cbar.set_label('Column Density [atoms/cm$^2$]', fontsize=12)

    ax.set_title(f'Sodium Column Density Map at TAA = {taa}°', fontsize=16)
    ax.set_xlabel('X-axis [$R_M$] (Sun is to the right)', fontsize=12)
    ax.set_ylabel('Y-axis [$R_M$]', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', color='white', alpha=0.5)
    ax.legend(handles=[mercury_circle], loc='upper right')

    plt.tight_layout()
    plt.show()


def find_and_plot_results(results_dir, taa_to_plot, grid_radius_rm):
    """
    結果フォルダを自動で検索し、ユーザーが選択したフォルダから
    指定されたTAAのファイルをプロットする。
    """
    # 1. results_dir 内のサブフォルダをリストアップ
    try:
        sub_folders = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    except FileNotFoundError:
        print(f"エラー: 親フォルダが見つかりません: {results_dir}")
        sys.exit()

    if not sub_folders:
        print(f"エラー: '{results_dir}' 内にシミュレーション結果のフォルダが見つかりません。")
        return

    # 2. プロット対象のフォルダを決定
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

    # 3. 選択されたフォルダ内で、指定されたTAAのファイルを探す
    full_folder_path = os.path.join(results_dir, target_sub_folder)
    target_filepath = None

    # ファイル名の形式 `...taa{TAA}_...` にマッチするものを探す
    search_pattern = f"taa{taa_to_plot}_"

    for filename in os.listdir(full_folder_path):
        if search_pattern in filename and filename.endswith('.npy'):
            target_filepath = os.path.join(full_folder_path, filename)
            break

    if target_filepath:
        visualize_density_map(target_filepath, grid_radius_rm, taa_to_plot)
    else:
        print(f"エラー: フォルダ '{target_sub_folder}' 内に TAA = {taa_to_plot} の .npy ファイルが見つかりません。")
        print("ファイル名が `...taa{TAA}_...` の形式になっているか確認してください。")


if __name__ == '__main__':
    find_and_plot_results(RESULTS_DIR, TAA_TO_PLOT, GRID_RADIUS_RM)