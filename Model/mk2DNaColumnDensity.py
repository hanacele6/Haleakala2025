import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

# --- ★★★ 設定項目 ★★★ ---

# シミュレーション結果が保存されている親フォルダを指定
RESULTS_DIR = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult"


BETA_VALUE = 0.5
Q_VALUE = 0.14
TAA_TO_PLOT = 150


# --- 以下のファイル名とフォルダ名は自動生成されます ---
# フォルダ名 (例: density_map_beta0.5_Q0.14)
SUB_FOLDER_NAME = f"density_map_beta{BETA_VALUE:.2f}_Q{Q_VALUE}"
# ファイル名テンプレート (例: density_map_taa{}_beta0.50_Q0.14.npy)
FILENAME_TEMPLATE = f"density_map_taa{{}}_beta{BETA_VALUE:.2f}_Q{Q_VALUE}.npy"


# シミュレーションで設定したグリッドの半径（水星半径単位）
GRID_RADIUS_RM = 5.0


# ----------------------------

def visualize_density_map(results_dir, sub_folder, taa, filename_template, grid_radius_rm):
    """
    指定されたTAAの.npyファイルを読み込み、2次元の密度マップとして可視化する。
    """
    # 1. 表示するファイルのフルパスを構築
    filename_with_ext = filename_template.format(taa)
    # 新しいフォルダ構造に合わせてパスを組み立てる
    filepath = os.path.join(results_dir, sub_folder, filename_with_ext)

    # 2. ファイルが存在するかチェック
    if not os.path.exists(filepath):
        print(f"エラー: ファイルが見つかりません。パスを確認してください。")
        print(f"-> {filepath}")
        return

    # 3. .npyファイルを読み込む
    print(f"ファイルを読み込んでいます: {filepath}")
    density_grid = np.load(filepath)

    # 4. 描画処理
    fig, ax = plt.subplots(figsize=(10, 8))

    # データの最小値が0の場合、対数スケールでエラーになるため、0より大きい最小値をクリップ値として使用
    positive_min = np.min(density_grid[density_grid > 0]) if np.any(density_grid > 0) else 1e-10

    # imshowで2次元配列を画像として表示
    im = ax.imshow(density_grid,
                   cmap='inferno',  # カラーマップ（'viridis', 'plasma', 'magma' などもおすすめ）
                   norm=LogNorm(vmin=positive_min, vmax=np.max(density_grid)),  # 対数スケールを指定
                   extent=[-grid_radius_rm, grid_radius_rm, -grid_radius_rm, grid_radius_rm],  # 軸の範囲を物理単位に
                   origin='lower'  # Y軸の原点を左下に設定
                   )

    # 5. グラフの装飾
    # 水星本体を表す円を追加 (半径1 RM)
    mercury_circle = plt.Circle((0, 0), 1, color='white', fill=False, linestyle='--', linewidth=1.2,
                                label='Mercury ($R_M=1$)')
    ax.add_artist(mercury_circle)

    # カラーバーを追加
    cbar = fig.colorbar(im, ax=ax, extend='min')
    cbar.set_label('Column Density [atoms/cm$^2$]', fontsize=12)

    # タイトルと軸ラベル
    ax.set_title(f'Sodium Column Density Map at TAA = {taa}°', fontsize=16)
    ax.set_xlabel(f'X-axis [$R_M$] (Sun is to the right)', fontsize=12)
    ax.set_ylabel(f'Y-axis [$R_M$]', fontsize=12)

    # アスペクト比を1:1に
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', color='white', alpha=0.5)
    ax.legend(handles=[mercury_circle], loc='upper right')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    visualize_density_map(RESULTS_DIR, SUB_FOLDER_NAME, TAA_TO_PLOT, FILENAME_TEMPLATE, GRID_RADIUS_RM)