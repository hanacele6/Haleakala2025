import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import glob
import re
from tqdm import tqdm

# --- 設定項目 ---

# ★★★ シミュレーション結果が保存されているフォルダの親フォルダを指定 ★★★
# 例: "C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult3D_test"
OUTPUT_DIRECTORY = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult3D_test"

# ★★★ シミュレーションで実行した際の `run_name` を指定 ★★★
# コードで自動生成されたフォルダ名です
RUN_NAME = "PSD_atm_R50_YEAR1.0_SPperCell10_Q2.0_new"  # ← ご自身の環境に合わせて変更してください

# ★★★ プロットを保存するフォルダ名 ★★★
PLOT_SAVE_DIR = "surface_plots"

# シミュレーションのグリッド設定 (シミュレーションコードと一致させる)
N_LON = 24
N_LAT = 24


# --- 可視化メイン処理 ---

def visualize_surface_density():
    """
    表面密度データを読み込み、経度-緯度マップとしてプロットして画像保存する。
    """
    # データの入力パスと画像の出力パスを構築
    data_path = os.path.join(OUTPUT_DIRECTORY, RUN_NAME)
    plot_path = os.path.join(data_path, PLOT_SAVE_DIR)
    os.makedirs(plot_path, exist_ok=True)
    print(f"データは '{data_path}' から読み込まれます。")
    print(f"プロットは '{plot_path}' に保存されます。")

    # 可視化対象のファイルリストを取得
    search_pattern = os.path.join(data_path, "surface_density_t*.npy")
    file_list = sorted(glob.glob(search_pattern))

    if not file_list:
        print(f"エラー: '{search_pattern}' に一致するファイルが見つかりませんでした。")
        print("`OUTPUT_DIRECTORY` と `RUN_NAME` の設定を確認してください。")
        return

    # 経度と緯度の軸を度数法で作成
    # プロット用に -180° から +180° の範囲にする
    lon_edges = np.linspace(-180, 180, N_LON + 1)
    lat_edges = np.linspace(-90, 90, N_LAT + 1)

    # 全ファイルの密度データの最小値と最大値を調べて、カラーバーの範囲を統一する
    vmin, vmax = 1e18, 0  # 密度の最小・最大値の初期化
    all_data = []
    for filepath in file_list:
        data = np.load(filepath)
        all_data.append(data)
        if np.max(data) > vmax:
            vmax = np.max(data)
        # 0より大きい最小値を探す
        if np.min(data[data > 0]) < vmin:
            vmin = np.min(data[data > 0])

    print(f"カラーバーの範囲を統一します: min={vmin:.2e}, max={vmax:.2e}")

    # 各ファイルをプロット
    for filepath, data in tqdm(zip(file_list, all_data), total=len(file_list), desc="Plotting"):
        # ファイル名から時間を抽出
        match = re.search(r't(\d+)', os.path.basename(filepath))
        time_h = int(match.group(1)) if match else 0

        # --- プロット作成 ---
        fig, ax = plt.subplots(figsize=(10, 5))

        # pcolormesh を使ってヒートマップを作成
        # データは (N_LON, N_LAT) なので、転置 (.T) して (N_LAT, N_LON) に合わせる
        # LogNormを使って対数スケールで表示すると、値の小さい部分の変化も分かりやすい
        c = ax.pcolormesh(lon_edges, lat_edges, data.T,
                          norm=LogNorm(vmin=vmin, vmax=vmax),
                          cmap='viridis', shading='auto')

        # カラーバーを追加
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label('Surface Density (atoms / m$^2$)')

        # タイトルと軸ラベルを設定
        ax.set_title(f'Mercury Surface Na Density at T = {time_h} hours')
        ax.set_xlabel('Longitude [degrees]')
        ax.set_ylabel('Latitude [degrees]')

        # 軸の範囲とアスペクト比を設定
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xticks(np.arange(-180, 181, 60))
        ax.set_yticks(np.arange(-90, 91, 30))
        ax.grid(True, linestyle='--', alpha=0.6)

        # 画像をファイルに保存
        output_filename = f"surface_density_map_t{time_h:05d}.png"
        plt.savefig(os.path.join(plot_path, output_filename), dpi=150)

        # メモリを解放するためにプロットを閉じる
        plt.close(fig)

    print("\n★★★ すべてのプロットが完了しました ★★★")


if __name__ == '__main__':
    visualize_surface_density()