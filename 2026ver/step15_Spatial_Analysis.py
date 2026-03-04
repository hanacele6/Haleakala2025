import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
import scipy.ndimage as ndimage
from matplotlib.colors import LogNorm

# ==============================================================================
# 設定パラメータ
# ==============================================================================
SPATIAL_CONFIG = {
    'nx': 10,  # X方向のファイバー数
    'ny': 12,  # Y方向のファイバー数
    'plate_scale': 1.0,  # 1ピクセル(ファイバー)あたりの秒角
    'default_diameter_arcsec': 6.0,  # CSVに直径がない場合のデフォルト値
}


# ==============================================================================
# Helper: エッジ検出関数
# ==============================================================================
def detect_edges_sobel(img_2d):
    """
    Sobelフィルタを用いて、画像の輝度勾配（エッジの強さ）を計算する。
    """
    # X方向、Y方向の勾配を計算
    grad_x = ndimage.sobel(img_2d, axis=1)
    grad_y = ndimage.sobel(img_2d, axis=0)

    # 勾配の大きさを求める (エッジの強さ)
    edge_strength = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # 正規化 (0〜1)
    if np.max(edge_strength) > 0:
        edge_strength /= np.max(edge_strength)

    return edge_strength


# ==============================================================================
# 中心推定関数 (Template Matching - Enhanced)
# ==============================================================================
def find_mercury_center(img_2d, r_pix):
    """
    12x10の輝度マップに対して、半径 r_pix の円テンプレートをマッチングさせ、
    真の中心座標 (cx, cy) を推定する。
    """
    ny, nx = img_2d.shape

    # 1. ざっくりとした初期位置（重心）を出す
    threshold = np.max(img_2d) * 0.2
    img_masked = np.where(img_2d > threshold, img_2d, 0)
    cy_init, cx_init = ndimage.center_of_mass(img_masked)

    if np.isnan(cy_init) or np.isnan(cx_init):
        cy_init, cx_init = ny / 2.0, nx / 2.0

    # 2. テンプレートマッチングによる高精度探索
    search_range = 2.0
    steps = 40

    best_score = -np.inf
    best_cx, best_cy = cx_init, cy_init

    y_idx, x_idx = np.indices((ny, nx))

    for dy in np.linspace(-search_range, search_range, steps):
        for dx in np.linspace(-search_range, search_range, steps):
            cx_test = cx_init + dx
            cy_test = cy_init + dy

            # 各ピクセルからテスト中心までの距離
            dist = np.sqrt((x_idx - cx_test) ** 2 + (y_idx - cy_test) ** 2)

            # スコア計算ロジックの強化:
            # 円周（リム）付近に光が集まっているかを重視

            # リムの幅（ピクセル）
            rim_width = 1.0

            # リム領域 (r-w/2 <= dist <= r+w/2)
            rim_mask = (dist >= r_pix - rim_width / 2) & (dist <= r_pix + rim_width / 2)

            # 水星のはるか外側 (ペナルティ領域)
            far_outside_mask = dist > r_pix + rim_width

            # スコア = リム領域の総光量 - 外側のはみ出しペナルティ
            score = np.sum(img_2d[rim_mask]) - 1.0 * np.sum(img_2d[far_outside_mask])

            if score > best_score:
                best_score = score
                best_cx = cx_test
                best_cy = cy_test

    return best_cx, best_cy


# ==============================================================================
# プロット関数 (確認用 - 大幅改良版)
# ==============================================================================
def plot_center_estimation(img_2d, cx, cy, r_pix, out_path, title):
    """
    12x10の画像の上に、計算されたエッジ（白い点線）と、
    推定された水星の輪郭（赤い円）をプロットする。
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # 1. 元のFlux画像をLogスケールで描画 (extentを使って座標系を合わせる)
    # Logスケールにすることで暗い部分のエッジも見やすくする
    norm = LogNorm(vmin=np.max(img_2d) * 0.01, vmax=np.max(img_2d)) if np.max(img_2d) > 0 else None
    im = ax.imshow(img_2d, origin='lower', cmap='viridis', norm=norm,
                   extent=[-0.5, SPATIAL_CONFIG['nx'] - 0.5, -0.5, SPATIAL_CONFIG['ny'] - 0.5])
    plt.colorbar(im, label='Total Flux (Log Scale)')

    # 2. ★計算されたエッジ（輝度勾配）を白い点線の等高線で重ね書き★
    # これが人間の目に見える「リム」の正体
    edge_strength = detect_edges_sobel(img_2d)
    # 勾配が強い上位3段階をプロット
    ax.contour(edge_strength, levels=[0.3, 0.5, 0.8], colors='white',
               linestyles='--', linewidths=1.0,
               extent=[-0.5, SPATIAL_CONFIG['nx'] - 0.5, -0.5, SPATIAL_CONFIG['ny'] - 0.5], alpha=0.7)

    # 3. 推定された水星の輪郭（赤い円）を描画
    circle = plt.Circle((cx, cy), r_pix, color='red', fill=False, linewidth=2.5, linestyle='-', label='Estimated Limb')
    ax.add_patch(circle)

    # 中心点にバツ印
    ax.plot(cx, cy, 'rx', markersize=12, markeredgewidth=2)

    # 凡例を追加
    ax.plot([], [], 'w--', label='Detected Edge (Gradient)')
    ax.legend(loc='upper right')

    ax.set_title(f"{title}\nCenter: ({cx:.2f}, {cy:.2f}), Radius: {r_pix:.2f} pix")
    ax.set_xlabel("Fiber X (pixels)")
    ax.set_ylabel("Fiber Y (pixels)")

    # グリッド（ファイバーの境界）を明確に描画
    ax.set_xticks(np.arange(SPATIAL_CONFIG['nx']) - 0.5, minor=True)
    ax.set_yticks(np.arange(SPATIAL_CONFIG['ny']) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', alpha=0.3)
    ax.tick_params(which='minor', bottom=False, left=False)

    plt.savefig(out_path, dpi=100)
    plt.close()


# ==============================================================================
# Main Logic
# ==============================================================================
def run(run_info, config):
    output_dir = Path(run_info["output_dir"])
    csv_file_path = run_info["csv_path"]

    print(f"\n--- 2D Spatial Analysis (Step 15: Center Estimation w/ Edge Visualization) ---")

    df = pd.DataFrame()
    if csv_file_path:
        try:
            if Path(csv_file_path).is_dir():
                cands = list(Path(csv_file_path).glob("*.csv"))
                if cands: df = pd.read_csv(cands[0])
            else:
                df = pd.read_csv(csv_file_path)
            if not df.empty:
                df = df[df['Type'] == 'MERCURY'].reset_index(drop=True)
        except Exception as e:
            print(f"  > CSV Read Error: {e}")

    spatial_dir = output_dir / "4_spatial_analysis"
    spatial_dir.mkdir(exist_ok=True)

    fits_dir = output_dir / "1_fits"
    if not fits_dir.exists():
        fits_dir = output_dir

    img_fits_files = list(fits_dir.glob("*_tr.img.fits"))

    if not img_fits_files:
        print(f"  > No *_tr.img.fits files found in {fits_dir}")
        return

    for fits_path in img_fits_files:
        try:
            with fits.open(fits_path) as hdul:
                img_data = hdul[0].data

            nx, ny = SPATIAL_CONFIG['nx'], SPATIAL_CONFIG['ny']

            if img_data.ndim == 1 and len(img_data) == nx * ny:
                img_2d = img_data.reshape((ny, nx))
            elif img_data.shape == (ny, nx):
                img_2d = img_data
            else:
                print(
                    f"  > Warning: Skipping {fits_path.name} due to unexpected shape {img_data.shape}. Expected ({ny}, {nx}).")
                continue

            dia_arcsec = SPATIAL_CONFIG['default_diameter_arcsec']
            if not df.empty:
                dia_arcsec = df.iloc[0].get('apparent_diameter_arcsec', dia_arcsec)

            r_pix = (dia_arcsec / 2.0) / SPATIAL_CONFIG['plate_scale']

            cx, cy = find_mercury_center(img_2d, r_pix)

            plot_name = spatial_dir / f"{fits_path.stem}_center_estimation.png"
            # 改良版プロット関数を呼び出し
            plot_center_estimation(img_2d, cx, cy, r_pix, plot_name, fits_path.name)

            print(f"  > Processed: {fits_path.name} -> Center({cx:.2f}, {cy:.2f})")

        except Exception as e:
            print(f"  > Error processing {fits_path.name}: {e}")


# ==============================================================================
# テスト用スタンドアロン実行ブロック
# ==============================================================================
if __name__ == '__main__':
    # ==========================================================================
    # ★ ここを書き換えてテストしたい日付を指定してください ★
    test_date = "20250819"
    # ==========================================================================

    project_base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025")
    csv_dir_name = "2026ver/csvs"
    output_dir_name = "test4_output"

    # パスの構築
    base_out = project_base_dir / output_dir_name / test_date
    csv_path = project_base_dir / csv_dir_name / f"mcparams{test_date}.csv"

    print(f"=== 試運転モード起動 (エッジ可視化版): 対象日 {test_date} ===")
    print(f"Output Dir : {base_out}")
    print(f"CSV Path   : {csv_path}")

    # モックアップの run_info を作成
    mock_run_info = {
        "output_dir": str(base_out),
        "csv_path": str(csv_path),
        "date": test_date
    }

    # 実行
    run(mock_run_info, {})