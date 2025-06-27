import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path

# --- ここを編集してください -------------------------------------------------
# 1. IDLが出力したFITSファイルのフルパスを指定
IDL_FITS_PATH = Path(r"C:\Users\hanac\University\Senior\Mercury\data\test\mc01_1_nhp.fits")
# 2. Pythonが出力したFITSファイルのフルパスを指定
PYTHON_FITS_PATH = Path(r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\output\test\mc01_1_nhp_py.fits")

# --- 追加: 特定のピクセルを確認するための座標 (1-based Y, X) ---
# Y=463, X=379 を指定
TARGET_PIXEL_Y = 463
TARGET_PIXEL_X = 379

# --------------------------------------------------------------------------

def compare_fits_results(idl_path, python_path, top_n_diff_pixels=10, target_y=None, target_x=None):
    """
    IDLとPythonで生成されたFITSファイルを比較し、差を可視化する関数。
    差が大きいピクセルの座標も表示し、指定されたピクセルも詳細に表示する。

    Args:
        idl_path (Path): IDLが出力したFITSファイルのパス。
        python_path (Path): Pythonが出力したFITSファイルのパス。
        top_n_diff_pixels (int): 差の絶対値が大きい上位N個のピクセルを表示する数。
        target_y (int, optional): 詳細を確認したいピクセルのY座標 (1-based)。Noneの場合は表示しない。
        target_x (int, optional): 詳細を確認したいピクセルのX座標 (1-based)。Noneの場合は表示しない。
    """
    # --- 1. ファイルの読み込み ---
    try:
        print(f"IDL FITSを読み込み中: {idl_path}")
        idl_data = fits.getdata(idl_path).astype(np.float64)
        print(f"Python FITSを読み込み中: {python_path}")
        python_data = fits.getdata(python_path).astype(np.float64)
    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません。パスを確認してください。\n{e}")
        return

    # --- 2. データ形状のチェック ---
    if idl_data.shape != python_data.shape:
        print("エラー: IDLとPythonのデータ形状が異なります。比較できません。")
        print(f"IDL shape: {idl_data.shape}, Python shape: {python_data.shape}")
        return

    print(f"\nデータ形状は一致しました: {idl_data.shape}")
    print("比較処理を開始します...")

    # --- 3. 差の計算と統計情報 ---
    difference = python_data - idl_data

    # NaNを0として扱う（nansumなどでNaNは0として扱われるため、統計値もそれに合わせる）
    diff_no_nan = np.nan_to_num(difference)

    print("\n--- 差の統計情報 (Python - IDL) ---")
    print(f"平均値: {np.mean(diff_no_nan):.4f}")
    print(f"標準偏差: {np.std(diff_no_nan):.4f}")
    print(f"最小値: {np.min(diff_no_nan):.4f}")
    print(f"最大値: {np.max(diff_no_nan):.4f}")
    print("------------------------------------")

    # --- 4. 指定ピクセルの詳細表示 ---
    if target_y is not None and target_x is not None:
        row_idx_0based = target_y - 1
        col_idx_0based = target_x - 1
        rows, cols = diff_no_nan.shape

        if 0 <= row_idx_0based < rows and 0 <= col_idx_0based < cols:
            print(f"\n--- 指定ピクセル (Y={target_y}, X={target_x}) の詳細情報 ---")
            diff_value = diff_no_nan[row_idx_0based, col_idx_0based]
            idl_value = idl_data[row_idx_0based, col_idx_0based]
            python_value = python_data[row_idx_0based, col_idx_0based]
            print(f"  差={diff_value:.4f}")
            print(f"  Pythonの値={python_value:.4f}")
            print(f"  IDLの値={idl_value:.4f}")
            print("-----------------------------------------------------")
        else:
            print(f"\n警告: 指定されたピクセル (Y={target_y}, X={target_x}) は画像の範囲外です。")

    # --- 5. 差が大きいピクセルの表示 --- (既存の機能)
    print(f"\n--- 差の絶対値が大きい上位 {top_n_diff_pixels} 個のピクセル ---")
    abs_diff_flat = np.abs(diff_no_nan).flatten()
    sorted_indices = np.argsort(abs_diff_flat)[::-1]

    rows, cols = diff_no_nan.shape
    for i in range(min(top_n_diff_pixels, len(sorted_indices))):
        flat_idx = sorted_indices[i]
        row_idx = flat_idx // cols
        col_idx = flat_idx % cols
        diff_value = diff_no_nan[row_idx, col_idx]
        idl_value = idl_data[row_idx, col_idx]
        python_value = python_data[row_idx, col_idx]
        print(f"  (Y={row_idx+1}, X={col_idx+1}): 差={diff_value:.4f} (Python={python_value:.4f}, IDL={idl_value:.4f})")
    print("-----------------------------------------------------")


    # --- 6. 可視化 --- (以下、既存のプロットコード)
    # プロットのスタイルを設定
    plt.style.use('default')
    plt.rcParams['font.family'] = 'Meiryo'  # 日本語フォント設定（環境に合わせて変更）
    plt.rcParams['figure.dpi'] = 100

    # 6a. 差の2D画像
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    vmax = np.nanpercentile(np.abs(difference), 99.8)
    vmin = -vmax
    im = ax1.imshow(difference, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    fig1.colorbar(im, ax=ax1, label='差 (Python - IDL)')
    ax1.set_title('データ全体の差の分布 (Python - IDL)')
    ax1.set_xlabel('波長方向のピクセル')
    ax1.set_ylabel('ファイバー番号')

    # 6b. 差のヒストグラム
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.hist(difference.flatten(), bins=100, range=(-200, 200), log=True)
    ax2.axvline(0, color='red', linestyle='--', linewidth=1)
    ax2.set_title('差のヒストグラム (Y軸は対数表示)')
    ax2.set_xlabel('差 (Python - IDL)')
    ax2.set_ylabel('ピクセル数')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 6c. 代表的なファイバーのスペクトル比較
    num_fibers = idl_data.shape[0]
    fiber_to_plot = num_fibers // 2  # 真ん中のファイバーを選択

    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                      gridspec_kw={'height_ratios': [3, 1]})

    ax3a.plot(idl_data[fiber_to_plot, :], label=f'IDL (ファイバー #{fiber_to_plot})', color='black', alpha=0.7)
    ax3a.plot(python_data[fiber_to_plot, :], label=f'Python (ファイバー #{fiber_to_plot})', color='dodgerblue',
              linestyle='--', alpha=0.8)
    ax3a.set_title(f'ファイバー #{fiber_to_plot} のスペクトル比較')
    ax3a.set_ylabel('強度')
    ax3a.legend()
    ax3a.grid(True, linestyle='--', linewidth=0.5)

    ax3b.plot(difference[fiber_to_plot, :], label='差 (Python - IDL)', color='red')
    ax3b.axhline(0, color='black', linestyle='--', linewidth=1)
    ax3b.set_xlabel('波長方向のピクセル')
    ax3b.set_ylabel('差')
    ax3b.legend()
    ax3b.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # TARGET_PIXEL_Y と TARGET_PIXEL_X を引数として渡す
    compare_fits_results(IDL_FITS_PATH, PYTHON_FITS_PATH,
                         target_y=TARGET_PIXEL_Y, target_x=TARGET_PIXEL_X,
                         top_n_diff_pixels=10)