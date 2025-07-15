import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path

# --- ここを編集してください -------------------------------------------------
# 1. IDLが出力したFITSファイルのフルパスを指定
#IDL_FITS_PATH = Path(r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\output\test\Hapketest_IDL.fits")
#IDL_FITS_PATH = Path(r"C:\Users\hanac\University\Senior\Mercury\data\test\mc01_1_nhp.fits")
#IDL_FITS_PATH = Path(r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\output\test\debug_fibl3_IDL.fit")
IDL_FITS_PATH = Path(r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\output\20241107\10001_tr_IDL.fit")
#IDL_FITS_PATH = Path(r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\sky_IDL.fit")
# 2. Pythonが出力したFITSファイルのフルパスを指定
#PYTHON_FITS_PATH = Path(r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\output\test\Hapketest_python.fits")
#PYTHON_FITS_PATH = Path(r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\output\test\mc01_1_nhp_py.fits")
#PYTHON_FITS_PATH = Path(r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\output\test\debug_fibl3_python.fit")
PYTHON_FITS_PATH = Path(r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\output\20241107\10001_tr_python.fit")
#PYTHON_FITS_PATH = Path(r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\sky_python_test.fit")
# --------------------------------------------------------------------------

def compare_fits_results(idl_path, python_path):
    """
    IDLとPythonで生成されたFITSファイルを比較し、差を可視化する関数
    """
    # --- 1. ファイルの読み込み ---
    try:
        print(f"IDL FITSを読み込み中: {idl_path}")
        idl_data = fits.getdata(idl_path).astype(np.float64)
        print(f"Python FITSを読み込み中: {python_path}")
        python_data = fits.getdata(python_path).astype(np.float64)
        #python_data = np.rot90(python_data_orig, k=-1)
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

    # --- 4. 可視化 ---

    # プロットのスタイルを設定
    plt.style.use('default')
    plt.rcParams['font.family'] = 'Meiryo'  # 日本語フォント設定（環境に合わせて変更）
    plt.rcParams['figure.dpi'] = 100

    # 4a. 差の2D画像
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    # 極端な値にカラーマップが引っ張られないように、99パーセンタイルで色の範囲を決定
    vmax = np.nanpercentile(np.abs(difference), 99.8)
    #vmax = np.nanpercentile(np.abs(difference), 100)
    vmin = -vmax
    im = ax1.imshow(difference, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    fig1.colorbar(im, ax=ax1, label='差 (Python - IDL)')
    ax1.set_title('データ全体の差の分布 (Python - IDL)')
    #ax1.set_xlabel('波長方向のピクセル')
    #ax1.set_ylabel('ファイバー番号')
    ax1.set_xlabel('画素')
    ax1.set_ylabel('画素')

    # 4b. 差のヒストグラム
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.hist(difference.flatten(), bins=100, range=(-1000, 1000), log=True)
    ax2.axvline(0, color='red', linestyle='--', linewidth=1)
    ax2.set_title('差のヒストグラム (Y軸は対数表示)')
    ax2.set_xlabel('差 (Python - IDL)')
    ax2.set_ylabel('ピクセル数')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 4c. 代表的なファイバーのスペクトル比較
    num_fibers = idl_data.shape[0]
    #fiber_to_plot = num_fibers // 2  # 真ん中のファイバーを選択
    fiber_to_plot = 50
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                      gridspec_kw={'height_ratios': [3, 1]})

    # 上段：IDLとPythonのスペクトルを重ねてプロット
    ax3a.plot(idl_data[fiber_to_plot, :], label=f'IDL (ファイバー #{fiber_to_plot})', color='black', alpha=0.7)
    ax3a.plot(python_data[fiber_to_plot, :], label=f'Python (ファイバー #{fiber_to_plot})', color='dodgerblue',
              linestyle='--', alpha=0.8)
    ax3a.set_title(f'ファイバー #{fiber_to_plot} のスペクトル比較')
    ax3a.set_ylabel('強度')
    ax3a.legend()
    ax3a.grid(True, linestyle='--', linewidth=0.5)

    # 下段：差のプロット
    ax3b.plot(difference[fiber_to_plot, :], label='差 (Python - IDL)', color='red')
    ax3b.axhline(0, color='black', linestyle='--', linewidth=1)
    ax3b.set_xlabel('波長方向のピクセル')
    ax3b.set_ylabel('差')
    ax3b.legend()
    ax3b.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    compare_fits_results(IDL_FITS_PATH, PYTHON_FITS_PATH)