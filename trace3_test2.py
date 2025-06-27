import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path

# --- ここを編集してください -------------------------------------------------
# 1. IDLが出力したFITSファイルのフルパスを指定
IDL_FITS_PATH = Path(r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\output\test\fibl_IDL_test.fit")
# 2. Pythonが出力したFITSファイルのフルパスを指定
PYTHON_FITS_PATH = Path(r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\output\test\fibl_Python_test.fit")
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
        # <--- 回転処理を元に戻しました
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

    # --- 3. 差の計算 ---
    difference = python_data - idl_data

    # --- 4. [詳細調査] 外れ値の情報を特定 ---
    print("\n--- [詳細調査] 差が最大・最小になる点の情報を特定します ---")

    # 差が最大になる場所を特定
    max_idx_flat = np.argmax(difference)
    max_coords = np.unravel_index(max_idx_flat, difference.shape)
    max_diff_val = difference[max_coords]
    idl_val_at_max = idl_data[max_coords]
    py_val_at_max = python_data[max_coords]

    # 差が最小になる場所を特定
    min_idx_flat = np.argmin(difference)
    min_coords = np.unravel_index(min_idx_flat, difference.shape)
    min_diff_val = difference[min_coords]
    idl_val_at_min = idl_data[min_coords]
    py_val_at_min = python_data[min_coords]

    # 結果を出力
    print("\n--- 差が最大になる点の詳細 ---")
    print(f"座標 (行, 列): {max_coords}  <- この座標は回転後のものです")
    print(f"IDLのfibl値:   {idl_val_at_max:.4f}")
    print(f"Pythonのfibl値: {py_val_at_max:.4f}")
    print(f"差 (Python - IDL): {max_diff_val:.4f}")

    print("\n--- 差が最小になる点の詳細 ---")
    print(f"座標 (行, 列): {min_coords}  <- この座標は回転後のものです")
    print(f"IDLのfibl値:   {idl_val_at_min:.4f}")
    print(f"Pythonのfibl値: {py_val_at_min:.4f}")
    print(f"差 (Python - IDL): {min_diff_val:.4f}")
    print("-----------------------------------------------------------\n")

    # --- 5. 差の統計情報 ---
    diff_no_nan = np.nan_to_num(difference)
    print("--- 差の統計情報 (Python - IDL) ---")
    print(f"平均値: {np.mean(diff_no_nan):.4f}")
    print(f"標準偏差: {np.std(diff_no_nan):.4f}")
    print(f"最小値: {np.min(diff_no_nan):.4f}")
    print(f"最大値: {np.max(diff_no_nan):.4f}")
    print("------------------------------------")


    # --- 6. 可視化 ---
    plt.style.use('default')
    plt.rcParams['font.family'] = 'Meiryo'
    plt.rcParams['figure.dpi'] = 100

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.hist(difference.flatten(), bins=100, range=(-1000, 1000), log=True)
    ax2.axvline(0, color='red', linestyle='--', linewidth=1)
    ax2.set_title('差のヒストグラム (Y軸は対数表示)')
    ax2.set_xlabel('差 (Python - IDL)')
    ax2.set_ylabel('ピクセル数')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    compare_fits_results(IDL_FITS_PATH, PYTHON_FITS_PATH)