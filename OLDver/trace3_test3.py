import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path

# --- ここを編集してください -------------------------------------------------
IDL_FITS_PATH = Path(r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\output\test\fibl_IDL_test.fit")
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

    # --- 4. [詳細調査] 特定の点の情報を表示 ---
    # <--- ここから追加
    print("\n--- [詳細調査] 特定の点(0,0)の値を比較します ---")
    coords_to_check = (0, 0)  # 調査したい座標 (行, 列) を指定

    # 座標が配列の範囲内にあるか確認
    if all(np.array(coords_to_check) < np.array(difference.shape)):
        idl_val_check = idl_data[coords_to_check]
        py_val_check = python_data[coords_to_check]
        diff_check = difference[coords_to_check]
        print(f"座標 {coords_to_check} での値:")
        print(f"  IDLのfibl値:   {idl_val_check:.4f}")
        print(f"  Pythonのfibl値: {py_val_check:.4f}")
        print(f"  差 (Python - IDL): {diff_check:.4f}")
    else:
        print(f"座標 {coords_to_check} は範囲外です。")
    print("--------------------------------------------------\n")
    # <--- ここまで追加

    # --- 5. 差の統計情報 ---
    # (変更なし)

    # --- 6. 可視化 ---
    # (変更なし)

    # (統計情報と可視化のコードは省略)
    diff_no_nan = np.nan_to_num(difference)
    print("--- 差の統計情報 (Python - IDL) ---")
    print(f"平均値: {np.mean(diff_no_nan):.4f}")
    print(f"標準偏差: {np.std(diff_no_nan):.4f}")
    print(f"最小値: {np.min(diff_no_nan):.4f}")
    print(f"最大値: {np.max(diff_no_nan):.4f}")
    print("------------------------------------")
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