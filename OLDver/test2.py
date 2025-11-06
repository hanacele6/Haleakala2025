import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os

# --- 設定項目 ---
# 比較したい2つのFITSファイルのパスを指定してください
# ご自身の環境に合わせてパスを修正してください
base_dir = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\output\20250501"

# IDLとPython、それぞれのfibl3デバッグファイル名を指定
idl_filename = "debug_fibl3_python_fib45.fit"
python_filename = "debug_fibl3_python_fib45.fit"

# プロファイルを直接比較したい行（ピクセル）を指定
# これまでのデバッグから、iy2=600 に設定
row_to_compare = 600


# --- ここからプログラム本体 ---

def compare_fits_files(idl_path, python_path, row_index):
    """2つのFITSファイルを読み込み、比較・可視化する関数"""

    # --- 1. ファイルの読み込み ---
    try:
        with fits.open(idl_path) as hdul:
            idl_data = hdul[0].data.astype(np.float64)  # 念のため倍精度に変換
        with fits.open(python_path) as hdul:
            python_data = hdul[0].data.astype(np.float64)
    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません。パスを確認してください。")
        print(e)
        return

    # --- 2. データ形状の確認 ---
    if idl_data.shape != python_data.shape:
        print("エラー: 2つのFITSファイルの配列形状が異なります。")
        print(f"IDL shape: {idl_data.shape}, Python shape: {python_data.shape}")
        return

    print(f"配列形状: {idl_data.shape}")

    # --- 3. 差分を計算 ---
    difference = idl_data - python_data

    # --- 4. 差の統計値を計算して表示 ---
    mean_diff = np.mean(difference)
    std_diff = np.std(difference)
    min_diff = np.min(difference)
    max_diff = np.max(difference)

    print("\n--- 差の統計値 (IDL - Python) ---")
    print(f"平均値:   {mean_diff:+.8f}")
    print(f"標準偏差: {std_diff:.8f}")
    print(f"最小値:   {min_diff:+.8f}")
    print(f"最大値:   {max_diff:+.8f}")
    print("------------------------------------")

    # --- 5. 可視化 ---
    # 日本語が文字化けしないようにフォントを設定
    plt.rcParams['font.family'] = 'Meiryo'  # or 'MS Gothic'など

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("fibl3 FITSファイルの比較 (IDL vs Python)", fontsize=16)

    # プロット1: 差分画像
    ax1 = fig.add_subplot(2, 2, 1)
    im = ax1.imshow(difference, aspect='auto', cmap='bwr',
                    vmin=-np.abs(difference).max(), vmax=np.abs(difference).max())
    fig.colorbar(im, ax=ax1)
    ax1.set_title("差分画像 (IDL - Python)")
    ax1.set_xlabel("空間方向のピクセル (ix2)")
    ax1.set_ylabel("波長方向のピクセル (iy2)")

    # プロット2: 差のヒストグラム
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(difference.flatten(), bins=100)
    ax2.set_title("差の分布 (ヒストグラム)")
    ax2.set_xlabel("差の値 (IDL - Python)")
    ax2.set_ylabel("ピクセル数")

    # プロット3: 中央列に沿った差のプロット
    ax3 = fig.add_subplot(2, 2, 3)
    central_column_index = difference.shape[1] // 2
    ax3.plot(difference[:, central_column_index])
    ax3.axhline(0, color='grey', linestyle='--')
    ax3.set_title(f"中央列 (ix2={central_column_index}) に沿った差")
    ax3.set_xlabel("波長方向のピクセル (iy2)")
    ax3.set_ylabel("差の値")
    ax3.grid(True)

    # プロット4: 特定の行のプロファイルを重ねて表示
    ax4 = fig.add_subplot(2, 2, 4)
    x_axis = np.arange(idl_data.shape[1])
    ax4.plot(x_axis, idl_data[row_index, :], 'o-', label=f"IDL (iy2={row_index})", alpha=0.8)
    ax4.plot(x_axis, python_data[row_index, :], 'x-', label=f"Python (iy2={row_index})", alpha=0.8)
    ax4.set_title(f"プロファイルの直接比較 (iy2={row_index})")
    ax4.set_xlabel("空間方向のピクセル (ix2)")
    ax4.set_ylabel("値")
    ax4.legend()
    ax4.grid(True)

    # レイアウトを調整して表示
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# --- メインの実行部分 ---
if __name__ == "__main__":
    idl_file_path = os.path.join(base_dir, idl_filename)
    python_file_path = os.path.join(base_dir, python_filename)

    compare_fits_files(idl_file_path, python_file_path, row_to_compare)