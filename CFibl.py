import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from pathlib import Path

# --- ファイルパスの設定 ---
# あなたの環境に合わせてパスを修正してください
base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
fileF1 = base_dir / "output/test"

idl_file = fileF1 / 'fibl_IDL_test.fit'
python_file = fileF1 / 'fibl_Python_test.fit'

# --- FITSファイルの読み込み ---
# astropyはFITSのデータ型を保持してくれるので、そのまま読み込みます
print(f"IDLのファイルを読み込み中: {idl_file}")
data_idl = fits.getdata(idl_file)

print(f"Pythonのファイルを読み込み中: {python_file}")
data_python = fits.getdata(python_file)

# --- データ型の確認（デバッグ用） ---
print(f"IDLデータの型: {data_idl.dtype}")
print(f"Pythonデータの型: {data_python.dtype}")

# --- 差分を計算 ---
# 差を取る前に、両方のデータ型をfloat64に揃えておくと安全です
diff = data_python.astype(np.float64) - data_idl.astype(np.float64)

print(f"差分の最大値: {np.max(diff)}")
print(f"差分の最小値: {np.min(diff)}")
print(f"差分の平均値: {np.mean(diff)}")

# --- ヒストグラムの描画 ---
plt.figure(figsize=(10, 7))
plt.hist(diff.flatten(), bins=101, range=(-50.5, 50.5)) # 整数ごとの差が見やすいように調整
plt.yscale('log')
plt.title('fiblの差のヒストグラム (Y軸は対数表示)')
plt.xlabel('差 (Python - IDL)')
plt.ylabel('ピクセル数')
plt.grid(True, which='both', linestyle='--')

# 差が0の位置に赤い線を引く
plt.axvline(0, color='red', linestyle='--')

plt.show()