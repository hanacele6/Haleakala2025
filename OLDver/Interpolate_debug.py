import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# --- 関数と定数の定義 ---
def gaussian(x, amplitude, mean, stddev, offset):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2) + offset

hwid = 5

# --- メイン処理 ---
print("--- Python Test: Second Step ---")

# 1. IDLが生成した共通のfiblファイルを読み込む
# 注意: IDLのFITSは(1327, 11)、Pythonの処理系は(11, 1327)なので、転置(.T)する
try:
    with fits.open('debug_fibl_idl_fib45.fit') as hdul:
        fibl = hdul[0].data.T
except FileNotFoundError:
    print("エラー: 'debug_fibl_idl_fib45.fit' が見つかりません。")
    exit()

# 2. fiblからfibl2を計算
fibl2 = np.sum(fibl, axis=1)

# 3. ガウスフィットを実行してarrを計算
x_fit = np.arange(hwid * 2 + 1)
initial_guess = [np.max(fibl2) - np.min(fibl2), float(hwid), 1.0, np.min(fibl2)]
try:
    params, _ = curve_fit(gaussian, x_fit, fibl2, p0=initial_guess)
    arr_py = params
except Exception as e:
    print(f"Pythonフィットエラー: {e}")
    arr_py = np.zeros(4)

print(f"Calculated Python arr: {np.array2string(arr_py, precision=8)}")

# 4. 計算したarrを使って2回目の補間を実行し、fibl4[600]を計算
iy2 = 600
y_data_fibl3 = fibl[:, iy2]
x_interp_data = np.arange(fibl.shape[0])
interp_func_fibl3 = interp1d(x_interp_data, y_data_fibl3, kind='linear', bounds_error=False, fill_value=(y_data_fibl3[0], y_data_fibl3[-1]))

fibl3_slice = np.zeros(hwid * 2 + 1)
for ix2 in range(-hwid, hwid + 1):
    fibl3_slice[ix2 + hwid] = interp_func_fibl3(arr_py[1] + ix2)

fibl4_600 = np.sum(fibl3_slice[hwid - 1:hwid + 2]) - (fibl3_slice[0] + fibl3_slice[hwid * 2]) / 2.0 * 5.0 # 5点合計版

print(f"Final Python fibl4[600]: {fibl4_600:.8f}")