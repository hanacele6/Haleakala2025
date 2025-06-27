import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from pathlib import Path
import os
import time

# --- 設定値 (IDLコードの変数に対応) ---
date = 'test'
fileF = Path("C:/Users/hanac/University/Senior/Mercury/")
base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
fileF1 = base_dir / f"output/{date}"
fileFd = fileF / f"data/{date}"

# --- FITSファイルの準備 ---
with fits.open(os.path.join(fileF1, f'flat1.fit')) as hdul:
    flat = hdul[0].data
with fits.open(os.path.join(fileF1, f'fppoly1.fit')) as hdul:
    fibp = hdul[0].data

# --- ファイルリストの生成 ---
base_filename_prefix = 'mc01-'
start_file_index = 1
end_file_index = 1
num_digits = 3
file_suffix = '_nhp.fits'

file = []
for i in range(start_file_index, end_file_index + 1):
    generated_filename = f'{base_filename_prefix}{i:0{num_digits}d}{file_suffix}'
    file.append(generated_filename)
ifilem = len(file)
print(f"Generated file list: {file}")

# --- 定数の定義 ---
iym = 1327
ifibm = 93  # fiber number
iys2 = 1 - 1
iye2 = 1327 - 1
iym2 = iye2 - iys2 + 1
hwid = 5


# --- Gauss関数定義 (curve_fit用) ---
def gaussian(x, amplitude, mean, stddev, offset):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2) + offset


# --- メイン処理 ---
poserr_path = os.path.join(fileF1, 'poserr_python.txt')
with open(poserr_path, 'w') as lunw2:
    total_start_time = time.time()

    for ifile in range(ifilem):
        fileid = file[ifile]
        file_start_time = time.time()
        print(f"\nファイル {ifile + 1}/{ifilem} を処理中: {fileid}")

        # --- FITSデータの読み込み ---
        data_file_path = os.path.join(fileFd, fileid)
        if not Path(data_file_path).exists():  # データファイルの存在確認
            print(f"警告: データファイル {data_file_path} が見つかりませんでした。スキップします。")
            continue  # ファイルが見つからない場合は次のループへ

        with fits.open(data_file_path) as hdul:
            b = hdul[0].data

        # --- 配列の初期化 ---
        fibl = np.zeros((hwid * 2 + 1, iym2), dtype=np.float64)
        fibl3 = np.zeros_like(fibl)
        fibl4 = np.zeros(iym2, dtype=np.float64)
        fiblall2 = np.zeros((ifibm, iym2), dtype=np.float64)

        for ifib in range(ifibm):
            fib_start_time = time.time()
            print(f"  ファイバー {ifib + 1}/{ifibm} を処理中 (ファイル {ifile + 1}/{ifilem})")
            fibpix = fibp[:, ifib]

            # --- 第1の内挿 ---
            for iy2 in range(iym2):
                x_data = np.arange(b.shape[1])
                y_data = b[iy2 + iys2, :]
                interp_func = interp1d(x_data, y_data, kind='linear', bounds_error=False,
                                       fill_value=(y_data[0], y_data[-1]))
                for ix2 in range(-hwid, hwid + 1):
                    fibl[ix2 + hwid, iy2] = interp_func(fibpix[iy2] + ix2)



            # --- ガウスフィット ---
            fibl2 = np.sum(fibl, axis=1)  # より効率的な合計の仕方
            x_fit = np.arange(hwid * 2 + 1)
            initial_guess = [np.max(fibl2) - np.min(fibl2), float(hwid), 1.0, np.min(fibl2)]

            # ★★★ ここでarrの値を設定 ★★★
            use_idl_arr = False# このフラグを切り替えて実験してください

            if ifib == 45 and use_idl_arr:
                print("  DEBUG (Fiber 45): IDLのarr値をハードコーディングして使用します。")
                arr = np.array([93812.25423050244, 5.084256050127824, 1.6240273773843095, -24920.2])
                #arr = np.array([67273.9, 5.08139, 1.62656, -24920.2]) orig_IDL
                #arr = np.array([93812.25423050244, 5.084256050127824, 1.6240273773843095, 2553771.7165144286]) orig_Python
            else:
                try:
                    params, covariance = curve_fit(gaussian, x_fit, fibl2, p0=initial_guess, maxfev=10000)
                    arr = params
                except RuntimeError as e:
                    print(f"  警告: ガウスフィットが失敗 (ifib={ifib}): {e}. 初期値を使用します。")
                    arr = initial_guess

            #if ifile == 0:
                # 'a'モード（追記）でファイルを開く
            #        with open(os.path.join(fileF1, 'gaussfit_params_Python.txt'), 'a') as f_gauss:
                 #ifib, amplitude, mean, stddev, offset
            #            f_gauss.write(f"{ifib},{arr[0]},{arr[1]},{arr[2]},{arr[3]}\n")

            # --- フィット結果の補正 ---
            if abs(arr[1] - hwid) > 1.5:
                lunw2.write(f"{ifile} {ifib} {arr[1]:.6f}\n")
                arr[1] = 5.0

            # --- 第2の内挿 & 最終計算 ---
            for iy2 in range(iym2):
                y_data_fibl3 = fibl[:, iy2]
                x_interp_data = np.arange(fibl.shape[0])
                interp_func_fibl3 = interp1d(x_interp_data, y_data_fibl3, kind='linear', bounds_error=False,
                                             fill_value=(y_data_fibl3[0], y_data_fibl3[-1]))

                for ix2 in range(-hwid, hwid + 1):
                    fibl3[ix2 + hwid, iy2] = interp_func_fibl3(arr[1] + ix2)

                fibl4[iy2] = np.sum(fibl3[hwid - 1:hwid + 2, iy2]) - (fibl3[0, iy2] + fibl3[hwid * 2, iy2]) / 2.0 * 3.0

            # ★★★ デバッグ用の保存は、iy2ループの外側に移動 ★★★
            if ifib == 45 and ifile == 0:
                if use_idl_arr:
                    save_filename = 'debug_fibl3_python_fib45_IDLarr2.fit'
                else:
                    save_filename = 'debug_fibl3_python_fib45_PYarr.fit'

                save_path = os.path.join(fileF1, save_filename)
                print(f'  DEBUG: Saving completed fibl3 for fiber 45 to {save_path}')
                hdu = fits.PrimaryHDU(fibl3.T)
                hdu.writeto(save_path, overwrite=True)

            # --- 結果の格納 ---
            fiblall2[ifib, :] = fibl4

            fib_end_time = time.time()
            print(f"  ファイバー {ifib + 1}/{ifibm} 処理完了。所要時間: {fib_end_time - fib_start_time:.2f}秒")

        # --- ファイルへの書き出し ---
        output_file_name = f'{10000 + ifile + 1:02d}_tr_python_test.fit'
        output_path = os.path.join(fileF1, output_file_name)
        fiblall2_flipped = fiblall2[:, ::-1]
        hdu = fits.PrimaryHDU(fiblall2_flipped)
        hdu.writeto(output_path, overwrite=True)

        file_end_time = time.time()
        print(f"ファイル {ifile + 1}/{ifilem} 処理完了。所要時間: {file_end_time - file_start_time:.2f}秒")

    total_end_time = time.time()
    print(f"\n全処理完了。全体所要時間: {total_end_time - total_start_time:.2f}秒")