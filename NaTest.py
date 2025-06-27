import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from pathlib import Path

# --- 関数定義 ---
def gaussian_with_linear(x, amplitude, mean, stddev, const, slope):
    """A Gaussian function with a linear background component."""
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2) + const + slope * x

# --- パスと定数の設定 ---
base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
fileF1 = base_dir / "output/test"
fileF2 = base_dir / "output/test"

is_num, ie_num = 10001, 10012

# --- 1. Naランプのコンポジットスペクトルを作成 ---
print("Step 1: Creating comparison (Na Lamp) spectrum...")
# ★★★ 修正点1: IDLと同じ、元の入力ファイルを指定します ★★★
try:
    with fits.open(fileF1 / f"{is_num}_f_python.fit") as hdul:
        template_shape = hdul[0].data.shape
except FileNotFoundError:
    print(f"エラー: 基準ファイル {is_num}_f.fit が見つかりません。")
    exit()

ixm, iym = template_shape[1], template_shape[0]
comp_spec = np.zeros((iym, ixm), dtype=np.float64)

for i in range(10009, 10013):
    lamp_path = fileF1 / f"{i}_f_python.fit"
    print(f"  Adding lamp frame: {lamp_path.name}")
    try:
        lamp_data = fits.getdata(lamp_path, ext=0).astype(np.float64)
        comp_spec += lamp_data
    except FileNotFoundError:
        print(f"  警告: Naランプファイル {lamp_path.name} が見つかりませんでした。スキップします。")

# --- 2. 2段階ガウスフィット ---
print("\nStep 2: Finding line centers in comparison spectrum...")
d2s, d2e = 549 - 1, 627 - 1
d1s, d1e = 797 - 1, 875 - 1
dw = 15
c21 = np.zeros(iym, dtype=np.float64)
c11 = np.zeros(iym, dtype=np.float64)

# --- プロット用のデータをループ内で保存する準備 ---
iy_to_diagnose = [0, 85] # 診断したい行のリスト
diag_data = {} # 診断データを格納する辞書

for iy in range(iym):
    # --- 第1段階：粗いフィット ---
    # ★★★ 修正点2: 輝線なので振幅の初期値を正にします ★★★
    # D2線
    x_coarse_d2 = np.arange(d2e - d2s + 1)
    y_coarse_d2 = comp_spec[iy, d2s:d2e + 1]
    #p0_coarse_d2 = [np.max(y_coarse_d2 - np.median(y_coarse_d2)), np.argmax(y_coarse_d2), 5.0, np.median(y_coarse_d2), 0]
    p0_coarse_d2 = [np.max(y_coarse_d2), np.argmax(y_coarse_d2), 5.0, np.median(y_coarse_d2), 0]
    try:
        popt_coarse_d2, _ = curve_fit(gaussian_with_linear, x_coarse_d2, y_coarse_d2, p0=p0_coarse_d2, method='lm', maxfev=5000)
        c2 = popt_coarse_d2[1] + d2s
    except RuntimeError:
        c2 = d2s + (d2e - d2s) / 2
    # D1線
    x_coarse_d1 = np.arange(d1e - d1s + 1)
    y_coarse_d1 = comp_spec[iy, d1s:d1e + 1]
    #p0_coarse_d1 = [np.max(y_coarse_d1 - np.median(y_coarse_d1)), np.argmax(y_coarse_d1), 5.0, np.median(y_coarse_d1), 0]
    p0_coarse_d1 = [np.max(y_coarse_d1), np.argmax(y_coarse_d1), 5.0, np.median(y_coarse_d1), 0]
    try:
        popt_coarse_d1, _ = curve_fit(gaussian_with_linear, x_coarse_d1, y_coarse_d1, p0=p0_coarse_d1, method='lm', maxfev=5000)
        c1 = popt_coarse_d1[1] + d1s
    except RuntimeError:
        c1 = d1s + (d1e - d1s) / 2

    # --- 第2段階：精密なフィット ---
    ic2, ic1 = int(np.round(c2)), int(np.round(c1))

    # D2線（精密）
    s, e = max(0, ic2 - dw), min(ixm - 1, ic2 + dw)
    x_fine, y_fine = np.arange(e - s + 1), comp_spec[iy, s:e + 1]
    if len(x_fine) > 5:
        p0_fine = [np.max(y_fine - np.median(y_fine)), dw, 5.0, np.median(y_fine), 0]
        try:
            popt, _ = curve_fit(gaussian_with_linear, x_fine, y_fine, p0=p0_fine, method='lm', maxfev=5000)
            c21[iy] = popt[1] + s
            if iy in iy_to_diagnose:
                diag_data.setdefault(iy, {})['d2'] = {'s':s, 'popt':popt}
        except RuntimeError: c21[iy] = c2
    else: c21[iy] = c2

    # D1線（精密）- バグ修正済み
    s, e = max(0, ic1 - dw), min(ixm - 1, ic1 + dw)
    x_fine, y_fine = np.arange(e - s + 1), comp_spec[iy, s:e + 1]
    if len(x_fine) > 5:
        p0_fine = [np.max(y_fine - np.median(y_fine)), dw, 5.0, np.median(y_fine), 0]
        try:
            popt, _ = curve_fit(gaussian_with_linear, x_fine, y_fine, p0=p0_fine, method='lm', maxfev=5000)
            c11[iy] = popt[1] + s
            if iy in iy_to_diagnose:
                diag_data.setdefault(iy, {})['d1'] = {'s':s, 'popt':popt}
        except RuntimeError: c11[iy] = c1
    else: c11[iy] = c1

# --- 3, 4, 5 (計算、補正、ファイル保存) ---
# この部分は変更がないため、元のコードをそのままお使いください。
# (ここでは簡潔にするため省略します)

# --- 3. ドップラーシフト量を計算 ---
print("\nStep 3: Calculating Doppler shift...")
cD2 = np.min(c21)
cD1 = np.min(c11)
dc = ((c21 - cD2) + (c11 - cD1)) / 2.0
print("Doppler shift calculation completed.")
# ... (以下、ファイル保存まで続く)


print("\nStep 6: Creating diagnostic plots with absolute coordinates...")
iy_to_diagnose = [0,85]  # 診断したい行のリスト

for iy_diag in iy_to_diagnose:
    # (この部分のコードは、以前のままでOKです)
    if iy_diag not in diag_data:
        print(f"No diagnostic data saved for iy={iy_diag}. Skipping plot.")
        continue

    print(f"\n--- Plotting for iy = {iy_diag} ---")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), tight_layout=True)
    fig.suptitle(f"Diagnostic Plot for Row iy = {iy_diag}", fontsize=16)

    # --- D2輝線のプロット ---
    ax1.set_title(f'D2 Line Fit')
    try:
        s2 = diag_data[iy_diag]['d2']['s']
        popt2 = diag_data[iy_diag]['d2']['popt']
        x_relative = np.arange(2 * dw + 1)
        x_absolute = x_relative + s2
        y_data2 = comp_spec[iy_diag, s2: s2 + 2 * dw + 1]
        python_fit_curve2 = gaussian_with_linear(x_relative, *popt2)
        python_center_abs2 = popt2[1] + s2

        ax1.plot(x_absolute, y_data2, 'o', color='gray', label='Raw Data')
        ax1.plot(x_absolute, python_fit_curve2, 'b-', label='Python Fit')
        ax1.axvline(python_center_abs2, color='blue', linestyle='--', label=f'Py Center: {python_center_abs2:.4f}')

        # ★★★ ここからがIDLファイル読み込みの修正箇所 ★★★
        idl_file_d2 = fileF2 / f'idl_diag_d2_iy{iy_diag}.txt'
        with open(idl_file_d2, 'r') as f:
            # 1行目を「真の中心（相対値）」として読み込む
            idl_center_rel = float(f.readline())
            # 残りをデータとして読み込む
            idl_data = np.loadtxt(f)

        idl_x_abs = idl_data[:, 0] + s2
        idl_y_fit = idl_data[:, 2]
        idl_center_abs = idl_center_rel + s2  # 絶対座標に変換

        ax1.plot(idl_x_abs, idl_y_fit, 'r--', label='IDL Fit')
        ax1.axvline(idl_center_abs, color='red', linestyle=':', label=f'IDL Center: {idl_center_abs:.4f}')

    except Exception as e:
        ax1.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax1.transAxes)
    ax1.legend()
    ax1.set_xlabel("Absolute Pixel Coordinate")
    ax1.set_ylabel("Intensity")

    # --- D1輝線のプロット (D2と同様に修正) ---
    ax2.set_title(f'D1 Line Fit')
    try:
        s1 = diag_data[iy_diag]['d1']['s']
        popt1 = diag_data[iy_diag]['d1']['popt']
        x_relative = np.arange(2 * dw + 1)
        x_absolute = x_relative + s1
        y_data1 = comp_spec[iy_diag, s1: s1 + 2 * dw + 1]
        python_fit_curve1 = gaussian_with_linear(x_relative, *popt1)
        python_center_abs1 = popt1[1] + s1

        ax2.plot(x_absolute, y_data1, 'o', color='gray', label='Raw Data')
        ax2.plot(x_absolute, python_fit_curve1, 'b-', label='Python Fit')
        ax2.axvline(python_center_abs1, color='blue', linestyle='--', label=f'Py Center: {python_center_abs1:.4f}')

        # ★★★ ここからがIDLファイル読み込みの修正箇所 ★★★
        idl_file_d1 = fileF2 / f'idl_diag_d1_iy{iy_diag}.txt'
        with open(idl_file_d1, 'r') as f:
            # 1行目を「真の中心（相対値）」として読み込む
            idl_center_rel = float(f.readline())
            # 残りをデータとして読み込む
            idl_data = np.loadtxt(f)

        idl_x_abs = idl_data[:, 0] + s1
        idl_y_fit = idl_data[:, 2]
        idl_center_abs = idl_center_rel + s1  # 絶対座標に変換

        ax2.plot(idl_x_abs, idl_y_fit, 'r--', label='IDL Fit')
        ax2.axvline(idl_center_abs, color='red', linestyle=':', label=f'IDL Center: {idl_center_abs:.4f}')

    except Exception as e:
        ax2.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax2.transAxes)
    ax2.legend()
    ax2.set_xlabel("Absolute Pixel Coordinate")

plt.show()