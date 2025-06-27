import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from pathlib import Path
import os


# --- 関数定義 ---
# nterms=5 に相当する、ガウス関数 + 線形バックグラウンドのモデル
def gaussian_with_linear(x, amplitude, mean, stddev, const, slope):
    """A Gaussian function with a linear background component."""
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2) + const + slope * x


# --- パスと定数の設定 ---
# ご自身の環境に合わせて修正してください
base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
fileF1 = base_dir / "output/test"  # 補正前データがあるディレクトリ
fileF2 = base_dir / "output/test"  # 補正後データの保存先

# 処理する科学データのファイル番号
is_num = 10001
ie_num = 10012

# --- 1. 比較用コンポジットスペクトルを作成 ---
print("Step 1: Creating comparison (sky) spectrum...")
# 基準ファイルから配列の形状を取得
try:
    with fits.open(fileF1 / f"{is_num}_f_IDL.fit") as hdul:
        template_shape = hdul[0].data.shape
except FileNotFoundError:
    print(f"エラー: 基準ファイル {is_num}_f.fit が見つかりません。")
    exit()

ixm, iym = template_shape[1], template_shape[0]  # NumPyの形状は (行, 列)
comp_spec = np.zeros((iym, ixm), dtype=np.float64)

# 複数のNaを読み込み、足し合わせる
for i in range(10009, 10013):  # 10009から10013まで
    sky_path = fileF1 / f"{i}_f_IDL.fit"
    print(f"  Adding sky frame: {sky_path.name}")
    try:
        sky_data = fits.getdata(sky_path, ext=0).astype(np.float64)
        comp_spec += sky_data
    except FileNotFoundError:
        print(f"  警告: スカイファイル {sky_path.name} が見つかりませんでした。スキップします。")

# --- 2. 2段階のガウスフィットで、各行の輝線の中心位置を精密に決定 ---
print("\nStep 2: Finding line centers in comparison spectrum for each row...")

# Na D線の存在するおおよそのピクセル範囲
d2s, d2e = 549 - 1, 627 - 1
d1s, d1e = 797 - 1, 875 - 1
dw = 15

# 各行の輝線中心位置を格納する配列
c21 = np.zeros(iym, dtype=np.float64)
c11 = np.zeros(iym, dtype=np.float64)

for iy in range(iym):
    # ★★★【修正点 1】論理的に正しい初期値(p0)を設定 ★★★
    # 吸収線(谷)を探すための正しい初期値:
    # amplitude: 谷の深さ (負の値)
    # mean: 谷底の位置
    def get_p0_for_absorption(y):
        if y.size == 0: return None # データが空の場合はNoneを返す
        amp = np.min(y) - np.median(y)
        mean = np.argmin(y)
        const = np.median(y)
        return [amp, mean, 5.0, const, 0]

    # --- 第1段階：粗いフィット ---
    # D2線
    y_coarse_d2 = comp_spec[iy, d2s:d2e + 1]
    p0_coarse_d2 = get_p0_for_absorption(y_coarse_d2)
    try:
        popt_coarse_d2, _ = curve_fit(gaussian_with_linear, np.arange(y_coarse_d2.size), y_coarse_d2, p0=p0_coarse_d2)
        c2 = popt_coarse_d2[1] + d2s
    except (RuntimeError, ValueError):
        c2 = d2s + (d2e - d2s) / 2

    # D1線
    y_coarse_d1 = comp_spec[iy, d1s:d1e + 1]
    p0_coarse_d1 = get_p0_for_absorption(y_coarse_d1)
    try:
        popt_coarse_d1, _ = curve_fit(gaussian_with_linear, np.arange(y_coarse_d1.size), y_coarse_d1, p0=p0_coarse_d1)
        c1 = popt_coarse_d1[1] + d1s
    except (RuntimeError, ValueError):
        c1 = d1s + (d1e - d1s) / 2


    # --- 第2段階：精密なフィット ---
    ic2 = int(np.round(c2))
    ic1 = int(np.round(c1))

    # D2線
    if (ic2 - dw >= 0) and (ic2 + dw < ixm):
        fine_s2, fine_e2 = ic2 - dw, ic2 + dw
        x_fine_d2 = np.arange(fine_e2 - fine_s2 + 1)
        y_fine_d2 = comp_spec[iy, fine_s2:fine_e2 + 1]
        p0_fine_d2 = [np.median(y_fine_d2) - np.min(y_fine_d2), np.argmin(y_fine_d2), 5.0, np.median(y_fine_d2), 0]
        try:
            popt_fine_d2, _ = curve_fit(gaussian_with_linear, x_fine_d2, y_fine_d2, p0=p0_fine_d2)
            c21[iy] = popt_fine_d2[1] + fine_s2
        except RuntimeError:
            c21[iy] = c2
    else:
        c21[iy] = c2

    # D1線
    if (ic2 - dw >= 0) and (ic2 + dw < ixm):
        fine_s1, fine_e1 = ic2 - dw, ic2 + dw
        x_fine_d1 = np.arange(fine_e1 - fine_s1 + 1)
        y_fine_d1 = comp_spec[iy, fine_s1:fine_e1 + 1]
        p0_fine_d1 = [np.median(y_fine_d1) - np.min(y_fine_d1), np.argmin(y_fine_d1), 5.0, np.median(y_fine_d1), 0]
        try:
            popt_fine_d1, _ = curve_fit(gaussian_with_linear, x_fine_d1, y_fine_d1, p0=p0_fine_d1)
            c11[iy] = popt_fine_d1[1] + fine_s1
        except RuntimeError:
            c11[iy] = c1
    else:
        c11[iy] = c1

# --- 3. ドップラーシフト量を計算 ---
print("\nStep 3: Calculating Doppler shift...")
cD2 = np.min(c21)
cD1 = np.min(c11)
dc = ((c21 - cD2) + (c11 - cD1)) / 2.0
print("Doppler shift calculation completed.")

# --- 4. 科学データにドップラー補正を適用 ---
print("\nStep 4: Applying Doppler correction to science frames...")
for i in range(is_num, ie_num + 1):
    science_path = fileF1 / f"{i}_f_IDL.fit"
    print(f"  Processing: {science_path.name}")
    try:
        Mc = fits.getdata(science_path, ext=0).astype(np.float64)
        Mc2 = np.zeros_like(Mc)

        # 各行をドップラーシフト量(dc)に応じて補間する
        x_coords = np.arange(ixm)
        for iy in range(iym):
            # 補間する元の行データ
            row_data = Mc[iy, :]
            # 補間関数を作成
            fill_value_tuple = (row_data[0], row_data[-1])
            interp_func = interp1d(x_coords, row_data, kind='linear', bounds_error=False, fill_value=fill_value_tuple)

            # 各ピクセルの補間先の座標を計算 (IDLの ix+dc[iy] に相当)
            new_coords = x_coords - dc[iy]  # シフトの向きに注意
            # 補間を実行
            Mc2[iy, :] = interp_func(new_coords)

        # 補正後のデータを保存
        output_path = fileF2 / f"{i}_Na_python_test.fit"
        fits.writeto(output_path, Mc2, overwrite=True)

    except FileNotFoundError:
        print(f"  警告: 科学データ {science_path.name} が見つかりませんでした。スキップします。")

# --- 5. キャリブレーション情報をファイルに保存 ---
print("\nStep 5: Saving calibration data files...")
# ドップラーシフト量を保存
np.savetxt(fileF2 / 'compdc_python.dat', np.column_stack([np.arange(iym), dc]), fmt='%16.8f')
# 波長キャリブレーションの基準値を保存
w2 = 589.1582 - 589.0
w1 = 589.7558 - 589.0
with open(fileF2 / 'pix2spec_python_test.dat', 'w') as f:
    f.write(f"{cD2:12.5f}{w2:12.5f}\n")
    f.write(f"{cD1:12.5f}{w1:12.5f}\n")

print("\nProcessing completed successfully.")