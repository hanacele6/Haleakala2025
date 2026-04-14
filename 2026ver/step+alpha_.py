import numpy as np
import pandas as pd
from pathlib import Path
from astropy.time import Time
from astroquery.jplhorizons import Horizons

# ==============================================================================
# 定数
# ==============================================================================
PI = np.pi
TARGET_WL = 589.7571833  # nm (Na D1)
FWHM_CONST = 0.005  # nm
C_KMS = 299792.458  # km/s

# 物理定数の計算 (先輩のコードから移植)
F_lambda_cgs = 5.18e14 * 1e7
lambda_cm = TARGET_WL * 1e-7
JL_nu = F_lambda_cgs * (lambda_cm ** 2 / (C_KMS * 1e5))
sigma_D1_nu = PI * (4.8032e-10) ** 2 / 9.109e-28 / (C_KMS * 1e5) * 0.327

PHYSICAL_CONVERSION_FACTOR = sigma_D1_nu * JL_nu


def calculate_gamma_convolution(solar_data, v_rad, target_wl, fwhm):
    """
    太陽スペクトルとガウス関数の畳み込み積分を行う。
    物理定数は掛けず、純粋な畳み込み係数(gamma0)のみを返す。
    """
    wl0 = solar_data[:, 0]
    sol = solar_data[:, 1]

    # ドップラーシフト (水星から見た太陽スペクトルの波長推移)
    wl_shifted = wl0 * (1.0 + v_rad / C_KMS)

    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    idx = np.argmin(np.abs(wl_shifted - target_wl))

    # 範囲切り出し
    hw = int(5 * sigma / np.mean(np.diff(wl0))) + 5
    s_idx = max(0, idx - hw)
    e_idx = min(len(wl0), idx + hw + 1)

    wl_crop = wl_shifted[s_idx:e_idx]
    sol_crop = sol[s_idx:e_idx]

    # ガウス関数
    phi = np.exp(-((wl_crop - target_wl) ** 2) / (2.0 * sigma ** 2))

    # 正規化
    if np.sum(phi) == 0: return 0.0
    phi_norm = phi / np.sum(phi)

    # 畳み込み
    gamma0 = np.sum(sol_crop * phi_norm)

    return gamma0


def get_ephemeris_from_horizons(date_str):
    """
    JPL Horizonsにアクセスし、指定日時の水星の太陽に対する軌道情報を取得する。
    """
    try:
        jd = Time(date_str).jd
        # location='@10' は太陽中心。これにより純粋な日心距離と太陽に対する視線速度が得られる
        obj = Horizons(id='199', location='@10', epochs=jd)
        eph = obj.ephemerides()

        r_au = eph['delta'][0]  # 太陽からの距離 [AU]
        v_r_kms = eph['delta_rate'][0]  # 太陽に対する視線速度 [km/s]
        return r_au, v_r_kms
    except Exception as e:
        print(f"  > [警告] JPL Horizonsの取得に失敗しました ({date_str}): {e}")
        return np.nan, np.nan


def process_excel_with_solar_spectrum(input_excel_path, output_excel_path, solar_spec_path):
    """
    Excelを読み込み、Horizons軌道データと太陽スペクトル畳み込みを用いて
    g-factorと輝度を計算・追加する。
    """
    print(f"--- 読み込み中: {Path(input_excel_path).name} ---")

    # 太陽スペクトルの読み込み
    try:
        solar_data = np.loadtxt(solar_spec_path)
        print(f"  > 太陽スペクトル読み込み完了: {solar_spec_path}")
    except Exception as e:
        print(f"エラー: 太陽スペクトルファイルの読み込みに失敗しました ({e})")
        return

    # Excelの読み込み
    try:
        df = pd.read_excel(input_excel_path)
    except Exception as e:
        print(f"エラー: Excelファイルの読み込みに失敗しました ({e})")
        return

    # 柱密度の単位を戻す ( * 10^11 )
    actual_cd_raw = df.iloc[:, 3] * 1e11

    gamma_vals = []
    g_factors = []
    r_au_list = []
    vr_kms_list = []

    print("--- 軌道取得および太陽スペクトル畳み込み計算を実行中 ---")
    for idx, row in df.iterrows():
        date_val = str(row.iloc[1])  # 2列目の日時を取得

        if pd.isna(row.iloc[1]) or date_val.strip() == "NaT" or date_val.strip() == "":
            gamma_vals.append(np.nan)
            g_factors.append(np.nan)
            r_au_list.append(np.nan)
            vr_kms_list.append(np.nan)
            continue

        # 1. Horizonsから軌道データを取得
        r_val, vr_val = get_ephemeris_from_horizons(date_val)

        if not np.isnan(vr_val):
            # 2. 視線速度を用いて太陽スペクトルを畳み込み積分
            gamma_val = calculate_gamma_convolution(solar_data, vr_val, TARGET_WL, FWHM_CONST)

            # 3. gamma値を 1AUでのg-factor に変換し、さらに距離の2乗で割って実空間のg-factorにする
            g_1au = gamma_val * PHYSICAL_CONVERSION_FACTOR
            g_val = g_1au / (r_val ** 2)
        else:
            gamma_val, g_val = np.nan, np.nan

        gamma_vals.append(gamma_val)
        g_factors.append(g_val)
        r_au_list.append(r_val)
        vr_kms_list.append(vr_val)

        print(f"  [{date_val}] v_rad: {vr_val:+.2f} km/s -> gamma: {gamma_val:.4f} -> g-factor: {g_val:.2f}")

    # 配列化
    g_factors = np.array(g_factors)

    # 4. 輝度 (kR) の計算 (投影面積のため柱密度を2.0倍)
    brightness_kr = (actual_cd_raw * 2.0 * g_factors) / 1e9

    # 5. データフレームに追加
    df['Sun_Distance_AU'] = r_au_list
    df['Radial_Velocity_kms'] = vr_kms_list
    df['Gamma_Val_Convolution'] = gamma_vals
    df['g_factor_Calculated'] = g_factors
    df['Brightness_kR_Calculated'] = brightness_kr

    # 6. 保存
    try:
        df.to_excel(output_excel_path, index=False)
        print(f"\n--- 完了: 厳密なg-factorと輝度を追加したファイルを保存しました ---")
        print(f" -> {output_excel_path}")
    except Exception as e:
        print(f"エラー: ファイルの保存に失敗しました ({e})")


# ==============================================================================
# 実行ブロック
# ==============================================================================
if __name__ == "__main__":
    # ★ 1. 太陽スペクトルデータのパス
    solar_spectrum_file = "C:/Users/hanac/univ/Mercury/Haleakala2025/SolarSpectrum.txt"

    # ★ 2. 読み込みたいExcelファイルのパス
    input_file = "C:/Users/hanac/univ/Mercury/DUSK_preview.xlsx"

    # ★ 3. 保存先のExcelファイルパス
    output_file = "C:/Users/hanac/univ/Mercury/Dusk_Brightness.xlsx"

    process_excel_with_solar_spectrum(input_file, output_file, solar_spectrum_file)