import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================================
# 物理定数・関数 (先輩のコードから移植)
# ==============================================================================
PI = np.pi
TARGET_WL = 589.7571833  # nm (Na D1)
FWHM_CONST = 0.005  # nm
C_KMS = 299792.458  # km/s

F_lambda_cgs = 5.18e14 * 1e7
lambda_cm = TARGET_WL * 1e-7
JL_nu = F_lambda_cgs * (lambda_cm ** 2 / (C_KMS * 1e5))
sigma_D1_nu = PI * (4.8032e-10) ** 2 / 9.109e-28 / (C_KMS * 1e5) * 0.327

PHYSICAL_CONVERSION_FACTOR = sigma_D1_nu * JL_nu

def calculate_gamma_convolution(solar_data, v_rad, target_wl, fwhm):
    """ 太陽スペクトルとガウス関数の畳み込み積分 """
    wl0 = solar_data[:, 0]
    sol = solar_data[:, 1]
    wl_shifted = wl0 * (1.0 + v_rad / C_KMS)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    idx = np.argmin(np.abs(wl_shifted - target_wl))
    
    hw = int(5 * sigma / np.mean(np.diff(wl0))) + 5
    s_idx = max(0, idx - hw)
    e_idx = min(len(wl0), idx + hw + 1)
    
    wl_crop = wl_shifted[s_idx:e_idx]
    sol_crop = sol[s_idx:e_idx]
    phi = np.exp(-((wl_crop - target_wl) ** 2) / (2.0 * sigma ** 2))
    
    if np.sum(phi) == 0: return 0.0
    phi_norm = phi / np.sum(phi)
    return np.sum(sol_crop * phi_norm)

# ==============================================================================
# プロットおよび理論線の計算
# ==============================================================================
def plot_mercury_brightness_with_pure_theory(dawn_excel_path, dusk_excel_path, solar_spec_path):
    print("--- 太陽スペクトルと観測データを読み込んでいます ---")
    
    # 太陽スペクトルの読み込み
    try:
        solar_data = np.loadtxt(solar_spec_path)
    except Exception as e:
        print(f"エラー: 太陽スペクトルの読み込みに失敗しました ({e})")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- 1. 観測データのプロット (実際のばらつきを含む点) ---
    if Path(dawn_excel_path).exists():
        df_dawn = pd.read_excel(dawn_excel_path)
        taa_col = df_dawn.columns[2]
        df_dawn_plot = df_dawn.dropna(subset=[taa_col, 'Brightness_kR_Calculated'])
        ax.plot(df_dawn_plot[taa_col], df_dawn_plot['Brightness_kR_Calculated'], 
                marker='o', linestyle='', color='blue', mfc='none', markersize=6, label='Dawn')

    if Path(dusk_excel_path).exists():
        df_dusk = pd.read_excel(dusk_excel_path)
        taa_col = df_dusk.columns[2]
        df_dusk_plot = df_dusk.dropna(subset=[taa_col, 'Brightness_kR_Calculated'])
        ax.plot(df_dusk_plot[taa_col], df_dusk_plot['Brightness_kR_Calculated'], 
                marker='o', linestyle='', color='red', markersize=6, label='Dusk')


    # --- 2. ★ 真の理論線の計算 (理想的なケプラー軌道に基づく) ---
    print("--- 理想軌道に基づく理論線を計算中 (TAA 0〜360度) ---")
    taa_array = np.arange(0, 361, 1) # 1度刻みで滑らかな線を引く
    
    # 水星の軌道要素 (JPL定数)
    e = 0.20563  # 離心率
    a = 0.387098 # 軌道長半径 (AU)
    V0_e = 10.058 # 視線速度の最大振幅 (km/s)
    num_a_1_e2 = a * (1 - e**2)

    theory_brightness = []
    CONSTANT_CD = 1.0e11  # 論文のピーク(約4700kR)に合わせるための一定柱密度
    
    for taa in taa_array:
        taa_rad = np.radians(taa)
        
        # 理想的な視線速度と日心距離をケプラーの法則から直接算出
        v_rad_ideal = V0_e * np.sin(taa_rad)
        r_au_ideal = num_a_1_e2 / (1 + e * np.cos(taa_rad))
        
        # 太陽スペクトルを使って純粋な理論g-factorを計算
        gamma = calculate_gamma_convolution(solar_data, v_rad_ideal, TARGET_WL, FWHM_CONST)
        g_1au = gamma * PHYSICAL_CONVERSION_FACTOR
        g_factor_ideal = g_1au / (r_au_ideal ** 2)
        
        # 理論輝度の計算
        brightness = (CONSTANT_CD * 2.0 * g_factor_ideal) / 1e9
        theory_brightness.append(brightness)

    # 理論線のプロット
    ax.plot(taa_array, theory_brightness, color='black', linestyle='--', linewidth=1.5, 
            label='Leblan Theory')

    # --- グラフの装飾 ---
    ax.set_title("Mercury Na D1 Brightness vs True Anomaly Angle", fontsize=14)
    ax.set_xlabel("True Anomaly Angle (°)", fontsize=12)
    ax.set_ylabel("Na D1 Intensity (kR)", fontsize=12)
    
    ax.set_xlim(0, 360)
    ax.set_xticks(range(0, 361, 60))
    
    ax.axvline(0, color='black', linestyle='-', alpha=0.8, linewidth=1.5)
    #ax.axvline(180, color='blue', linestyle='--', alpha=0.5, label='Aphelion (180°)')
    ax.axvline(360, color='black', linestyle='-', alpha=0.8, linewidth=1.5)

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    print("--- プロット完了 ---")

# ==============================================================================
if __name__ == "__main__":
    solar_spectrum_file = "C:/Users/hanac/univ/Mercury/Haleakala2025/SolarSpectrum.txt"
    dawn_file = "C:/Users/hanac/univ/Mercury/Dawn_Brightness.xlsx"
    dusk_file = "C:/Users/hanac/univ/Mercury/Dusk_Brightness.xlsx"
    
    plot_mercury_brightness_with_pure_theory(dawn_file, dusk_file, solar_spectrum_file)