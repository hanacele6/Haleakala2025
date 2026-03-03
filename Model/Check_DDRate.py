# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. 物理定数・共通設定
# ==============================================================================
PHYSICAL_CONSTANTS = {
    'PI': np.pi,
    'AU': 1.496e11,
    'RM': 2.440e6,
    'KB_EV': 8.617e-5,
    'MERCURY_SEMI_MAJOR_AXIS_AU': 0.387098,
    'MERCURY_ECCENTRICITY': 0.205630,
}

HEMISPHERE_AREA = 2 * PHYSICAL_CONSTANTS['PI'] * (PHYSICAL_CONSTANTS['RM'] ** 2)
QUARTER_SPHERE_AREA = PHYSICAL_CONSTANTS['PI'] * (PHYSICAL_CONSTANTS['RM'] ** 2)

TEMP_BASE = 100.0
TEMP_AMP = 600.0
DIFF_REF_TEMP = 700.0

# ==============================================================================
# 2. モデル定義エリア（ここで2つの条件を設定）
# ==============================================================================

# --- モデルA (現在の設定など) ---
MODEL_A = {
    'LABEL': 'Model A',
    'E_A_EV': 0.65,  # 活性化エネルギー
    'REF_FLUX': 5.0e7 * (100.0 ** 2),  # 基準フラックス (入力は cm^-2 なので m^-2 に換算)
}

# --- モデルB (比較したい設定) ---
MODEL_B = {
    'LABEL': 'Model B',
    'E_A_EV': 0.65,  # 活性化エネルギーを上げてみる
    'REF_FLUX': 7.0e7 * (100.0 ** 2),  # Fluxは同じにする
}


# ==============================================================================
# 3. 計算ロジック・関数定義
# ==============================================================================

def calc_pre_factor(params):
    """頻度因子(A)を計算"""
    return params['REF_FLUX'] / np.exp(-params['E_A_EV'] / (PHYSICAL_CONSTANTS['KB_EV'] * DIFF_REF_TEMP))


# 各モデルにPre-factorを計算して埋め込む
MODEL_A['PRE_FACTOR'] = calc_pre_factor(MODEL_A)
MODEL_B['PRE_FACTOR'] = calc_pre_factor(MODEL_B)


def calculate_au_from_taa(taa_deg):
    """TAAから距離(AU)"""
    a = PHYSICAL_CONSTANTS['MERCURY_SEMI_MAJOR_AXIS_AU']
    e = PHYSICAL_CONSTANTS['MERCURY_ECCENTRICITY']
    rad = np.deg2rad(taa_deg)
    return a * (1 - e ** 2) / (1 + e * np.cos(rad))


def get_subsolar_temperature(au_distance):
    """太陽直下点温度"""
    scaling = np.sqrt(0.306 / au_distance)
    return TEMP_BASE + TEMP_AMP * (1.0 ** 0.25) * scaling


def calculate_diffusion_flux(temp_k, params):
    """拡散フラックス [atoms/m^2/s]"""
    if temp_k <= 100.0: return 0.0
    return params['PRE_FACTOR'] * np.exp(-params['E_A_EV'] / (PHYSICAL_CONSTANTS['KB_EV'] * temp_k))


def integrate_total_source_rate(au_distance, params):
    """半球積分による総放出率 [atoms/s]"""
    n_steps = 90
    thetas = np.linspace(0, np.pi / 2, n_steps)
    d_theta = thetas[1] - thetas[0]
    scaling = np.sqrt(0.306 / au_distance)

    total_rate = 0.0
    rm = PHYSICAL_CONSTANTS['RM']

    for theta in thetas:
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        temp_local = TEMP_BASE + TEMP_AMP * (cos_t ** 0.25) * scaling
        flux = calculate_diffusion_flux(temp_local, params)
        area_element = 2 * np.pi * (rm ** 2) * sin_t * d_theta
        total_rate += flux * area_element
    return total_rate


# ==============================================================================
# 4. シミュレーション実行関数
# ==============================================================================

def run_simulation(params):
    """
    指定されたパラメータセットでTAA 0-360度の計算を行う
    """
    taa_list = np.arange(0, 361, 2)
    res_flux = []
    res_total = []

    for taa in taa_list:
        # 頭打ちなしなので、常に実際の距離を使用
        au_dist = calculate_au_from_taa(taa)

        # 直下点温度 & Flux
        temp = get_subsolar_temperature(au_dist)
        flux = calculate_diffusion_flux(temp, params)

        # 総放出率
        total = integrate_total_source_rate(au_dist, params)

        # 保存 (Fluxは cm^-2 に変換して見やすく)
        res_flux.append(flux / 1.0e4)
        res_total.append(total)

    return taa_list, res_flux, res_total


# ==============================================================================
# 5. メイン処理 & プロット
# ==============================================================================

print("--- Calculation Start ---")

# 計算実行
taa_a, flux_a, total_a = run_simulation(MODEL_A)
taa_b, flux_b, total_b = run_simulation(MODEL_B)

# プロット作成
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# --- Plot 1: Subsolar Flux ---
ax1.plot(taa_a, flux_a, 'b-', linewidth=2, label=MODEL_A['LABEL'])
ax1.plot(taa_b, flux_b, 'r--', linewidth=2, label=MODEL_B['LABEL'])
ax1.set_ylabel(r'Subsolar Flux [$cm^{-2} s^{-1}$]', fontsize=11)
ax1.set_title('Comparison: Subsolar Flux', fontsize=14)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper right')

# --- Plot 2: Total Source Rate ---
ax2.plot(taa_a, total_a, 'b-', linewidth=2, label=MODEL_A['LABEL'])
ax2.plot(taa_b, total_b, 'r--', linewidth=2, label=MODEL_B['LABEL'])
ax2.set_ylabel(r'Total Source Rate [$atoms/s$]', fontsize=11)
ax2.set_xlabel('True Anomaly Angle (TAA) [deg]', fontsize=12)
ax2.set_title('Comparison: Total Source Rate', fontsize=14)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right')

# 軸ラベル（近日点・遠日点）
for ax in [ax1, ax2]:
    y_min, y_max = ax.get_ylim()
    ax.text(0, y_min, ' Perihelion', rotation=90, va='bottom', color='gray')
    ax.text(180, y_min, ' Aphelion', rotation=90, va='bottom', color='gray')

plt.tight_layout()
plt.show()

# --- 最大値比較 ---
print(f"\n--- Result Comparison (at Perihelion/Peak) ---")
print(f"[{MODEL_A['LABEL']}] Max Flux: {max(flux_a):.2e}")
print(f"[{MODEL_B['LABEL']}] Max Flux: {max(flux_b):.2e}")
print(f"Ratio (B/A): {max(flux_b) / max(flux_a):.2f}")