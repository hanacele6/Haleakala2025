# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. 物理定数・共通設定
# ==============================================================================
PHYSICAL_CONSTANTS = {
    'PI': np.pi,
    'AU': 1.496e11,  # [m] 1天文単位
    'RM': 2.440e6,  # [m] 水星半径
    'KB_EV': 8.617e-5,  # [eV/K] ボルツマン定数
    'MERCURY_SEMI_MAJOR_AXIS_AU': 0.387098,
    'MERCURY_ECCENTRICITY': 0.205630,
}

# 半球（昼側全体）の表面積 [m^2] = 2 * pi * R^2
HEMISPHERE_AREA = 2 * PHYSICAL_CONSTANTS['PI'] * (PHYSICAL_CONSTANTS['RM'] ** 2)

# 1/4球（明け方・夕方それぞれ）の表面積 [m^2] = pi * R^2
QUARTER_SPHERE_AREA = PHYSICAL_CONSTANTS['PI'] * (PHYSICAL_CONSTANTS['RM'] ** 2)

# 温度モデル設定 (Leblanc et al. 共通)
TEMP_BASE = 100.0
TEMP_AMP = 600.0
DIFF_REF_TEMP = 700.0  # [K] 拡散係数の基準温度 (共通)

# ==============================================================================
# 2. パラメータ設定エリア
# ==============================================================================

# --- モデルA: カットオフなし (Standard) ---
PARAMS_STD = {
    'LABEL': 'Model A (Standard)',
    'E_A_EV': 0.40,
    'REF_FLUX': 5.0e7 * (100.0 ** 2),
    'USE_CLAMP': False,
    'CLAMP_START': None,
    'CLAMP_END': None
}

# --- モデルB: カットオフあり (Clamped) ---
PARAMS_CLAMP = {
    'LABEL': 'Model B (Clamped)',
    'E_A_EV': 0.8,
    'REF_FLUX': 1.0e8 * (100.0 ** 2),
    'USE_CLAMP': True,
    'CLAMP_START': 70.0,
    'CLAMP_END': 290.0
}


# ==============================================================================
# 3. 計算ロジック・関数定義
# ==============================================================================

def calc_pre_factor(params):
    """パラメータ辞書から頻度因子(A)を計算"""
    return params['REF_FLUX'] / np.exp(-params['E_A_EV'] / (PHYSICAL_CONSTANTS['KB_EV'] * DIFF_REF_TEMP))


# 各モデルのPre-factorを計算して辞書に追加登録
PARAMS_STD['PRE_FACTOR'] = calc_pre_factor(PARAMS_STD)
PARAMS_CLAMP['PRE_FACTOR'] = calc_pre_factor(PARAMS_CLAMP)


def calculate_au_from_taa(taa_deg):
    """TAA(度)から水星の太陽距離(AU)を計算"""
    a = PHYSICAL_CONSTANTS['MERCURY_SEMI_MAJOR_AXIS_AU']
    e = PHYSICAL_CONSTANTS['MERCURY_ECCENTRICITY']
    rad = np.deg2rad(taa_deg)
    r = a * (1 - e ** 2) / (1 + e * np.cos(rad))
    return r


def get_subsolar_temperature(au_distance):
    """指定距離における太陽直下点温度を計算"""
    scaling = np.sqrt(0.306 / au_distance)
    return TEMP_BASE + TEMP_AMP * (1.0 ** 0.25) * scaling


def calculate_diffusion_flux(temp_k, params):
    """
    温度とモデルパラメータを受け取って拡散フラックスを計算
    Return: [atoms/m^2/s]
    """
    if temp_k <= 100.0:
        return 0.0

    pre_factor = params['PRE_FACTOR']
    e_a = params['E_A_EV']

    return pre_factor * np.exp(-e_a / (PHYSICAL_CONSTANTS['KB_EV'] * temp_k))


def integrate_total_source_rate(au_distance, params):
    """
    指定AU・指定パラメータセットに基づいて昼側表面全体の総放出率 [atoms/s] を計算
    ※現在の温度モデルは太陽直下点対称なので、
      Dawn側総量 = Dusk側総量 = Total / 2 となります。
    """
    n_steps = 90
    thetas = np.linspace(0, np.pi / 2, n_steps)  # 天頂角 0~90度
    d_theta = thetas[1] - thetas[0]

    scaling = np.sqrt(0.306 / au_distance)

    total_rate = 0.0
    rm = PHYSICAL_CONSTANTS['RM']

    for theta in thetas:
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # 局所温度
        temp_local = TEMP_BASE + TEMP_AMP * (cos_t ** 0.25) * scaling

        # 局所フラックス
        flux = calculate_diffusion_flux(temp_local, params)

        # 面積要素 dS (リング状) = 2*pi*R^2 * sin(theta) * d_theta
        area_element = 2 * np.pi * (rm ** 2) * sin_t * d_theta

        total_rate += flux * area_element

    return total_rate


# ==============================================================================
# 4. メイン計算処理
# ==============================================================================

# 事前計算: Model B用のCutoff距離
au_cutoff_val = calculate_au_from_taa(PARAMS_CLAMP['CLAMP_START'])

taa_list = np.arange(0, 361, 1)

# 結果格納用（Dawn/Duskを追加）
results = {
    'std': {'flux': [], 'total': [], 'avg_dawn': [], 'avg_dusk': []},
    'clamp': {'flux': [], 'total': [], 'avg_dawn': [], 'avg_dusk': []}
}

print("--- Calculation Start ---")

for taa in taa_list:
    # 1. 実際の距離
    au_actual = calculate_au_from_taa(taa)


    # --- 共通計算関数 ---
    def process_model(au_input, params, storage_key):
        # A. 直下点温度・フラックス
        temp = get_subsolar_temperature(au_input)
        flux_subsolar = calculate_diffusion_flux(temp, params)

        # B. 昼側総放出率 [atoms/s]
        rate_total = integrate_total_source_rate(au_input, params)

        # C. 平均フラックス [atoms/m^2/s]
        # ※現在のモデルは対称なので、DawnもDuskも全球平均と同じになります。
        #   Average = (Total Rate / 2) / (Quarter Area) = Total Rate / Hemisphere Area

        rate_dawn = rate_total / 2.0
        rate_dusk = rate_total / 2.0

        avg_flux_dawn = rate_dawn / QUARTER_SPHERE_AREA
        avg_flux_dusk = rate_dusk / QUARTER_SPHERE_AREA

        # 保存 (Fluxは cm^-2 に変換)
        results[storage_key]['flux'].append(flux_subsolar / 1.0e4)
        results[storage_key]['total'].append(rate_total)
        results[storage_key]['avg_dawn'].append(avg_flux_dawn / 1.0e4)
        results[storage_key]['avg_dusk'].append(avg_flux_dusk / 1.0e4)


    # -------------------------------------------------------
    # Model A (Standard)
    # -------------------------------------------------------
    process_model(au_actual, PARAMS_STD, 'std')

    # -------------------------------------------------------
    # Model B (Clamped)
    # -------------------------------------------------------
    start = PARAMS_CLAMP['CLAMP_START']
    end = PARAMS_CLAMP['CLAMP_END']
    if start <= taa <= end:
        au_use = au_actual
    else:
        au_use = au_cutoff_val

    process_model(au_use, PARAMS_CLAMP, 'clamp')

# ==============================================================================
# 5. プロット作成 (3段構成に変更)
# ==============================================================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14), sharex=True)

# --- Plot 1: Subsolar Flux (直下点フラックス) ---
ax1.plot(taa_list, results['std']['flux'], 'k--', alpha=0.7, label=PARAMS_STD['LABEL'])
ax1.plot(taa_list, results['clamp']['flux'], 'r-', linewidth=2.5, label=PARAMS_CLAMP['LABEL'])
ax1.set_ylabel(r'Subsolar Flux [$cm^{-2} s^{-1}$]', fontsize=11)
ax1.set_title('Sodium Diffusion Analysis', fontsize=14)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper right')

# --- Plot 2: Total Source Rate (総放出量) ---
ax2.plot(taa_list, results['std']['total'], 'k--', alpha=0.7, label='Model A Total')
ax2.plot(taa_list, results['clamp']['total'], 'b-', linewidth=2.5, label='Model B Total')
ax2.set_ylabel(r'Total Source Rate [$atoms/s$]', fontsize=11)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right')

# --- Plot 3: Average Flux (Dawn vs Dusk) ---
# ※対称モデルのため重なりますが、ラベルを分けて表示します
# Model A
ax3.plot(taa_list, results['std']['avg_dawn'], 'k--', alpha=0.7, label='Model A Dawn Avg')
ax3.plot(taa_list, results['std']['avg_dusk'], 'y:', alpha=0.7, label='Model A Dusk Avg')
# Model B
ax3.plot(taa_list, results['clamp']['avg_dawn'], 'g-', linewidth=2.5, label='Model B Dawn Avg')
ax3.plot(taa_list, results['clamp']['avg_dusk'], 'y:', linewidth=2.5, label='Model B Dusk Avg')

ax3.set_xlabel('True Anomaly Angle (TAA) [deg]', fontsize=12)
ax3.set_ylabel(r'Average Flux [$cm^{-2} s^{-1}$]', fontsize=11)
ax3.set_title('Regional Average Flux (Dawn vs Dusk)', fontsize=12)
ax3.grid(True, linestyle=':', alpha=0.6)
ax3.legend(loc='upper right')

# --- 共通: 補助線とラベル ---
for ax in [ax1, ax2, ax3]:
    # Clamp範囲
    ax.axvline(x=PARAMS_CLAMP['CLAMP_START'], color='g', linestyle=':', alpha=0.5)
    ax.axvline(x=PARAMS_CLAMP['CLAMP_END'], color='g', linestyle=':', alpha=0.5)

    # 近日点・遠日点ラベル
    y_min, y_max = ax.get_ylim()
    ax.text(0, y_min, ' Perihelion', rotation=90, verticalalignment='bottom', color='gray', fontsize=8)
    ax.text(180, y_min, ' Aphelion', rotation=90, verticalalignment='bottom', color='gray', fontsize=8)

plt.tight_layout()
plt.show()

# --- 数値確認用 ---
print("\n--- Results Summary (Max Values) ---")
print(f"Model A Subsolar Flux : {max(results['std']['flux']):.2e}")
print(f"Model A Dawn Avg Flux : {max(results['std']['avg_dawn']):.2e}")
print("-" * 30)
print(f"Model B Subsolar Flux : {max(results['clamp']['flux']):.2e}")
print(f"Model B Dawn Avg Flux : {max(results['clamp']['avg_dawn']):.2e}")
print("※現在の静的温度モデルでは、Dawn(明け方)とDusk(夕方)の平均値は等しくなります。")