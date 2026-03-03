import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 設定パラメータ
# ==============================================================================
TARGET_U_EV = 1.4  # ★ここでUを指定 (例: 0.65, 0.85, 1.85, 2.7)
LAG_ANGLE = 5.0  # 夜明け線から日照側に何度進んだ地点で評価するか

# ==============================================================================
# 物理定数
# ==============================================================================
PHYSICAL_CONSTANTS = {
    'PI': np.pi,
    'AU_M': 1.496e11,
    'K_BOLTZMANN': 1.380649e-23,
    'EV_TO_JOULE': 1.602e-19,
    'ROTATION_PERIOD': 58.6462 * 86400,
    'MERCURY_SEMI_MAJOR_AXIS_AU': 0.387098,
    'MERCURY_ECCENTRICITY': 0.205630,
    'MASS_SUN': 1.989e30,
    'G': 6.6743e-11,
}

SIMULATION_SETTINGS = {
    'TEMP_BASE': 100.0,
    'TEMP_AMP': 600.0,
    'TEMP_NIGHT': 100.0,
}


# ==============================================================================
# 計算関数
# ==============================================================================

def calculate_thermal_desorption_rate(temp_K, U_eV):
    if temp_K < 10.0: return 0.0
    VIB_FREQ = 1e13
    KB = PHYSICAL_CONSTANTS['K_BOLTZMANN']
    EV_J = PHYSICAL_CONSTANTS['EV_TO_JOULE']
    U_JOULE = U_eV * EV_J
    exponent = -U_JOULE / (KB * temp_K)
    if exponent < -700: return 0.0
    return VIB_FREQ * np.exp(exponent)


def get_orbit_analytical(taa_deg):
    e = PHYSICAL_CONSTANTS['MERCURY_ECCENTRICITY']
    a_au = PHYSICAL_CONSTANTS['MERCURY_SEMI_MAJOR_AXIS_AU']
    a_m = a_au * PHYSICAL_CONSTANTS['AU_M']
    rad = np.deg2rad(taa_deg)

    r_m = a_m * (1 - e ** 2) / (1 + e * np.cos(rad))
    r_au = r_m / PHYSICAL_CONSTANTS['AU_M']

    mu = PHYSICAL_CONSTANTS['G'] * PHYSICAL_CONSTANTS['MASS_SUN']
    h = np.sqrt(mu * a_m * (1 - e ** 2))
    omega_orb = h / (r_m ** 2)
    return r_au, omega_orb


def get_surface_temp_at_lag(r_au, lag_angle_deg):
    scaling = (0.306 / r_au)**(2)
    zenith_angle_rad = np.deg2rad(90.0 - lag_angle_deg)
    cos_theta = np.cos(zenith_angle_rad)
    if cos_theta <= 0: return SIMULATION_SETTINGS['TEMP_NIGHT']
    return SIMULATION_SETTINGS['TEMP_BASE'] + \
        SIMULATION_SETTINGS['TEMP_AMP'] * (cos_theta ** 0.25) * scaling


# ==============================================================================
# メイン処理
# ==============================================================================

def main_verification_single_plot():
    taa_list = np.arange(0, 361, 0.5)  # 解像度を少し上げました
    ratios = []

    omega_spin = 2 * np.pi / PHYSICAL_CONSTANTS['ROTATION_PERIOD']

    print(f"Calculating Ratio for U = {TARGET_U_EV} eV at Lag = {LAG_ANGLE} deg...")

    for taa in taa_list:
        # 1. 軌道・供給速度計算
        r_au, omega_orb = get_orbit_analytical(taa)
        omega_app = omega_spin - omega_orb
        supply_speed = abs(omega_app)
        if supply_speed < 1e-12: supply_speed = 1e-12  # ゼロ割回避

        # 2. 温度・脱離率計算
        temp = get_surface_temp_at_lag(r_au, LAG_ANGLE)
        rate = calculate_thermal_desorption_rate(temp, TARGET_U_EV)

        # 3. 比の計算
        ratio = rate * supply_speed
        ratios.append(ratio)

    # ==========================================================================
    # プロット (Ratioのみ)
    # ==========================================================================
    plt.figure(figsize=(10, 6))

    plt.plot(taa_list, ratios, color='b', linewidth=2, label=f'U = {TARGET_U_EV} eV')

    plt.yscale('log')
    plt.title(f'Desorption Rate / Supply Speed Ratio (U={TARGET_U_EV} eV)', fontsize=14)
    plt.xlabel('True Anomaly Angle [deg]', fontsize=12)
    plt.ylabel('Ratio (Rate / Supply Speed)', fontsize=12)
    plt.xlim(0, 360)

    # ガイドライン
    #plt.axhline(1.0, color='r', linestyle='--', linewidth=1.5, label='Equilibrium Limit (Ratio=1)')

    # ピーク(24度付近)の強調
    peak_idx = np.argmax(ratios)
    peak_taa = taa_list[peak_idx]
    plt.axvline(peak_taa, color='gray', linestyle=':', alpha=0.5)
    plt.text(peak_taa + 5, np.max(ratios) * 0.5, f"Peak at ~{peak_taa:.1f}°\n(Solar Stop)", color='gray')

    # 領域の説明
    #plt.text(0.02, 0.95, "Ratio >> 1 : Supply Limited (Rapid Evaporation)\nRatio << 1 : Desorption Limited",
    #         transform=plt.gca().transAxes, verticalalignment='top',
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(fontsize=12, loc='lower right')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main_verification_single_plot()