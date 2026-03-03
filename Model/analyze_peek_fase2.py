import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 設定：ここをいじって「いい感じの山」になるUを探す
# ==============================================================================
# 先生の予想では、低いU (1.4eV付近) なら150度付近に山が来るはず
TARGET_U_EV = 1.4
LAG_ANGLE = 5.0

# 物理定数
PHYSICAL_CONSTANTS = {
    'PI': np.pi, 'AU_M': 1.496e11, 'K_BOLTZMANN': 1.380649e-23,
    'EV_TO_JOULE': 1.602e-19, 'ROTATION_PERIOD': 58.6462 * 86400,
    'MERCURY_SEMI_MAJOR_AXIS_AU': 0.387098, 'MERCURY_ECCENTRICITY': 0.205630,
    'MASS_SUN': 1.989e30, 'G': 6.6743e-11,
}
SIMULATION_SETTINGS = {'TEMP_BASE': 100.0, 'TEMP_AMP': 600.0, 'TEMP_NIGHT': 100.0}


def calculate_thermal_desorption_rate(temp_K, U_eV):
    if temp_K < 10.0: return 0.0
    VIB_FREQ = 1e13
    KB = PHYSICAL_CONSTANTS['K_BOLTZMANN']
    U_JOULE = U_eV * PHYSICAL_CONSTANTS['EV_TO_JOULE']
    exponent = -U_JOULE / (KB * temp_K)
    if exponent < -700: return 0.0
    return VIB_FREQ * np.exp(exponent)

def get_orbit_analytical(taa_deg):
    e = PHYSICAL_CONSTANTS['MERCURY_ECCENTRICITY']
    a_m = PHYSICAL_CONSTANTS['MERCURY_SEMI_MAJOR_AXIS_AU'] * PHYSICAL_CONSTANTS['AU_M']
    rad = np.deg2rad(taa_deg)
    r_m = a_m * (1 - e ** 2) / (1 + e * np.cos(rad))
    mu = PHYSICAL_CONSTANTS['G'] * PHYSICAL_CONSTANTS['MASS_SUN']
    h = np.sqrt(mu * a_m * (1 - e ** 2))
    return r_m / PHYSICAL_CONSTANTS['AU_M'], h / (r_m ** 2)


def get_surface_temp_at_lag(r_au, lag_angle_deg):
    scaling = (0.306 / r_au)**(1/2)
    zenith_angle_rad = np.deg2rad(90.0 - lag_angle_deg)
    cos_theta = np.cos(zenith_angle_rad)
    return SIMULATION_SETTINGS['TEMP_BASE'] + SIMULATION_SETTINGS['TEMP_AMP'] * (
                cos_theta ** 0.25) * scaling if cos_theta > 0 else SIMULATION_SETTINGS['TEMP_NIGHT']


# ==============================================================================
# メイン処理：先生のアイデア「掛け算」の実装
# ==============================================================================
def main_sensei_idea():
    taa_list = np.arange(0, 361, 1.0)
    products = []

    omega_spin = 2 * np.pi / PHYSICAL_CONSTANTS['ROTATION_PERIOD']

    print(f"Calculating Product (Speed * Rate) for U = {TARGET_U_EV} eV...")

    for taa in taa_list:
        r_au, omega_orb = get_orbit_analytical(taa)

        # 1. 見かけの自転速度 (上がっていくやつ)
        omega_app = omega_spin - omega_orb
        supply_speed = abs(omega_app)

        # 2. 温度による脱離率 (下がっていくやつ)
        temp = get_surface_temp_at_lag(r_au, LAG_ANGLE)
        rate = calculate_thermal_desorption_rate(temp, TARGET_U_EV)

        # ★先生の提案：単純な掛け算
        # 「供給速度」×「脱離しやすさ」
        product = supply_speed * rate
        products.append(product)

    # プロット
    plt.figure(figsize=(10, 6))


    plt.plot(taa_list, products, color='green', linewidth=2,
             label=f'U={TARGET_U_EV}eV')

    # ガイドライン
    plt.yscale('log')


    plt.title(f'Shape of "Rotation Speed x Desorption Rate" (U={TARGET_U_EV} eV)', fontsize=14)
    plt.xlabel('True Anomaly Angle [deg]', fontsize=12)
    plt.ylabel('Intensity', fontsize=12)
    plt.xlim(0, 360)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main_sensei_idea()