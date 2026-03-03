import numpy as np

# ==============================================================================
# 1. 計算設定
# ==============================================================================
# 物理定数
CONSTANTS = {
    'RM': 2.440e6,  # 水星半径 [m]
    'AU_M': 1.496e11,  # 1AU [m]
    'KB': 1.380649e-23,
    'EV_J': 1.602e-19,
    'VIB_FREQ': 1e13,
}

# ユーザーのシミュレーション設定
SETTINGS = {
    'TEMP_BASE': 100.0,
    'TEMP_AMP': 600.0,
    'U_EV': 1.85,  # 固定値
    'DIST_AU': 0.467,  # ★基準とする季節（ここでは遠日点 Aphelion を採用）
    # 現在のあなたのコードの設定値
    'CURRENT_INIT_DENS': 7.5e14 * (100 ** 2) * 0.0053,  # [atoms/m^2]
}

# 先輩の論文(安田2015)のターゲット値
# 遠日点でのPSDによる総生成率の概算
# Flux(Mean) ≈ 2.5e7 atoms/cm^2/s * Area_Effective
# ここでは簡易的に、昼面半球の総量をターゲットとします
TARGET_FLUX_PER_CM2 = 2.5e7  # [atoms/cm^2/s] (遠日点)


# ==============================================================================
# 2. 計算ロジック
# ==============================================================================
def calculate_temp(cos_theta, r_au):
    """Leblancモデル温度"""
    if cos_theta <= 0: return SETTINGS['TEMP_BASE']
    scaling = np.sqrt(0.306 / r_au)
    return SETTINGS['TEMP_BASE'] + SETTINGS['TEMP_AMP'] * (cos_theta ** 0.25) * scaling


def calculate_rate(temp, u_ev):
    """TD Rate (Fixed U)"""
    if temp < 100: return 0.0
    exponent = -(u_ev * CONSTANTS['EV_J']) / (CONSTANTS['KB'] * temp)
    if exponent < -500: return 0.0
    return CONSTANTS['VIB_FREQ'] * np.exp(exponent)


def calculate_global_source_rate(surface_density_m2):
    """
    指定された表面密度において、水星全体(昼面)から1秒間に放出される
    総原子数 [atoms/s] を数値積分で求める
    """
    total_atoms_per_sec = 0.0

    # グリッド積分 (簡易的に緯度経度でループ)
    n_lon = 72
    n_lat = 36
    d_lon = 2 * np.pi / n_lon
    d_lat = np.pi / n_lat  # -pi/2 to pi/2

    r_au = SETTINGS['DIST_AU']

    for i in range(n_lon):
        lon = -np.pi + (i + 0.5) * d_lon
        for j in range(n_lat):
            lat = -np.pi / 2 + (j + 0.5) * d_lat

            # 面積要素 dS = R^2 cos(lat) dlat dlon
            area = (CONSTANTS['RM'] ** 2) * np.cos(lat) * d_lat * d_lon

            # 太陽天頂角 (Subsolar at 0,0)
            cos_theta = np.cos(lat) * np.cos(lon)

            if cos_theta > 0:
                temp = calculate_temp(cos_theta, r_au)
                rate = calculate_rate(temp, SETTINGS['U_EV'])

                # 放出量 = Rate * Density * Area
                flux_amount = rate * surface_density_m2 * area
                total_atoms_per_sec += flux_amount

    return total_atoms_per_sec


def main_calculation():
    print(f"--- Surface Density Calculator (Targeting Aphelion) ---")
    print(f"Condition: Dist = {SETTINGS['DIST_AU']} AU, U = {SETTINGS['U_EV']} eV")

    # 1. 先輩のターゲット総生成量を計算 (PSDモデルベース)
    # PSDはcos_thetaに比例して出るため、有効面積は πR^2
    # Total Source ~ Flux_subsolar * πR^2 (近似)
    effective_area = np.pi * (CONSTANTS['RM'] ** 2)
    # cm^2 -> m^2 換算 (1e4倍)
    target_flux_m2 = TARGET_FLUX_PER_CM2 * 1e4
    target_total_source = target_flux_m2 * effective_area

    print(f"Target Total Source (Thesis PSD): {target_total_source:.2e} [atoms/s]")

    # 2. 現在の設定でのTD総生成量を計算
    current_dens = SETTINGS['CURRENT_INIT_DENS']
    current_total_source = calculate_global_source_rate(current_dens)

    print(f"Current TD Source ({current_dens:.2e} m^-2): {current_total_source:.2e} [atoms/s]")

    # 3. 必要倍率と推奨密度の算出
    if current_total_source == 0:
        print("Error: Current source is 0. Check U value or Temp.")
        return

    ratio = target_total_source / current_total_source
    recommended_dens = current_dens * ratio

    print("-" * 40)
    print(f"【結論: 推奨される初期表面密度】")
    print(f"今の設定だと、ターゲットに対して {1 / ratio:.1f} 分の1 しか出ていません。")
    print(f"観測と同等の総量を出すには、密度を {ratio:.1f} 倍にする必要があります。")
    print(f"")
    print(f"Recommended INIT_SURF_DENS = {recommended_dens:.3e} [atoms/m^2]")
    print(f"                           = {recommended_dens / 1e4:.3e} [atoms/cm^2]")
    print("-" * 40)
    print("※この値をコードの INIT_SURF_DENS に設定してください。")
    print("※ただし、TDは局所集中型なので、この密度にすると直下点は一瞬で枯渇し、")
    print("  逆に80度付近からの放出がメインの供給源になる可能性があります。")


if __name__ == "__main__":
    main_calculation()