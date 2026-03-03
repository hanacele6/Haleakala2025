import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. 物理定数・設定
# ==============================================================================
PHYSICAL_CONSTANTS = {
    'PI': np.pi,
    'AU': 1.496e11,
    'C': 299792458.0,
    'H': 6.62607015e-34,
    'GM_SUN': 1.32712440018e20,  # m^3/s^2
    'MASS_NA': 3.8175e-26,
}

# 水星軌道要素 (近似値)
MERCURY_A_AU = 0.387098
MERCURY_E = 0.205630


# ==============================================================================
# 2. 簡易太陽スペクトルモデル生成 (Sodium D1/D2)
# ==============================================================================
# 実際のデータファイルがないため、ガウス関数でD1/D2の吸収線を模擬します。
def generate_mock_spectrum():
    # 波長範囲 (nm)
    wl = np.linspace(588.5, 590.5, 5000)

    # 基準強度 (Continuum = 1.0)
    gamma = np.ones_like(wl)

    # D2 Line parameters (589.158 nm)
    w_d2 = 589.1582
    sigma_d2 = 0.008  # 吸収線の幅
    depth_d2 = 0.95  # 5%まで落ち込む深い吸収線

    # D1 Line parameters (589.756 nm)
    w_d1 = 589.7558
    sigma_d1 = 0.008
    depth_d1 = 0.95

    # 吸収線を追加 (ガウシアン)
    gamma -= depth_d2 * np.exp(-0.5 * ((wl - w_d2) / sigma_d2) ** 2)
    gamma -= depth_d1 * np.exp(-0.5 * ((wl - w_d1) / sigma_d1) ** 2)

    return wl, np.clip(gamma, 0.01, 1.0)


# スペクトルデータの準備
wl_mock, gamma_mock = generate_mock_spectrum()

# ユーザーコード相当のスペクトルデータ辞書
spec_data_mock = {
    'wl': wl_mock,
    'gamma': gamma_mock,
    # 定数は比率を見るだけなのでダミー値で調整
    'sigma0_perdnu2': 2e-4,
    'sigma0_perdnu1': 1e-4,
    'JL': 5.0e14
}


# ==============================================================================
# 3. 放射圧計算関数 (Original vs Modified)
# ==============================================================================
def calculate_radiation_acceleration(TAA_deg, AU, V_rad_ms, atom_vel_x, spec_data, use_subtraction=False):
    """
    atom_vel_x: 水星に対する原子の速度 (太陽方向が正)
    use_subtraction: Trueなら修正版(vel - Vrad), Falseなら現状版(vel + Vrad)
    """

    # --- ここが検証対象のロジック ---
    if use_subtraction:
        # [修正案] 太陽から遠ざかる(V_rad>0)と赤方偏移するので、
        # 青側(短波長)の光を吸収する必要がある -> マイナスが必要
        velocity_for_doppler = atom_vel_x - V_rad_ms
    else:
        # [現状のコード]
        velocity_for_doppler = atom_vel_x + V_rad_ms
    # -----------------------------

    # ドップラーシフト後の波長 (Source Frame)
    w_na_d2 = 589.1582e-9 * (1.0 + velocity_for_doppler / PHYSICAL_CONSTANTS['C'])
    w_na_d1 = 589.7558e-9 * (1.0 + velocity_for_doppler / PHYSICAL_CONSTANTS['C'])

    wl = spec_data['wl']
    gamma = spec_data['gamma']

    # 線形補間
    gamma2 = np.interp(w_na_d2 * 1e9, wl, gamma)
    gamma1 = np.interp(w_na_d1 * 1e9, wl, gamma)

    # 水星位置でのフラックス
    F_at_Merc = (spec_data['JL'] * 1e13) / (AU ** 2)

    # 加速度計算
    term_d1 = (PHYSICAL_CONSTANTS['H'] / w_na_d1) * spec_data['sigma0_perdnu1'] * \
              (F_at_Merc * gamma1 * w_na_d1 ** 2 / PHYSICAL_CONSTANTS['C'])
    term_d2 = (PHYSICAL_CONSTANTS['H'] / w_na_d2) * spec_data['sigma0_perdnu2'] * \
              (F_at_Merc * gamma2 * w_na_d2 ** 2 / PHYSICAL_CONSTANTS['C'])

    return (term_d1 + term_d2) / PHYSICAL_CONSTANTS['MASS_NA']


# ==============================================================================
# 4. メインループ & プロット
# ==============================================================================
taa_list = np.arange(0, 361, 2)
v_rad_list = []

# ケース1: 静止原子 (vel = 0)
acc_static_orig = []
acc_static_mod = []

# ケース2: 太陽へ向かう原子 (vel = 3 km/s)
# ※ TAA=90付近のV_rad(~6-10km/s)と拮抗する速度でテスト
ATOM_VEL = 3000.0
acc_move_orig = []
acc_move_mod = []

for taa_deg in taa_list:
    theta = np.deg2rad(taa_deg)

    # --- 簡易軌道計算 ---
    # 距離 r
    r_val_m = (MERCURY_A_AU * PHYSICAL_CONSTANTS['AU']) * (1 - MERCURY_E ** 2) / (1 + MERCURY_E * np.cos(theta))
    au_val = r_val_m / PHYSICAL_CONSTANTS['AU']

    # 視線速度 V_rad (dr/dt)
    p_val = (MERCURY_A_AU * PHYSICAL_CONSTANTS['AU']) * (1 - MERCURY_E ** 2)
    v_factor = np.sqrt(PHYSICAL_CONSTANTS['GM_SUN'] / p_val)
    v_rad = v_factor * MERCURY_E * np.sin(theta)  # TAA 0-180で正(遠ざかる)

    v_rad_list.append(v_rad)

    # --- 放射圧計算 ---
    # 1. 静止原子
    acc_static_orig.append(calculate_radiation_acceleration(taa_deg, au_val, v_rad, 0.0, spec_data_mock, False))
    acc_static_mod.append(calculate_radiation_acceleration(taa_deg, au_val, v_rad, 0.0, spec_data_mock, True))

    # 2. 運動原子 (+3km/s)
    acc_move_orig.append(calculate_radiation_acceleration(taa_deg, au_val, v_rad, ATOM_VEL, spec_data_mock, False))
    acc_move_mod.append(calculate_radiation_acceleration(taa_deg, au_val, v_rad, ATOM_VEL, spec_data_mock, True))

# --- プロット描画 ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# 上段: 静止原子
ax1.set_title(f"Case 1: Atom Velocity = 0 m/s (Static on Surface)", fontsize=14)
ax1.plot(taa_list, acc_static_orig, 'b--', label='Original Code (+Vrad)')
ax1.plot(taa_list, acc_static_mod, 'g-', label='Modified Code (-Vrad)', alpha=0.7, linewidth=2)
ax1.set_ylabel('Acceleration ($m/s^2$)')
ax1.legend()
ax1.grid(True, alpha=0.3)
# V_radを右軸に
ax1r = ax1.twinx()
ax1r.plot(taa_list, np.array(v_rad_list) / 1000, 'r:', alpha=0.5, label='Mercury Radial Velocity')
ax1r.set_ylabel('V_rad (km/s)', color='red')
ax1r.text(10, -5, "Note: If spectrum is symmetric,\nstatic atoms might look similar,\nbut physics is inverted.",
          bbox=dict(facecolor='white', alpha=0.8))

# 下段: 運動原子
ax2.set_title(f"Case 2: Atom Velocity = +3.0 km/s (Towards Sun)", fontsize=14)
ax2.plot(taa_list, acc_move_orig, 'b--', label='Original Code (+Vrad)')
ax2.plot(taa_list, acc_move_mod, 'g-', label='Modified Code (-Vrad)', alpha=0.7, linewidth=2)
ax2.set_ylabel('Acceleration ($m/s^2$)')
ax2.set_xlabel('True Anomaly Angle (deg)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()