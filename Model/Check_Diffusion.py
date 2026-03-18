import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# パラメータ設定
# ==========================================
KB_EV_CONST = 8.617e-5
DIFF_REF_FLUX = 2.0e7 * (100.0 ** 2)
DIFF_REF_TEMP = 700.0
T_BASE = 100.0
T_AMP = 600.0

EA_OLD = 0.4
EA_NEW = 0.65

# --- Sizmannモデル(Proposed)用の物理パラメータ ---
EA_TH = 0.95  # Sizmann 熱拡散 (TD)
EA_RED = 0.48  # Sizmann イオン誘起拡散 (RED: 0.5 * Hv^M)

KAPPA = 4.0

# --- SWS (太陽風スパッタリング/イオン供給) パラメータ ---
SWS_FLUX_1AU = 10.0 * (100 ** 3) * 400e3 * 4  # 約 1.6e13 [ions/m^2/s]
SWS_YIELD = 0.06
SWS_SOURCE_1AU = SWS_FLUX_1AU * SWS_YIELD  # 1AUでのベース供給量 (約 9.6e11)

R_PERI_AU = 0.387098 * (1 - 0.205630)


# ==========================================
# 物理計算関数
# ==========================================
def calc_r_au(taa_deg):
    rad = np.deg2rad(taa_deg)
    a, e = 0.387098, 0.205630
    return a * (1 - e ** 2) / (1 + e * np.cos(rad))


def calc_temp(sza_rad, r_au):
    eff_cos = max(0.0, np.cos(sza_rad))
    scaling = np.sqrt(0.306 / r_au)
    return T_BASE + T_AMP * (eff_cos ** 0.25) * scaling


def get_prefactor(ea_ev):
    return DIFF_REF_FLUX / np.exp(-ea_ev / (KB_EV_CONST * DIFF_REF_TEMP))


def calc_thermal_flux(temp, ea_ev, prefactor):
    if temp < 100.0: return 1e-30
    return prefactor * np.exp(-ea_ev / (KB_EV_CONST * temp))


def is_sw_region(lon_deg, lat_deg):
    in_lon = -40 <= lon_deg <= 40
    in_lat = (20 <= lat_deg <= 80) or (-80 <= lat_deg <= -20)
    return in_lon and in_lat


# ==========================================
# 1. TAA vs 全球供給量
# ==========================================
pref_old = get_prefactor(EA_OLD)
pref_new = get_prefactor(EA_NEW)
pref_th = get_prefactor(EA_TH)

taas = np.arange(0, 360, 10)
global_old, global_new = [], []
global_burger, global_prop = [], []

for taa in taas:
    r_au = calc_r_au(taa)

    # 既存のBurgerモデル用スケーリング
    sw_flux_scale = (R_PERI_AU / r_au) ** 2

    # 【追加】RED用の局所太陽風ソース (1/r^2 依存)
    sws_source_local = SWS_SOURCE_1AU / (r_au ** 2)

    tot_old, tot_new, tot_b, tot_p = 0, 0, 0, 0
    for lon in range(-180, 180, 5):
        for lat in range(-90, 90, 5):
            lon_rad, lat_rad = np.deg2rad(lon), np.deg2rad(lat)
            cos_sza = np.cos(lat_rad) * np.cos(lon_rad)
            sza_rad = np.arccos(np.clip(cos_sza, -1.0, 1.0))

            t = calc_temp(sza_rad, r_au)
            area = np.cos(lat_rad)

            f_old = calc_thermal_flux(t, EA_OLD, pref_old)
            f_new = calc_thermal_flux(t, EA_NEW, pref_new)
            f_th = calc_thermal_flux(t, EA_TH, pref_th)

            sw_flag = is_sw_region(lon, lat)

            # Burger Style (元のまま)
            f_burger = f_new * (1 + KAPPA * sw_flux_scale) if sw_flag else f_new

            # Proposed Style (Sizmann: TD + RED)
            if sw_flag:
                # 太陽風の絶対供給量(sws_source_local)に、Ea=0.48eVの熱活性化項を乗算
                red_term = (sws_source_local * KAPPA) * np.exp(-EA_RED / (KB_EV_CONST * t)) if t >= 100.0 else 1e-30
                f_prop = f_th + red_term
            else:
                f_prop = f_th

            tot_old += f_old * area
            tot_new += f_new * area
            tot_b += f_burger * area
            tot_p += f_prop * area

    global_old.append(tot_old)
    global_new.append(tot_new)
    global_burger.append(tot_b)
    global_prop.append(tot_p)

# === 第1ウィンドウ: 全球供給量 ===
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(taas, global_old, label=f'Old (Ea={EA_OLD}eV)', color='black', linestyle='--')
ax1.plot(taas, global_new, label=f'High Ea (Ea={EA_NEW}eV)', color='gray')
ax1.plot(taas, global_burger, label='Burger Style', color='blue')
ax1.plot(taas, global_prop, label=f'Proposed (Sizmann: TD {EA_TH}eV + RED {EA_RED}eV)', color='red')
ax1.set_yscale('log')
ax1.set_xlabel('True Anomaly Angle (deg)')
ax1.set_ylabel('Relative Global Source Rate')
ax1.set_title('Global Diffusion Source vs TAA')
ax1.legend()
ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.set_ylim(bottom=1e12)
plt.tight_layout()
plt.show()

# ==========================================
# 2. 天頂角(緯度)プロファイル (複数のTAAで比較)
# ==========================================
target_taas = [0, 90, 180, 270]
lats = np.arange(-90, 91, 2)

fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, t_taa in enumerate(target_taas):
    r_au_val = calc_r_au(t_taa)
    sw_flux_scale_val = (R_PERI_AU / r_au_val) ** 2

    # 【追加】RED用の局所太陽風ソース
    sws_source_local_val = SWS_SOURCE_1AU / (r_au_val ** 2)

    p_old, p_new, p_burger, p_prop = [], [], [], []
    for lat in lats:
        lat_rad = np.deg2rad(lat)
        cos_sza = np.cos(lat_rad) * np.cos(0)
        sza_rad = np.arccos(np.clip(cos_sza, -1.0, 1.0))
        t = calc_temp(sza_rad, r_au_val)

        f_old = calc_thermal_flux(t, EA_OLD, pref_old)
        f_new = calc_thermal_flux(t, EA_NEW, pref_new)
        f_th = calc_thermal_flux(t, EA_TH, pref_th)

        sw_flag = is_sw_region(0, lat)

        f_burger = f_new * (1 + KAPPA * sw_flux_scale_val) if sw_flag else f_new

        if sw_flag:
            red_term = (sws_source_local_val * KAPPA) * np.exp(-EA_RED / (KB_EV_CONST * t)) if t >= 100.0 else 1e-30
            f_prop = f_th + red_term
        else:
            f_prop = f_th

        p_old.append(f_old)
        p_new.append(f_new)
        p_burger.append(f_burger)
        p_prop.append(f_prop)

    ax = axes[i]
    ax.plot(lats, p_old, label=f'Old (Ea={EA_OLD}eV)', color='black', linestyle='--')
    ax.plot(lats, p_new, label=f'High Ea (Ea={EA_NEW}eV)', color='gray')
    ax.plot(lats, p_burger, label='Burger Style', color='blue')
    ax.plot(lats, p_prop, label=f'Proposed (TD+RED)', color='red')

    ax.set_yscale('log')
    ax.set_title(f'TAA={t_taa}° (r={r_au_val:.3f} AU)')
    ax.set_xlabel('Latitude (deg) [Subsolar line]')
    ax.set_ylabel('Local Flux')

    ax.axvspan(20, 80, color='yellow', alpha=0.2, label='SW Influx Region' if i == 0 else "")
    ax.axvspan(-80, -20, color='yellow', alpha=0.2)

    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.set_ylim(bottom=1e7)

    if i == 0:
        ax.legend(loc='lower center', fontsize=9)
plt.tight_layout(h_pad=4.0, w_pad=2.0)
plt.show()