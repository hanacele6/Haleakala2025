import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. 物理定数・設定
# ==============================================================================
CONSTANTS = {
    'AU_M': 1.496e11,  # 1 AU [m]
    'KB_EV': 8.617e-5,  # Boltzmann定数 [eV/K]
    'MERCURY_SEMI_MAJOR_AU': 0.387098,
    'MERCURY_ECCENTRICITY': 0.205630,
    'U_EV': 1.85,  # 束縛エネルギー (Binding Energy)
}

# --- 温度モデル係数 (以前のコード準拠) ---
# T_subsolar = 100 + 600 * sqrt(0.306 / r_au)
TEMP_BASE = 100.0
TEMP_AMP = 600.0
REF_DIST_AU = 0.306


# ==============================================================================
# 2. 計算関数（半球積分）
# ==============================================================================
def get_orbit_r_au(taa_deg):
    """TAAから水星の太陽距離(AU)を計算"""
    a = CONSTANTS['MERCURY_SEMI_MAJOR_AU']
    e = CONSTANTS['MERCURY_ECCENTRICITY']
    rad = np.deg2rad(taa_deg)
    r = a * (1 - e ** 2) / (1 + e * np.cos(rad))
    return r


def integrate_hemisphere_rates(r_au):
    """
    指定距離における、昼面半球全体の総生成率（相対値）を計算
    """
    # --- A. グリッド生成 (緯度 theta, 経度 phi) ---
    # 昼面のみ: 経度 -90~90, 緯度 -90~90
    # 計算コストと精度のバランスで 50x50 分割程度にする
    num_points = 100
    lat = np.linspace(-np.pi / 2, np.pi / 2, num_points)
    lon = np.linspace(-np.pi / 2, np.pi / 2, num_points)

    # メッシュグリッド作成
    LON, LAT = np.meshgrid(lon, lat)

    # 面積要素 (球面上での重み) dS = cos(lat) d(lat) d(lon)
    # ※定数は最終的に正規化するので無視してよい
    area_weight = np.cos(LAT)

    # 太陽天頂角 (SZA) chi のコサイン
    # cos(chi) = cos(lat) * cos(lon) （Subsolarが (0,0) の場合）
    cos_chi = np.cos(LAT) * np.cos(LON)

    # 昼面のみ計算（cos_chi < 0 は除外＝0にする）
    cos_chi = np.maximum(0, cos_chi)

    # --- B. 温度分布計算 ---
    # 1. Subsolar点温度 (シミュレーションコード準拠)
    scaling = np.sqrt(REF_DIST_AU / r_au)
    t_subsolar = TEMP_BASE + TEMP_AMP * scaling

    # 2. 表面温度分布 T = T_ss * (cos_chi)^0.25
    # 夜側や低温部がエラーにならないよう処理
    local_temp = t_subsolar * (cos_chi ** 0.25)

    # ベース温度（夜間温度）を下回らないようにクリップ（今回は昼面積分なので影響小だが念のため）
    local_temp = np.maximum(100.0, local_temp)

    # --- C. PSD (光刺激脱離) の総量 ---
    # Local Flux ∝ (1/r^2) * cos(chi)
    # Total Rate ∝ (1/r^2) * ∫ cos(chi) dS
    # ※ ∫ cos(chi) dS は「投影面積」なので一定(πR^2)。つまり結局 1/r^2 に比例する。
    total_psd = (1.0 / (r_au ** 2)) * np.sum(cos_chi * area_weight)

    # --- D. TD (熱脱離) の総量 ---
    # Local Flux ∝ exp( -U / kT )
    # Total Rate ∝ ∫ exp( -U / kT_local ) dS
    local_flux_td = np.exp(-CONSTANTS['U_EV'] / (CONSTANTS['KB_EV'] * local_temp))

    # グリッドごとに面積重みを掛けて合計
    total_td = np.sum(local_flux_td * area_weight)

    return total_psd, total_td


# ==============================================================================
# 3. メイン処理
# ==============================================================================
def main_compare_hemisphere_integrated():
    taa_list = np.arange(0, 361, 5)  # 5度刻み

    psd_list = []
    td_list = []
    r_list = []

    print("Calculating hemisphere integration over orbit...")
    for taa in taa_list:
        r = get_orbit_r_au(taa)
        p, t = integrate_hemisphere_rates(r)

        psd_list.append(p)
        td_list.append(t)
        r_list.append(r)

    psd_list = np.array(psd_list)
    td_list = np.array(td_list)

    # --- 正規化 (近日点 TAA=0 で最大になると仮定して 1.0 にする) ---
    psd_norm = psd_list / np.max(psd_list)
    td_norm = td_list / np.max(td_list)

    # --- プロット ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # PSD
    ax.plot(taa_list, psd_norm, color='blue', linestyle='--', linewidth=2,
            label='PSD (Photon-Stimulated)\n$\propto r^{-2}$ (Geometric)')

    # TD
    ax.plot(taa_list, td_norm, color='red', linestyle='-', linewidth=2,
            label=f'TD (Thermal Desorption)\n$U={CONSTANTS["U_EV"]} eV$, Integrated over Hemisphere')

    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 60))
    ax.set_xlabel("True Anomaly Angle (TAA) [deg]", fontsize=14)
    ax.set_ylabel("Total Source Rate (Normalized)", fontsize=14)
    #ax.set_title(f"Mercury Exosphere Source Variation (Hemisphere Integrated)\nU = {CONSTANTS['U_EV']} eV", fontsize=14)

    ax.grid(True, linestyle='--', alpha=0.6)
    #ax.legend(fontsize=11)

    # 補助線: 遠日点
    #ax.axvline(180, color='gray', linestyle=':', alpha=0.5)
    #ax.text(180, 0.5, ' Aphelion', color='gray', va='center')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main_compare_hemisphere_integrated()