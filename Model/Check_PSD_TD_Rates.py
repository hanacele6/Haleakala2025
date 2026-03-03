import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. 物理定数・設定
# ==============================================================================
CONSTANTS = {
    'AU_M': 1.496e11,
    'KB_EV': 8.617e-5,  # Boltzmann定数 [eV/K]
    'MERCURY_SEMI_MAJOR_AU': 0.387098,
    'MERCURY_ECCENTRICITY': 0.205630,
}

# --- 温度モデル係数 ---
TEMP_BASE = 100.0
TEMP_AMP = 600.0
REF_DIST_AU = 0.306


# ==============================================================================
# 2. 計算関数
# ==============================================================================
def get_orbit_r_au(taa_deg):
    """TAAから水星の太陽距離(AU)を計算"""
    a = CONSTANTS['MERCURY_SEMI_MAJOR_AU']
    e = CONSTANTS['MERCURY_ECCENTRICITY']
    rad = np.deg2rad(taa_deg)
    r = a * (1 - e ** 2) / (1 + e * np.cos(rad))
    return r


def integrate_hemisphere_rates(r_au, u_ev):
    """
    指定距離・指定Binding Energyにおける、昼面半球全体の総生成率（相対値）を計算
    """
    # --- A. グリッド生成 (50x50) ---
    num_points = 50
    lat = np.linspace(-np.pi / 2, np.pi / 2, num_points)
    lon = np.linspace(-np.pi / 2, np.pi / 2, num_points)
    LON, LAT = np.meshgrid(lon, lat)

    # 面積要素 dS
    area_weight = np.cos(LAT)

    # 太陽天頂角 (SZA) chi のコサイン
    cos_chi = np.cos(LAT) * np.cos(LON)
    cos_chi = np.maximum(0, cos_chi)  # 昼面のみ

    # --- B. 温度分布計算 ---
    scaling = np.sqrt(REF_DIST_AU / r_au)
    t_subsolar = TEMP_BASE + TEMP_AMP * scaling
    local_temp = t_subsolar * (cos_chi ** 0.25)
    local_temp = np.maximum(100.0, local_temp)

    # --- C. PSD (光刺激脱離) の総量 ---
    # 単純に 1/r^2 に比例 (幾何学的変動のみ見るため、積分値は簡易計算でも傾向は同じだが一応積分)
    total_psd = (1.0 / (r_au ** 2)) * np.sum(cos_chi * area_weight)

    # --- D. TD (熱脱離) の総量 ---
    # U を引数から使用
    local_flux_td = np.exp(-u_ev / (CONSTANTS['KB_EV'] * local_temp))
    total_td = np.sum(local_flux_td * area_weight)

    return total_psd, total_td


# ==============================================================================
# 3. メイン処理
# ==============================================================================
def main_compare_u_impact():
    taa_list = np.arange(0, 361, 5)
    r_list = [get_orbit_r_au(taa) for taa in taa_list]

    # 比較したい Binding Energy のリスト (eV)
    # 1.85 eV: 典型的なNaの脱離エネルギー想定
    # 2.70 eV: より強く結合している場合 (岩石質に近い等)
    u_list = [1.85, 2.7]

    # 結果格納用
    psd_results = []
    td_results = {u: [] for u in u_list}

    print("Calculating orbit variations...")

    for r in r_list:
        # PSDはUに依存しないので、代表して u_list[0] を渡すが結果は同じ
        # TDは計算コストが高いので、ここでループ内で U を変えて計算

        # まずPSD計算 (Uはダミー)
        p, _ = integrate_hemisphere_rates(r, u_ev=1.0)
        psd_results.append(p)

        # 各UについてTD計算
        for u in u_list:
            _, t = integrate_hemisphere_rates(r, u_ev=u)
            td_results[u].append(t)

    # numpy配列化 & 正規化
    psd_arr = np.array(psd_results)
    psd_norm = psd_arr / np.max(psd_arr)

    # --- プロット ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. PSD Plot
    ax.plot(taa_list, psd_norm, color='black', linestyle='--', linewidth=2.5, alpha=0.7,
            label='PSD ($\propto r^{-2}$)')

    # 2. TD Plots (Loop)
    colors = ['tab:red', 'tab:blue', 'tab:green']  # 必要なら色を増やす
    for i, u in enumerate(u_list):
        td_arr = np.array(td_results[u])
        td_norm = td_arr / np.max(td_arr)  # 各自の最大値で正規化して形状比較

        c = colors[i % len(colors)]
        ax.plot(taa_list, td_norm, color=c, linewidth=2,
                label=f'TD ($U={u}$ eV)')

    # 装飾
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 60))
    ax.set_xlabel("True Anomaly Angle (TAA) [deg]", fontsize=14)
    ax.set_ylabel("Normalized Source Rate", fontsize=14)
    #ax.set_title("Orbital Variation of Mercury's Source Rates\nEffect of Binding Energy $U$ on Thermal Desorption",
    #             fontsize=15)

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=12, loc='upper right')

    # 補助情報
    #ax.text(180, 0.05, 'Aphelion (Far)', ha='center', color='gray')
    #ax.text(0, 0.05, 'Perihelion (Near)', ha='left', color='gray')
    #ax.text(360, 0.05, 'Perihelion', ha='right', color='gray')

    ax.set_yscale('log')

    # 対数グラフだと0は描画できないので、下限を少し設定しておくと見やすいです
    ax.set_ylim(1e-4, 1.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main_compare_u_impact()