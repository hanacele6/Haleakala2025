import numpy as np
import matplotlib.pyplot as plt

# --- 定数 ---
GM = 2.2032e13
RM = 2.440e6
MASS_NA = 3.8175e-26
KB = 1.380649e-23
EV_TO_JOULE = 1.602e-19
G_SURF = GM / (RM ** 2)

# 軌道
PERIHELION_AU = 0.307
APHELION_AU = 0.467


# =============================================================================
# 物理モデル関数
# =============================================================================

def get_temp_at_sza(sza_deg, au):
    """SZAと距離AUから表面温度を計算"""
    T_BASE = 100.0
    T_AMP = 600.0
    scaling = np.sqrt(0.306 / au)

    rad = np.deg2rad(sza_deg)
    cos_theta = np.cos(rad)
    if cos_theta <= 0: return T_BASE
    return T_BASE + T_AMP * (cos_theta ** 0.25) * scaling


def calculate_sticking_prob(temp_k):
    """付着確率"""
    if temp_k <= 0: return 1.0
    A = 0.0804
    B = 458.0
    porosity = 0.8
    p = A * np.exp(B / temp_k)
    peff = p / (1.0 - (1.0 - p) * porosity)
    return min(peff, 1.0)


def calculate_td_flux_simple(temp_k):
    """加重平均計算用の簡易フラックス"""
    if temp_k < 100: return 0.0
    U_eV = 1.85
    return np.exp(- (U_eV * EV_TO_JOULE) / (KB * temp_k))


def calculate_weighted_avg_temp(au, n_steps=1000):
    """Global解析用の「放出粒子平均温度」を計算"""
    theta = np.linspace(0, np.pi / 2, n_steps)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    T_BASE = 100.0
    T_AMP = 600.0
    scaling = np.sqrt(0.306 / au)
    temps = T_BASE + T_AMP * (cos_theta ** 0.25) * scaling

    fluxes = np.array([calculate_td_flux_simple(t) for t in temps])

    weighted_sum = np.sum(temps * fluxes * sin_theta)
    total_weight = np.sum(fluxes * sin_theta)

    if total_weight == 0: return 100.0
    return weighted_sum / total_weight


# =============================================================================
# シミュレーション関数
# =============================================================================

def simulate_particles(temp_k, n_samples=5000):
    """
    指定温度での粒子シミュレーション
    戻り値: (総寿命リスト, 単発寿命リスト, 付着確率)
    """
    p_stick = calculate_sticking_prob(temp_k)
    kT = KB * temp_k

    total_lifetimes = []
    single_lifetimes = []

    for _ in range(n_samples):
        t_total = 0.0
        t_first = 0.0

        # バウンドループ
        for hop_count in range(1000):
            # 初速度分布
            E = np.random.gamma(2.0, kT)
            v = np.sqrt(2 * E / MASS_NA)
            # 角度分布
            u = np.random.random()
            vz = v * np.sqrt(1.0 - u)

            t_flight = 2.0 * vz / G_SURF

            # 最初のジャンプ時間を記録
            if hop_count == 0:
                t_first = t_flight

            t_total += t_flight

            # 付着判定
            if np.random.random() < p_stick:
                break

        total_lifetimes.append(t_total)
        single_lifetimes.append(t_first)

    return np.array(total_lifetimes), np.array(single_lifetimes), p_stick


def simulate_particles_range(au, sza_min, sza_max, n_samples=5000):
    """
    SZA範囲内での粒子シミュレーション
    戻り値: (総寿命リスト, 単発寿命リスト, 代表温度)
    """
    # 範囲内からランダムサンプリング
    sza_samples = np.random.uniform(sza_min, sza_max, n_samples)

    total_lifetimes = []
    single_lifetimes = []
    avg_temp_accum = 0.0

    for sza in sza_samples:
        temp_k = get_temp_at_sza(sza, au)
        avg_temp_accum += temp_k
        p_stick = calculate_sticking_prob(temp_k)
        kT = KB * temp_k

        t_total = 0.0
        t_first = 0.0

        for hop_count in range(1000):
            E = np.random.gamma(2.0, kT)
            v = np.sqrt(2 * E / MASS_NA)
            u = np.random.random()
            vz = v * np.sqrt(1.0 - u)

            t_flight = 2.0 * vz / G_SURF

            if hop_count == 0: t_first = t_flight
            t_total += t_flight

            if np.random.random() < p_stick:
                break

        total_lifetimes.append(t_total)
        single_lifetimes.append(t_first)

    mean_temp = avg_temp_accum / n_samples
    # この範囲での平均的な付着確率（表示用）
    p_stick_mean = calculate_sticking_prob(mean_temp)

    return np.array(total_lifetimes), np.array(single_lifetimes), mean_temp, p_stick_mean


# =============================================================================
# 描画用関数 (1つのウィンドウに左右の図を描画)
# =============================================================================

def plot_analysis_window(title_main,
                         peri_data, aphe_data,
                         dt_range=np.linspace(10, 1500, 150)):
    """
    ウィンドウ生成関数
    peri_data = (total_lt, single_lt, temp, p_stick)
    aphe_data = (total_lt, single_lt, temp, p_stick)
    """
    # アンパック
    pt_total, pt_single, pt_temp, pt_p = peri_data
    at_total, at_single, at_temp, at_p = aphe_data

    # ウィンドウ作成 (左右2枚)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title_main, fontsize=16)

    # --- 左図: 落下率 vs DT ---
    # 近日点 (Red)
    sorted_pt = np.sort(pt_total)
    y_pt = np.arange(len(sorted_pt)) / len(sorted_pt)
    ax1.plot(dt_range, np.interp(dt_range, sorted_pt, y_pt),
             'r-', linewidth=2, label=f'Peri Total (p={pt_p:.2f})')

    sorted_ps = np.sort(pt_single)
    y_ps = np.arange(len(sorted_ps)) / len(sorted_ps)
    ax1.plot(dt_range, np.interp(dt_range, sorted_ps, y_ps),
             'r:', alpha=0.6, label='Peri Single')

    # 遠日点 (Blue)
    sorted_at = np.sort(at_total)
    y_at = np.arange(len(sorted_at)) / len(sorted_at)
    ax1.plot(dt_range, np.interp(dt_range, sorted_at, y_at),
             'b-', linewidth=2, label=f'Aphe Total (p={at_p:.2f})')

    sorted_as = np.sort(at_single)
    y_as = np.arange(len(sorted_as)) / len(sorted_as)
    ax1.plot(dt_range, np.interp(dt_range, sorted_as, y_as),
             'b:', alpha=0.6, label='Aphe Single')

    ax1.axvline(500, color='k', linestyle='--', label='DT=500s')
    ax1.set_title("Return Rate vs Time Step")
    ax1.set_xlabel("Time Step dt [s]")
    ax1.set_ylabel("Fraction Returning within dt")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, 1500)
    ax1.set_ylim(0, 1.05)

    # --- 右図: 寿命分布 (ヒストグラム) ---
    bins = np.linspace(0, 2000, 50)

    ax2.hist(pt_single, bins=bins, density=True, color='red', alpha=0.1, label='Peri Single')
    ax2.hist(pt_total, bins=bins, density=True, histtype='step', color='red', linewidth=2,
             label=f'Peri Total ({pt_temp:.0f}K)')

    ax2.hist(at_single, bins=bins, density=True, color='blue', alpha=0.1, label='Aphe Single')
    ax2.hist(at_total, bins=bins, density=True, histtype='step', color='blue', linewidth=2,
             label=f'Aphe Total ({at_temp:.0f}K)')

    ax2.axvline(500, color='k', linestyle='--', label='DT=500s')

    # 500s以内の割合テキスト
    ratio_p = np.sum(pt_total < 500) / len(pt_total) * 100
    ratio_a = np.sum(at_total < 500) / len(at_total) * 100
    stats = f"<500s:\nPeri: {ratio_p:.1f}%\nAphe: {ratio_a:.1f}%"
    ax2.text(0.95, 0.6, stats, transform=ax2.transAxes, ha='right',
             bbox=dict(facecolor='white', alpha=0.8))

    ax2.set_title("Lifetime Distribution")
    ax2.set_xlabel("Lifetime [s]")
    ax2.set_ylabel("Probability Density")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    # plt.show() はループの最後、または個別に呼び出す


# =============================================================================
# メイン実行
# =============================================================================

# --- 1. Global Window (Flux Weighted Average) ---
print("Generating Window 1: Global Average...")
T_peri_avg = calculate_weighted_avg_temp(PERIHELION_AU)
T_aphe_avg = calculate_weighted_avg_temp(APHELION_AU)

# 固定温度シミュレーション
# --- 1. Global Window (Flux Weighted Average) ---
print("Generating Window 1: Global Average...")
T_peri_avg = calculate_weighted_avg_temp(PERIHELION_AU)
T_aphe_avg = calculate_weighted_avg_temp(APHELION_AU)

# 固定温度シミュレーションの結果を取得
# simulate_particles は (total_lt, single_lt, p_stick) を返す
r_peri = simulate_particles(T_peri_avg)
r_aphe = simulate_particles(T_aphe_avg)

# plot関数が期待する (total, single, temp, p_stick) の順序に並べ替え
d_peri_glob = (r_peri[0], r_peri[1], T_peri_avg, r_peri[2])
d_aphe_glob = (r_aphe[0], r_aphe[1], T_aphe_avg, r_aphe[2])

plot_analysis_window("Window 1: Global Average (Flux-Weighted)", d_peri_glob, d_aphe_glob)

plot_analysis_window("Window 1: Global Average (Flux-Weighted)", d_peri_glob, d_aphe_glob)

# --- 2~6. SZA Range Windows ---
sza_ranges = [
    (0, 20),
    (20, 40),
    (40, 60),
    (60, 80),
    (80, 90)
]

for i, (sza_min, sza_max) in enumerate(sza_ranges):
    win_num = i + 2
    print(f"Generating Window {win_num}: SZA {sza_min}-{sza_max}...")

    # 範囲シミュレーション
    # simulate_particles_range の戻り値は (total, single, temp, p_stick) になっているのでそのまま展開
    d_peri_sza = simulate_particles_range(PERIHELION_AU, sza_min, sza_max)
    d_aphe_sza = simulate_particles_range(APHELION_AU, sza_min, sza_max)

    title = f"Window {win_num}: SZA Range {sza_min}° - {sza_max}°"
    plot_analysis_window(title, d_peri_sza, d_aphe_sza)

# 最後にまとめて表示
plt.show()