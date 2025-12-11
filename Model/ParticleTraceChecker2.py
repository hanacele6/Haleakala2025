import numpy as np
import matplotlib.pyplot as plt

# --- 定数 ---
GM = 2.2032e13
RM = 2.440e6
MASS_NA = 3.8175e-26
KB = 1.380649e-23
EV_TO_JOULE = 1.602e-19
G_SURF = GM / (RM ** 2)  # 重力加速度 ~3.7 m/s^2


# --- 物理モデル関数 ---

def calculate_sticking_probability(surface_temp_K):
    """シミュレーションコードと同じ付着確率モデル"""
    A = 0.0804
    B = 458.0
    porosity = 0.8
    if surface_temp_K <= 0: return 1.0
    p_stick = A * np.exp(B / surface_temp_K)
    p_stick_eff = p_stick / (1.0 - (1.0 - p_stick) * porosity)
    return min(p_stick_eff, 1.0)


def get_flight_times_td_simulation_logic(temp_k, n_samples=100000):
    """
    シミュレーションコードのロジックを完全に再現したTD粒子の滞空時間計算

    Logic form simulation code:
      1. E = np.random.gamma(2.0, kT)
      2. v = sqrt(2*E/m)
      3. cos_theta = sqrt(1 - u)  (Lambertian)
      4. vz = v * cos_theta
    """
    kT = KB * temp_k

    # 1. エネルギー分布 (Gamma 2.0)
    E = np.random.gamma(2.0, kT, n_samples)

    # 2. 速度の大きさ
    v_mag = np.sqrt(2.0 * E / MASS_NA)

    # 3. 放出角度 (ランベルト則: cos_theta = sqrt(1-u))
    # シミュレーションコード: cos_theta = np.sqrt(1 - u2)
    u2 = np.random.random(n_samples)
    cos_theta = np.sqrt(1.0 - u2)

    # 4. 垂直速度成分
    vz = v_mag * cos_theta

    # 滞空時間 t = 2 * vz / g
    flight_times = 2.0 * vz / G_SURF
    return flight_times


def get_total_lifetime_with_bouncing(temp_k, n_samples=100000):
    """バウンドを含めた【総寿命】の分布"""
    p_stick = calculate_sticking_probability(temp_k)
    total_lifetimes = np.zeros(n_samples)
    active_indices = np.arange(n_samples)

    while len(active_indices) > 0:
        # シミュレーションと全く同じロジックで飛行時間を生成
        current_flight_times = get_flight_times_td_simulation_logic(temp_k, len(active_indices))
        total_lifetimes[active_indices] += current_flight_times

        r = np.random.random(len(active_indices))
        bouncing_mask = r >= p_stick
        active_indices = active_indices[bouncing_mask]

        if len(active_indices) < n_samples * 0.0001 and len(active_indices) < 10:
            break

    return total_lifetimes, p_stick


def get_flight_times_sws(u_ev=0.27, n_samples=100000):
    """SWS (変更なし)"""
    energies = []
    while len(energies) < n_samples:
        needed = n_samples - len(energies)
        E_try = np.random.uniform(0, 10.0, needed * 2)
        f_max = (u_ev / 2.0) / (u_ev / 2.0 + u_ev) ** 3
        f_val = E_try / (E_try + u_ev) ** 3

        accept = np.random.random(len(E_try)) * f_max <= f_val
        energies.extend(E_try[accept])

    energies = np.array(energies[:n_samples])
    v_mag = np.sqrt(2 * energies * EV_TO_JOULE / MASS_NA)
    cos_theta = np.random.random(n_samples)
    vz = v_mag * cos_theta
    flight_times = 2.0 * vz / G_SURF
    return flight_times


# === メイン処理 ===
temps = [100, 575, 700]
dt_range = np.linspace(10, 1000, 100)

plt.figure(figsize=(14, 6))

# --- グラフ1: タイムステップ vs 落下率 (バウンド考慮) ---
plt.subplot(1, 2, 1)

colors = {100: 'blue', 575: 'green', 700: 'orange'}

for T in temps:
    # 1. 単発飛行 (Solid Line)
    ft_single = get_flight_times_td_simulation_logic(T)

    sorted_ft = np.sort(ft_single)
    y_vals = np.arange(len(sorted_ft)) / len(sorted_ft)
    ratios_single = np.interp(dt_range, sorted_ft, y_vals)

    plt.plot(dt_range, ratios_single, color=colors[T], alpha=0.5, linestyle='-', label=f'TD {T}K (Single Hop)')

    # 2. バウンド込み寿命 (Dashed Line)
    if T > 100:
        ft_total, p_stick = get_total_lifetime_with_bouncing(T)

        sorted_total = np.sort(ft_total)
        y_vals_total = np.arange(len(sorted_total)) / len(sorted_total)
        ratios_total = np.interp(dt_range, sorted_total, y_vals_total)

        plt.plot(dt_range, ratios_total, color=colors[T], linewidth=2, linestyle='--',
                 label=f'TD {T}K (Bounce, P_stick={p_stick:.2f})')

# SWS (比較用)
ft_sws = get_flight_times_sws()
sorted_sws = np.sort(ft_sws)
y_vals_sws = np.arange(len(sorted_sws)) / len(sorted_sws)
ratios_sws = np.interp(dt_range, sorted_sws, y_vals_sws)

plt.plot(dt_range, ratios_sws, label='SWS (Sputtering)', linestyle=':', color='black', linewidth=2)

# 500s ライン
plt.axvline(500, color='red', alpha=0.5, linestyle=':')
plt.text(520, 0.3, 'Target DT=500s', color='red', fontsize=12)

plt.title("Effect of Bouncing on Return Rate (< DT) [Simulation Logic]")
plt.xlabel("Time Step [s]")
plt.ylabel("Returning Rate")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=9)

# --- グラフ2: 寿命のヒストグラム (分布の変化) ---
plt.subplot(1, 2, 2)

# T=575K
ft_575_single = get_flight_times_td_simulation_logic(575)
ft_575_total, ps575 = get_total_lifetime_with_bouncing(575)

plt.hist(ft_575_single, bins=50, range=(0, 1500), density=True, alpha=0.3, color='green', label='TD 575K (Single)')
plt.hist(ft_575_total, bins=50, range=(0, 1500), density=True, histtype='step', linewidth=2, color='green',
         label=f'TD 575K (Bounce, Mean={np.mean(ft_575_total):.0f}s)')

# T=700K
ft_700_single = get_flight_times_td_simulation_logic(700)
ft_700_total, ps700 = get_total_lifetime_with_bouncing(700)

plt.hist(ft_700_single, bins=50, range=(0, 1500), density=True, alpha=0.3, color='orange', label='TD 700K (Single)')
plt.hist(ft_700_total, bins=50, range=(0, 1500), density=True, histtype='step', linewidth=2, color='orange',
         label=f'TD 700K (Bounce, Mean={np.mean(ft_700_total):.0f}s)')

# 500s リミット線
plt.axvline(500, color='red', linestyle='--', label='500s Limit')

plt.title("Lifetime Distribution Shift [Simulation Logic]")
plt.xlabel("Total Lifetime [s]")
plt.ylabel("Probability Density")
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()