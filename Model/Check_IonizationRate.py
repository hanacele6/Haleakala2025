import numpy as np


def verify_ion_recycling_impact():
    """
    イオンリサイクル率を変えた時に、表面密度（粒子の寿命）が
    何倍に伸びるかをモンテカルロ法で簡易計算する。
    """
    # --- パラメータ設定 ---
    N_SAMPLES = 100000  # 試行粒子数
    TEMP_TD = 500.0  # TDの代表温度 [K]
    RECYCLING_RATE = 0.15  # ★ここを変更して試す (例: 0.15 = 15%が戻ってくる)

    # 物理定数
    MASS_NA = 3.8175e-26
    KB = 1.380649e-23
    GM = 2.2032e13
    RM = 2.440e6
    T1AU = 54500.0  # 光電離寿命 [s] (近日点補正なしの1AU値で概算)
    AU_FACTOR = 0.387  # 水星の軌道長半径 (AU)

    # 水星位置での寿命
    TAU_ION = T1AU * (AU_FACTOR ** 2)
    G_SURF = GM / (RM ** 2)
    V_ESC = np.sqrt(2 * GM / RM)

    print(f"--- Ion Recycling Impact Verification ---")
    print(f"Temperature: {TEMP_TD} K")
    print(f"Ionization Tau: {TAU_ION:.1f} s")
    print(f"Assumed Recycling Rate: {RECYCLING_RATE * 100:.1f} %")
    print("-" * 40)

    # 速度分布生成 (Maxwellian)
    # E ~ Gamma(2, kT) -> v = sqrt(2E/m)
    E = np.random.gamma(2.0, KB * TEMP_TD, N_SAMPLES)
    v_mag = np.sqrt(2 * E / MASS_NA)

    # 角度分布 (Lambertian) -> 垂直速度 vz = v * cos(theta), cos(theta)=sqrt(1-u)
    # 平均的な滞空時間を知りたいので簡易的に垂直成分だけ見る
    u = np.random.random(N_SAMPLES)
    cos_theta = np.sqrt(1 - u)
    v_z = v_mag * cos_theta

    # --- 1ステップあたりの確率計算 ---

    # A. 脱出確率 (P_escape)
    # TD(500K)ではほぼ0だが、計算上含める
    is_escape = v_mag >= V_ESC
    p_escape_avg = np.mean(is_escape)

    # B. イオン化確率 (P_ion)
    # 滞空時間 t_flight = 2 * vz / g
    # イオン化確率 = 1 - exp(-t / tau)
    # 脱出する粒子は無限時間飛ぶので100%イオン化するが、Loss判定としては「脱出」が先。
    # ここでは「脱出しなかった粒子」の滞空時間中のイオン化を考える。

    t_flight = 2.0 * v_z / G_SURF
    # 脱出するやつの時間は無視（Loss判定済みだから）
    p_ion_per_bounce = 1.0 - np.exp(-t_flight / TAU_ION)

    # 脱出しなかった粒子群の中での平均イオン化確率
    p_ion_avg = np.mean(p_ion_per_bounce[~is_escape])

    # --- 総合ロス率の比較 ---

    # 1. 現状 (No Recycling)
    # Loss = Escape + (Non-Escape * Ionization)
    loss_rate_current = p_escape_avg + (1.0 - p_escape_avg) * p_ion_avg

    # 2. 修正後 (With Recycling)
    # イオン化したうちの R% は戻ってくる = Lossにならない
    # Effective Ion Loss = Ionization * (1 - R)
    loss_rate_new = p_escape_avg + (1.0 - p_escape_avg) * (p_ion_avg * (1.0 - RECYCLING_RATE))

    # --- 結果算出 ---
    # 密度(寿命)は 1/Loss に比例する
    density_factor = loss_rate_current / loss_rate_new

    print(f"Escape Probability (per hop):   {p_escape_avg:.6f}")
    print(f"Ionization Prob (per hop):      {p_ion_avg:.6f}")
    print(f"----------------------------------------")
    print(f"Loss Rate (Current):            {loss_rate_current:.6f}")
    print(f"Loss Rate (With Recycling):     {loss_rate_new:.6f}")
    print(f"----------------------------------------")
    print(f"Expected Density Increase:      x {density_factor:.3f} times")

    # 簡易理論値: 脱出が0なら Gain = 1/(1-R)
    theoretical_max = 1.0 / (1.0 - RECYCLING_RATE)
    print(f"(Theoretical Max if Escape=0:   x {theoretical_max:.3f} times)")


if __name__ == "__main__":
    verify_ion_recycling_impact()