import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


def calculate_mercury_mmv_rates():
    """
    鈴木 et al. (2020) および Leblanc & Johnson (2003) に基づく
    水星のMMV（微小隕石衝突気化）によるNa放出量の絶対値計算コード
    """

    # --- 1. 定数とパラメータ設定 ---
    AU_M = 1.495978707e11  # 1 AU [m]
    GM_SUN = 1.3271244e20  # 太陽重力定数 [m^3/s^2]
    R_MERCURY = 2440e3  # 水星半径 [m]
    M_NA_KG = 3.817e-26  # Na原子の質量 [kg] (約23 u)

    # レゴリス中のNa含有率 (Leblanc 2003 [861] を参照, Suzuki [277] もこれに準拠)
    C_NA = 0.0053

    # 水星の軌道要素
    A_AU = 0.387098  # 軌道長半径 [AU]
    ECC = 0.205630  # 離心率

    # 真近点角 (TAA) の配列 0〜360度
    taa_deg = np.linspace(0, 360, 361)
    taa_rad = np.radians(taa_deg)

    # 太陽からの距離 R [AU] および [m]
    r_au = A_AU * (1 - ECC ** 2) / (1 + ECC * np.cos(taa_rad))
    r_m = r_au * AU_M

    # 水星の公転速度 V0 [m/s] (Suzuki Eq.8 の V0)
    # Vis-vivaの式: v = sqrt(GM * (2/r - 1/a))
    v0 = np.sqrt(GM_SUN * (2 / r_m - 1 / (A_AU * AU_M)))

    # --- 2. Suzuki et al. (2020) モデルの計算 ---

    # ダスト密度のパラメータ (Suzuki [296-297])
    # j=1: Jupiter Family Comets / Asteroids
    # j=2: Halley Type Comets
    # j=3: Oort Cloud Comets / Interstellar (等方性)

    f = [0.45, 0.50, 0.05]  # 割合 f_j
    chi = [1.00, 1.45, 2.00]  # 距離依存性の指数 chi_j
    sigma_deg = [7.0, 33.0, 0]  # 傾斜角分布の広がり sigma_j (j=3は定義なし)
    c_const = [10.3, 2.19, 0]  # 正規化定数 c_j (j=3は定義なし)

    # ダスト数密度 n_mm(R) の計算 [Suzuki Eq.9 & 10]
    # n_mm = sum( f_j * R^-chi_j * Integral ) * 10^-4
    # ここでは黄道面 (beta=0) を仮定して積分します。
    # beta=0 の場合、分母 sqrt(sin^2 i - sin^2 beta) は sin i となり、
    # 被積分関数は h_j(i) / sin i となります。

    # 積分の計算 (Rに依存しない項なので先に計算)
    integrals = []

    # j=1 (JFC)
    s1_rad = np.radians(sigma_deg[0])

    # h1(i) = c1 * exp(-i^2 / 2sigma^2) * sin(i)
    # beta=0 で割ると sin(i) が消えるため、被積分関数は exp(...) のみになる
    def integrand1(i_rad):
        return c_const[0] * np.exp(-(i_rad) ** 2 / (2 * s1_rad ** 2))

    val1, _ = quad(integrand1, 0, np.pi)
    integrals.append(val1)

    # j=2 (HTC)
    s2_rad = np.radians(sigma_deg[1])

    def integrand2(i_rad):
        return c_const[1] * np.exp(-(i_rad) ** 2 / (2 * s2_rad ** 2))

    val2, _ = quad(integrand2, 0, np.pi)
    integrals.append(val2)

    # j=3 (OCC) 等方性
    # h3(i) = 0.5 * sin(i)
    # beta=0 で割ると 0.5 となる。0からpiまで積分すると 0.5 * pi
    integrals.append(0.5 * np.pi)

    # 各距離 R での密度 n_mm [m^-3] を計算
    n_mm = np.zeros_like(r_au)
    for j in range(3):
        # 論文 Eq.9: R^-chi * Integral * 10^-4
        # (注: Killen & Hahn 2015等に基づき、ここでのRはAU単位、10^-4でm^-3に変換と解釈)
        term = f[j] * (r_au ** -chi[j]) * integrals[j]
        n_mm += term
    n_mm *= 1.0e-4

    # 平均蒸発質量 M_vapor [kg] (Suzuki [276])
    # "7e-15 * (R / 1au)"
    m_vapor = 7.0e-15 * r_au

    # 総放出率 Rate [atoms/s] (Suzuki Eq.8 に基づく)
    # Rate = (衝突フラックス [events/m^2/s]) * (断面積 [m^2]) * (放出原子数/event)
    # 衝突フラックス = n_mm * V0
    # 放出原子数/event = M_vapor * C_NA / M_NA_KG

    flux_events = n_mm * v0
    cross_section = np.pi * R_MERCURY ** 2
    atoms_per_event = m_vapor * C_NA / M_NA_KG

    rate_suzuki = flux_events * cross_section * atoms_per_event

    # --- 3. Leblanc & Johnson (2003) モデルの計算 ---
    # 近日点 (TAA=0) で 5.0e23 atoms/s, 距離の-1.9乗に比例 [Leblanc p.268]
    r_peri_au = A_AU * (1 - ECC)
    rate_leblanc = 5.0e23 * (r_peri_au / r_au) ** 1.9

    # --- 4. プロット ---
    plt.figure(figsize=(10, 6))

    plt.plot(taa_deg, rate_suzuki, label='Suzuki et al. (2020) Model',
             color='blue', linewidth=2)
    plt.plot(taa_deg, rate_leblanc, label='Leblanc (2003) Model',
             color='red', linewidth=2, linestyle='--')

    #plt.title('Mercury Na Exosphere Production Rate via MMV (Absolute Value)', fontsize=14)
    plt.xlabel('True Anomaly Angle (TAA) [deg]', fontsize=16)
    plt.ylabel('Production Rate [atoms/s]', fontsize=16)
    plt.yscale('log')  # 桁が変わるため対数軸推奨
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.legend(fontsize=11)

    # 軸の設定変更箇所
    plt.xlim(0, 360)                      # x軸の範囲を0から360に固定
    plt.xticks(np.arange(0, 361, 60), fontsize=12)    # 60度刻みのメモリを設定

    # 近日点・遠日点の補助線
    #plt.axvline(x=0, color='k', linestyle=':', alpha=0.6)
    #plt.text(5, min(rate_suzuki) * 1.1, 'Perihelion', fontsize=10)
    #plt.axvline(x=180, color='k', linestyle=':', alpha=0.6)
    #plt.text(185, min(rate_suzuki) * 1.1, 'Aphelion', fontsize=10)

    plt.tight_layout()
    plt.show()

    # 数値の確認用出力
    print(f"Suzuki Model at Perihelion: {rate_suzuki[0]:.3e} atoms/s")
    print(f"Suzuki Model at Aphelion:   {rate_suzuki[180]:.3e} atoms/s")
    print(f"Leblanc Model at Perihelion:{rate_leblanc[0]:.3e} atoms/s")


if __name__ == "__main__":
    calculate_mercury_mmv_rates()