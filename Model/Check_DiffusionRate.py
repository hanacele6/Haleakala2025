import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# パラメータ設定 (Killen et al. 2004, 2022 Based)
# ==============================================================================
# 物理定数
KB_EV = 8.617e-5  # Boltzmann constant [eV/K]

# シミュレーション設定
GRAIN_RADIUS_CM = 60.0e-4  # 粒子半径 a = 60 um
TEMP_LIST = [400, 500, 600, 700]  # 検証する温度 [K]

# 拡散パラメータ (Impact Glass)
DIFF_D0 = 1.0e-3  # [cm^2/s]
DIFF_EA_EV = 0.75  # [eV]

# 近似モデルの設定
DIFF_LENGTH_CM = 1.0e-4  # 枯渇層厚 delta = 1 um (粒子半径の1/60程度)

# 時間軸 (ログスケール)
# 1秒から 1水星年(約88日 ~ 7.6e6秒) まで
time_sec = np.logspace(0, 7, 500)


# ==============================================================================
# 計算関数
# ==============================================================================

def get_diff_coeff(temp_k):
    """アレニウスの式で拡散係数 D を計算"""
    if temp_k <= 0: return 0.0
    return DIFF_D0 * np.exp(-DIFF_EA_EV / (KB_EV * temp_k))


def exact_sphere_flux_normalized(t, D, a):
    """Crank (1975) 球体拡散の厳密解からフラックス比率 (dF/dt) を計算"""
    sum_term = np.zeros_like(t)
    pi_sq = np.pi ** 2
    factor = pi_sq * D / (a ** 2)

    # n=1000まで計算（収束判定付き）
    for n in range(1, 1000):
        term = np.exp(- (n ** 2) * factor * t)
        sum_term += term
        if np.max(term) < 1e-20 * np.max(sum_term):
            break

    rate = (6 * D / (a ** 2)) * sum_term
    return rate


def approx_fick_flux_normalized(D, a, delta):
    """Fickの法則による近似フラックス (dF/dt相当に換算)"""
    return (3 * D) / (a * delta)


# ==============================================================================
# メイン実行・プロット
# ==============================================================================
plt.figure(figsize=(10, 6))

for T in TEMP_LIST:
    D = get_diff_coeff(T)

    # 厳密解
    rate_exact = exact_sphere_flux_normalized(time_sec, D, GRAIN_RADIUS_CM)

    # 近似解
    rate_approx = approx_fick_flux_normalized(D, GRAIN_RADIUS_CM, DIFF_LENGTH_CM)

    # プロット (ここを修正)
    # まず実線(厳密解)をプロットし、戻り値linesから色情報を取得する
    lines = plt.loglog(time_sec, rate_exact, label=f'Exact (Crank) T={T}K')
    line_color = lines[0].get_color()

    # 取得した色を使って近似解(破線)を描く
    plt.axhline(y=rate_approx, linestyle='--', color=line_color, alpha=0.7, label=f'Approx (Fick) T={T}K')

    # 交差する時間を確認
    idx_cross = np.argwhere(rate_exact < rate_approx)
    if len(idx_cross) > 0:
        t_cross = time_sec[idx_cross[0][0]]
        plt.plot(t_cross, rate_approx, 'o', color=line_color)
        print(f"[T={T}K] D={D:.2e} cm2/s | 交差時間 t ≈ {t_cross:.2e} sec ({t_cross / 86400:.2f} days)")
    else:
        print(f"[T={T}K] 交差せず (厳密解が常に高い範囲)")

# グラフ装飾
plt.title(
    f"Comparison of Diffusion Models\nSphere Radius a={GRAIN_RADIUS_CM * 1e4:.0f}um, Depletion Layer delta={DIFF_LENGTH_CM * 1e4:.0f}um")
plt.xlabel("Time since exposure [sec]")
plt.ylabel("Normalized Release Rate dF/dt [1/s]")
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()

# 時間スケール目安
plt.axvline(x=50 * 60, color='gray', linestyle=':', label='Simulation Step (Typical)')
plt.axvline(x=88 * 24 * 3600, color='k', linestyle='-', alpha=0.3, label='1 Mercury Year')
plt.text(88 * 24 * 3600, plt.ylim()[0] * 2, ' 1 Year', verticalalignment='bottom')

plt.tight_layout()
plt.show()