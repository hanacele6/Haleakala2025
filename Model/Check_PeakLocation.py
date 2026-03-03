import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# Mura et al. (2023) Based Analytical Model
# ==============================================================================
# 目的: "Planetary Rotation" (Dawn Sweep) 効果による季節変動を再現する
# ==============================================================================

# --- 物理定数 ---
AU_M = 1.496e11
R_M = 2440e3
GM_SUN = 1.327e20
R_PERI_AU = 0.307
R_APH_AU = 0.467

# --- 軌道・自転パラメータ ---
# 水星の公転周期と自転周期 (3:2共鳴)
P_orb = 87.969 * 86400
P_spin = 58.646 * 86400
n_mean = 2 * np.pi / P_orb  # 平均公転角速度
omega_spin = 2 * np.pi / P_spin  # 自転角速度 (一定)


def get_g_factor(v_rad_km_s):
    # 簡易的なg-factorモデル (視線速度0で暗くなる)
    # 実際はもっと複雑ですが、ガウス関数で近似
    depth = 0.6
    width = 5.0  # km/s
    return 1.0 - depth * np.exp(-0.5 * (v_rad_km_s / width) ** 2)


def run_mura_model(accumulation_dependency=-2.0):
    # TAA 0~360度
    taa_deg = np.linspace(0, 360, 360)
    taa_rad = np.deg2rad(taa_deg)

    # 1. 軌道計算
    e = 0.2056
    a = 0.387 * AU_M
    r = a * (1 - e ** 2) / (1 + e * np.cos(taa_rad))
    r_au = r / AU_M

    # 2. 公転角速度 (nu_dot)
    # 角運動量保存: r^2 * nu_dot = h = const
    h = np.sqrt(GM_SUN * a * (1 - e ** 2))
    nu_dot = h / (r ** 2)

    # 視線速度 (ドップラー用)
    v_rad = np.sqrt(GM_SUN / a / (1 - e ** 2)) * e * np.sin(taa_rad) / 1000.0  # km/s

    # 3. Mura's Dawn Source Mechanism
    # 相対的な自転速度 (夜明け線が地表を走る速度)
    # Omega_rel = Omega_spin - nu_dot
    omega_rel = omega_spin - nu_dot

    # 夜明け線速度が負になる(逆行する)場合は0とする(供給なし)
    omega_rel = np.maximum(omega_rel, 0.0)

    # 夜側での蓄積 (Accumulation)
    # 論文によると Plasma や Micrometeoroid が起源 -> 距離依存性あり
    # 一般的に Flux ~ r^(-2) と仮定
    accumulation_rate = r_au ** accumulation_dependency

    # Dawn Flux = (Sweep Rate) * (Accumulated Density)
    # 単純化モデル: Flux ~ Omega_rel * Accumulation
    flux_dawn = omega_rel * accumulation_rate

    # 4. 柱密度 (Column Density)
    # Density = Flux * Lifetime
    # 光イオン化寿命 tau ~ r^2
    tau = r_au ** 2.0
    column_density = flux_dawn * tau

    # 5. 輝度 (Intensity / Brightness)
    # Intensity = Density * g-factor
    g = get_g_factor(v_rad)
    intensity = column_density * g

    return taa_deg, column_density, intensity, omega_rel


# --- 実行とプロット ---
# ケーススタディ: 蓄積レートの距離依存性を変えてみる
# Case A: r^-2 (標準的な太陽風/プラズマ)
deg, dens_A, int_A, omega = run_mura_model(-2.0)

plt.figure(figsize=(10, 8))

# 1. Column Density & Intensity
plt.subplot(2, 1, 1)
# 正規化
plt.plot(deg, dens_A / np.max(dens_A), 'b--', label='Column Density (Peak ~180-220)', linewidth=1.5)
plt.plot(deg, int_A / np.max(int_A), 'r-', label='Observed Intensity (Peak ~150)', linewidth=3)

# 150度ライン
plt.axvline(150, color='green', linestyle=':', label='Target 150 deg')
plt.axvline(180, color='gray', linestyle='-', alpha=0.3)

plt.title('Mura et al. (2023) Mechanism Check', fontsize=14)
plt.ylabel('Normalized Value')
plt.legend(loc='upper right')
plt.grid(True)
plt.xlim(0, 360)

# ピーク位置表示
pk_dens = deg[np.argmax(dens_A)]
pk_int = deg[np.argmax(int_A)]
plt.text(pk_dens + 5, 0.9, f'Dens Peak: {pk_dens:.0f}', color='blue')
plt.text(pk_int - 40, 0.95, f'Int Peak: {pk_int:.0f}', color='red', fontweight='bold')

# 2. メカニズム解説 (角速度差)
plt.subplot(2, 1, 2)
# Omega_rel と nu_dot の比較
plt.plot(deg, omega, 'k-', label='Relative Rotation (Dawn Sweep Speed)')
plt.ylabel('Angular Velocity [rad/s]')
plt.xlabel('True Anomaly Angle (TAA) [deg]')
plt.legend()
plt.grid(True)
plt.title('Why Aphelion? -> Dawn Terminator moves fastest at 180 deg')
plt.xlim(0, 360)

plt.tight_layout()
plt.show()