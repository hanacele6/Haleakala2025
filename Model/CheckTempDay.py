import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. 物理定数・設定 (Deutsch et al. 2025 / Hayne et al. 2017 準拠)
# ==============================================================================
# 軌道・回転パラメータ
ROTATION_PERIOD = 176.0 * 24.0 * 3600.0  # 水星の1太陽日 [cite: 150]
CURRENT_AU = 0.306  # 近日点 (Figure 1b はホットポール/赤道付近の激しい変化を想定)

# 太陽定数とスケーリング
SOLAR_CONSTANT_1AU = 1361.0
SOLAR_FLUX_MAX = SOLAR_CONSTANT_1AU / (CURRENT_AU ** 2)

# Reference Model (LeBlanc) 用パラメータ (Deutsch Fig 2a の最大温度 ~700K に合わせる)
# 距離スケーリング係数 (1AU基準に対する倍率の2乗ではなく、フラックスベースで計算するためここは係数調整用)
SCALING_FACTOR_REF = np.sqrt(0.306 / CURRENT_AU)
#SCALING_FACTOR_REF = (0.306 / CURRENT_AU)**2
SCALING_PARAMS_T_BASE = 600.0
SCALING_PARAMS_T_AMP = 100.0
SCALING_PARAMS_T_NIGHT = 100.0

# --- 熱物理モデル用物性値 (Deutsch et al. Eq 1 & Hayne et al. 2017) ---
SIGMA = 5.670374419e-8
ALBEDO = 0.12  # [cite: 157]
EMISSIVITY = 0.95  # [cite: 157]
SPECIFIC_HEAT = 840.0  # [J/kg K] (定数近似)

# 【重要】密度の深さ依存性パラメータ
RHO_S = 1100.0  # 表面密度 [kg/m^3] (Hayne 2017 moon value fitting)
RHO_D = 1800.0  # 深部密度 [kg/m^3]
H_PARAMETER = 0.07  # 背景レゴリスのHパラメータ [m]

# 熱伝導率 (密度に依存させる簡易モデル)
# 表面(スカスカ)は低く、深部(カチカチ)は高い
K_SURFACE = 0.001  # 非常に低い (断熱的)
K_DEEP = 0.02  # 岩盤に近い値

# 計算グリッド設定
DT = 600.0
STEPS_PER_DAY = int(ROTATION_PERIOD / DT)
DEPTH = 0.8  # 計算深さ [m]
NUM_LAYERS = 60
# 深さ方向のグリッド (表面付近の精度を上げるため不均一グリッド推奨だが、今回は簡易的に均一)
DZ = DEPTH / NUM_LAYERS


# ==============================================================================
# 2. 熱物理モデル (Deutsch仕様: 密度プロファイル導入)
# ==============================================================================
class MercuryThermalModelDeutsch:
    def __init__(self):
        self.T_profile = np.full(NUM_LAYERS, 350.0)

        # 深さ配列 (各層の中心深さ)
        self.z_depths = np.linspace(DZ / 2, DEPTH - DZ / 2, NUM_LAYERS)

        # --- 密度プロファイルの計算  ---
        # rho(z) = rho_d - (rho_d - rho_s) * exp(-z/H)
        self.rho_profile = RHO_D - (RHO_D - RHO_S) * np.exp(-self.z_depths / H_PARAMETER)

        # --- 熱伝導率プロファイルの計算 (簡易モデル) ---
        # 密度が高いほど熱伝導率が高いと仮定
        # 単純な線形補間: rho_s -> K_surf, rho_d -> K_deep
        rho_ratio = (self.rho_profile - RHO_S) / (RHO_D - RHO_S)
        self.k_profile = K_SURFACE + (K_DEEP - K_SURFACE) * rho_ratio

    def update(self, solar_flux_input):
        T_new = self.T_profile.copy()

        # --- 表面 (Layer 0) ---
        S_in = (1.0 - ALBEDO) * solar_flux_input
        T_sub = self.T_profile[1]
        T_surf = self.T_profile[0]

        K_surf = self.k_profile[0]

        # 表面エネルギー収支
        for _ in range(5):
            conduction = K_surf * (T_surf - T_sub) / DZ
            radiation = EMISSIVITY * SIGMA * (T_surf ** 4)
            residual = S_in - radiation - conduction
            dF_dT = -4 * EMISSIVITY * SIGMA * (T_surf ** 3) - (K_surf / DZ)
            T_surf = T_surf - (residual / dF_dT)

        T_new[0] = T_surf

        # --- 内部 (Layer 1..N) ---
        # 密度(rho)と熱伝導率(k)が層ごとに異なるため、拡散係数 alpha も層ごとに異なる
        # heat_capacity_volumetric = rho * c

        for i in range(1, NUM_LAYERS - 1):
            # 空間微分 (中心差分)
            # 熱流束の勾配: d/dz (K * dT/dz)
            # ここでは簡易的に K を一定とみなした差分法を使うが、本来はKの勾配も必要
            # Deutsch再現の肝は「表面の密度が低い」ことなので、K_localを使用

            K_local = self.k_profile[i]
            rho_local = self.rho_profile[i]

            # 拡散項: alpha * (T_next - 2T + T_prev) / dz^2
            alpha = K_local / (rho_local * SPECIFIC_HEAT)
            diffusion = alpha * (self.T_profile[i + 1] - 2 * self.T_profile[i] + self.T_profile[i - 1]) / (DZ ** 2)

            T_new[i] = self.T_profile[i] + diffusion * DT

        T_new[-1] = T_new[-2]
        self.T_profile = T_new
        return T_surf


# ==============================================================================
# 3. シミュレーション実行
# ==============================================================================
def main():
    model = MercuryThermalModelDeutsch()

    times_hours = []
    temps_tp = []
    temps_user_ref = []

    # スピンアップ (2サイクル)
    total_cycles = 2

    for cycle in range(total_cycles):
        record = (cycle == total_cycles - 1)

        for step in range(STEPS_PER_DAY):
            t_norm = step / STEPS_PER_DAY
            hour_angle = (t_norm - 0.5) * 2 * np.pi
            cos_theta = np.cos(hour_angle)

            # 日照 (物理モデル用)
            solar_flux = SOLAR_FLUX_MAX * max(0.0, cos_theta)

            # 1. Thermophysical Model (Deutsch - Regolith)
            T_tp = model.update(solar_flux)

            # 2. Reference Model (LeBlanc) - 比較用
            if cos_theta > 0:
                T_ref = SCALING_PARAMS_T_BASE* SCALING_FACTOR_REF + SCALING_PARAMS_T_AMP * (cos_theta ** 0.25)
            else:
                T_ref = SCALING_PARAMS_T_NIGHT

            if record:
                local_time_hr = t_norm * 24.0
                times_hours.append(local_time_hr)
                temps_tp.append(T_tp)
                temps_user_ref.append(T_ref)

    # プロット
    shift = -int(len(times_hours) / 2)
    t_plot = np.linspace(0, 24, len(times_hours))
    y_tp = np.roll(temps_tp, shift)
    y_ref = np.roll(temps_user_ref, shift)

    plt.figure(figsize=(10, 6))

    # Deutsch Fig 1b の "Regolith" (Solid line) を再現
    plt.plot(t_plot, y_tp, color='black', linewidth=2.5, label='Deutsch Model')

    # 比較用 (あなたの元のReferenceモデル)
    plt.plot(t_plot, y_ref, color='blue', linestyle='--', alpha=0.6, label='My Model')

    plt.title("Reproduction of Deutsch et al. (2025) Figure 1(b): Mercury Surface Temp", fontsize=14)
    plt.xlabel("Local Time (hr) [0=Noon, 6=Sunset]", fontsize=12)
    plt.ylabel("Temperature (K)", fontsize=12)
    plt.xlim(0, 24)
    plt.ylim(0, 800)
    plt.xticks([0, 6, 12, 18, 24], ['Noon', 'Sunset', 'Midnight', 'Sunrise', 'Noon'])
    plt.grid(True, linestyle=':', alpha=0.6)

    # Deutsch論文の特徴的な部分への注釈
    #plt.annotate('Rapid Drop\n(Low Density Surface)',
    #             xy=(6.5, 300), xytext=(8, 500),
    #             arrowprops=dict(facecolor='black', arrowstyle='->'), ha='center')

    #plt.annotate('Slow Equilibration\n(H=0.07m effect)',
    #             xy=(15, 110), xytext=(15, 250),
    #             arrowprops=dict(facecolor='black', arrowstyle='->'), ha='center')

    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()