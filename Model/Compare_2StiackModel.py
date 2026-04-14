import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 物理定数
# =============================================================================
MASS_NA_AMU = 23.0
MASS_TARGET_AMU = 16.0  # 酸素メインを想定
U_0 = 1.85              # 化学吸着の井戸の深さ [eV]
KB_EV = 8.617e-5        # ボルツマン定数 [eV/K]

# 速度計算用の変換係数 (eV, amu -> km/s)
# E = 1/2 m v^2 より、v = sqrt(2E/m)
# 1 eV = 1.602e-19 J, 1 amu = 1.6605e-27 kg
V_FACTOR = np.sqrt((2.0 * 1.602e-19) / (1.0 * 1.6605e-27)) / 1000.0

# =============================================================================
# モデルA: Zangwill / Baule による力学的な散逸判定
# =============================================================================
def evaluate_baule_model(E_i_eV, theta_i_rad):
    # 質量比とエネルギー伝達係数
    mu = MASS_NA_AMU / MASS_TARGET_AMU
    transfer_coeff = (4 * mu) / ((1 + mu)**2)
    
    # 法線成分のエネルギー (Zangwill p.356)
    E_n = E_i_eV * (np.cos(theta_i_rad)**2)
    E_t = E_i_eV * (np.sin(theta_i_rad)**2) # 水平成分（保存される）
    
    # 井戸の底でのトータル衝突エネルギー
    E_impact = E_n + U_0
    
    # フォノンへ散逸するエネルギー
    delta_E = transfer_coeff * E_impact
    
    if delta_E >= E_n:
        return True, 0.0, 0.0  # 吸着(Trapped)
    else:
        # 非弾性散乱（エネルギーを一部失ってバウンド）
        E_n_out = E_n - delta_E
        
        # 速度への変換 [km/s]
        v_n_out = np.sqrt(E_n_out / MASS_NA_AMU) * V_FACTOR
        v_t_out = np.sqrt(E_t / MASS_NA_AMU) * V_FACTOR
        
        # x方向（進行方向）を正、z方向（上向き）を正とする
        return False, v_t_out, v_n_out

# =============================================================================
# モデルB: 空隙率を考慮した経験的確率判定（従来の王道モデル）
# =============================================================================
def calculate_sticking_probability(surface_temp_K):
    A = 0.0804
    B = 458.0
    porosity = 0.8 
    if surface_temp_K <= 0: return 1.0
    p_stick = A * np.exp(B / surface_temp_K)
    p_stick_eff = p_stick / (1.0 - (1.0 - p_stick) * porosity)
    return min(p_stick_eff, 1.0)

def evaluate_empirical_model(surface_temp_K):
    prob = calculate_sticking_probability(surface_temp_K)
    
    if np.random.random() < prob:
        return True, 0.0, 0.0  # 吸着(Trapped)
    else:
        # 弾かれた場合：入射の記憶を失い、表面温度によるランバート反射（熱的バウンド）
        kT = KB_EV * surface_temp_K
        # 脱離フラックスのエネルギー分布 (Gamma(2, kT))
        E_th = np.random.gamma(2.0, kT)
        v_th = np.sqrt(E_th / MASS_NA_AMU) * V_FACTOR
        
        # ランバート反射角（3Dの半球からサンプリングし、xz平面に投影）
        # thetaは法線からの角度
        sin_theta = np.sqrt(np.random.random())
        cos_theta = np.sqrt(1.0 - sin_theta**2)
        
        # 水平方向は全方位ランダムなので、進行方向(x)への投影を考慮
        phi = np.random.uniform(0, 2*np.pi)
        
        v_t_out = v_th * sin_theta * np.cos(phi)
        v_n_out = v_th * cos_theta
        
        return False, v_t_out, v_n_out

# =============================================================================
# メインシミュレーションと描画
# =============================================================================
def run_simulation_and_plot():
    # --- テスト条件 ---
    N_particles = 1000
    T_surface = 400.0  # 表面温度 [K]
    
    # 状況設定：高エネルギーのPSD由来の粒子が、浅い角度で降ってきたとする
    E_incident_base = 1.0   # 入射エネルギー [eV] (約6000K相当)
    theta_incident_base = np.deg2rad(0) # 入射角（法線から60度 = 浅い角度）
    
    # プロット準備
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    fig.suptitle(f"Mercury Na Scattering Profile (Ei={E_incident_base} eV, Angle={60} deg, Ts={T_surface} K)", fontsize=14)
    
    for ax in axes:
        ax.set_xlim(-2, 4)
        ax.set_ylim(0, 4)
        ax.set_xlabel("Horizontal Velocity Vx [km/s]")
        ax.set_ylabel("Vertical Velocity Vz [km/s]")
        ax.axhline(0, color='black', linewidth=2)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.7)

    # --- モデルA (Baule) の計算 ---
    vx_A, vz_A = [], []
    trapped_A = 0
    for _ in range(N_particles):
        # 表面の微小な粗さを模擬するため、入射角にわずかなバラツキ(±3度)を持たせる
        theta_i = np.random.normal(theta_incident_base, np.deg2rad(3))
        
        is_trapped, vx, vz = evaluate_baule_model(E_incident_base, theta_i)
        if is_trapped:
            trapped_A += 1
        else:
            vx_A.append(vx)
            vz_A.append(vz)

    # --- モデルB (Empirical) の計算 ---
    vx_B, vz_B = [], []
    trapped_B = 0
    for _ in range(N_particles):
        is_trapped, vx, vz = evaluate_empirical_model(T_surface)
        if is_trapped:
            trapped_B += 1
        else:
            vx_B.append(vx)
            vz_B.append(vz)

    # --- 描画 ---
    # 入射ベクトル（目安）の描画
    v_in_mag = np.sqrt(E_incident_base / MASS_NA_AMU) * V_FACTOR
    v_in_x = v_in_mag * np.sin(theta_incident_base)
    v_in_z = -v_in_mag * np.cos(theta_incident_base) # 下向き
    
    for ax in axes:
        # 入射方向を示す矢印 (原点に向かってくる)
        ax.annotate('', xy=(0, 0), xytext=(-v_in_x, -v_in_z), 
                    arrowprops=dict(facecolor='red', shrink=0, width=1.5, headwidth=6), zorder=10)
        ax.text(-v_in_x, -v_in_z + 0.2, "Incident\nParticle", color='red', ha='center')

    # モデルAの散乱ローブ
    axes[0].scatter(vx_A, vz_A, s=5, color='blue', alpha=0.5)
    axes[0].set_title(f"Model A: Baule (Kinematic Bounce)\nTrapped: {trapped_A/N_particles*100:.1f} %")
    
    # モデルBの散乱ローブ
    axes[1].scatter(vx_B, vz_B, s=5, color='green', alpha=0.5)
    axes[1].set_title(f"Model B: Empirical (Thermal Bounce)\nTrapped: {trapped_B/N_particles*100:.1f} %")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation_and_plot()