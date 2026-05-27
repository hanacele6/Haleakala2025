import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 水星外気圏 1.5D アナリティカルモデル (経度解像・見かけの太陽運動を考慮)
# ==============================================================================

# --- 物理・軌道定数 ---
KB = 8.617e-5
NU = 1e13
E_ECC = 0.2056
A_AU = 0.387
T_ORBIT_DAYS = 87.969

# --- 空間・エネルギーグリッド ---
N_LON = 36  # 経度を10度刻み
LONS_RAD = np.linspace(0, 2*np.pi, N_LON, endpoint=False)
N_BINS = 20
U_BINS = np.linspace(1.4, 2.7, N_BINS)

def run_15d_model(u_mu=1.80, u_sigma=0.25, supply_rate=1.0):
    # 初期化: 各経度にガウス分布で在庫を配置
    weights = np.exp(-0.5 * ((U_BINS - u_mu) / u_sigma)**2)
    weights /= np.sum(weights)
    surface_density = np.zeros((N_LON, N_BINS))
    for i in range(N_LON):
        surface_density[i, :] = 1000.0 * weights

    # TAAを2周分（720度）回す。1度刻み。
    taa_deg = np.arange(0, 720, 1)
    taa_rad = np.radians(taa_deg)
    
    time_days = 0.0
    
    # 記録用 (2周目 TAA=360~719)
    res_abundance = np.zeros(360)
    res_source = np.zeros(360)

    for i in range(len(taa_deg)):
        nu = taa_rad[i]
        
        # 1. 軌道上の経過時間 dt を計算 (面積速度一定則からの導出)
        dt_days = (T_ORBIT_DAYS / (2 * np.pi)) * ((1 - E_ECC**2)**1.5) / ((1 + E_ECC * np.cos(nu))**2) * np.radians(1)
        time_days += dt_days
        dt_sec = dt_days * 86400
        
        # 2. 自転と太陽直下経度の計算 (3:2 共鳴)
        theta_rot = (1.5 * 2 * np.pi / T_ORBIT_DAYS) * time_days
        subsolar_lon = theta_rot - nu  # 見かけの太陽直下経度
        
        # 3. 距離と最大温度
        r_au = A_AU * (1 - E_ECC**2) / (1 + E_ECC * np.cos(nu))
        t_max = 700 * np.sqrt((1 + E_ECC * np.cos(nu)) / (1 + E_ECC))
        
        total_emission_step = 0.0
        
        # 4. 経度ごとの熱脱離計算
        for j, lon in enumerate(LONS_RAD):
            # 太陽からの局所時角
            hour_angle = lon - subsolar_lon
            cos_theta = np.cos(hour_angle)
            
            # 温度の算出
            if cos_theta > 0:
                temp = t_max * (cos_theta**0.25)
            else:
                temp = 100.0
                
            # 各エネルギービンの枯渇と放出
            for k, u in enumerate(U_BINS):
                rate = NU * np.exp(-u / (KB * temp))
                # 厳密な減衰式でマイナス化を防止
                emission_frac = 1.0 - np.exp(-rate * dt_sec)
                
                emission = surface_density[j, k] * emission_frac
                surface_density[j, k] -= emission
                total_emission_step += emission
                
                # 内部拡散からの供給 (全体に一定レートで補充)
                supply = supply_rate * dt_days * weights[k]
                surface_density[j, k] += supply

        # 5. 外気圏量の算出 ( Source × 寿命(r^2) )
        abundance = total_emission_step * (r_au**2)
        
        # 6. 2周目のデータを記録
        if taa_deg[i] >= 360:
            idx = int(taa_deg[i] % 360)
            res_source[idx] = total_emission_step
            res_abundance[idx] = abundance

    return res_source, res_abundance

# --- シミュレーション実行と描画 ---
# mu を変えて比較する
mu_values = [2.2, 1.85, 1.4]
colors = ['gray', 'blue', 'red']

plt.figure(figsize=(10, 6))

for mu, color in zip(mu_values, colors):
    source, abundance = run_15d_model(u_mu=mu, u_sigma=0.25, supply_rate=0.5)
    # 比較のため正規化
    abundance_norm = abundance / np.max(abundance)
    
    peak_taa = np.argmax(abundance_norm)
    
    plt.plot(np.arange(0, 360), abundance_norm, label=f'$\mu$ = {mu} eV (Peak: TAA {peak_taa}°)', color=color, lw=2)
    plt.plot(peak_taa, abundance_norm[peak_taa], marker='o', color=color)

plt.axvline(180, color='black', linestyle=':', alpha=0.5, label='Aphelion (TAA 180°)')
plt.title("Exospheric Abundance vs TAA (1.5D Spatial Depletion Model)")
plt.xlabel("True Anomaly Angle (TAA) [deg]")
plt.ylabel("Normalized Abundance")
plt.xlim(0, 360)
plt.ylim(0, 1.1)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()