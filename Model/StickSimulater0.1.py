import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
try:
    import japanize_matplotlib  # 文字化け対策
except ImportError:
    pass

# --- 1. パラメータ設定 ---
# ① 物理吸着 (Physisorption) 
eps_p = 0.5
sig_p = 2.8

# ② 表面への化学吸着 (Chemisorption) 
eps_c = 1.85
sig_c = 1.6
E_dc = 0.8      

# ③ バルク内の化学結合 (Bulk Lattice)
eps_b = 6.0
sig_b = 0.9
E_db = 2.5      

mass = 1.0
E_init = 2.0    
gamma = 0.4     

z_start = 6.0
v_start = -np.sqrt(2 * E_init / mass)

# --- 2. 物理モデル（3つのポテンシャルの合成） ---
def calc_potentials(z):
    z = np.maximum(z, 0.1)
    V_p = 4 * eps_p * ((sig_p/z)**12 - (sig_p/z)**6)
    V_c = 4 * eps_c * ((sig_c/z)**12 - (sig_c/z)**6) + E_dc
    V_b = 4 * eps_b * ((sig_b/z)**12 - (sig_b/z)**6) + E_db
    return V_p, V_c, V_b

def triple_well_potential(z):
    V_p, V_c, V_b = calc_potentials(z)
    return np.minimum(np.minimum(V_p, V_c), V_b)

def triple_well_force(z):
    z = max(z, 0.1)
    V_p, V_c, V_b = calc_potentials(z)
    F_p = (24 * eps_p / z) * (2 * (sig_p/z)**12 - (sig_p/z)**6)
    F_c = (24 * eps_c / z) * (2 * (sig_c/z)**12 - (sig_c/z)**6)
    F_b = (24 * eps_b / z) * (2 * (sig_b/z)**12 - (sig_b/z)**6)
    
    if V_b < V_c and V_b < V_p:
        return F_b
    elif V_c < V_p:
        return F_c
    else:
        return F_p

def equations(t, y):
    z, v = y
    force_friction = -gamma * v if z < 4.0 else 0.0
    dzdt = v
    dvdt = (triple_well_force(z) + force_friction) / mass
    return [dzdt, dvdt]

# --- 3. シミュレーション実行 ---
t_span = (0, 30)
t_eval = np.linspace(t_span[0], t_span[1], 3000)
sol = solve_ivp(equations, t_span, [z_start, v_start], t_eval=t_eval, method='Radau')

z_t = sol.y[0]
v_t = sol.y[1]
total_energy = 0.5 * mass * v_t**2 + np.array([triple_well_potential(z) for z in z_t])

# --- 4. グラフ描画 ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=False)

z_min_p = sig_p * (2**(1/6))
z_min_c = sig_c * (2**(1/6))
z_min_b = sig_b * (2**(1/6))

# ==========================================
# 【上図】ポテンシャルエネルギーの形状とPSD遷移
# ==========================================
z_plot = np.linspace(0.8, 6.0, 1000)
V_p_plot, V_c_plot, V_b_plot = calc_potentials(z_plot)
actual_potential = np.minimum(np.minimum(V_p_plot, V_c_plot), V_b_plot)

ax1.plot(z_plot, V_p_plot, 'b--', alpha=0.5, label='物理吸着の曲線 (V_p)')
ax1.plot(z_plot, V_c_plot, 'g--', alpha=0.5, label='表面化学吸着の曲線 (V_c, ~1.85 eV)')
ax1.plot(z_plot, V_b_plot, 'r--', alpha=0.5, label='バルク化学結合の曲線 (V_b, 4~8 eV)')

ax1.plot(z_plot, actual_potential, 'k-', linewidth=2.5, label='実際の合成ポテンシャル (基底状態)')
ax1.fill_between(z_plot, actual_potential, 6.0, color='lightgray', alpha=0.3)

ax1.axvline(x=z_min_b, color='r', linestyle=':', alpha=0.7)
ax1.axvline(x=z_min_c, color='g', linestyle=':', alpha=0.7)
ax1.axvline(x=z_min_p, color='b', linestyle=':', alpha=0.7)

# --- ★ここを変更：PSDの起点を z_min_c から z_min_p へ ---
E_photon = 4.5
V_base = triple_well_potential(z_min_p)  # 基準を物理吸着の底に
# 遷移先のエネルギーの絶対値（ゼロ基準）
V_exc_peak = V_base + E_photon 

# 無限遠で0に漸近するように、ピーク値から減衰させる
V_exc_plot = V_exc_peak * np.exp(-2.0 * (z_plot - z_min_p))

ax1.plot(z_plot, V_exc_plot, '-', color='magenta', linewidth=2.5, label='励起状態 $V^*(z)$')

V_exc_at_z_min_p = 3.0 + V_base + E_photon
ax1.annotate('', xy=(z_min_p, V_exc_at_z_min_p), xytext=(z_min_p, V_base),
             arrowprops=dict(arrowstyle='->', color='purple', lw=2.5))
ax1.text(z_min_p + 0.1, V_base + E_photon/2, f'光子吸収 (垂直遷移)\n(ソース: 物理吸着の井戸)', 
         color='purple', fontsize=11, fontweight='bold')
# ----------------------------------------------------

ax1.set_ylim(-4.5, 6.0)
ax1.set_xlim(0.8, 6.0)
ax1.axhline(y=0, color='black', linewidth=0.8)
ax1.set_ylabel('エネルギー (eV)')
ax1.set_xlabel('表面からの距離')
ax1.set_title('ポテンシャルエネルギーと光刺激脱離（PSD）のメカニズム')
ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
ax1.grid(True)

# ==========================================
# 【下図】時間変化に伴う粒子の位置 z
# ==========================================
ax2.plot(t_eval, z_t, color='orange', linewidth=2, label='粒子の位置 z(t)')
ax2.axhline(y=z_min_b, color='r', linestyle=':', alpha=0.7, label='バルクの底の位置')
ax2.axhline(y=z_min_c, color='g', linestyle=':', alpha=0.7, label='化学吸着の底の位置')
ax2.axhline(y=z_min_p, color='b', linestyle=':', alpha=0.7, label='物理吸着の底の位置')

ax2.set_xlim(t_span[0], t_span[1])
ax2.set_ylim(0.8, 6.0)
ax2.set_xlabel('時間 t')
ax2.set_ylabel('表面からの距離 z')
ax2.set_title(f'粒子の運動軌跡（初期エネルギー: {E_init} eV, 摩擦係数: {gamma}）')
ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
ax2.grid(True)

plt.tight_layout()
plt.show()