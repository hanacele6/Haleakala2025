import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import japanize_matplotlib  # 文字化け対策（インストール済みの場合）

# --- 1. パラメータ設定 ---
# 物理吸着 (Physisorption) - 遠くて浅い
eps_p = 0.3
sig_p = 2.5

# 化学吸着 (Chemisorption) - 近くて深いが、解離の壁(E_d)がある
eps_c = 3.0
sig_c = 1.2
E_d = 1.0      # 分子の解離エネルギー（壁の高さの要因）

mass = 1.0

# ユーザーが変更して遊ぶパラメータ
E_init = 1.8   # 初期エネルギー（例：0.5なら物理吸着、1.5なら化学吸着へ）
gamma = 0.3    # 表面の摩擦（エネルギー散逸）

z_start = 6.0
v_start = -np.sqrt(2 * E_init / mass)

# --- 2. 物理モデル（2つのポテンシャルの合成） ---
def calc_potentials(z):
    V_p = 4 * eps_p * ((sig_p/z)**12 - (sig_p/z)**6)
    V_c = 4 * eps_c * ((sig_c/z)**12 - (sig_c/z)**6) + E_d
    return V_p, V_c

def double_well_potential(z):
    """2つのポテンシャルのうち、より低い（安定な）方を実際のポテンシャルとする"""
    V_p, V_c = calc_potentials(z)
    return np.minimum(V_p, V_c)

def double_well_force(z):
    """ポテンシャルの勾配（力）を計算"""
    V_p, V_c = calc_potentials(z)
    F_p = (24 * eps_p / z) * (2 * (sig_p/z)**12 - (sig_p/z)**6)
    F_c = (24 * eps_c / z) * (2 * (sig_c/z)**12 - (sig_c/z)**6)
    
    # 粒子がいる位置で、どちらの曲線上を走っているかで力を切り替える
    if V_p < V_c:
        return F_p
    else:
        return F_c

def equations(t, y):
    z, v = y
    # 表面近傍だけで摩擦が働くと仮定
    force_friction = -gamma * v if z < 4.0 else 0.0
    
    dzdt = v
    dvdt = (double_well_force(z) + force_friction) / mass
    return [dzdt, dvdt]

# --- 3. シミュレーション実行 ---
t_span = (0, 30)
t_eval = np.linspace(t_span[0], t_span[1], 2000)
# 交差点（壁）で力が不連続になるため、Radau法などの硬い方程式向けソルバーが安定します
sol = solve_ivp(equations, t_span, [z_start, v_start], t_eval=t_eval, method='Radau')

z_t = sol.y[0]
v_t = sol.y[1]
kinetic_energy = 0.5 * mass * v_t**2
# 軌道上のポテンシャルエネルギーを計算
potential_energy = np.array([double_well_potential(z) for z in z_t])
total_energy = kinetic_energy + potential_energy

# --- 4. グラフ描画 ---
fig, ax = plt.subplots(figsize=(10, 6))

# 背景のポテンシャル曲線を描画
z_plot = np.linspace(0.9, 6.0, 1000)
V_p_plot, V_c_plot = calc_potentials(z_plot)
ax.plot(z_plot, V_p_plot, 'b--', alpha=0.5, label='物理吸着 (V_p)')
ax.plot(z_plot, V_c_plot, 'r--', alpha=0.5, label='化学吸着 (V_c)')
ax.plot(z_plot, np.minimum(V_p_plot, V_c_plot), 'k-', linewidth=2, label='実際のポテンシャル (Double Well)')

# 粒子の全エネルギー軌跡を描画
ax.plot(z_t, total_energy, color='orange', marker='.', markersize=1, linestyle='-', label='粒子の全エネルギー軌跡')

ax.set_ylim(-3.5, 3.5)
ax.set_xlim(0.8, 6.0)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xlabel('表面からの距離 z')
ax.set_ylabel('エネルギー')
ax.set_title(f'初期エネルギー: {E_init}, 摩擦係数: {gamma}')
ax.legend(loc='upper right')
ax.grid(True)

plt.tight_layout()
plt.show()