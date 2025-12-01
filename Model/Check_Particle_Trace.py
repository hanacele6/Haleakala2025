import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 物理定数 (ユーザーコードより抜粋)
# ==============================================================================
PHYSICAL_CONSTANTS = {
    'GM_MERCURY': 2.2032e13,  # [m^3/s^2]
    'RM': 2.440e6,  # 水星半径 [m]
    'MASS_NA': 22.98976928 * 1.66054e-27,  # [kg] (エネルギー計算用)
}


def calculate_gravity_acceleration(pos):
    """水星重力のみを計算"""
    r_sq = np.sum(pos ** 2)
    r = np.sqrt(r_sq)
    if r == 0:
        return np.array([0.0, 0.0, 0.0])
    # F = -GM/r^3 * r_vec
    acc = -PHYSICAL_CONSTANTS['GM_MERCURY'] * pos / (r ** 3)
    return acc


def run_simulation(dt, initial_pos, initial_vel, max_duration=10000):
    """
    指定された dt で単一粒子の軌道を計算する関数
    (修正版: エネルギーと時間の配列サイズを一致させる)
    """
    pos = initial_pos.copy()
    vel = initial_vel.copy()

    current_time = 0.0

    # --- 初期状態の保存 ---
    trajectory = [pos.copy()]
    times = [0.0]

    # 初期エネルギー計算
    r = np.linalg.norm(pos)
    v_sq = np.sum(vel ** 2)
    pe = -PHYSICAL_CONSTANTS['GM_MERCURY'] / r
    ke = 0.5 * v_sq
    total_e = ke + pe
    energies = [total_e]  # 初期エネルギーをリストに入れる

    while current_time < max_duration:
        # RK4 Step
        k1_vel = dt * calculate_gravity_acceleration(pos)
        k1_pos = dt * vel

        k2_vel = dt * calculate_gravity_acceleration(pos + 0.5 * k1_pos)
        k2_pos = dt * (vel + 0.5 * k1_vel)

        k3_vel = dt * calculate_gravity_acceleration(pos + 0.5 * k2_pos)
        k3_pos = dt * (vel + 0.5 * k2_vel)

        k4_vel = dt * calculate_gravity_acceleration(pos + k3_pos)
        k4_pos = dt * (vel + k3_vel)

        pos += (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos) / 6.0
        vel += (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel) / 6.0

        current_time += dt

        # --- 更新後の状態を保存 ---
        trajectory.append(pos.copy())
        times.append(current_time)

        # 更新後のエネルギー計算
        r = np.linalg.norm(pos)
        v_sq = np.sum(vel ** 2)
        pe = -PHYSICAL_CONSTANTS['GM_MERCURY'] / r
        ke = 0.5 * v_sq
        total_e = ke + pe
        energies.append(total_e)

        # 衝突判定
        if np.linalg.norm(pos) < PHYSICAL_CONSTANTS['RM']:
            break

    return np.array(trajectory), np.array(times), np.array(energies)

# ==============================================================================
# メイン検証処理
# ==============================================================================

# 初期条件の設定
# 速度: 2.5 km/s (典型的な脱出しない粒子の速度), 角度: 45度
v0 = 2500.0  # [m/s]
angle_deg = 45.0
angle_rad = np.deg2rad(angle_deg)

start_pos = np.array([PHYSICAL_CONSTANTS['RM'], 0.0, 0.0])
start_vel = np.array([v0 * np.sin(angle_rad), v0 * np.cos(angle_rad), 0.0])

# 検証するタイムステップのリスト
dt_list = [1, 50, 100, 500, 1000]
colors = ['black', 'green', 'blue', 'orange', 'red']
labels = [f'dt={dt}s' for dt in dt_list]

plt.figure(figsize=(14, 6))

# --- Plot 1: 軌道 (X vs Y) ---
plt.subplot(1, 2, 1)
for i, dt in enumerate(dt_list):
    traj, t, e = run_simulation(dt, start_pos, start_vel)

    # 表面からの高度をプロットするため、回転などは考慮せず単純なXY平面投影
    x_vals = traj[:, 0]
    y_vals = traj[:, 1]

    # 水星表面を描画 (最初のループのみ)
    if i == 0:
        theta = np.linspace(0, np.pi / 2, 100)
        mx = PHYSICAL_CONSTANTS['RM'] * np.cos(theta)
        my = PHYSICAL_CONSTANTS['RM'] * np.sin(theta)
        plt.plot(mx, my, 'k--', alpha=0.3, label='Surface')

    # dt=1s を正解とみなして実線、他は点線やマーカーで区別
    ls = '-' if dt == 1 else '--'
    marker = '.' if dt >= 100 else None

    plt.plot(x_vals, y_vals, label=labels[i], color=colors[i], linestyle=ls, marker=marker, markersize=4)

plt.title(f'Trajectory Comparison (v0={v0 / 1000:.1f} km/s)')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.axis('equal')
plt.grid(True)
plt.legend()

# --- Plot 2: エネルギー保存の誤差 ---
plt.subplot(1, 2, 2)
for i, dt in enumerate(dt_list):
    traj, t, e = run_simulation(dt, start_pos, start_vel)

    # 初期エネルギーに対する相対誤差
    e0 = e[0]
    e_error = np.abs((e - e0) / e0)

    # 対数グラフで表示 (0が含まれると警告が出るため、少し工夫)
    valid_idx = e_error > 0
    if np.any(valid_idx):
        plt.semilogy(t[valid_idx], e_error[valid_idx], label=labels[i], color=colors[i])

plt.title('Energy Conservation Error (|E(t) - E0| / |E0|)')
plt.xlabel('Time [s]')
plt.ylabel('Relative Energy Error')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()

plt.tight_layout()
plt.show()