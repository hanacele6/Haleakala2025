# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# --- 定数 ---
CONST = {
    'RM': 2.440e6,  # 水星半径 [m]
    'GM': 2.2032e13,  # 重力定数
}


def get_acceleration(pos):
    """重力加速度"""
    r_sq = np.sum(pos ** 2)
    r_mag = np.sqrt(r_sq)
    return -CONST['GM'] * pos / (r_mag ** 3)


def rk4_step(pos, vel, dt):
    """RK4で1ステップだけ進める関数"""
    k1_v = dt * get_acceleration(pos)
    k1_p = dt * vel

    k2_v = dt * get_acceleration(pos + 0.5 * k1_p)
    k2_p = dt * (vel + 0.5 * k1_v)

    k3_v = dt * get_acceleration(pos + 0.5 * k2_p)
    k3_p = dt * (vel + 0.5 * k2_v)

    k4_v = dt * get_acceleration(pos + k3_p)
    k4_p = dt * (vel + k3_v)

    new_pos = pos + (k1_p + 2 * k2_p + 2 * k3_p + k4_p) / 6.0
    new_vel = vel + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6.0

    return new_pos, new_vel


def to_alt_dist(pos_arr):
    """グラフ用座標変換"""
    if pos_arr.ndim == 1: pos_arr = pos_arr[None, :]
    r = np.linalg.norm(pos_arr, axis=1)
    alt = r - CONST['RM']
    dist = np.sqrt(pos_arr[:, 1] ** 2 + pos_arr[:, 2] ** 2)
    return dist / 1000.0, alt / 1000.0  # km


def visualize_rk_steps():
    # 初期条件
    v0_mag = 1000.0
    angle = np.deg2rad(45)
    pos_start = np.array([CONST['RM'], 0.0, 0.0])
    vel_start = np.array([v0_mag * np.cos(angle), 0.0, v0_mag * np.sin(angle)])

    TOTAL_TIME = 500.0

    plt.figure(figsize=(10, 6))

    # ------------------------------------------------
    # パターン1: 高精度 (dt=10s x 50回) - 黒い実線
    # ------------------------------------------------
    path = [pos_start]
    p, v = pos_start.copy(), vel_start.copy()
    dt = 10.0
    for _ in range(int(TOTAL_TIME / dt)):
        p, v = rk4_step(p, v, dt)
        path.append(p)

    d, a = to_alt_dist(np.array(path))
    plt.plot(d, a, 'k-', linewidth=3, alpha=0.3, label='High Res (dt=10s)')

    # ------------------------------------------------
    # パターン2: 粗い刻み (dt=100s x 5回) - 赤い点線
    # ------------------------------------------------
    # これなら「RK4の軌道」が見えます！
    path = [pos_start]
    p, v = pos_start.copy(), vel_start.copy()
    dt = 100.0
    for _ in range(int(TOTAL_TIME / dt)):
        p, v = rk4_step(p, v, dt)
        path.append(p)

    d, a = to_alt_dist(np.array(path))
    plt.plot(d, a, 'r--o', label='Medium RK4 (dt=100s x 5)')

    # ------------------------------------------------
    # パターン3: 元のコード (dt=500s x 1回) - 青い矢印
    # ------------------------------------------------
    # 途中経過がないので、どうしても直線に見えてしまう
    p_end, v_end = rk4_step(pos_start, vel_start, TOTAL_TIME)

    d_start, a_start = to_alt_dist(pos_start)
    d_end, a_end = to_alt_dist(p_end)

    plt.plot([d_start, d_end], [a_start, a_end], 'b-', marker='x', markersize=10, label='Original (dt=500s x 1)')

    # --- グラフ設定 ---
    plt.title(f"Trajectory Visualization by Step Size (Total {TOTAL_TIME}s)")
    plt.xlabel("Horizontal Distance [km]")
    plt.ylabel("Altitude [km]")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    visualize_rk_steps()