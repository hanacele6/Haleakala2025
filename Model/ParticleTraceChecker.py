# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. 物理定数・設定
# ==============================================================================
CONST = {
    'PI': np.pi,
    'RM': 2.440e6,  # 水星半径 [m]
    'GM': 2.2032e13,  # 重力定数 (水星)
    'KB': 1.380649e-23,  # ボルツマン定数
    'M_NA': 3.8175e-26,  # Na質量 [kg]
    'AU_M': 1.496e11,  # 1AU [m]
}

# グリッド設定
GRID_OPTS = {
    'SURF_LON': 72,
    'SURF_LAT': 36,
    'SPATIAL_RES': 101,
}


# ==============================================================================
# 2. 物理計算ヘルパー関数 (改良版RK4)
# ==============================================================================
def get_temp_perihelion_subsolar():
    r_au = 0.307
    return 100.0 + 600.0 * (1.0 ** 0.25) * ((0.306 / r_au) ** 2)


def generate_particles(num_particles, temp_k, lon_center, lat_center, d_lon, d_lat, mode='random'):
    if mode == 'center':
        rnd_lon = np.full(num_particles, lon_center)
        rnd_lat = np.full(num_particles, lat_center)
    else:
        rnd_lon = (np.random.random(num_particles) - 0.5) * d_lon + lon_center
        rnd_lat = (np.random.random(num_particles) - 0.5) * d_lat + lat_center

    x = CONST['RM'] * np.cos(rnd_lat) * np.cos(rnd_lon)
    y = CONST['RM'] * np.cos(rnd_lat) * np.sin(rnd_lon)
    z = CONST['RM'] * np.sin(rnd_lat)
    pos = np.stack([x, y, z], axis=1)

    kT = CONST['KB'] * temp_k
    E = np.random.gamma(2.0, kT, num_particles)
    spd = np.sqrt(2.0 * E / CONST['M_NA'])

    u1, u2 = np.random.random(num_particles), np.random.random(num_particles)
    phi, sin_theta = 2 * CONST['PI'] * u1, np.sqrt(u2)

    lx = sin_theta * np.cos(phi)
    ly = sin_theta * np.sin(phi)
    lz = np.sqrt(1 - u2)

    nx, ny, nz = x / CONST['RM'], y / CONST['RM'], z / CONST['RM']
    world_up = np.array([0, 0, 1])
    vec_e = np.cross(world_up, np.stack([nx, ny, nz], axis=1))
    norm_e = np.linalg.norm(vec_e, axis=1, keepdims=True)
    vec_e = np.where(norm_e > 1e-6, vec_e / norm_e, np.array([1, 0, 0]))
    vec_n = np.cross(np.stack([nx, ny, nz], axis=1), vec_e)

    vel = (lx[:, None] * vec_e + ly[:, None] * vec_n + lz[:, None] * np.stack([nx, ny, nz], axis=1)) * spd[:, None]
    return pos, vel


def get_acceleration(pos):
    r_sq = np.sum(pos ** 2, axis=1)[:, None]
    r_mag = np.sqrt(r_sq)

    # 中心(0,0,0)でのゼロ除算回避
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = -CONST['GM'] * pos / (r_mag ** 3)

    # ★修正箇所★
    # r_mag == 0 は (N, 1) の形状ですが、acc は (N, 3) なので
    # そのままインデックスに使うとエラーになります。
    # .flatten() して (N,) の1次元配列にすれば行を指定できます。
    mask = (r_mag == 0).flatten()
    acc[mask] = 0.0

    return acc


def run_trajectory_rk4(pos, vel, duration, dt_step):
    """
    着地時の位置補間機能付き RK4
    """
    num_steps = int(duration / dt_step)
    # 軌跡配列: [step, particle_id, xyz]
    trajectory = np.zeros((num_steps + 1, len(pos), 3))
    trajectory[0] = pos

    curr_pos, curr_vel = pos.copy(), vel.copy()

    # すでに着地しているかどうかのフラグ
    is_stopped = np.zeros(len(pos), dtype=bool)

    for i in range(num_steps):
        # 既に止まっている粒子は動かさない（前の位置を維持）
        if np.all(is_stopped):
            trajectory[i + 1:] = curr_pos
            break

        # RK4 計算
        k1_v = dt_step * get_acceleration(curr_pos)
        k1_p = dt_step * curr_vel
        k2_v = dt_step * get_acceleration(curr_pos + 0.5 * k1_p)
        k2_p = dt_step * (curr_vel + 0.5 * k1_v)
        k3_v = dt_step * get_acceleration(curr_pos + 0.5 * k2_p)
        k3_p = dt_step * (curr_vel + 0.5 * k2_v)
        k4_v = dt_step * get_acceleration(curr_pos + k3_p)
        k4_p = dt_step * (curr_vel + k3_v)

        next_pos = curr_pos + (k1_p + 2 * k2_p + 2 * k3_p + k4_p) / 6.0
        next_vel = curr_vel + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6.0

        # --- 着地判定と補間 ---
        dist_curr = np.linalg.norm(curr_pos, axis=1)
        dist_next = np.linalg.norm(next_pos, axis=1)

        # 今回のステップで新しく地下に潜った粒子
        just_landed = (~is_stopped) & (dist_next < CONST['RM'])

        if np.any(just_landed):
            r_c = dist_curr[just_landed]
            r_n = dist_next[just_landed]

            # ゼロ除算防止
            denom = r_n - r_c
            denom[denom == 0] = -1e-9

            fraction = (CONST['RM'] - r_c) / denom
            fraction = np.clip(fraction, 0.0, 1.0)

            delta_pos = next_pos[just_landed] - curr_pos[just_landed]
            hit_pos = curr_pos[just_landed] + delta_pos * fraction[:, None]

            next_pos[just_landed] = hit_pos
            next_vel[just_landed] = 0.0
            is_stopped[just_landed] = True

        # 既に止まっている粒子は位置更新しない
        already_stopped = is_stopped & (~just_landed)
        next_pos[already_stopped] = trajectory[i][already_stopped]
        next_vel[already_stopped] = 0.0

        curr_pos = next_pos
        curr_vel = next_vel
        trajectory[i + 1] = curr_pos

    return trajectory, curr_vel


# ==============================================================================
# 3. メイン可視化ロジック
# ==============================================================================
def visualize_detailed_migration():
    # --- 設定 ---
    # 'random': グリッド内でランダムに放出
    # 'center': グリッドの中心一点から放出
    EMISSION_MODE = 'center'

    N_PARTICLES = 10000
    N_TRACE = 30
    TOTAL_TIME = 500.0
    DT_STEP = 10.0

    d_lon = 2 * np.pi / GRID_OPTS['SURF_LON']
    d_lat = np.pi / GRID_OPTS['SURF_LAT']

    print(f"Generating particles... Mode: {EMISSION_MODE}")
    temp = get_temp_perihelion_subsolar()
    pos0, vel0 = generate_particles(N_PARTICLES, temp, 0.0, 0.0, d_lon, d_lat, mode=EMISSION_MODE)

    # 軌道計算
    traj, vel_end = run_trajectory_rk4(pos0, vel0, TOTAL_TIME, DT_STEP)
    pos_end = traj[-1]

    # --- 着地判定 ---
    speed_end = np.linalg.norm(vel_end, axis=1)
    is_landed = speed_end == 0
    is_flying = ~is_landed
    num_landed = np.sum(is_landed)

    # 共通の座標変換関数
    def to_deg(p):
        lon = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
        r = np.linalg.norm(p, axis=1)
        r[r == 0] = 1.0
        lat = np.degrees(np.arcsin(np.clip(p[:, 2] / r, -1, 1)))
        return lon, lat

    def to_ew_alt(p_arr):
        r = np.linalg.norm(p_arr, axis=1)
        alt = (r - CONST['RM']) / 1000.0  # km
        lon = np.arctan2(p_arr[:, 1], p_arr[:, 0])
        d_lon_rad = lon - 0.0
        d_lon_rad = (d_lon_rad + np.pi) % (2 * np.pi) - np.pi
        dist_ew = (CONST['RM'] * d_lon_rad) / 1000.0
        return dist_ew, alt

    # --------------------------------------------------------------------------
    # Figure 1: Top-Down View
    # --------------------------------------------------------------------------
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.set_title(f"Top-Down View ({EMISSION_MODE.capitalize()} Emission)")

    lon_end_deg, lat_end_deg = to_deg(pos_end)
    grid_w_deg = np.degrees(d_lon)
    grid_h_deg = np.degrees(d_lat)

    is_inside_grid = (np.abs(lon_end_deg) <= grid_w_deg / 2) & (np.abs(lat_end_deg) <= grid_h_deg / 2)
    num_out = np.sum(~is_inside_grid)
    percent_out = (num_out / N_PARTICLES) * 100.0

    if np.any(is_flying):
        ax1.scatter(lon_end_deg[is_flying], lat_end_deg[is_flying],
                    s=5, c='blue', alpha=0.4, label='Flying')
    if np.any(is_landed):
        ax1.scatter(lon_end_deg[is_landed], lat_end_deg[is_landed],
                    s=15, c='red', marker='x', alpha=0.8, label='Landed')

    for i in range(-4, 5):
        offset_x = (i + 0.5) * grid_w_deg
        offset_x_m = (i - 0.5) * grid_w_deg
        ax1.axvline(offset_x, color='red', linestyle='-', alpha=0.3)
        ax1.axvline(offset_x_m, color='red', linestyle='-', alpha=0.3)
        offset_y = (i + 0.5) * grid_h_deg
        offset_y_m = (i - 0.5) * grid_h_deg
        ax1.axhline(offset_y, color='red', linestyle='-', alpha=0.3)
        ax1.axhline(offset_y_m, color='red', linestyle='-', alpha=0.3)

    src_rect = plt.Rectangle((-grid_w_deg / 2, -grid_h_deg / 2), grid_w_deg, grid_h_deg,
                             edgecolor='red', facecolor='none', lw=2, label='Source Grid')
    ax1.add_patch(src_rect)

    ax1.text(0.95, 0.95, f"Out of Grid: {percent_out:.1f}%\nLanded: {num_landed / N_PARTICLES * 100:.1f}%",
             transform=ax1.transAxes, ha='right', va='top',
             fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))

    ax1.set_xlabel("Longitude [deg]")
    ax1.set_ylabel("Latitude [deg]")
    ax1.set_aspect('equal')
    ax1.set_xlim(-15, 15)
    ax1.set_ylim(-10, 10)
    ax1.grid(False)

    # --------------------------------------------------------------------------
    # Figure 2: Side View (East-West Cross Section)
    # --------------------------------------------------------------------------
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.set_title(f"East-West Cross Section ({EMISSION_MODE.capitalize()})")

    ew_end, alt_end = to_ew_alt(pos_end)

    if np.any(is_flying):
        ax2.scatter(ew_end[is_flying], alt_end[is_flying], s=5, c='blue', alpha=0.3, label='Flying')
    if np.any(is_landed):
        ax2.scatter(ew_end[is_landed], alt_end[is_landed], s=20, c='red', marker='x', alpha=0.8, label='Landed')

    for i in range(N_TRACE):
        color = 'red' if is_landed[i] else 'black'
        alpha = 0.8 if is_landed[i] else 0.4

        ew_hist, alt_hist = to_ew_alt(traj[:, i, :])
        ax2.plot(ew_hist, alt_hist, c=color, alpha=alpha, lw=1)

    grid_km_w = (CONST['RM'] * d_lon) / 1000.0
    for i in range(0, 5):
        bound_pos = (i + 0.5) * grid_km_w
        ax2.axvline(bound_pos, color='red', linestyle='--', alpha=0.4)
        bound_neg = -(i + 0.5) * grid_km_w
        ax2.axvline(bound_neg, color='red', linestyle='--', alpha=0.4)

    spatial_cell_size = (10.0 * CONST['RM'] / GRID_OPTS['SPATIAL_RES']) / 1000.0
    for i in range(1, 6):
        h_grid = i * spatial_cell_size
        ax2.axhline(h_grid, color='blue', linestyle=':', alpha=0.6, label='Spatial Grid' if i == 1 else "")

    ax2.set_xlabel("East-West Distance [km] (West <--- 0 ---> East)")
    ax2.set_ylabel("Altitude [km]")
    ax2.set_xlim(-600, 600)
    ax2.set_ylim(0, 600)
    ax2.legend(loc='upper right')

    plt.show()


if __name__ == "__main__":
    visualize_detailed_migration()