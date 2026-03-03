# !pip install plotly scipy numpy # 初回のみ実行

import numpy as np
from scipy.stats import maxwell
import plotly.graph_objects as go
import plotly.io as pio

# --- 表示設定 ---
pio.renderers.default = 'browser'

# ==============================================================================
# 物理定数とシミュレーションパラメータ (変更なし)
# ==============================================================================
PI = np.pi

PHYSICAL_CONSTANTS = {
    'MASS_NA': 22.98976928 * 1.66054e-27,
    'K_BOLTZMANN': 1.380649e-23,
    'G': 6.6743e-11,
    'MASS_MERCURY': 3.302e23,
    'RADIUS_MERCURY': 2.440e6
}

TEMPERATURE = 1500.0
FLIGHT_DURATION = 5000
DT = 10
USE_GRAVITY = False

ATOMS_PER_SUPERPARTICLE = 1e24
F_UV_1AU = 1.5e14 * (100)**2
Q_PSD = 2.0e-20 / (100)**2
AU = 1.496e11
r0 = 0.4 * AU
SIGMA_NA_INITIAL = 1.5e23 / (1e3)**2

N_LAT = 24
N_LON = 48

# ==============================================================================
# サンプリング関数 (変更なし)
# ==============================================================================
def sample_emission_angle_lambertian():
    u1, u2 = np.random.random(2)
    cos_theta = np.sqrt(1 - u2)
    sin_theta = np.sqrt(u2)
    phi = 2 * PI * u1
    return sin_theta, cos_theta, phi

def sample_maxwellian_speed(mass_kg, temp_k):
    scale_param = np.sqrt(PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_k / mass_kg)
    return maxwell.rvs(scale=scale_param)

# ==============================================================================
# 粒子放出位置の生成 (変更なし)
# ==============================================================================
print("光刺激脱離モデルに基づいて、放出される粒子を生成しています...")
initial_positions = []
lat_rad = np.linspace(-PI / 2, PI / 2, N_LAT)
lon_rad = np.linspace(-PI, PI, N_LON)
dlat = lat_rad[1] - lat_rad[0]
dlon = lon_rad[1] - lon_rad[0]

for i in range(N_LAT):
    for j in range(N_LON):
        lat_center = lat_rad[i]
        lon_center = lon_rad[j]
        cos_Z = np.cos(lat_center) * np.cos(lon_center)
        if cos_Z <= 0:
            continue
        cell_area = (PHYSICAL_CONSTANTS['RADIUS_MERCURY'] ** 2) * np.cos(lat_center) * dlat * dlon
        F_UV = F_UV_1AU * (AU / r0) ** 2
        R_PSD = F_UV * Q_PSD * cos_Z * SIGMA_NA_INITIAL
        n_ejected_total = R_PSD * cell_area * DT
        num_superparticles_to_plot = int(np.floor(n_ejected_total / ATOMS_PER_SUPERPARTICLE))
        if num_superparticles_to_plot == 0:
            continue
        for _ in range(num_superparticles_to_plot):
            lat_point = lat_center + (np.random.rand() - 0.5) * dlat
            lon_point = lon_center + (np.random.rand() - 0.5) * dlon
            radius_m = PHYSICAL_CONSTANTS['RADIUS_MERCURY']
            x = radius_m * np.cos(lat_point) * np.cos(lon_point)
            y = radius_m * np.cos(lat_point) * np.sin(lon_point)
            z = radius_m * np.sin(lat_point)
            initial_positions.append(np.array([x, y, z]))

print(f"合計 {len(initial_positions)} 個のスーパーパーティクルを生成しました。")
print("各粒子の軌道計算を開始します...")

# ==============================================================================
# メインのシミュレーション処理 (変更なし)
# ==============================================================================
all_trajectories = []
for initial_pos in initial_positions:
    pos = initial_pos.copy()
    speed = sample_maxwellian_speed(PHYSICAL_CONSTANTS['MASS_NA'], TEMPERATURE)
    sin_theta_dir, cos_theta_dir, phi_dir = sample_emission_angle_lambertian()
    vel_local = np.array([
        speed * sin_theta_dir * np.cos(phi_dir),
        speed * sin_theta_dir * np.sin(phi_dir),
        speed * cos_theta_dir
    ])
    local_z_axis = pos / np.linalg.norm(pos)
    world_up = np.array([0., 0., 1.])
    if np.allclose(local_z_axis, world_up) or np.allclose(local_z_axis, -world_up):
        world_up = np.array([0., 1., 0.])
    local_x_axis = np.cross(world_up, local_z_axis)
    local_x_axis /= np.linalg.norm(local_x_axis)
    local_y_axis = np.cross(local_z_axis, local_x_axis)
    vel = (vel_local[0] * local_x_axis + vel_local[1] * local_y_axis + vel_local[2] * local_z_axis)
    particle_trajectory = []
    for t in np.arange(0, FLIGHT_DURATION, DT):
        current_radius_sq = np.sum(pos ** 2)
        if current_radius_sq < PHYSICAL_CONSTANTS['RADIUS_MERCURY'] ** 2 and t > 0:
            break
        particle_trajectory.append(pos.copy())
        if USE_GRAVITY:
            g_accel = -PHYSICAL_CONSTANTS['G'] * PHYSICAL_CONSTANTS['MASS_MERCURY'] * pos / (current_radius_sq ** 1.5)
            vel += g_accel * DT
        pos += vel * DT
    all_trajectories.append(particle_trajectory)

print("軌道計算が完了しました。可視化処理を開始します。")

# ==============================================================================
# ★★★ Plotlyによる可視化 (ここを修正) ★★★
# ==============================================================================
# 水星半径 [m] を変数として保持
radius = PHYSICAL_CONSTANTS['RADIUS_MERCURY']

# --- 水星の球体データを作成 (メートル単位) ---
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_sphere_m = radius * np.outer(np.cos(u), np.sin(v))
y_sphere_m = radius * np.outer(np.sin(u), np.sin(v))
z_sphere_m = radius * np.outer(np.ones(np.size(u)), np.cos(v))

sphere_trace = go.Surface(
    x=x_sphere_m / radius, y=y_sphere_m / radius, z=z_sphere_m / radius,
    colorscale='Greys', opacity=0.8, showscale=False, name='Mercury'
)

# --- 軌道データをリストに格納 (メートル単位) ---
lines_x_m, lines_y_m, lines_z_m = [], [], []
for trajectory in all_trajectories:
    if not trajectory: continue
    x, y, z = zip(*trajectory)
    lines_x_m.extend(x)
    lines_y_m.extend(y)
    lines_z_m.extend(z)
    lines_x_m.append(None)
    lines_y_m.append(None)
    lines_z_m.append(None)

# (リスト内包表記を使って、Noneを維持したまま割り算)
lines_x_scaled = [val / radius if val is not None else None for val in lines_x_m]
lines_y_scaled = [val / radius if val is not None else None for val in lines_y_m]
lines_z_scaled = [val / radius if val is not None else None for val in lines_z_m]

lines_trace = go.Scatter3d(
    x=lines_x_scaled, y=lines_y_scaled, z=lines_z_scaled,
    mode='lines', line=dict(color='orange', width=2), name='Trajectories'
)

fig = go.Figure(data=[sphere_trace, lines_trace])

fig.update_layout(
    title=f'Na Particle Trajectories from Mercury (Gravity: {USE_GRAVITY}, PSD Model)',
    scene=dict(
        xaxis_title='X [R_mercury]',
        yaxis_title='Y [R_mercury]',
        zaxis_title='Z [R_mercury]',
        aspectmode='data'
    )
)

fig.show()