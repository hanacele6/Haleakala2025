# !pip install plotly scipy numpy # 初回のみ実行

import numpy as np
from scipy.stats import maxwell
import plotly.graph_objects as go
import plotly.io as pio

# --- 表示設定 ---
pio.renderers.default = 'browser'

# ==============================================================================
# 物理定数とシミュレーションパラメータ
# ==============================================================================
PI = np.pi

# --- 物理定数 ---
PHYSICAL_CONSTANTS = {
    'MASS_NA': 22.98976928 * 1.66054e-27,
    'K_BOLTZMANN': 1.380649e-23,
    'G': 6.6743e-11,
    'MASS_MERCURY': 3.302e23,

    'RADIUS_MERCURY': 2.440e6
}

# --- シミュレーションパラメータ ---
TEMPERATURE = 1500.0
N_PARTICLES = 100
FLIGHT_DURATION = 5000
DT = 10


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
# メインのシミュレーション処理
# ==============================================================================
all_trajectories = []
for _ in range(N_PARTICLES):
    # --- ステップ1: 初期位置を水星表面に設定 ---

    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★ ご要望通り、放出範囲を経度-90°〜+90°の半球に戻しました ★
    phi_pos = PI * np.random.random() - (PI / 2.0)
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

    cos_theta_pos = 2 * np.random.random() - 1.0
    sin_theta_pos = np.sqrt(1.0 - cos_theta_pos ** 2)

    radius = PHYSICAL_CONSTANTS['RADIUS_MERCURY']
    pos = np.array([
        radius * sin_theta_pos * np.cos(phi_pos),
        radius * sin_theta_pos * np.sin(phi_pos),
        radius * cos_theta_pos
    ])

    # --- ステップ2: 初速を計算 (変更なし) ---
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

    # --- ステップ3: 時間発展シミュレーション (変更なし) ---
    particle_trajectory = []
    for t in np.arange(0, FLIGHT_DURATION, DT):
        current_radius_sq = np.sum(pos ** 2)
        if current_radius_sq < radius ** 2 and t > 0:
            break
        particle_trajectory.append(pos.copy())
        g_accel = -PHYSICAL_CONSTANTS['G'] * PHYSICAL_CONSTANTS['MASS_MERCURY'] * pos / (current_radius_sq ** 1.5)
        vel += g_accel * DT
        pos += vel * DT
    all_trajectories.append(particle_trajectory)

# ==============================================================================
# Plotlyによる可視化 (変更なし)
# ==============================================================================
radius = PHYSICAL_CONSTANTS['RADIUS_MERCURY']
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_sphere = radius * np.outer(np.cos(u), np.sin(v))
y_sphere = radius * np.outer(np.sin(u), np.sin(v))
z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v))

sphere_trace = go.Surface(
    x=x_sphere, y=y_sphere, z=z_sphere,
    colorscale='Greys', opacity=0.8, showscale=False, name='Mercury'
)

lines_x, lines_y, lines_z = [], [], []
for trajectory in all_trajectories:
    if not trajectory: continue
    x, y, z = zip(*trajectory)
    lines_x.extend(x)
    lines_y.extend(y)
    lines_z.extend(z)
    lines_x.append(None)
    lines_y.append(None)
    lines_z.append(None)

lines_trace = go.Scatter3d(
    x=lines_x, y=lines_y, z=lines_z,
    mode='lines', line=dict(color='blue', width=2), name='Trajectories'
)

fig = go.Figure(data=[sphere_trace, lines_trace])
fig.update_layout(
    title=f'Particle Trajectories with Gravity of Mercury (Day-side Emission)',
    scene=dict(
        xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
        aspectmode='data'
    )
)

fig.show()