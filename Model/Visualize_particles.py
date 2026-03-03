# !pip install plotly scipy numpy # 初回のみ実行

import numpy as np
from scipy.stats import maxwell
import plotly.graph_objects as go
import plotly.io as pio # plotly.ioをインポート

pio.renderers.default = 'browser'
# --- シミュレーション部分は変更なし ---
PI = np.pi

# <<< 変更点 1: 物理定数に水星の半径を追加 ---
PHYSICAL_CONSTANTS = {
    'MASS_NA': 22.98976928 * 1.66054e-27,
    'K_BOLTZMANN': 1.380649e-23,
    'RADIUS_MERCURY': 2.440e6  # 水星の半径 (m)
}

# <<< 変更点 2: パラメータを調整 ---
TEMPERATURE = 1500.0
N_PARTICLES = 100
FLIGHT_DURATION = 5000.0 # 軌跡が見えるように飛行時間を長くする

def sample_emission_position():
    phi = PI * np.random.random() - (PI / 2.0)
    cos_theta = 2 * np.random.random() - 1.0
    sin_theta = np.sqrt(1.0 - cos_theta ** 2)
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = cos_theta
    return np.array([x, y, z])

def sample_emission_angle_lambertian():
    u1, u2 = np.random.random(2)
    cos_theta = np.sqrt(1 - u2)
    sin_theta = np.sqrt(u2)
    phi = 2 * PI * u1
    return sin_theta, cos_theta, phi

def sample_maxwellian_speed(mass_kg, temp_k):
    scale_param = np.sqrt(PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_k / mass_kg)
    return maxwell.rvs(scale=scale_param)

trajectories = []
radius = PHYSICAL_CONSTANTS['RADIUS_MERCURY'] # <<< 半径を変数として定義

for _ in range(N_PARTICLES):
    # <<< 変更点 3: 初期位置を水星の半径にスケーリング ---
    pos0_unit = sample_emission_position()
    pos0 = pos0_unit * radius # 単位ベクトルに半径を掛ける

    speed = sample_maxwellian_speed(PHYSICAL_CONSTANTS['MASS_NA'], TEMPERATURE)
    sin_theta_dir, cos_theta_dir, phi_dir = sample_emission_angle_lambertian()
    vel_local = np.array([
        speed * sin_theta_dir * np.cos(phi_dir),
        speed * sin_theta_dir * np.sin(phi_dir),
        speed * cos_theta_dir
    ])
    local_z_axis = pos0 / np.linalg.norm(pos0)
    world_up = np.array([0., 0., 1.])
    if np.allclose(local_z_axis, world_up) or np.allclose(local_z_axis, -world_up):
        world_up = np.array([0., 1., 0.])
    local_x_axis = np.cross(world_up, local_z_axis)
    local_x_axis /= np.linalg.norm(local_x_axis)
    local_y_axis = np.cross(local_z_axis, local_x_axis)
    vel_world = (vel_local[0] * local_x_axis + vel_local[1] * local_y_axis + vel_local[2] * local_z_axis)
    pos1 = pos0 + vel_world * FLIGHT_DURATION
    trajectories.append((pos0, pos1))

# ==============================================================================
# Plotlyによる可視化
# ==============================================================================

# <<< 変更点 4: 球体を水星の半径にスケーリング ---
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_sphere = radius * np.outer(np.cos(u), np.sin(v))
y_sphere = radius * np.outer(np.sin(u), np.sin(v))
z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v))

sphere_trace = go.Surface(
    x=x_sphere, y=y_sphere, z=z_sphere,
    colorscale='Greys', opacity=0.8, showscale=False,
    name='Mercury'
)

# 2. 軌跡（線）と放出点（点）のデータを用意
lines_x, lines_y, lines_z = [], [], []
points_x, points_y, points_z = [], [], []

for p0, p1 in trajectories:
    lines_x.extend([p0[0], p1[0], None])
    lines_y.extend([p0[1], p1[1], None])
    lines_z.extend([p0[2], p1[2], None])
    points_x.append(p0[0])
    points_y.append(p0[1])
    points_z.append(p0[2])

# 3. 軌跡（線）を作成
lines_trace = go.Scatter3d(
    x=lines_x, y=lines_y, z=lines_z,
    mode='lines',
    line=dict(color='blue', width=3),
    name='Trajectories'
)

# 4. 放出点（点）を作成
points_trace = go.Scatter3d(
    x=points_x, y=points_y, z=points_z,
    mode='markers',
    marker=dict(color='red', size=3),
    name='Emission Points'
)

# 5. レイアウトを整えて描画
fig = go.Figure(data=[sphere_trace, lines_trace])

# <<< 変更点 5: タイトルと軸ラベルを更新 ---
fig.update_layout(
    title=f'Trajectories of {N_PARTICLES} Particles from Mercury-sized Sphere',
    scene=dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (m)',
        aspectmode='data' # アスペクト比を1:1:1に
    )
)

fig.show()