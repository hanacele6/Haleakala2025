# -*- coding: utf-8 -*-
"""
水星外気圏 3D粒子軌道シミュレーション (ver 2025.10.21)
★★★ MIVモデル - 生成位置プロット版 ★★★

概要:
このスクリプトは、MIV（微小隕石衝突気化）モデルに基づいて
水星表面の粒子生成位置をサンプリングし、
その位置をPlotlyを使用して3D球面上に可視化します。

★★★ 変更点 ★★★
- 軌道計算、加速度計算、軌道ファイル読み込みのロジックをすべて削除。
- 粒子が生成された初期位置のみを収集し、3D散布図としてプロットします。
- hidesurface=True を使用し、球体の裏側を隠すように修正。
- points_trace の Y座標指定のバグを修正。
- ★★★ 修正: デフォルトのカメラアングルを+X軸（太陽）視点に変更 ★★★

座標系: (変更なし)
- 原点: 水星の中心
- +X軸: 水星から太陽へ向かう方向
- +Z軸: 水星の公転軌道面に対して北向き
- +Y軸: 右手系を完成させる方向（Dusk側）
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from typing import Dict, List, Tuple

# --- 表示設定 ---
pio.renderers.default = 'browser'

# ==============================================================================
# 物理定数とシミュレーションパラメータ
# ==============================================================================
PI = np.pi
AU = 1.496e11  # Astronomical Unit [m]

# Physical constants used in the simulation
PHYSICAL_CONSTANTS: Dict[str, float] = {
    'MASS_NA': 22.98976928 * 1.66054e-27,
    'K_BOLTZMANN': 1.380649e-23,
    'RADIUS_MERCURY': 2.440e6,  # Radius of Mercury [m]
    'C': 299792458.0,
    'H': 6.62607015e-34,
    'E_CHARGE': 1.602176634e-19,
    'ME': 9.1093897e-31,
    'EPSILON_0': 8.854187817e-12
}

# --- MIV model specific constant ---
MIV_TEMPERATURE = 3000.0  # (Used only for dummy velocity calculation)

# --- Number of particles to generate and plot ---
TOTAL_PARTICLES_TO_LAUNCH = 1000


# ==============================================================================
# Helper functions based on physical models (only those needed for MIV generation)
# ==============================================================================

def sample_maxwellian_speed(mass_kg, temp_k):
    """ Samples speed [m/s] from a Maxwellian distribution (not used for plotting positions). """
    scale_param = np.sqrt(PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_k / mass_kg)
    vx, vy, vz = np.random.normal(0, scale_param, 3)
    return np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)


def sample_lambertian_direction_local():
    """ Samples a direction vector following Lambertian distribution (not used for plotting positions). """
    u1, u2 = np.random.random(2)
    phi = 2 * PI * u1
    cos_theta = np.sqrt(1 - u2)
    sin_theta = np.sqrt(u2)
    return np.array([sin_theta * np.cos(phi),
                     sin_theta * np.sin(phi),
                     cos_theta])


def transform_local_to_world(local_vec, normal_vector):
    """ Transforms a vector from local to world coordinates (not used for plotting positions). """
    local_z_axis = normal_vector / np.linalg.norm(normal_vector)
    world_up = np.array([0., 0., 1.])
    if np.allclose(local_z_axis, world_up) or np.allclose(local_z_axis, -world_up):
        world_up = np.array([0., 1., 0.])
    local_x_axis = np.cross(world_up, local_z_axis)
    local_x_axis /= np.linalg.norm(local_x_axis)
    local_y_axis = np.cross(local_z_axis, local_x_axis)
    return (local_vec[0] * local_x_axis +
            local_vec[1] * local_y_axis +
            local_vec[2] * local_z_axis)


def lonlat_to_xyz(lon_rad, lat_rad, radius):
    """ Converts longitude/latitude [rad] to 3D Cartesian coordinates [m] (Essential). """
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return np.array([x, y, z])


# ==============================================================================
# Particle Generation (MIV Model)
# ==============================================================================
print(f"--- MODIFIED: Generating {TOTAL_PARTICLES_TO_LAUNCH} particles based on MIV model ---")

# List to store initial positions [m] and velocities [m/s]
particle_properties: List[Tuple[np.ndarray, np.ndarray]] = []
radius_m = PHYSICAL_CONSTANTS['RADIUS_MERCURY']

# Rejection sampling constant M
M_rejection = 4.0 / 3.0

print("Generating particles based on MIV longitude and latitude distribution...")

for _ in range(TOTAL_PARTICLES_TO_LAUNCH):

    # --- 1. Longitude Sampling (Rejection Method) ---
    while True:
        random_lon_rad = np.random.uniform(-PI, PI)
        prob_accept = (1.0 - (1.0 / 3.0) * np.sin(random_lon_rad)) / M_rejection
        if np.random.random() < prob_accept:
            break

    # --- 2. Latitude Sampling (Area-uniform) ---
    random_lat_rad = np.arcsin(np.random.uniform(-1.0, 1.0))

    # --- 3. Calculate Initial Position and (dummy) Velocity ---
    initial_pos = lonlat_to_xyz(random_lon_rad, random_lat_rad, radius_m)

    # Dummy velocity calculation to maintain original logic structure
    surface_normal = initial_pos / np.linalg.norm(initial_pos)
    speed = sample_maxwellian_speed(PHYSICAL_CONSTANTS['MASS_NA'], MIV_TEMPERATURE)
    initial_vel = speed * transform_local_to_world(sample_lambertian_direction_local(),
                                                   surface_normal)

    # Add particle's initial state (position, velocity) to the list
    particle_properties.append((initial_pos, initial_vel))

print(f"Total {len(particle_properties)} superparticles generated.")
print("Starting visualization process.")

# ==============================================================================
# Plotly Visualization (Generation Positions Only)
# ==============================================================================

radius = PHYSICAL_CONSTANTS['RADIUS_MERCURY']

# --- 1. Create Mercury Sphere Mesh ---
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_sphere_m = radius * np.outer(np.cos(u), np.sin(v))
y_sphere_m = radius * np.outer(np.sin(u), np.sin(v))
z_sphere_m = radius * np.outer(np.ones(np.size(u)), np.cos(v))

# Scale coordinates to Mercury Radii for plotting
x_sphere_scaled = x_sphere_m / radius
y_sphere_scaled = y_sphere_m / radius
z_sphere_scaled = z_sphere_m / radius

sphere_trace = go.Surface(
    x=x_sphere_scaled, y=y_sphere_scaled, z=z_sphere_scaled,
    colorscale='Greys',
    opacity=1.0,  # Set to opaque
    showscale=False,
    name='Mercury Surface',
    lightposition=dict(x=10000, y=0, z=0),
    lighting=dict(ambient=0.2, diffuse=1.0, specular=0.0),

    # hidesurface=True は球体自体を消してしまうため削除
)

# --- 2. Create Particle Generation Positions Data (Scatter Plot) ---
positions_m = [prop[0] for prop in particle_properties]
plot_radius_factor = 1.02
points_x_scaled = [(pos[0] / radius) * plot_radius_factor for pos in positions_m]
points_y_scaled = [(pos[1] / radius) * plot_radius_factor for pos in positions_m]
points_z_scaled = [(pos[2] / radius) * plot_radius_factor for pos in positions_m]

points_trace = go.Scatter3d(
    # Y-coordinate bug fixed
    x=points_x_scaled, y=points_y_scaled, z=points_z_scaled,
    mode='markers',
    marker=dict(
        color='red',
        size=2,
        opacity=0.8
    ),
    name='Generation Points'
)

# --- 3. Create and Display the Graph ---
fig = go.Figure(data=[sphere_trace, points_trace])

# --- ★★★ MODIFIED: Set default camera view ★★★ ---
camera = dict(
    up=dict(x=0, y=0, z=1),  # Z-axis is "up"
    center=dict(x=0, y=0, z=0),  # Look at the origin
    eye=dict(x=2.5, y=0, z=0)  # View from +X (Sun)
)
# --- ★★★ END MODIFIED ★★★ ---

fig.update_layout(
    title=f'MIV Particle Generation Positions (N={TOTAL_PARTICLES_TO_LAUNCH})',
    scene=dict(
        xaxis_title='X [R_M] (to Sun)',
        yaxis_title='Y [R_M] (Dusk)',
        zaxis_title='Z [R_M] (North)',
        aspectmode='data',  # This ensures the 1:1:1 aspect ratio
        camera=camera  # Apply the new camera angle
    ),
    legend=dict(x=0, y=1)
)

fig.show()