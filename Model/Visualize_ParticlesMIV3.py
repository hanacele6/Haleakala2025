# -*- coding: utf-8 -*-
"""
水星外気圏 3D粒子軌道シミュレーション (ver 2025.10.21)
★★★ MIVモデル - 生成位置プロット版 (TAA連動) ★★★

概要:
このスクリプトは、MIV（微小隕石衝突気化）モデルに基づいて
水星表面の粒子生成位置をサンプリングし、
その位置をPlotlyを使用して3D球面上に可視化します。

★★★ 変更点 ★★★
- 軌道ファイル('orbit2025_v5.txt')の読み込みを復活。
- ユーザーが TARGET_TAA を指定可能に。
- MIVの物理モデルに基づき、TAA(太陽距離)に応じて
- プロットする粒子総数(TOTAL_PARTICLES_TO_LAUNCH)を
- 自動的にスケーリングする機能を追加。
- (例: TAA=0 (近日点) で粒子数最大、TAA=180 (遠日点) で最小)

座標系: (変更なし)
- 原点: 水星の中心
- +X軸: 水星から太陽へ向かう方向
- +Z軸: 水星の公転軌道面に対して北向き
- +Y軸: 右手系を完成させる方向（Dusk側）
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import sys
from typing import Dict, List, Tuple

# --- 表示設定 ---
pio.renderers.default = 'browser'

# ==============================================================================
# 物理定数とシミュレーションパラメータ
# ==============================================================================
PI = np.pi
AU = 1.496e11  # Astronomical Unit [m]

# Physical constants
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

# --- MIV model specific constants ---
MIV_TEMPERATURE = 3000.0
# (ComplexSim No.3 より) 太陽距離依存のべき指数 (R_Hel ^ -1.9)
MIV_R_HEL_POWER_LAW = 1.9

# --- ★★★ MODIFIED: TAA連動設定 ★★★ ---
# ユーザーが変更するTAA
TARGET_TAA = 180  # (例: 0.0, 90.0, 180.0 など)

# TAA=0 (近日点) のときの基準となる粒子数
BASE_PARTICLES_AT_PERIHELION = 5000
# --- ★★★ END MODIFIED ★★★ ---


# ==============================================================================
# ★★★ MODIFIED: 外部ファイルから軌道情報を読み込む ★★★
# ==============================================================================
# グローバル変数 (TAA, AU_val)
TAA: float = 0.0
AU_val: float = 0.0
AU_PERIHELION: float = 0.0  # 近日点距離 [AU]

try:
    filename = 'orbit2025_v5.txt'
    # skiprows=1 でヘッダーを読み飛ばす
    orbit_data = np.loadtxt(filename, skiprows=1)

    if orbit_data.ndim == 1:
        orbit_data = orbit_data.reshape(1, -1)
    if orbit_data.shape[0] == 0:
        raise ValueError(f"{filename} にはデータが含まれていません。")

    # 1列目 (TAA) と 2列目 (AU) を取得
    taa_column = orbit_data[:, 0]
    au_column = orbit_data[:, 1]

    # 1. 近日点距離 (AUの最小値) を取得
    AU_PERIHELION = np.min(au_column)

    # 2. 指定された TARGET_TAA に最も近い行を探す
    target_index = np.argmin(np.abs(taa_column - TARGET_TAA))

    # 3. 選択された行から TAA と AU_val を取得
    selected_row = orbit_data[target_index]
    TAA, AU_val = selected_row[0], selected_row[1]  # 0:TAA, 1:AU

    print(f"軌道情報ファイル '{filename}' を読み込みました。")
    print(f"-> 軌道上の近日点距離 (AU_peri) = {AU_PERIHELION:.4f} AU")
    print(f"-> 目標 TAA {TARGET_TAA:.1f} 度に最も近い TAA = {TAA:.1f} 度の軌道情報を使用します。")
    print(f"-> この地点の太陽距離 (AU_val) = {AU_val:.4f} AU")

except FileNotFoundError:
    print(f"エラー: '{filename}' が見つかりません。", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"軌道情報ファイルの読み込み中にエラーが発生しました: {e}", file=sys.stderr)
    sys.exit(1)

# --- 4. TAA(AU)に基づいてプロットする粒子総数をスケーリング ---
# Flux(AU) ∝ (AU_peri / AU)^MIV_R_HEL_POWER_LAW
scaling_factor = (AU_PERIHELION / AU_val) ** MIV_R_HEL_POWER_LAW
TOTAL_PARTICLES_TO_LAUNCH = int(BASE_PARTICLES_AT_PERIHELION * scaling_factor)

print(
    f"-> 粒子数をスケーリングします: {BASE_PARTICLES_AT_PERIHELION} (近日点) * {scaling_factor:.3f} = {TOTAL_PARTICLES_TO_LAUNCH} 個")


# ==============================================================================


# ==============================================================================
# Helper functions (変更なし)
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
print(f"--- MIVモデルに基づき {TOTAL_PARTICLES_TO_LAUNCH} 個の粒子を生成します ---")

# List to store initial positions [m] and velocities [m/s]
particle_properties: List[Tuple[np.ndarray, np.ndarray]] = []
radius_m = PHYSICAL_CONSTANTS['RADIUS_MERCURY']

# Rejection sampling constant M (2:1 Ratio)
M_rejection = 4.0 / 3.0

print("Generating particles based on MIV longitude (2:1) and latitude distribution...")

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
)

# --- 2. Create Particle Generation Positions Data (Scatter Plot) ---
positions_m = [prop[0] for prop in particle_properties]

# Plot points slightly outside the surface (1.02 * R_M)
plot_radius_factor = 1.02
points_x_scaled = [(pos[0] / radius) * plot_radius_factor for pos in positions_m]
points_y_scaled = [(pos[1] / radius) * plot_radius_factor for pos in positions_m]
points_z_scaled = [(pos[2] / radius) * plot_radius_factor for pos in positions_m]

points_trace = go.Scatter3d(
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

# Set default camera view (from +X axis)
camera = dict(
    up=dict(x=0, y=0, z=1),  # Z-axis is "up"
    center=dict(x=0, y=0, z=0),  # Look at the origin
    eye=dict(x=2.5, y=0, z=0)  # View from +X (Sun)
)

fig.update_layout(
    # ★★★ MODIFIED: Title に TAA と 粒子総数を表示 ★★★
    title=f'MIV Particle Generation Positions (TAA={TAA:.1f}°, N={TOTAL_PARTICLES_TO_LAUNCH})',
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