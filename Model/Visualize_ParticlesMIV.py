# -*- coding: utf-8 -*-
"""
水星外気圏 3D粒子軌道シミュレーション (ver 2025.10.21)
★★★ 微小隕石衝突気化 (MIV) モデル版 ★★★

概要:
このスクリプトは、水星の表面から放出された粒子（ナトリウム原子を想定）の
3次元軌道を計算し、Plotlyを使用して可視化します。

主な機能:
- ★★★ 変更 (MIV) ★★★
- 粒子生成モデルを「微小隕石衝突気化(MIV)」に変更。
- 粒子は全球から生成されます。
- 生成位置は、公転の進行方向（Dawn側, -Y軸側）でフラックスが
- 2倍になるよう、経度依存性を持たせます。
- 粒子の初速は、固定温度(MIV_TEMPERATURE)のマクスウェル分布に従います。
- 放出角度は、ランバート（余弦則）分布に従います。
- --------------------------
- 外部ファイル('orbit2025_v5.txt')から軌道パラメータを読み込みます。
- 以下の物理モデル（加速度）の影響をフラグでON/OFF切り替え可能です。
- 計算された全粒子の軌道を3Dグラフで表示します。

座標系: (変更なし)
- 原点: 水星の中心
- +X軸: 水星から太陽へ向かう方向
- +Z軸: 水星の公転軌道面に対して北向き
- +Y軸: 右手系を完成させる方向（Dusk側）
- ※回転座標系
"""

import numpy as np
# from scipy.stats import maxwell # -> 移植する関数で代替
import plotly.graph_objects as go
import plotly.io as pio
import sys
from typing import Dict, Any, List, Tuple, Union, Set

# --- 表示設定 ---
pio.renderers.default = 'browser'

# ==============================================================================
# 物理定数とシミュレーションパラメータ
# ==============================================================================
PI = np.pi
AU = 1.496e11  # 天文単位 (Astronomical Unit) [m]

# シミュレーションで使用する物理定数
PHYSICAL_CONSTANTS: Dict[str, float] = {
    'MASS_NA': 22.98976928 * 1.66054e-27,  # Na原子の質量 [kg]
    'K_BOLTZMANN': 1.380649e-23,  # ボルツマン定数 [J/K]
    'G': 6.6743e-11,  # 万有引力定数 [N m^2/kg^2]
    'MASS_MERCURY': 3.302e23,  # 水星の質量 [kg]
    'RADIUS_MERCURY': 2.440e6,  # 水星の半径 [m]
    'MASS_SUN': 1.989e30,  # 太陽の質量 [kg]
    'C': 299792458.0,  # 光速 [m/s]
    'H': 6.62607015e-34,  # プランク定数 [J・s]
    'E_CHARGE': 1.602176634e-19,  # 電気素量 [C]
    'ME': 9.1093897e-31,  # 電子の質量 [kg]
    'EPSILON_0': 8.854187817e-12  # 真空の誘電率 [F/m]
}

# --- ★★★ 変更 (MIV) ★★★ ---
# MIV (微小隕石衝突気化) モデル用の物理定数を追加
MIV_TEMPERATURE = 3000.0  # MIVによる放出粒子のマクスウェル分布温度 [K]
# --- ★★★ 変更 (MIV) ここまで ★★★ ---


# --- シミュレーション設定 (グローバル変数) ---
FLIGHT_DURATION = 5000  # 粒子の最大飛行（追跡）時間 [s]
DT = 10  # 時間ステップ [s]

# --- 物理モデルのトグルスイッチ ---
USE_GRAVITY = False
USE_RADIATION_PRESSURE = False
USE_SOLAR_GRAVITY = False
USE_CORIOLIS_FORCES = False

TARGET_TAA = 0.0

# --- ★★★ 変更 (MIV) ★★★ ---
# N_LAT, N_LON はMIVでは不要
# 生成する総粒子数
TOTAL_PARTICLES_TO_LAUNCH = 500
# --- ★★★ 変更 (MIV) ここまで ★★★ ---


# ==============================================================================
# 外部ファイルから軌道情報とスペクトルデータを読み込む
# ==============================================================================

# (このセクションは変更なし)
# --- グローバル変数として軌道情報を格納 ---
r0: float = 0.0
AU_val: float = 0.0
V_MERCURY_RADIAL: float = 0.0
V_MERCURY_TANGENTIAL: float = 0.0
TAA: float = 0.0

try:
    filename = 'orbit2025_v5.txt'
    orbit_data = np.loadtxt(filename, skiprows=1)
    if orbit_data.ndim == 1:
        orbit_data = orbit_data.reshape(1, -1)
    if orbit_data.shape[0] == 0:
        raise ValueError(f"{filename} にはデータが含まれていません。")
    taa_column = orbit_data[:, 0]
    target_index = np.argmin(np.abs(taa_column - TARGET_TAA))
    selected_row = orbit_data[target_index]
    TAA, AU_val, _, V_radial_ms, V_tangential_ms = selected_row
    r0 = AU_val * AU
    V_MERCURY_RADIAL = V_radial_ms
    V_MERCURY_TANGENTIAL = V_tangential_ms
    print(f"軌道情報ファイル '{filename}' を読み込みました。")
    print(f"-> 目標 TAA {TARGET_TAA:.1f} 度に最も近い TAA = {TAA:.1f} 度の軌道情報を使用します。")
    print(f"-> 太陽距離 = {AU_val:.3f} AU")
    print(f"-> V_radial = {V_MERCURY_RADIAL:.1f} m/s, V_tangential = {V_MERCURY_TANGENTIAL:.1f} m/s")
except FileNotFoundError:
    print(f"エラー: '{filename}' が見つかりません。", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"軌道情報ファイルの読み込み中にエラーが発生しました: {e}", file=sys.stderr)
    sys.exit(1)

# --- 太陽スペクトルデータの読み込み ---
# (このセクションは変更なし)
spec_data_dict: Dict[str, Any] = {}
if USE_RADIATION_PRESSURE:
    try:
        spec_data_np = np.loadtxt('SolarSpectrum_Na0.txt', usecols=(0, 3))
        wl_angstrom = spec_data_np[:, 0]
        gamma = spec_data_np[:, 1]
        if not np.all(np.diff(wl_angstrom) > 0):
            sort_indices = np.argsort(wl_angstrom)
            wl_angstrom, gamma = wl_angstrom[sort_indices], gamma[sort_indices]
        sigma_const = PHYSICAL_CONSTANTS['E_CHARGE'] ** 2 / (
                4 * PHYSICAL_CONSTANTS['ME'] * PHYSICAL_CONSTANTS['C'] * PHYSICAL_CONSTANTS['EPSILON_0'])
        spec_data_dict = {
            'wl': wl_angstrom, 'gamma': gamma,
            'sigma0_perdnu1': sigma_const * 0.320,
            'sigma0_perdnu2': sigma_const * 0.641,
            'JL': 5.18e14 * 1e4
        }
        print("太陽スペクトルデータを正常に読み込みました。")
    except FileNotFoundError:
        print("エラー: 'SolarSpectrum_Na0.txt' が見つかりません。", file=sys.stderr)
        USE_RADIATION_PRESSURE = False


# ==============================================================================
# 加速度計算の関数
# ==============================================================================

# (このセクション (calculate_radiation_acceleration, get_total_acceleration) は変更なし)
def calculate_radiation_acceleration(
        pos: np.ndarray, vel: np.ndarray, spec_data: Dict[str, Any],
        sun_distance_au: float, orbital_radial_velocity_ms: float
) -> np.ndarray:
    x, y, z = pos
    if x < 0 and np.sqrt(y ** 2 + z ** 2) < PHYSICAL_CONSTANTS['RADIUS_MERCURY']:
        return np.array([0.0, 0.0, 0.0])
    velocity_for_doppler = vel[0] + orbital_radial_velocity_ms
    w_na_d2 = 589.1582e-9 * (1.0 - velocity_for_doppler / PHYSICAL_CONSTANTS['C'])
    w_na_d1 = 589.7558e-9 * (1.0 - velocity_for_doppler / PHYSICAL_CONSTANTS['C'])
    wl, gamma, sigma0_perdnu1, sigma0_perdnu2, JL = spec_data.values()
    if not (wl[0] * 1e-9 <= w_na_d2 < wl[-1] * 1e-9 and wl[0] * 1e-9 <= w_na_d1 < wl[-1] * 1e-9):
        return np.array([0.0, 0.0, 0.0])
    gamma2 = np.interp(w_na_d2 * 1e9, wl, gamma)
    gamma1 = np.interp(w_na_d1 * 1e9, wl, gamma)
    F_lambda_1AU_m = JL * 1e9
    F_lambda_d1 = F_lambda_1AU_m / (sun_distance_au ** 2) * gamma1
    F_nu_d1 = F_lambda_d1 * (w_na_d1 ** 2) / PHYSICAL_CONSTANTS['C']
    J1 = sigma0_perdnu1 * F_nu_d1
    F_lambda_d2 = F_lambda_1AU_m / (sun_distance_au ** 2) * gamma2
    F_nu_d2 = F_lambda_d2 * (w_na_d2 ** 2) / PHYSICAL_CONSTANTS['C']
    J2 = sigma0_perdnu2 * F_nu_d2
    b = (1 / PHYSICAL_CONSTANTS['MASS_NA']) * (
            (PHYSICAL_CONSTANTS['H'] / w_na_d1) * J1 + (PHYSICAL_CONSTANTS['H'] / w_na_d2) * J2)
    return np.array([-b, 0.0, 0.0])


def get_total_acceleration(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    accel = np.array([0.0, 0.0, 0.0])
    G = PHYSICAL_CONSTANTS['G']
    if USE_GRAVITY:
        r_sq = np.sum(pos ** 2)
        if r_sq > 0:
            grav_accel = -G * PHYSICAL_CONSTANTS['MASS_MERCURY'] * pos / (r_sq ** 1.5)
            accel += grav_accel
    if USE_SOLAR_GRAVITY:
        x, y, z = pos
        M_SUN = PHYSICAL_CONSTANTS['MASS_SUN']
        r_ps_vec = np.array([r0 - x, -y, -z])
        r_ps_mag_sq = np.sum(r_ps_vec ** 2)
        if r_ps_mag_sq > 0:
            r_ps_mag = np.sqrt(r_ps_mag_sq)
            accel_sun = (G * M_SUN * r_ps_vec) / (r_ps_mag ** 3)
            accel += accel_sun
    if USE_RADIATION_PRESSURE:
        rad_accel = calculate_radiation_acceleration(
            pos, vel, spec_data_dict, AU_val, V_MERCURY_RADIAL
        )
        accel += rad_accel
    if USE_CORIOLIS_FORCES:
        omega_val = V_MERCURY_TANGENTIAL / r0
        accel_combined_centrifugal = np.array([
            omega_val ** 2 * (pos[0] - r0),
            omega_val ** 2 * pos[1],
            0.0
        ])
        accel += accel_combined_centrifugal
        accel_coriolis = np.array([
            2 * omega_val * (vel[1] - V_MERCURY_TANGENTIAL),
            -2 * omega_val * vel[0],
            0.0
        ])
        accel += accel_coriolis
    return accel


# ==============================================================================
# ★★★ 変更 (MIV) ★★★
# 物理モデルに基づくヘルパー関数群 (ComplexSimから移植)
# ==============================================================================

# def calculate_surface_temperature(...) # MIVの生成には不要

def sample_maxwellian_speed(mass_kg, temp_k):
    """ マクスウェル分布に従う速さ [m/s] をサンプリング """
    scale_param = np.sqrt(PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_k / mass_kg)
    # 3次元の正規分布から速度ベクトル(vx, vy, vz)を生成し、その大きさを返す
    vx, vy, vz = np.random.normal(0, scale_param, 3)
    return np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)


def sample_lambertian_direction_local():
    """ ランバート（余弦則）分布に従う方向ベクトルをローカル座標系 (Z軸=法線) で生成 """
    u1, u2 = np.random.random(2)
    phi = 2 * PI * u1
    cos_theta = np.sqrt(1 - u2)
    sin_theta = np.sqrt(u2)
    return np.array([sin_theta * np.cos(phi),
                     sin_theta * np.sin(phi),
                     cos_theta])


def transform_local_to_world(local_vec, normal_vector):
    """ ローカル座標系 (法線=Z) のベクトルをワールド座標系に変換 """
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
    """ 経度・緯度 [rad] を三次元直交座標 [m] に変換 """
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return np.array([x, y, z])


# ==============================================================================
# ★★★ 変更 (MIV) ★★★
# 粒子放出位置の生成 (MIVモデル)
# ==============================================================================
print(f"--- MODIFIED: MIVモデルに基づき {TOTAL_PARTICLES_TO_LAUNCH} 個の粒子を生成します ---")

# 粒子の初期位置 [m] と初期速度 [m/s] を格納するリスト
# (pos, vel) のタプルを格納
particle_properties: List[Tuple[np.ndarray, np.ndarray]] = []
radius_m = PHYSICAL_CONSTANTS['RADIUS_MERCURY']

# MIVロジック (ComplexSim No.3, line 620)
# P(lon) ∝ (1 - (1/3)sin(lon))
# 座標系: +X=Sun, +Y=Dusk, -Y=Dawn.
# sin(lon) は +Y (Dusk) で正、-Y (Dawn) で負。
# よって (1 - (1/3)sin(lon)) は Dawn側で大きくなり、Dusk側で小さくなる。
# これは「Dawn側でフラックスが2倍」という物理と一致する。
#
# リジェクションサンプリング用の定数 M
# P(lon) の最大値は lon = -pi/2 (Dawn) のときで、 1 - (1/3)*(-1) = 4/3
# P(lon) の最小値は lon = +pi/2 (Dusk) のときで、 1 - (1/3)*(+1) = 2/3
# (Dusk/Dawn = (2/3) / (4/3) = 1/2。DawnがDuskの2倍。OK)
#
# P(lon)を正規化していないが、
# P(lon) ∝ (1 - (1/3)sin(lon))
# 提案分布 g(lon) = 1/(2pi) (一様分布)
# M = max( P(lon) / g(lon) ) ∝ max( 1 - (1/3)sin(lon) ) = 4/3
#
# よって、採択確率 f = P(lon) / (M * g(lon))
#                  ∝ (1 - (1/3)sin(lon)) / (4/3)
M_rejection = 4.0 / 3.0

print("MIVの経度・緯度分布に基づき粒子を生成中...")

for _ in range(TOTAL_PARTICLES_TO_LAUNCH):

    # --- 1. 経度のサンプリング (リジェクション法) ---
    while True:
        # 候補となる経度 lon を [-pi, pi] から一様に選ぶ
        random_lon_rad = np.random.uniform(-PI, PI)

        # 採択確率を計算
        # (ComplexSim No.3, line 622 のロジックをそのまま使用)
        prob_accept = (1.0 - (1.0 / 3.0) * np.sin(random_lon_rad)) / M_rejection

        # 0-1の一様乱数がこの確率より小さければ採択
        if np.random.random() < prob_accept:
            break  # while ループを抜ける

    # --- 2. 緯度のサンプリング (面積均等) ---
    # sin(lat) を [-1, 1] から一様に選び、arcsin をとる
    random_lat_rad = np.arcsin(np.random.uniform(-1.0, 1.0))

    # --- 3. 初期位置・速度の計算 ---
    initial_pos = lonlat_to_xyz(random_lon_rad, random_lat_rad, radius_m)
    surface_normal = initial_pos / np.linalg.norm(initial_pos)

    # 速度: MIV用の固定温度 (MIV_TEMPERATURE) でのマクスウェル分布
    speed = sample_maxwellian_speed(PHYSICAL_CONSTANTS['MASS_NA'], MIV_TEMPERATURE)

    # 速度ベクトル (ローカル座標系 -> ワールド座標系)
    initial_vel = speed * transform_local_to_world(sample_lambertian_direction_local(),
                                                   surface_normal)

    # 粒子の初期状態 (位置, 速度) をリストに追加
    particle_properties.append((initial_pos, initial_vel))

print(f"合計 {len(particle_properties)} 個のスーパーパーティクルを生成しました。")
print("各粒子の軌道計算を開始します...")

# ==============================================================================
# メインシミュレーション
# ==============================================================================

# 全粒子の軌道データ（位置ベクトルのリスト）を格納するリスト
all_trajectories: List[List[np.ndarray]] = []

# --- ★★★ 変更 (MIV) ★★★ ---
# 生成された初期位置と初期速度のリストに対して軌道計算を実行
for initial_pos, initial_vel in particle_properties:

    # 位置ベクトル `pos` と速度ベクトル `vel` を初期化
    pos = initial_pos.copy()  # [m]
    vel = initial_vel.copy()  # [m/s]

    # --- 削除 (MIV) ---
    # 以前の「MODIFIED: 速度と角度を固定」ブロックを削除
    # --- 削除 (MIV) ここまで ---

    # この粒子の軌道（位置ベクトルのリスト）を格納するリスト
    particle_trajectory: List[np.ndarray] = []
    # --- ★★★ 変更 (MIV) ここまで ★★★ ---

    # 時間 0 から FLIGHT_DURATION まで、DT 刻みでループ
    for t in np.arange(0, FLIGHT_DURATION, DT):

        # 粒子が水星表面より内側に入ったら、衝突（吸着）とみなす
        if np.sum(pos ** 2) < PHYSICAL_CONSTANTS['RADIUS_MERCURY'] ** 2 and t > 0:
            break  # この粒子の追跡を終了

        # 現在の位置を軌道リストに追加
        particle_trajectory.append(pos.copy())

        # --- 4次のルンゲ＝クッタ（RK4）法による数値積分 ---
        # (この RK4 ブロックは変更なし)
        a1 = get_total_acceleration(pos, vel)
        k1_vel = DT * a1
        k1_pos = DT * vel
        a2 = get_total_acceleration(pos + 0.5 * k1_pos, vel + 0.5 * k1_vel)
        k2_vel = DT * a2
        k2_pos = DT * (vel + 0.5 * k1_vel)
        a3 = get_total_acceleration(pos + 0.5 * k2_pos, vel + 0.5 * k2_vel)
        k3_vel = DT * a3
        k3_pos = DT * (vel + 0.5 * k2_vel)
        a4 = get_total_acceleration(pos + k3_pos, vel + k3_vel)
        k4_vel = DT * a4
        k4_pos = DT * (vel + k3_vel)
        pos += (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos) / 6.0
        vel += (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel) / 6.0
        # --- RK4 終了 ---

    # この粒子の全軌道データを、全軌道リストに追加
    all_trajectories.append(particle_trajectory)

print("軌道計算が完了しました。可視化処理を開始します。")

# ==============================================================================
# Plotlyによる可視化
# ==============================================================================

# (このセクションは変更なし)
radius = PHYSICAL_CONSTANTS['RADIUS_MERCURY']

# --- 1. 水星の球体メッシュを作成 ---
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_sphere_m = radius * np.outer(np.cos(u), np.sin(v))
y_sphere_m = radius * np.outer(np.sin(u), np.sin(v))
z_sphere_m = radius * np.outer(np.ones(np.size(u)), np.cos(v))

sphere_trace = go.Surface(
    x=x_sphere_m / radius, y=y_sphere_m / radius, z=z_sphere_m / radius,
    colorscale='Greys', opacity=0.8, showscale=False, name='水星',
    lightposition=dict(x=10000, y=0, z=0),
    lighting=dict(ambient=0.2, diffuse=1.0, specular=0.0)
)

# --- 2. 粒子の軌道データ（ライン）を作成 ---
lines_x_m: List[Union[float, None]] = []
lines_y_m: List[Union[float, None]] = []
lines_z_m: List[Union[float, None]] = []

for trajectory in all_trajectories:
    if not trajectory: continue
    x, y, z = zip(*trajectory)
    lines_x_m.extend(x)
    lines_y_m.extend(y)
    lines_z_m.extend(z)
    lines_x_m.append(None)
    lines_y_m.append(None)
    lines_z_m.append(None)

lines_x_scaled = [val / radius if val is not None else None for val in lines_x_m]
lines_y_scaled = [val / radius if val is not None else None for val in lines_y_m]
lines_z_scaled = [val / radius if val is not None else None for val in lines_z_m]

lines_trace = go.Scatter3d(
    x=lines_x_scaled, y=lines_y_scaled, z=lines_z_scaled,
    mode='lines',
    line=dict(color='orange', width=2),
    name='粒子軌道'
)

# --- 3. グラフの作成と表示 ---
fig = go.Figure(data=[sphere_trace, lines_trace])
fig.update_layout(
    title=f'水星粒子軌道 (MIVモデル, TAA={TAA:.1f}°, Coriolis:{USE_CORIOLIS_FORCES})',
    scene=dict(
        xaxis_title='X [R_M]',
        yaxis_title='Y [R_M]',
        zaxis_title='Z [R_M]',
        aspectmode='data'
    ),
    legend=dict(x=0, y=1)
)
fig.show()