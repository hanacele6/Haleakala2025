import numpy as np
from scipy.stats import maxwell
import plotly.graph_objects as go
import plotly.io as pio
import sys

# --- 表示設定 ---
pio.renderers.default = 'browser'

# ==============================================================================
# 物理定数とシミュレーションパラメータ
# ==============================================================================
PI = np.pi
AU = 1.496e11  # 天文単位 [m]

PHYSICAL_CONSTANTS = {
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

# --- シミュレーション設定 ---
TEMPERATURE = 1500.0  # 初期速度を決める表面温度 [K]
FLIGHT_DURATION = 5000  # 粒子の最大飛行時間 [s]
DT = 10  # 時間ステップ [s]
USE_GRAVITY = False  # 水星の重力を考慮するかどうかの設定
USE_RADIATION_PRESSURE = False  # 太陽放射圧を考慮するかどうかの設定
USE_SOLAR_GRAVITY = True  # 太陽の重力を考慮するかどうかの設定

# --- 発生源モデルのパラメータ ---
ATOMS_PER_SUPERPARTICLE = 1e24
F_UV_1AU = 1.5e14 * (100) ** 2
Q_PSD = 2.0e-20 / (100) ** 2
SIGMA_NA_INITIAL = 1.5e23 / (1e3) ** 2

N_LAT = 24
N_LON = 48

# ==============================================================================
# 外部ファイルから軌道情報とスペクトルデータを読み込む
# ==============================================================================
try:
    with open('orbit360.txt', 'r') as f:
        first_line = f.readline()
        if not first_line:
            raise ValueError("orbit360.txt is empty or could not be read.")
        TAA, AU_val, _, _, Vms_ms = map(float, first_line.split())
    r0 = AU_val * AU
    V_MERCURY_ORBIT = Vms_ms
    print(f"軌道情報ファイル 'orbit360.txt' を読み込みました。")
    print(f"-> TAA = {TAA:.1f} 度, 太陽距離 = {AU_val:.3f} AU の軌道情報を使用します。")
except FileNotFoundError:
    print("エラー: 'orbit360.txt' が見つかりません。", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"軌道情報ファイルの読み込み中にエラーが発生しました: {e}", file=sys.stderr)
    sys.exit(1)

spec_data_dict = {}
if USE_RADIATION_PRESSURE:
    try:
        spec_data_np = np.loadtxt('SolarSpectrum_Na0.txt', usecols=(0, 3))
        wl, gamma = spec_data_np[:, 0], spec_data_np[:, 1]
        if not np.all(np.diff(wl) > 0):
            sort_indices = np.argsort(wl)
            wl, gamma = wl[sort_indices], gamma[sort_indices]
        sigma_const = PHYSICAL_CONSTANTS['E_CHARGE'] ** 2 / (
                4 * PHYSICAL_CONSTANTS['ME'] * PHYSICAL_CONSTANTS['C'] * PHYSICAL_CONSTANTS['EPSILON_0'])
        spec_data_dict = {
            'wl': wl, 'gamma': gamma,
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
def calculate_radiation_acceleration(pos, vel, spec_data, sun_distance_au, orbital_velocity_ms):
    x, y, z = pos
    if x < 0 and np.sqrt(y ** 2 + z ** 2) < PHYSICAL_CONSTANTS['RADIUS_MERCURY']:
        return np.array([0.0, 0.0, 0.0])
    velocity_for_doppler = vel[0] + orbital_velocity_ms
    w_na_d2 = 589.1582e-9 * (1.0 - velocity_for_doppler / PHYSICAL_CONSTANTS['C'])
    w_na_d1 = 589.7558e-9 * (1.0 - velocity_for_doppler / PHYSICAL_CONSTANTS['C'])
    wl, gamma, sigma0_perdnu1, sigma0_perdnu2, JL = spec_data.values()
    if not (wl[0] * 1e-9 <= w_na_d2 < wl[-1] * 1e-9 and wl[0] * 1e-9 <= w_na_d1 < wl[-1] * 1e-9):
        return np.array([0.0, 0.0, 0.0])
    gamma2, gamma1 = np.interp(w_na_d2 * 1e9, wl, gamma), np.interp(w_na_d1 * 1e9, wl, gamma)
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


def get_total_acceleration(pos, vel):
    """指定された位置と速度における総加速度（重力＋放射圧）を計算する"""
    accel = np.array([0.0, 0.0, 0.0])
    G = PHYSICAL_CONSTANTS['G']

    # 1. 水星の重力
    if USE_GRAVITY:
        r_sq = np.sum(pos ** 2)
        if r_sq > 0:
            grav_accel = -G * PHYSICAL_CONSTANTS['MASS_MERCURY'] * pos / (r_sq ** 1.5)
            accel += grav_accel

    # 2. 太陽の重力
    if USE_SOLAR_GRAVITY:
        # このシミュレーションは水星中心の座標系 (+Xが太陽方向) で行われている。

        x, y, z = pos
        M_SUN = PHYSICAL_CONSTANTS['MASS_SUN']

        # 粒子から太陽へのベクトル (r_ps)
        # 太陽の位置: [r0, 0, 0]
        # 粒子の位置: [x, y, z]
        r_ps_vec = np.array([r0 - x, -y, -z])
        r_ps_mag_sq = np.sum(r_ps_vec ** 2)

        # ゼロ割を避ける
        if r_ps_mag_sq == 0:
            pass  # 原理的に太陽と粒子が重なることはないが念のため
        else:
            r_ps_mag = np.sqrt(r_ps_mag_sq)

            # 水星から太陽へのベクトル (r_Ms)
            r_Ms_vec = np.array([r0, 0.0, 0.0])

            # 粒子が太陽から受ける重力加速度 (a_p)
            # a_p = G * M_sun * r_ps_vec / |r_ps_vec|^3
            accel_sun_on_particle = (G * M_SUN * r_ps_vec) / (r_ps_mag ** 3)

            # 水星が太陽から受ける重力加速度 (a_M)
            # a_M = G * M_sun * r_Ms_vec / |r_Ms_vec|^3
            # (r0 はグローバル変数から読み込まれる)
            accel_sun_on_mercury = (G * M_SUN * r_Ms_vec) / (r0 ** 3)

            # 水星中心座標系での、太陽による相対加速度（潮汐力）
            # a_tidal = a_p - a_M
            accel += (accel_sun_on_particle - accel_sun_on_mercury)

    # 3. 太陽放射圧
    if USE_RADIATION_PRESSURE:
        rad_accel = calculate_radiation_acceleration(pos, vel, spec_data_dict, AU_val, V_MERCURY_ORBIT)
        accel += rad_accel

    return accel


# ==============================================================================
# サンプリング関数
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
# 粒子放出位置の生成
# ==============================================================================
print("光刺激脱離モデルに基づいて、放出される粒子を生成しています...")
initial_positions = []
lat_rad = np.linspace(-PI / 2, PI / 2, N_LAT)
lon_rad = np.linspace(-PI, PI, N_LON)
dlat = lat_rad[1] - lat_rad[0]
dlon = lon_rad[1] - lon_rad[0]
for i in range(N_LAT):
    for j in range(N_LON):
        lat_center, lon_center = lat_rad[i], lon_rad[j]
        cos_Z = np.cos(lat_center) * np.cos(lon_center)
        if cos_Z <= 0: continue
        cell_area = (PHYSICAL_CONSTANTS['RADIUS_MERCURY'] ** 2) * np.cos(lat_center) * dlat * dlon
        F_UV = F_UV_1AU * (AU / r0) ** 2
        R_PSD = F_UV * Q_PSD * cos_Z * SIGMA_NA_INITIAL
        n_ejected_total = R_PSD * cell_area * DT
        num_superparticles_to_plot = int(np.floor(n_ejected_total / ATOMS_PER_SUPERPARTICLE))
        if num_superparticles_to_plot == 0: continue
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
# メインシミュレーション
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
        if np.sum(pos ** 2) < PHYSICAL_CONSTANTS['RADIUS_MERCURY'] ** 2 and t > 0:
            break
        particle_trajectory.append(pos.copy())

        # k1
        k1_vel = DT * get_total_acceleration(pos, vel)
        k1_pos = DT * vel
        # k2
        k2_vel = DT * get_total_acceleration(pos + 0.5 * k1_pos, vel + 0.5 * k1_vel)
        k2_pos = DT * (vel + 0.5 * k1_vel)
        # k3
        k3_vel = DT * get_total_acceleration(pos + 0.5 * k2_pos, vel + 0.5 * k2_vel)
        k3_pos = DT * (vel + 0.5 * k2_vel)
        # k4
        k4_vel = DT * get_total_acceleration(pos + k3_pos, vel + k3_vel)
        k4_pos = DT * (vel + k3_vel)

        # 最終的な位置と速度の更新
        pos += (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos) / 6.0
        vel += (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel) / 6.0

    all_trajectories.append(particle_trajectory)

print("軌道計算が完了しました。可視化処理を開始します。")

# ==============================================================================
# Plotlyによる可視化
# ==============================================================================
radius = PHYSICAL_CONSTANTS['RADIUS_MERCURY']
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_sphere_m = radius * np.outer(np.cos(u), np.sin(v))
y_sphere_m = radius * np.outer(np.sin(u), np.sin(v))
z_sphere_m = radius * np.outer(np.ones(np.size(u)), np.cos(v))
sphere_trace = go.Surface(
    x=x_sphere_m / radius, y=y_sphere_m / radius, z=z_sphere_m / radius,
    colorscale='Greys', opacity=0.8, showscale=False, name='水星'
)
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
lines_x_scaled = [val / radius if val is not None else None for val in lines_x_m]
lines_y_scaled = [val / radius if val is not None else None for val in lines_y_m]
lines_z_scaled = [val / radius if val is not None else None for val in lines_z_m]
lines_trace = go.Scatter3d(
    x=lines_x_scaled, y=lines_y_scaled, z=lines_z_scaled,
    mode='lines', line=dict(color='orange', width=2), name='粒子軌道'
)
fig = go.Figure(data=[sphere_trace, lines_trace])
fig.update_layout(
    title=f'(TAA={TAA:.1f}°, Gravity:{USE_GRAVITY}, Rad.Pressure:{USE_RADIATION_PRESSURE}, Solar.Grav:{USE_SOLAR_GRAVITY})',
    scene=dict(
        xaxis_title='X [R_M]',
        yaxis_title='Y [R_M]',
        zaxis_title='Z [R_M]',
        aspectmode='data'
    ),
    legend=dict(x=0, y=1)
)
fig.show()