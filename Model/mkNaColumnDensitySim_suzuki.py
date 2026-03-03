import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 0. 物理定数と基本パラメータ (論文参照)
# ==============================================================================
G = 6.674e-11  # 万有引力定数 (m^3 kg^-1 s^-2)
M_MERCURY = 3.301e23  # 水星の質量 (kg)
R_MERCURY = 2440e3  # 水星の半径 (m)
M_SUN = 1.989e30  # 太陽の質量 (kg)
AU = 1.496e11  # 天文単位 (m)
K_B = 1.38e-23  # ボルツマン定数 (J/K)
M_NA = 22.99 * 1.66e-27  # ナトリウム原子の質量 (kg)


# ==============================================================================
# 1. 粒子の運動を計算する関数 (論文 式11 参照)
# ==============================================================================
def calculate_trajectory(pos0, vel0, mercury_pos, dt=30.0, max_steps=1000, rad_accel=0.5):
    """1つの粒子の軌道を計算する"""
    pos = np.array(pos0, dtype=float)
    vel = np.array(vel0, dtype=float)

    trajectory = [pos.copy()]

    for _ in range(max_steps):
        r_from_mercury = pos - mercury_pos
        dist_mercury = np.linalg.norm(r_from_mercury)

        # <<< 修正点: 衝突判定をループの最初から移動
        # 粒子は地表からスタートするため、少なくとも1ステップは動かしてから衝突を判定する
        if len(trajectory) > 1 and dist_mercury <= R_MERCURY:
            break

        # 太陽からの距離ベクトル
        r_from_sun = pos
        dist_sun = np.linalg.norm(r_from_sun)

        # 水星と太陽からの重力加速度
        accel_mercury = -G * M_MERCURY * r_from_mercury / dist_mercury ** 3
        accel_sun = -G * M_SUN * r_from_sun / dist_sun ** 3

        # 太陽放射圧による加速度 (反太陽方向)
        accel_rad = rad_accel * r_from_sun / dist_sun

        # 合計加速度
        accel = accel_sun + accel_mercury + accel_rad

        # リープフロッグ法で位置と速度を更新
        vel += accel * dt
        pos += vel * dt

        trajectory.append(pos.copy())

    return np.array(trajectory)


# ==============================================================================
# 2. 放出条件を定義する関数
# ==============================================================================

def generate_ejection_vectors(num, ejection_angle_dist='cosine'):
    """指定された角度分布で放出ベクトルを生成する"""
    # 方位角は0から2πまで一様
    azimuth = 2 * np.pi * np.random.rand(num)

    # 天頂角 (射出角)
    if ejection_angle_dist == 'cosine':
        # cos(theta)分布の場合、sqrt(u) (uは[0,1]の一様乱数) を使うと正しい分布になる
        cos_theta = np.sqrt(np.random.rand(num))
    else:  # 等方的な場合など
        cos_theta = np.random.rand(num)

    sin_theta = np.sqrt(1 - cos_theta ** 2)

    # 局所座標系での速度ベクトル (z軸を表面の法線方向とする)
    vx = sin_theta * np.cos(azimuth)
    vy = sin_theta * np.sin(azimuth)
    vz = cos_theta

    return np.vstack([vx, vy, vz]).T


def sample_maxwellian_speed(T, num=1):
    """マクスウェル分布から速度をサンプリングする"""
    # 厳密なサンプリングを行う (3次元マクスウェル分布は3つの正規分布の合成)
    sigma = np.sqrt(K_B * T / M_NA)
    vx = np.random.normal(0, sigma, num)
    vy = np.random.normal(0, sigma, num)
    vz = np.random.normal(0, sigma, num)
    return np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)


def sample_thompson_energy(U, num=1):
    """シグムンド-トンプソン分布からエネルギーをサンプlingする (簡易版)"""
    # 論文の分布(式6)は E/(E+U)^3 に比例する。これを簡易的にサンプリング
    # 逆関数法を用いるため、累積分布関数の逆関数を求める必要があるが複雑
    # ここでは指数分布で代用する (ピークがU付近になるように調整)
    energy_J = np.random.exponential(scale=U, size=num)
    return energy_J


# --- 各プロセスごとの粒子生成関数 ---

def eject_particle_psd(mercury_radius):
    """光刺激脱離(PSD)による粒子を生成する (論文 3.1.2節)"""
    # 放出領域: 昼側のどこか (簡単のため太陽直下点とする)
    pos0 = np.array([mercury_radius, 0, 0])

    # 速度分布: 1500Kのマクスウェル分布
    T_psd = 1500  # K
    speed = sample_maxwellian_speed(T_psd)

    # 放出角度: コサイン分布
    direction = generate_ejection_vectors(1, 'cosine')[0]

    # グローバル座標系での速度ベクトルに変換
    vel0 = speed * direction

    return pos0, vel0


def eject_particle_sws(mercury_radius):
    """イオンスパッタリング(SWS)による粒子を生成する (論文 3.1.3節)"""
    # 放出領域: 昼側の中緯度カスプ領域 (例として北緯45度)
    lat = np.deg2rad(45)
    lon = np.deg2rad(0)  # 正午
    pos0 = np.array([
        mercury_radius * np.cos(lat) * np.cos(lon),
        mercury_radius * np.cos(lat) * np.sin(lon),
        mercury_radius * np.sin(lat)
    ])

    # 速度分布: シグムンド-トンプソン分布
    U_sws_eV = 0.27  # eV
    energy_J = sample_thompson_energy(U_sws_eV * 1.602e-19)
    speed = np.sqrt(2 * energy_J / M_NA)

    # 放出角度: コサイン分布
    # 局所的な法線ベクトル
    normal_vec = pos0 / np.linalg.norm(pos0)

    # 放出方向ベクトルを生成し、法線ベクトルを基準に座標変換する
    # (簡易版: ここではグローバル座標での放出方向をそのまま使う)
    # 厳密には、法線ベクトルをz軸とする局所座標からグローバル座標への回転が必要
    direction_local = generate_ejection_vectors(1, 'cosine')[0]
    vel0 = speed * direction_local

    return pos0, vel0


# ==============================================================================
# 3. シミュレーションの実行と可視化
# ==============================================================================
if __name__ == '__main__':
    N_PARTICLES = 20  # シミュレーションする粒子数 (各プロセスごと)

    # 水星の軌道位置 (簡単のため近日点付近とする)
    mercury_orbit_radius = 0.31 * AU
    mercury_position = np.array([mercury_orbit_radius, 0, 0])

    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # 水星を描画
    mercury_circle = plt.Circle((0, 0), R_MERCURY, color='gray', label='Mercury')
    ax.add_patch(mercury_circle)

    print("Simulating PSD particles (orange)...")
    for i in range(N_PARTICLES):
        initial_pos_local, initial_vel_local = eject_particle_psd(R_MERCURY)
        trajectory = calculate_trajectory(mercury_position + initial_pos_local, initial_vel_local, mercury_position)
        traj_relative = trajectory - mercury_position
        ax.plot(traj_relative[:, 0], traj_relative[:, 2], color='orange', alpha=0.6)  # X-Z平面に投影

    print("Simulating SWS particles (cyan)...")
    for i in range(N_PARTICLES):
        initial_pos_local, initial_vel_local = eject_particle_sws(R_MERCURY)
        trajectory = calculate_trajectory(mercury_position + initial_pos_local, initial_vel_local, mercury_position)
        traj_relative = trajectory - mercury_position
        ax.plot(traj_relative[:, 0], traj_relative[:, 2], color='cyan', alpha=0.6)  # X-Z平面に投影

    # グラフの体裁を整える
    ax.set_aspect('equal')
    limit = R_MERCURY * 4
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_xlabel('Sun Direction (m)')
    ax.set_ylabel('North-South Direction (m)')
    ax.set_title('Simplified Na Exosphere Simulation (2D X-Z Projection)')
    ax.grid(True)

    # 凡例用のダミープロット
    ax.plot([], [], color='orange', label='PSD Trajectory')
    ax.plot([], [], color='cyan', label='SWS Trajectory')
    ax.legend()

    plt.show()