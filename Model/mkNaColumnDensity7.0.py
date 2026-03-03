# -*- coding: utf-8 -*-
"""
水星ナトリウム大気 3次元時間発展モンテカルロシミュレーションコード

概要:
    このスクリプトは、水星のナトリウム大気のふるまいを、
    時間発展を考慮してシミュレートするモンテカルロ法に基づいたプログラムです。
    水星の公転運動に伴う太陽との位置関係の変化を反映しながら、
    表面からのナトリウム原子の生成、宇宙空間での運動、そして消滅過程を追跡します。

主な物理モデル:
    1.  粒子生成:
        - 光刺激脱離 (Photo-Stimulated Desorption, PSD) モデルを採用。
        - 太陽からの紫外線フラックス、太陽天頂角、表面のナトリウム密度に基づき、
          日照側の表面から粒子が生成されます。
        - ★★★ 本バージョンでは、表面密度は常に一定（無限供給源）と仮定します。★★★

    2.  初期速度:
        - 放出される原子の速さは、指定された表面温度におけるマクスウェル分布に従います。
        - 放出角度は、表面の法線方向を基準としたランバート（余弦則）分布に従います。

    3.  軌道計算:
        - 粒子にかかる力は、水星の重力と太陽光の放射圧を考慮します。
        - 軌道積分には4次のルンゲ＝クッタ（RK4）法を使用し、高い精度を確保します。

    4.  消滅過程:
        - 光電離: 日照側を飛行する粒子は、確率的に太陽光によってイオン化され、
                   シミュレーションから除去されます。
        - 表面衝突: 表面に再衝突した粒子は、表面温度に応じた確率で吸着（消滅）するか、
                     熱的にエネルギーを交換して再放出（バウンド）されます。

    5.  出力:
        - シミュレーションは水星の公転周期に沿って進行します。
        - 指定された真近点離角 (TAA) に達するたびに、全粒子の状態をスナップショットとして保存します。
        - ★★★ スナップショットデータは、3次元の立方体グリッド上の密度分布として集計・保存されます。★★★

必要な外部ファイル:
    - orbit360.txt: 水星の軌道パラメータ（太陽距離、速度など）を記述したファイル。
    - SolarSpectrum_Na0.txt: 太陽光のスペクトルデータ。放射圧計算に使用。

実行方法:
    python this_script_name.py
"""

import numpy as np
import sys
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

# ==============================================================================
# 物理定数
# ==============================================================================
# この辞書は、シミュレーション全体で使用される物理定数を一元管理します。
# 単位は国際単位系（SI）に統一されています。
PHYSICAL_CONSTANTS = {
    'PI': np.pi,  # 円周率
    'AU': 1.496e11,  # 天文単位 [m]
    'MASS_NA': 22.98976928 * 1.66054e-27,  # ナトリウム原子の質量 [kg]
    'K_BOLTZMANN': 1.380649e-23,  # ボルツマン定数 [J/K]
    'GM_MERCURY': 2.2032e13,  # 水星の万有引力定数 G * 水星質量 M [m^3/s^2]
    'RM': 2.440e6,  # 水星の半径 [m]
    'C': 299792458.0,  # 光速 [m/s]
    'H': 6.62607015e-34,  # プランク定数 [J·s]
    'E_CHARGE': 1.602176634e-19,  # 電気素量 [C]
    'ME': 9.1093897e-31,  # 電子の質量 [kg]
    'EPSILON_0': 8.854187817e-12,  # 真空の誘電率 [F/m]
}


# ==============================================================================
# 物理モデルに基づくヘルパー関数群
# ==============================================================================

def calculate_surface_temperature(lon_rad, lat_rad, AU, subsolar_lon_rad):
    """
    水星表面の局所的な温度を計算します。

    日照側では太陽天頂角と太陽距離に依存し、夜側では最低温度に固定されます。

    Args:
        lon_rad (float): 計算対象地点の経度 [rad]
        lat_rad (float): 計算対象地点の緯度 [rad]
        AU (float): 現在の太陽からの距離 [天文単位]
        subsolar_lon_rad (float): 太陽直下点の経度 [rad]

    Returns:
        float: 表面温度 [K]
    """
    T0 = 100.0  # 夜側の最低温度
    T1 = 600.0  # 日照による最大温度上昇
    # 太陽天頂角の余弦を計算
    cos_theta = np.cos(lat_rad) * np.cos(lon_rad - subsolar_lon_rad)
    if cos_theta <= 0:
        return T0  # 夜側
    # 日照側の温度を計算
    return T0 + T1 * (cos_theta ** 0.25) * ((0.306 / AU) ** 2)


def calculate_sticking_probability(surface_temp_K):
    """
    表面温度に基づき、ナトリウム原子が表面に吸着する確率を計算します。
    温度が低いほど吸着しやすくなります。

    Args:
        surface_temp_K (float): 衝突地点の表面温度 [K]

    Returns:
        float: 吸着確率 (0から1の範囲)
    """
    A = 0.08
    B = 458.0
    porosity = 0.8  # 表面の多孔性
    if surface_temp_K <= 0: return 1.0
    p_stick = A * np.exp(B / surface_temp_K)
    # 多孔性を考慮した実効的な吸着確率
    p_stick_eff = p_stick / (1.0 - (1.0 - p_stick) * porosity)
    return min(p_stick_eff, 1.0)


def sample_maxwellian_speed(mass_kg, temp_k):
    """
    マクスウェル分布に従う速さをサンプリングします。
    熱運動する粒子の典型的な速さを与えます。

    Args:
        mass_kg (float): 粒子の質量 [kg]
        temp_k (float): 温度 [K]

    Returns:
        float: サンプリングされた速さ [m/s]
    """
    # スケールパラメータ (最確速)
    scale_param = np.sqrt(PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_k / mass_kg)
    # 3次元の正規分布から速度ベクトルを生成し、その大きさ（速さ）を返す
    vx, vy, vz = np.random.normal(0, scale_param, 3)
    return np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)


def sample_lambertian_direction_local():
    """
    ランバート（余弦則）分布に従う方向ベクトルをローカル座標系で生成します。
    ローカルZ軸が表面の法線方向に対応し、Z軸に近い方向ほど選ばれやすくなります。

    Returns:
        np.ndarray: 3次元の方向ベクトル（正規化済み）
    """
    u1, u2 = np.random.random(2)
    cos_theta = np.sqrt(1 - u2)
    sin_theta = np.sqrt(u2)
    phi = 2 * PHYSICAL_CONSTANTS['PI'] * u1
    # ローカル座標系 (x, y, z)
    return np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])


def transform_local_to_world(local_vec, normal_vector):
    """
    ローカル座標系のベクトルをワールド座標系に変換します。

    Args:
        local_vec (np.ndarray): ローカル座標系でのベクトル
        normal_vector (np.ndarray): ワールド座標系での法線ベクトル（ローカルZ軸に相当）

    Returns:
        np.ndarray: ワールド座標系に変換されたベクトル
    """
    local_z_axis = normal_vector / np.linalg.norm(normal_vector)
    # 任意の外積計算用のベクトルを選択（ただし法線と平行でないもの）
    world_up = np.array([0., 0., 1.])
    if np.allclose(local_z_axis, world_up) or np.allclose(local_z_axis, -world_up):
        world_up = np.array([0., 1., 0.])

    # 直交する基底ベクトルを計算
    local_x_axis = np.cross(world_up, local_z_axis)
    local_x_axis /= np.linalg.norm(local_x_axis)
    local_y_axis = np.cross(local_z_axis, local_x_axis)

    # ローカルベクトルをワールド座標系に射影
    return (local_vec[0] * local_x_axis +
            local_vec[1] * local_y_axis +
            local_vec[2] * local_z_axis)


def preprocess_orbit_data(filename, mercury_year_sec):
    """
    軌道データファイルを読み込み、シミュレーション時間軸を追加します。

    Args:
        filename (str): 軌道データファイル名
        mercury_year_sec (float): 水星の1年の秒数

    Returns:
        np.ndarray: 時間軸が追加された軌道データ
    """
    data = np.loadtxt(filename)
    time_axis = np.linspace(0, mercury_year_sec, len(data), endpoint=False)
    return np.column_stack((time_axis, data))


def get_orbital_params(time_sec, orbit_data, mercury_year_sec):
    """
    指定された時刻における水星の軌道パラメータと太陽直下点経度を取得します。
    データ点の中間時刻は線形補間によって計算されます。

    Args:
        time_sec (float): シミュレーション開始からの経過時間 [s]
        orbit_data (np.ndarray): 前処理済みの軌道データ
        mercury_year_sec (float): 水星の1年の秒数

    Returns:
        tuple: (TAA[deg], AU, 軌道速度[m/s], 太陽直下点経度[rad])
    """
    ROTATION_PERIOD_SEC = 58.646 * 24 * 3600  # 水星の自転周期
    current_time_in_orbit = time_sec % mercury_year_sec

    # 線形補間
    taa = np.interp(current_time_in_orbit, orbit_data[:, 0], orbit_data[:, 1])
    au = np.interp(current_time_in_orbit, orbit_data[:, 0], orbit_data[:, 2])
    vms = np.interp(current_time_in_orbit, orbit_data[:, 0], orbit_data[:, 5])

    # 太陽直下点経度は自転に基づいて計算
    subsolar_lon_rad = (2 * np.pi * time_sec / ROTATION_PERIOD_SEC) % (2 * np.pi)

    return taa, au, vms, subsolar_lon_rad


def lonlat_to_xyz(lon_rad, lat_rad, radius):
    """
    経度・緯度 [rad] を三次元直交座標 [m] に変換します。
    """
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return np.array([x, y, z])


# ==============================================================================
# コア追跡関数 (並列処理の対象)
# ==============================================================================

def _calculate_acceleration(pos, vel, Vms_ms, AU, spec_data):
    """
    【内部関数】粒子にかかる加速度（重力＋放射圧）を計算します。

    Args:
        pos (np.ndarray): 粒子の位置ベクトル [m]
        vel (np.ndarray): 粒子の速度ベクトル [m/s]
        Vms_ms (float): 水星の公転速度 [m/s] (太陽方向を正とする)
        AU (float): 太陽距離 [天文単位]
        spec_data (dict): 太陽スペクトルデータ

    Returns:
        np.ndarray: 加速度ベクトル [m/s^2]
    """
    x, y, z = pos

    ### 放射圧計算 ###
    # 太陽は-X方向から来ると仮定, 水星は-X方向に公転していると定義
    # よって、粒子から見た太陽の相対速度は (vel_x - (-V_merc)) = vel_x + V_merc
    velocity_for_doppler = vel[0] + Vms_ms
    # ドップラーシフトを考慮した波長
    w_na_d2 = 589.1582e-9 * (1.0 - velocity_for_doppler / PHYSICAL_CONSTANTS['C'])
    w_na_d1 = 589.7558e-9 * (1.0 - velocity_for_doppler / PHYSICAL_CONSTANTS['C'])

    b = 0.0  # 放射圧による加速度
    wl, gamma, sigma0_perdnu2, sigma0_perdnu1, JL = spec_data.values()
    # スペクトルデータの範囲内かチェック
    if (wl[0] * 1e-9 <= w_na_d2 < wl[-1] * 1e-9) and \
            (wl[0] * 1e-9 <= w_na_d1 < wl[-1] * 1e-9):

        gamma2 = np.interp(w_na_d2 * 1e9, wl, gamma)
        gamma1 = np.interp(w_na_d1 * 1e9, wl, gamma)
        F_lambda_1AU_m = JL * 1e4 * 1e9
        F_lambda_d1_at_Mercury = F_lambda_1AU_m / (AU ** 2) * gamma1
        F_nu_d1 = F_lambda_d1_at_Mercury * (w_na_d1 ** 2) / PHYSICAL_CONSTANTS['C']
        J1 = sigma0_perdnu1 * F_nu_d1
        F_lambda_d2_at_Mercury = F_lambda_1AU_m / (AU ** 2) * gamma2
        F_nu_d2 = F_lambda_d2_at_Mercury * (w_na_d2 ** 2) / PHYSICAL_CONSTANTS['C']
        J2 = sigma0_perdnu2 * F_nu_d2
        b = 1 / PHYSICAL_CONSTANTS['MASS_NA'] * (
                    (PHYSICAL_CONSTANTS['H'] / w_na_d1) * J1 + (PHYSICAL_CONSTANTS['H'] / w_na_d2) * J2)

    # 水星の影に入っている場合は放射圧を0に
    if x < 0 and np.sqrt(y ** 2 + z ** 2) < PHYSICAL_CONSTANTS['RM']:
        b = 0.0

    accel_srp = np.array([-b, 0.0, 0.0])  # 逆太陽方向（-X）へ加速

    ### 重力計算 ###
    r_sq = np.sum(pos ** 2)
    accel_g = -PHYSICAL_CONSTANTS['GM_MERCURY'] * pos / (r_sq ** 1.5) if r_sq > 0 else np.array([0., 0., 0.])

    return accel_srp + accel_g


def simulate_particle_for_one_step(args):
    """
    一個のスーパーパーティクルを、指定された時間 (duration) だけ追跡します。
    この関数は並列処理の各プロセスで実行されます。

    Args:
        args (dict): シミュレーションに必要な全てのパラメータを含む辞書。

    Returns:
        dict: 粒子の最終状態 ('status') と情報 ('final_state') を含む辞書。
    """
    # 引数を展開
    settings, spec_data = args['settings'], args['spec']
    TAA, AU, Vms_ms, subsolar_lon_rad = args['orbit']
    duration, DT = args['duration'], settings['DT']
    RM = PHYSICAL_CONSTANTS['RM']
    R_MAX = RM * settings['GRID_RADIUS_RM']  # シミュレーション空間の最大半径
    pos, vel, weight = args['particle_state']['pos'], args['particle_state']['vel'], args['particle_state']['weight']

    # 1AUでの電離寿命を現在の太陽距離にスケーリング
    tau_ionization = settings['T1AU'] * AU ** 2

    num_steps = int(duration / DT)
    for _ in range(num_steps):

        # 1. 光電離判定
        if pos[0] > 0 and np.random.random() < (1.0 - np.exp(-DT / tau_ionization)):
            return {'status': 'ionized', 'final_state': None}

        # 2. 軌道計算 (4次ルンゲ＝クッタ法)
        pos_prev = pos.copy()
        k1_vel = DT * _calculate_acceleration(pos, vel, Vms_ms, AU, spec_data)
        k1_pos = DT * vel
        k2_vel = DT * _calculate_acceleration(pos + 0.5 * k1_pos, vel + 0.5 * k1_vel, Vms_ms, AU, spec_data)
        k2_pos = DT * (vel + 0.5 * k1_vel)
        k3_vel = DT * _calculate_acceleration(pos + 0.5 * k2_pos, vel + 0.5 * k2_vel, Vms_ms, AU, spec_data)
        k3_pos = DT * (vel + 0.5 * k2_vel)
        k4_vel = DT * _calculate_acceleration(pos + k3_pos, vel + k3_vel, Vms_ms, AU, spec_data)
        k4_pos = DT * (vel + k3_vel)
        pos += (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos) / 6.0
        vel += (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel) / 6.0

        # 3. 境界条件の判定
        r_current = np.linalg.norm(pos)

        # 3a. シミュレーション領域外への脱出
        if r_current > R_MAX:
            return {'status': 'escaped', 'final_state': None}

        # 3b. 表面への衝突
        if r_current <= RM:
            impact_lon = np.arctan2(pos_prev[1], pos_prev[0])
            impact_lat = np.arcsin(np.clip(pos_prev[2] / np.linalg.norm(pos_prev), -1.0, 1.0))
            temp_at_impact = calculate_surface_temperature(impact_lon, impact_lat, AU, subsolar_lon_rad)

            # 3b-i. 吸着して消滅
            if np.random.random() < calculate_sticking_probability(temp_at_impact):
                return {'status': 'stuck', 'final_state': None}

            # 3b-ii. 熱的に再放出（バウンド）
            else:
                E_in = 0.5 * PHYSICAL_CONSTANTS['MASS_NA'] * np.sum(vel ** 2)
                E_T = PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_at_impact
                # 熱緩和係数BETAに基づいて衝突後のエネルギーを決定
                E_out = settings['BETA'] * E_T + (1.0 - settings['BETA']) * E_in
                v_out_speed = np.sqrt(2 * E_out / PHYSICAL_CONSTANTS['MASS_NA']) if E_out > 0 else 0.0

                impact_normal = pos_prev / np.linalg.norm(pos_prev)
                rebound_direction = transform_local_to_world(sample_lambertian_direction_local(), impact_normal)
                vel = v_out_speed * rebound_direction
                pos = RM * impact_normal  # 粒子を表面に戻す
                continue  # 次の積分ステップへ

    # durationを生き延びた場合
    return {'status': 'alive', 'final_state': {'pos': pos, 'vel': vel, 'weight': weight}}


# ==============================================================================
# メイン制御関数
# ==============================================================================

def main_snapshot_simulation():
    """
    シミュレーション全体を制御するメイン関数。
    """
    start_time = time.time()

    # --- シミュレーション設定 ---
    OUTPUT_DIRECTORY = r"./SimulationResult_202510"
    N_LON, N_LAT = 48, 24  # 表面グリッドの分割数（粒子生成用）

    # ★★★ 表面密度は常にこの値で固定（無限供給源モデル）★★★
    #CONSTANT_SURFACE_DENSITY = 1.5e17  # [atoms/m^2] (suzuki et al., 2019)
    CONSTANT_SURFACE_DENSITY = 7.5e17 * 0.053  # [atoms/m^2] (killen et al., 2001)

    TIME_STEP_SEC = 1000  # シミュレーションのメインループの時間刻み [s]
    TOTAL_SIM_YEARS = 1.0  # 総シミュレーション時間 [水星年]
    TARGET_TAA_DEGREES = np.arange(0, 360, 1)  # スナップショットを保存するTAA [度] (計算を始めるTAA, 計算を終えるTAA, 何度毎にスナップショットを保存するか)
    ATOMS_PER_SUPERPARTICLE = 1e24 # 超粒子1個は現実の10^25個の原子に相当

    # --- 粒子生成モデルのパラメータ ---
    F_UV_1AU_PER_M2 = 1.5e14 * (100) ** 2  # 1AUでの紫外線光子フラックス [photons/m^2/s]
    #Q_PSD_M2 = 2.0e-20 / (100) ** 2  # 光刺激脱離の断面積 [m^2] (suzuki et al., 2019)
    Q_PSD_M2 = 1.4e-21 / (100) ** 2  # 光刺激脱離の断面積 [m^2] (killen et al., 2004)
    SOURCE_TEMPERATURE = 1500.0  # 光刺激脱離の速度分布に用いる温度 [K]

    # --- ★★★ 立方体グリッドの設定 ★★★ ---
    GRID_RESOLUTION = 101  # グリッドの解像度（各軸のセル数, 奇数推奨）
    GRID_MAX_RM = 5.0  # グリッドの最大範囲（水星半径単位, 例: 5.0 -> -5RMから+5RMまで）

    # --- その他の設定 ---
    settings = {
        'BETA': 0.5,  # 表面衝突時の熱緩和係数　β
        'T1AU': 168918.0,  # 1AUでの光電離寿命 [s]
        'DT': 1000.0,  # 軌道積分の時間刻み [s]
        'N_LON': N_LON, 'N_LAT': N_LAT,
        'GRID_RADIUS_RM': GRID_MAX_RM + 1.0  # 粒子が脱出する半径（グリッドより少し大きく）
    }

    run_name = f"Grid{GRID_RESOLUTION}_Range{int(GRID_MAX_RM)}RM_SP{ATOMS_PER_SUPERPARTICLE:.0e}_2"
    target_output_dir = os.path.join(OUTPUT_DIRECTORY, run_name)
    os.makedirs(target_output_dir, exist_ok=True)
    print(f"結果は '{target_output_dir}' に保存されます。")

    ### 準備 ###
    MERCURY_YEAR_SEC = 87.97 * 24 * 3600
    time_steps = np.arange(0, TOTAL_SIM_YEARS * MERCURY_YEAR_SEC, TIME_STEP_SEC)

    # 表面グリッドの定義（粒子生成用）
    lon_edges = np.linspace(-np.pi, np.pi, N_LON + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    dlon = lon_edges[1] - lon_edges[0]
    cell_areas_m2 = (PHYSICAL_CONSTANTS['RM'] ** 2) * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))

    # 外部ファイルの読み込み
    try:
        spec_data_np = np.loadtxt('SolarSpectrum_Na0.txt', usecols=(0, 3))
        orbit_data = preprocess_orbit_data('orbit360.txt', MERCURY_YEAR_SEC)
    except FileNotFoundError as e:
        print(f"エラー: データファイル '{e.filename}' が見つかりません。");
        sys.exit()

    # スペクトルデータの前処理
    wl, gamma = spec_data_np[:, 0], spec_data_np[:, 1]
    if not np.all(np.diff(wl) > 0):
        sort_indices = np.argsort(wl)
        wl, gamma = wl[sort_indices], gamma[sort_indices]
    sigma_const = PHYSICAL_CONSTANTS['E_CHARGE'] ** 2 / (
                4 * PHYSICAL_CONSTANTS['ME'] * PHYSICAL_CONSTANTS['C'] * PHYSICAL_CONSTANTS['EPSILON_0'])
    spec_data_dict = {'wl': wl, 'gamma': gamma, 'sigma0_perdnu2': sigma_const * 0.641,
                      'sigma0_perdnu1': sigma_const * 0.320, 'JL': 5.18e14}

    ### メインループ ###
    active_particles = []
    previous_taa = -1
    target_taa_idx = 0
    with tqdm(total=len(time_steps), desc="Time Evolution") as pbar:
        for t_sec in time_steps:
            # 1. 現在時刻の軌道パラメータを取得
            TAA, AU, Vms_ms, subsolar_lon_rad = get_orbital_params(t_sec, orbit_data, MERCURY_YEAR_SEC)
            pbar.set_description(f"TAA={TAA:.1f} | N_particles={len(active_particles)}")

            # 2. 表面から新しい粒子を生成
            newly_launched_particles = []
            for i_lon in range(N_LON):
                for i_lat in range(N_LAT):
                    lon_center_rad = (lon_edges[i_lon] + lon_edges[i_lon + 1]) / 2
                    lat_center_rad = (lat_edges[i_lat] + lat_edges[i_lat + 1]) / 2
                    cos_Z = np.cos(lat_center_rad) * np.cos(lon_center_rad - subsolar_lon_rad)
                    if cos_Z <= 0: continue

                    F_UV_current_per_m2 = F_UV_1AU_PER_M2 / (AU ** 2)
                    desorption_rate_per_m2_s = F_UV_current_per_m2 * Q_PSD_M2 * cos_Z * CONSTANT_SURFACE_DENSITY
                    n_atoms_to_desorb = desorption_rate_per_m2_s * cell_areas_m2[i_lat] * TIME_STEP_SEC
                    if n_atoms_to_desorb <= 0: continue

                    # 現実の原子数を、超粒子1個あたりの原子数で割る
                    num_sps_to_launch_float = n_atoms_to_desorb / ATOMS_PER_SUPERPARTICLE

                    num_to_launch_int = int(num_sps_to_launch_float)
                    if np.random.random() < (num_sps_to_launch_float - num_to_launch_int):
                        num_to_launch_int += 1

                    if num_to_launch_int == 0: continue

                    for _ in range(num_to_launch_int):
                        # セル内でランダムな位置から放出
                        random_lon_rad = np.random.uniform(lon_edges[i_lon], lon_edges[i_lon + 1])
                        sin_lat_min, sin_lat_max = np.sin(lat_edges[i_lat]), np.sin(lat_edges[i_lat + 1])
                        random_lat_rad = np.arcsin(np.random.uniform(sin_lat_min, sin_lat_max))

                        initial_pos = lonlat_to_xyz(random_lon_rad, random_lat_rad, PHYSICAL_CONSTANTS['RM'])
                        surface_normal = initial_pos / np.linalg.norm(initial_pos)

                        speed = sample_maxwellian_speed(PHYSICAL_CONSTANTS['MASS_NA'], SOURCE_TEMPERATURE)
                        initial_vel = speed * transform_local_to_world(sample_lambertian_direction_local(),
                                                                       surface_normal)

                        newly_launched_particles.append({
                            'pos': initial_pos,
                            'vel': initial_vel,
                            'weight': ATOMS_PER_SUPERPARTICLE  # 重みは定数
                        })

            active_particles.extend(newly_launched_particles)

            # 3. 全ての粒子を1ステップ進める (並列処理)
            tasks = [{'settings': settings, 'spec': spec_data_dict, 'particle_state': p,
                      'orbit': (TAA, AU, Vms_ms, subsolar_lon_rad), 'duration': TIME_STEP_SEC} for p in
                     active_particles]
            next_active_particles = []
            if tasks:
                # CPUコア数-1を上限として並列処理
                with Pool(processes=max(1, cpu_count() - 1)) as pool:
                    results = list(pool.imap(simulate_particle_for_one_step, tasks, chunksize=100))

                for res in results:
                    if res['status'] == 'alive':
                        next_active_particles.append(res['final_state'])

            active_particles = next_active_particles

            # 4. スナップショット保存判定
            if TAA < previous_taa: target_taa_idx = 0
            save_this_step = False
            if target_taa_idx < len(TARGET_TAA_DEGREES):
                current_target_taa = TARGET_TAA_DEGREES[target_taa_idx]
                if previous_taa < current_target_taa <= TAA:
                    save_this_step = True
                    target_taa_idx += 1

            # 5. ★★★ 立方体グリッドに集計して保存 ★★★
            if save_this_step:
                pbar.write(f"\n>>> Saving grid snapshot at TAA={TAA:.1f} ({len(active_particles)} particles) <<<")

                # グリッドの準備
                density_grid = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION, GRID_RESOLUTION), dtype=np.float32)
                grid_min = -GRID_MAX_RM * PHYSICAL_CONSTANTS['RM']
                grid_max = GRID_MAX_RM * PHYSICAL_CONSTANTS['RM']
                cell_size = (grid_max - grid_min) / GRID_RESOLUTION

                # 粒子をグリッドに割り当てる
                for p in active_particles:
                    pos = p['pos']
                    # 座標値からグリッドのインデックスを計算
                    ix = int((pos[0] - grid_min) / cell_size)
                    iy = int((pos[1] - grid_min) / cell_size)
                    iz = int((pos[2] - grid_min) / cell_size)

                    # グリッド範囲内かチェック
                    if 0 <= ix < GRID_RESOLUTION and 0 <= iy < GRID_RESOLUTION and 0 <= iz < GRID_RESOLUTION:
                        density_grid[ix, iy, iz] += p['weight']

                # グリッドセルの体積で割って、数密度に変換 [atoms/m^3]
                cell_volume_m3 = cell_size ** 3
                density_grid /= cell_volume_m3

                # ファイルに保存
                save_time_h = t_sec / 3600
                filename = f"density_grid_t{int(save_time_h):05d}_taa{int(TAA):03d}.npy"
                np.save(os.path.join(target_output_dir, filename), density_grid)

            previous_taa = TAA
            pbar.update(1)

    end_time = time.time()
    print(f"\n★★★ シミュレーションが完了しました ★★★")
    print(f"総計算時間: {(end_time - start_time) / 3600:.2f} 時間")


if __name__ == '__main__':
    # 実行前に必要なファイルの存在を確認
    for f in ['orbit360.txt', 'SolarSpectrum_Na0.txt']:
        if not os.path.exists(f):
            print(f"エラー: 必須ファイル '{f}' が見つかりません。スクリプトと同じディレクトリに配置してください。")
            sys.exit()
    main_snapshot_simulation()