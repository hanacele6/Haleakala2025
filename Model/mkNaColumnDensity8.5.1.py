# -*- coding: utf-8 -*-
"""
==============================================================================
プロジェクト: 水星ナトリウム外気圏 3次元モンテカルロシミュレーション
==============================================================================

概要:
    水星表面から放出されるナトリウム原子を追跡し、外気圏の構造（密度分布）を計算します。
    自転、公転、および各種物理プロセス（放出、重力、輻射圧、光イオン化）を考慮しています。

主な機能:
    1. 放出プロセス:
       - 熱脱離 (TD: Thermal Desorption)
       - 光刺激脱離 (PSD: Photon-Stimulated Desorption)
       - 太陽風スパッタリング (SWS: Solar Wind Sputtering)
       - 微小隕石衝突気化 (MMV: Meteoroid Micro-Vaporization)

    2. 輸送プロセス (運動方程式):
       - 水星重力 & 太陽重力
       - 太陽輻射圧 (SRP: Solar Radiation Pressure) ※ドップラーシフト考慮
       - コリオリ力 & 遠心力 (非慣性系)

    3. 消滅・境界条件:
       - 光イオン化による消失
       - 表面への再衝突 (吸着 or 再放出)
       - 逃走 (ヒル圏外への脱出)

    4. 平衡解モード (Equilibrium Mode):
       - 計算時間を短縮するため、粒子の生成量と消失量が釣り合う状態（平衡）を
         動的に計算し、表面密度を更新します。

出力:
    - 3次元密度グリッド (.npy)
    - 表面密度マップ (.npy)

==============================================================================
"""

import numpy as np
import sys
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

# ==============================================================================
# 1. 物理定数・天文定数 (SI単位系: m, kg, s, A, K)
# ==============================================================================
PHYSICAL_CONSTANTS = {
    'PI': np.pi,
    'AU': 1.496e11,  # 1天文単位 [m]
    'MASS_NA': 3.8175e-26,  # Na原子質量 (22.989 u) [kg]
    'K_BOLTZMANN': 1.380649e-23,  # ボルツマン定数 [J/K]
    'GM_MERCURY': 2.2032e13,  # 水星重力定数 (GM) [m^3/s^2]
    'RM': 2.440e6,  # 水星半径 [m]
    'C': 299792458.0,  # 光速 [m/s]
    'H': 6.62607015e-34,  # プランク定数 [J·s]
    'E_CHARGE': 1.602e-19,  # 電気素量 [C]
    'ME': 9.109e-31,  # 電子質量 [kg]
    'EPSILON_0': 8.854e-12,  # 真空の誘電率 [F/m]
    'G': 6.6743e-11,  # 万有引力定数
    'MASS_SUN': 1.989e30,  # 太陽質量 [kg]
    'EV_TO_JOULE': 1.602e-19,  # eV -> J 変換係数
    'ROTATION_PERIOD': 58.6462 * 86400,  # 水星自転周期 (3:2共鳴) [s]
    'ORBITAL_PERIOD': 87.969 * 86400,  # 水星公転周期 [s]
}


# ==============================================================================
# 2. 物理モデル・ヘルパー関数群
# ==============================================================================

def calculate_surface_temperature_leblanc(lon_rad, lat_rad, AU, subsolar_lon_rad):
    """
    水星表面の局所的な温度を計算します。

    日照側では太陽天頂角(SZA)と太陽距離(AU)に依存し、
    夜側では最低温度(T0)に固定されます。

    (注: 太陽固定座標系では subsolar_lon_rad は常に 0.0)

    Args:
        lon_rad (float): 計算対象地点の経度 (惑星固定座標系) [rad]
        lat_rad (float): 計算対象地点の緯度 (惑星固定座標系) [rad]
        AU (float): 現在の太陽からの距離 [天文単位]
        subsolar_lon_rad (float): 太陽直下点の経度 (惑星固定座標系) [rad]

    Returns:
        float: 表面温度 [K]
    """
    T0 = 100.0  # 夜側の最低温度 [K]
    T1 = 600.0  # 日照による最大温度上昇の係数 [K]

    # 太陽天頂角の余弦 (cos(SZA)) を計算
    # (lon - subsolar_lon_rad) は、太陽直下点からの経度差
    cos_theta = np.cos(lat_rad) * np.cos(lon_rad - 0.0)

    if cos_theta <= 0:
        return T0  # 夜側

    return T0 + T1 * (cos_theta ** 0.25) * ((0.306 / AU) ** 2)


def calculate_sticking_probability(surface_temp_K):
    """
    吸着確率 (Sticking Probability) の計算

    解説:
        粒子が表面に衝突した際、その場に留まる（吸着する）確率です。
        表面温度が高いほど吸着しにくくなります。
        Yakshinskiy & Madey (1999) などの実験式に基づきます。

    Args:
        surface_temp_K: 衝突地点の表面温度 [K]

    Returns:
        float: 吸着確率 (0.0 ~ 1.0)
    """
    A = 0.0804
    B = 458.0
    porosity = 0.8  # 表面の多孔質度（実効的な表面積を増やす効果）

    if surface_temp_K <= 0: return 1.0

    p_stick = A * np.exp(B / surface_temp_K)
    # 多孔質補正: 穴に入り込んで何度も壁に当たると吸着しやすくなる効果
    p_stick_eff = p_stick / (1.0 - (1.0 - p_stick) * porosity)

    return min(p_stick_eff, 1.0)


def calculate_thermal_desorption_rate(surface_temp_K):
    """
    熱脱離率 (Thermal Desorption Rate) の計算

    解説:
        表面に吸着している原子が、熱エネルギーを得て飛び出す頻度 [1/s]。
        アレニウスの式 (Rate = freq * exp(-E/kT)) に従います。

    Args:
        surface_temp_K: 表面温度 [K]

    Returns:
        float: 脱離率 [atoms/s per atom]
    """
    if surface_temp_K < 10.0: return 0.0

    VIB_FREQ = 1e13  # 原子の振動数 [Hz]
    BINDING_ENERGY_EV = 1.85  # ナトリウムの結合エネルギー [eV]
    BINDING_ENERGY_J = BINDING_ENERGY_EV * PHYSICAL_CONSTANTS['EV_TO_JOULE']
    k_B = PHYSICAL_CONSTANTS['K_BOLTZMANN']

    exponent = -BINDING_ENERGY_J / (k_B * surface_temp_K)

    # オーバーフロー防止
    if exponent < -700: return 0.0

    return VIB_FREQ * np.exp(exponent)


def calculate_mmv_flux(AU):
    """
    微小隕石衝突気化 (MMV) のフラックス計算

    解説:
        微小隕石が水星表面に衝突し、岩石を蒸発させてNaを放出させるプロセス。
        太陽に近いほど隕石フラックスが高いと仮定しています (R^-1.9 則など)。

    Returns:
        float: 放出フラックス [atoms/m^2/s]
    """
    TOTAL_FLUX_AT_PERI = 5e23  # 近日点での全球総放出量 [atoms/s] (仮定値)
    PERIHELION_AU = 0.307
    AREA = 4 * PHYSICAL_CONSTANTS['PI'] * (PHYSICAL_CONSTANTS['RM'] ** 2)

    avg_flux_peri = TOTAL_FLUX_AT_PERI / AREA

    # 距離依存性 (Borin et al. 2009等参照)
    C = avg_flux_peri * (PERIHELION_AU ** 1.9)
    return C * (AU ** (-1.9))


def sample_speed_from_flux_distribution(mass_kg, temp_k):
    """
    熱的な速度分布（Maxwell-Boltzmann Flux分布）からのサンプリング

    解説:
        表面から熱的に放出される粒子の速度 v は、f(v) ~ v^3 * exp(-mv^2/2kT) に従います。
        これはガンマ分布を用いて効率的に生成できます。
    """
    kT = PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_k
    # ガンマ分布(形状パラメータ2.0)からエネルギーEをサンプリング
    E = np.random.gamma(2.0, kT)
    return np.sqrt(2.0 * E / mass_kg)


def sample_thompson_sigmund_energy(U_eV, E_max_eV=5.0):
    """
    スパッタリング放出エネルギー (Thompson-Sigmund分布)

    解説:
        高エネルギー粒子（太陽風イオン）の衝突による放出。
        分布は f(E) ~ E / (E + U)^3 に従います（Uは結合エネルギー）。
        高エネルギーのテールを持つ分布になります。
    """
    # 棄却法 (Rejection Sampling) による乱数生成
    f_max = (U_eV / 2.0) / (U_eV / 2.0 + U_eV) ** 3
    while True:
        E_try = np.random.uniform(0, E_max_eV)
        f_val = E_try / (E_try + U_eV) ** 3
        if np.random.uniform(0, f_max) <= f_val:
            return E_try


def sample_lambertian_direction_local():
    """
    ランバート反射（余弦則）に従う放出方向ベクトル（局所座標系）

    解説:
        天頂方向（真上）への放出が最も確率が高くなる分布。
        Local座標: z軸が法線方向。
    """
    u1, u2 = np.random.random(2)
    phi = 2 * PHYSICAL_CONSTANTS['PI'] * u1  # 方位角
    cos_theta = np.sqrt(1 - u2)
    sin_theta = np.sqrt(u2)  # 天頂角のsin
    # x, y, z (zが上)
    return np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])


def transform_local_to_world(local_vec, normal_vector):
    """
    局所座標系(Local)から水星中心座標系(World)へのベクトル変換

    解説:
        LocalのZ軸(0,0,1)を、実際の地表面法線ベクトル(normal_vector)に合わせる回転を行います。
    """
    local_z = normal_vector / np.linalg.norm(normal_vector)

    # 補助ベクトル (World Up)
    world_up = np.array([0., 0., 1.])
    if np.abs(np.dot(local_z, world_up)) > 0.99:
        world_up = np.array([0., 1., 0.])  # 特異点回避

    local_x = np.cross(world_up, local_z)
    local_x /= np.linalg.norm(local_x)
    local_y = np.cross(local_z, local_x)

    # 基底変換行列の適用
    return local_vec[0] * local_x + local_vec[1] * local_y + local_vec[2] * local_z


def get_orbital_params_cyclic(time_sec, orbit_data, t_perihelion_file):
    """
    軌道パラメータの取得 (ループ再生対応)

    解説:
        シミュレーション時間が軌道データファイル(1水星年分)を超えた場合、
        データをループさせて適切な位置・速度を取得します。

    Args:
        time_sec: シミュレーション開始からの経過時間
        orbit_data: 読み込んだ軌道データ配列
        t_perihelion_file: ファイル内の近日点通過時刻

    Returns:
        taa_deg: 真近点角 (True Anomaly)
        au: 太陽距離
        v_rad, v_tan: 動径速度, 接線速度
        subsolar_lon_rad: 太陽直下経度 (自転の計算に使用)
    """
    cycle_sec = PHYSICAL_CONSTANTS['ORBITAL_PERIOD']

    # 経過時間を1周期内に折り畳む
    dt_from_peri = (time_sec - t_perihelion_file)
    time_in_cycle = dt_from_peri % cycle_sec

    # ファイル内の時刻列を参照して補間
    time_col_original = orbit_data[:, 2]
    idx_peri = np.argmin(np.abs(orbit_data[:, 0]))
    t_peri_in_file = time_col_original[idx_peri]
    t_lookup = t_peri_in_file + time_in_cycle

    # 線形補間
    taa_deg = np.interp(t_lookup, time_col_original, orbit_data[:, 0])
    au = np.interp(t_lookup, time_col_original, orbit_data[:, 1])
    v_rad = np.interp(t_lookup, time_col_original, orbit_data[:, 3])
    v_tan = np.interp(t_lookup, time_col_original, orbit_data[:, 4])

    # 自転角の計算 (Subsolar Longitude)
    # TAA (公転角) - Rotation (自転角)
    taa_rad = np.deg2rad(taa_deg)
    omega_rot = 2 * np.pi / PHYSICAL_CONSTANTS['ROTATION_PERIOD']
    rotation_angle = omega_rot * (time_sec - t_perihelion_file)

    subsolar_lon_rad = taa_rad - rotation_angle
    # -pi ~ pi の範囲に正規化
    subsolar_lon_rad = (subsolar_lon_rad + np.pi) % (2 * np.pi) - np.pi

    return taa_deg, au, v_rad, v_tan, subsolar_lon_rad


def lonlat_to_xyz(lon_rad, lat_rad, radius):
    """球座標(Lon, Lat) -> 直交座標(x, y, z)変換"""
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return np.array([x, y, z])


# ==============================================================================
# 3. 粒子運動計算エンジン (並列化対応)
# ==============================================================================

def _calculate_acceleration(pos, vel, V_radial_ms, V_tangential_ms, AU, spec_data, settings):
    """
    粒子に働く加速度の総和を計算

    考慮される力:
    1. 輻射圧 (SRP):
       - Na原子が太陽光を吸収して運動量を得る力。
       - ドップラー効果により吸収効率が劇的に変化する。
    2. 水星重力 (Gravity)
    3. 太陽重力 (Solar Gravity)
    4. 非慣性系の力 (Coriolis, Centrifugal):
       - 座標系が太陽-水星固定系(または回転系)の場合に必要。
       - ※現在の実装では、計算系に応じてON/OFFする想定。
    """
    x, y, z = pos
    r0 = AU * PHYSICAL_CONSTANTS['AU']

    # --- A. 輻射圧 (SRP) 計算 ---
    # 太陽に対する相対視線速度 (ドップラーシフト用)
    # vel[0]はx方向(太陽方向)の速度。V_radial_msは水星自体の公転による視線速度。
    velocity_for_doppler = vel[0] + V_radial_ms

    # ドップラーシフトしたNaの吸収波長 (D1, D2線)
    w_na_d2 = 589.1582e-9 * (1.0 + velocity_for_doppler / PHYSICAL_CONSTANTS['C'])
    w_na_d1 = 589.7558e-9 * (1.0 + velocity_for_doppler / PHYSICAL_CONSTANTS['C'])

    b = 0.0  # 輻射圧による加速度の大きさ
    wl, gamma, sigma0_perdnu2, sigma0_perdnu1, JL = spec_data.values()

    # 太陽スペクトルデータ範囲内であればg-factor(加速度)を計算
    if (wl[0] * 1e-9 <= w_na_d2 < wl[-1] * 1e-9) and (wl[0] * 1e-9 <= w_na_d1 < wl[-1] * 1e-9):
        # スペクトル強度(gamma)を補間取得
        gamma2 = np.interp(w_na_d2 * 1e9, wl, gamma)
        gamma1 = np.interp(w_na_d1 * 1e9, wl, gamma)

        # 1AUでの光子フラックス -> 現在距離でのフラックス
        F_lambda_1AU = JL * 1e13
        F_at_Merc = F_lambda_1AU / (AU ** 2)

        # 加速度 = (運動量変化率) / 質量
        term_d1 = (PHYSICAL_CONSTANTS['H'] / w_na_d1) * sigma0_perdnu1 * \
                  (F_at_Merc * gamma1 * w_na_d1 ** 2 / PHYSICAL_CONSTANTS['C'])
        term_d2 = (PHYSICAL_CONSTANTS['H'] / w_na_d2) * sigma0_perdnu2 * \
                  (F_at_Merc * gamma2 * w_na_d2 ** 2 / PHYSICAL_CONSTANTS['C'])

        b = (term_d1 + term_d2) / PHYSICAL_CONSTANTS['MASS_NA']

    # 影の判定: 水星の背後(x < 0)かつ円筒領域内ならSRPはゼロ
    if x < 0 and np.sqrt(y ** 2 + z ** 2) < PHYSICAL_CONSTANTS['RM']:
        b = 0.0

    # SRPは太陽から遠ざかる方向(-x方向)に働く
    accel_srp = np.array([-b, 0.0, 0.0])

    # --- B. 重力 (Mercury) ---
    r_sq = np.sum(pos ** 2)
    accel_g = -PHYSICAL_CONSTANTS['GM_MERCURY'] * pos / (r_sq ** 1.5) if r_sq > 0 else np.zeros(3)

    # --- C. 太陽重力 ---
    accel_sun = np.zeros(3)
    if settings.get('USE_SOLAR_GRAVITY', False):
        G = PHYSICAL_CONSTANTS['G']
        M_SUN = PHYSICAL_CONSTANTS['MASS_SUN']
        # 太陽の位置ベクトル (x軸正方向にあると仮定する場合の相対ベクトル)
        # ※座標系の定義に依存するため注意。ここでは簡易的に計算。
        r_ps_vec = np.array([r0 - pos[0], -pos[1], -pos[2]])
        r_ps_mag_sq = np.sum(r_ps_vec ** 2)
        if r_ps_mag_sq > 0:
            accel_sun = (G * M_SUN * r_ps_vec) / (r_ps_mag_sq ** 1.5)

    # --- D. 非慣性力 (Coriolis & Centrifugal) ---
    # 回転座標系(太陽-水星固定)で解く場合に必要
    accel_cor = np.zeros(3)
    accel_cen = np.zeros(3)
    if settings.get('USE_CORIOLIS_FORCES', False):
        if r0 > 0:
            omega_val = V_tangential_ms / r0  # 公転角速度
            omega_sq = omega_val ** 2
            # 遠心力
            accel_cen = np.array([(omega_val ** 2) * (pos[0] - r0), omega_sq * pos[1], 0.0])
            # コリオリ力 F = -2 m (omega x v)
            two_omega = 2 * omega_val
            accel_cor = np.array([two_omega * vel[1], -two_omega * vel[0], 0.0])

    return accel_srp + accel_g + accel_sun + accel_cen + accel_cor


def simulate_particle_for_one_step(args):
    """
    1粒子の時間発展計算 (並列化タスクの単位)

    アルゴリズム:
        4次のルンゲ・クッタ法 (RK4) を用いて運動方程式を積分します。

    Args:
        args (dict): 必要な全パラメータを含む辞書
            - settings: シミュレーション設定
            - spec: スペクトルデータ
            - orbit: 現在の軌道要素
            - particle_state: 現在の位置・速度
            - duration: 進める時間ステップ幅 [s]

    Returns:
        dict: 更新後の状態
            - status: 'alive', 'ionized', 'escaped', 'stuck'
            - final_state: 位置・速度・重み
    """
    settings, spec_data = args['settings'], args['spec']
    TAA, AU, V_rad, V_tan, subsolar_lon = args['orbit']
    total_duration = args['duration']

    # 積分精度を保つための内部微小時間ステップ
    DT_INTEGRATION = 500.0

    if total_duration <= 0:
        return {'status': 'alive', 'final_state': args['particle_state']}

    num_steps = int(np.ceil(total_duration / DT_INTEGRATION))
    dt_per_step = total_duration / num_steps

    pos = args['particle_state']['pos'].copy()
    vel = args['particle_state']['vel'].copy()
    weight = args['particle_state']['weight']

    # 光イオン化の時定数
    tau_ion = settings['T1AU'] * AU ** 2

    RM = PHYSICAL_CONSTANTS['RM']
    R_MAX = RM * settings['GRID_RADIUS_RM']

    pos_start = pos.copy()

    # --- RK4 積分ループ ---
    for _ in range(num_steps):
        current_dt = dt_per_step

        # 1. 光イオン化判定 (確率的消滅)
        # 影に入っていない(pos[0]>0)場合のみ
        if pos[0] > 0:
            if np.random.random() < (1.0 - np.exp(-current_dt / tau_ion)):
                return {'status': 'ionized', 'final_state': None}

        # 2. ルンゲ・クッタ法による位置・速度更新
        # k1
        k1_v = current_dt * _calculate_acceleration(pos, vel, V_rad, V_tan, AU, spec_data, settings)
        k1_p = current_dt * vel
        # k2
        k2_v = current_dt * _calculate_acceleration(pos + 0.5 * k1_p, vel + 0.5 * k1_v, V_rad, V_tan, AU, spec_data,
                                                    settings)
        k2_p = current_dt * (vel + 0.5 * k1_v)
        # k3
        k3_v = current_dt * _calculate_acceleration(pos + 0.5 * k2_p, vel + 0.5 * k2_v, V_rad, V_tan, AU, spec_data,
                                                    settings)
        k3_p = current_dt * (vel + 0.5 * k2_v)
        # k4
        k4_v = current_dt * _calculate_acceleration(pos + k3_p, vel + k3_v, V_rad, V_tan, AU, spec_data, settings)
        k4_p = current_dt * (vel + k3_v)

        pos += (k1_p + 2 * k2_p + 2 * k3_p + k4_p) / 6.0
        vel += (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6.0

        r_now = np.linalg.norm(pos)

        # 3. 逃走判定 (シミュレーション領域外へ)
        if r_now > R_MAX:
            return {'status': 'escaped', 'final_state': None}

        # 4. 表面衝突判定
        if r_now <= RM:
            pos_impact = pos_start  # 厳密には交点を求めるべきだが、start地点で近似

            # 衝突地点の経度・緯度・温度計算
            lon_rot = np.arctan2(pos_impact[1], pos_impact[0])
            lat_rot = np.arcsin(np.clip(pos_impact[2] / np.linalg.norm(pos_impact), -1, 1))
            lon_fixed = (lon_rot + subsolar_lon + np.pi) % (2 * np.pi) - np.pi
            temp_impact = calculate_surface_temperature_leblanc(lon_fixed, lat_rot, AU, subsolar_lon)

            # 吸着判定
            if np.random.random() < calculate_sticking_probability(temp_impact):
                return {'status': 'stuck', 'pos_at_impact': pos_impact, 'weight': weight}
            else:
                # バウンス (非弾性衝突 / 熱的再放出)
                # 衝突エネルギーの一部を失い、表面温度にある程度なじんで再放出されるモデル(Accomodation)
                E_in = 0.5 * PHYSICAL_CONSTANTS['MASS_NA'] * np.sum(vel ** 2)
                E_T = PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_impact
                # エネルギー加重平均
                E_out = settings['BETA'] * E_T + (1.0 - settings['BETA']) * E_in
                v_out = np.sqrt(2 * E_out / PHYSICAL_CONSTANTS['MASS_NA']) if E_out > 0 else 0

                norm = pos_impact / np.linalg.norm(pos_impact)
                # ランバート反射方向へ
                rebound_dir = transform_local_to_world(sample_lambertian_direction_local(), norm)

                pos = (RM + 1.0) * norm  # 埋まらないように少し浮かせる
                vel = v_out * rebound_dir

        pos_start = pos.copy()

    return {'status': 'alive', 'final_state': {'pos': pos, 'vel': vel, 'weight': weight}}


# ==============================================================================
# 4. メインルーチン
# ==============================================================================
def main_snapshot_simulation():
    start_time = time.time()

    # --- A. シミュレーション設定 ---
    OUTPUT_DIRECTORY = r"./SimulationResult_202510"

    # 表面グリッド解像度
    N_LON_FIXED, N_LAT = 72, 36
    INIT_SURF_DENS = 7.5e14 * (100 ** 2) * 0.0053  # 初期表面密度 (単位注意: /m^2)
    MAIN_DT = 500.0  # メインループの時間刻み [s]

    SPIN_UP_YEARS = 1.0  # 助走期間 (平衡状態にするための空回し)
    TOTAL_SIM_YEARS = 1.0  # 実際のデータ取得期間

    # データ保存を行うタイミング (True Anomaly [deg])
    TARGET_TAA = np.arange(0, 360, 1)  # 1度刻みで保存

    # スーパーパーティクル(SP)の目標数 (統計数を確保するため)
    TARGET_SPS = {'TD': 1000, 'PSD': 1000, 'SWS': 1000, 'MMV': 1000}

    # 3D空間密度グリッド設定
    GRID_RESOLUTION = 201  # 101x101x101
    GRID_MAX_RM = 5.0  # 半径5水星半径まで

    # 各種パラメータ
    F_UV_1AU = 1.5e14 * (100 ** 2)  # UVフラックス
    Q_PSD = 1.0e-20 / (100 ** 2)  # PSD断面積
    TEMP_PSD = 1500.0  # PSD放出温度(等価温度)
    TEMP_MMV = 3000.0  # MMV放出温度

    SWS_PARAMS = {
        'FLUX_1AU': 10.0 * 100 ** 3 * 400e3,  # 太陽風フラックス
        'YIELD': 0.06,  # スパッタリング収率
        'U_eV': 0.27,  # 結合エネルギー
        'REF_DENS': 7.5e14 * 100 ** 2,  # 参照表面密度
        'LON_RANGE': np.deg2rad([-40, 40]),  # 太陽風が当たる経度範囲
        'LAT_N_RANGE': np.deg2rad([30, 60]),  # 開口部(カスプ)緯度範囲
        'LAT_S_RANGE': np.deg2rad([-60, -30]),
    }

    settings = {
        'BETA': 0.5, 'T1AU': 168918.0, 'DT': MAIN_DT,
        'GRID_RADIUS_RM': GRID_MAX_RM + 1.0,
        'USE_SOLAR_GRAVITY': True, 'USE_CORIOLIS_FORCES': True
    }

    # --- B. 初期化処理 ---
    run_name = f"DynamicGrid{N_LON_FIXED}x{N_LAT}_13.0"
    target_output_dir = os.path.join(OUTPUT_DIRECTORY, run_name)
    os.makedirs(target_output_dir, exist_ok=True)
    print(f"Simulation Start. Results will be saved to: {target_output_dir}")

    # 表面グリッド（セル面積計算）
    lon_edges = np.linspace(-np.pi, np.pi, N_LON_FIXED + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    dlon = lon_edges[1] - lon_edges[0]
    cell_areas = (PHYSICAL_CONSTANTS['RM'] ** 2) * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))

    # 表面密度配列の初期化
    surface_density = np.full((N_LON_FIXED, N_LAT), INIT_SURF_DENS, dtype=np.float64)

    # 外部データ読み込み (スペクトル & 軌道)
    try:
        # ※ファイルが存在することを確認してください
        spec_np = np.loadtxt('SolarSpectrum_Na0.txt', usecols=(0, 3))
        orbit_data = np.loadtxt('orbit2025_v6.txt')

        # === 【重要修正】TAA (0列目) の角度連続化処理 ===
        # TAAが 360 -> 0 のようにリセットされている場合、
        # np.interp が 180 と補間してしまうのを防ぐため、unwrap して連続的な値にします。
        orbit_data[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(orbit_data[:, 0])))
        # ===============================================

    except Exception as e:
        print(f"Error loading external files: {e}")
        print("Please ensure 'SolarSpectrum_Na0.txt' and 'orbit2025_v6.txt' exist.")
        return

    # スペクトルデータの整理
    wl, gamma = spec_np[:, 0], spec_np[:, 1]
    if wl[1] < wl[0]:  # 波長が昇順でなければソート
        idx = np.argsort(wl)
        wl, gamma = wl[idx], gamma[idx]

    # 吸収断面積の定数項
    const_sigma = PHYSICAL_CONSTANTS['E_CHARGE'] ** 2 / (
            4 * PHYSICAL_CONSTANTS['ME'] * PHYSICAL_CONSTANTS['C'] * PHYSICAL_CONSTANTS['EPSILON_0'])
    spec_dict = {
        'wl': wl, 'gamma': gamma,
        'sigma0_perdnu2': const_sigma * 0.641,  # D2線強度
        'sigma0_perdnu1': const_sigma * 0.320,  # D1線強度
        'JL': 5.18e14  # 光子束定数
    }

    # 時間設定
    MERCURY_YEAR = PHYSICAL_CONSTANTS['ORBITAL_PERIOD']
    idx_peri = np.argmin(np.abs(orbit_data[:, 0]))  # 近日点のインデックス
    t_peri_file = orbit_data[idx_peri, 2]  # 近日点通過時刻

    t_start_run = t_peri_file
    t_end_run = t_start_run + TOTAL_SIM_YEARS * MERCURY_YEAR
    # 助走期間分だけ開始時刻を早める
    t_start_spinup = t_start_run - SPIN_UP_YEARS * MERCURY_YEAR

    t_curr = t_start_spinup

    # 粒子管理リスト
    active_particles = []
    prev_taa = -999

    # 「グリッドに戻ってきた量」の記録用
    prev_gained_grid = np.zeros_like(surface_density)

    # --- C. タイムステップループ ---
    total_steps = int((t_end_run - t_start_spinup) / MAIN_DT)

    with tqdm(total=total_steps, desc="Simulating") as pbar:
        while t_curr < t_end_run:

            # 1. 現在時刻の軌道・回転情報を取得
            TAA, AU, V_rad, V_tan, sub_lon = get_orbital_params_cyclic(t_curr, orbit_data, t_peri_file)

            new_particles = []  # このステップで新規生成される粒子

            # 各プロセスのレート計算用グリッド
            loss_grid = np.zeros_like(surface_density)
            rate_psd = np.zeros_like(surface_density)
            rate_td = np.zeros_like(surface_density)
            rate_sws = np.zeros_like(surface_density)

            # 平衡密度記録用 (NaNで初期化)
            na_eq_record = np.full((N_LON_FIXED, N_LAT), np.nan)

            f_uv = F_UV_1AU / (AU ** 2)
            sw_flux = SWS_PARAMS['FLUX_1AU'] / (AU ** 2)
            mmv_flux = calculate_mmv_flux(AU)

            # --- MMV (全球的生成) ---
            # 表面密度に依存せず、常に宇宙空間から降り注ぐため独立計算
            n_mmv = mmv_flux * 4 * np.pi * PHYSICAL_CONSTANTS['RM'] ** 2 * MAIN_DT
            w_mmv = max(1.0, n_mmv / TARGET_SPS['MMV'])
            if n_mmv > 0:
                num_p = int(n_mmv / w_mmv)
                if np.random.random() < (n_mmv / w_mmv - num_p): num_p += 1
                for _ in range(num_p):
                    # 時間のランダム化: DTの間でランダムな時刻に発生
                    dt_init = MAIN_DT * np.random.random()

                    # ランダムな位置と速度
                    while True:
                        lr = np.random.uniform(-np.pi, np.pi)
                        # 緯度方向の重み付け (面積要素考慮)
                        if np.random.random() < (1 - 1 / 3 * np.sin(lr)) * 0.75: break
                    latr = np.arcsin(np.random.uniform(-1, 1))

                    pos = lonlat_to_xyz(lr, latr, PHYSICAL_CONSTANTS['RM'])
                    norm = pos / PHYSICAL_CONSTANTS['RM']
                    spd = sample_speed_from_flux_distribution(PHYSICAL_CONSTANTS['MASS_NA'], TEMP_MMV)
                    vel = spd * transform_local_to_world(sample_lambertian_direction_local(), norm)

                    new_particles.append({
                        'pos': pos, 'vel': vel, 'weight': w_mmv, 'dt_remaining': dt_init
                    })

            # --- 表面依存プロセス (PSD, TD, SWS) ---
            atoms_psd_step = 0
            atoms_td_step = 0
            atoms_sws_step = 0

            # グリッドごとの放出率計算ループ
            for i in range(N_LON_FIXED):
                for j in range(N_LAT):
                    # グリッド中心座標
                    lon_f = (lon_edges[i] + lon_edges[i + 1]) / 2
                    lat_f = (lat_edges[j] + lat_edges[j + 1]) / 2

                    # PSD: 太陽光が当たっているか (cos_z > 0)
                    cos_z = np.cos(lat_f) * np.cos(lon_f - sub_lon)
                    if cos_z > 0:
                        rate_psd[i, j] = f_uv * Q_PSD * cos_z

                    # TD: 表面温度に依存
                    temp = calculate_surface_temperature_leblanc(lon_f, lat_f, AU, sub_lon)
                    rate_td[i, j] = calculate_thermal_desorption_rate(temp)

                    # SWS: 太陽風マップ領域内か
                    lon_sun = (lon_f - sub_lon + np.pi) % (2 * np.pi) - np.pi
                    in_lon = SWS_PARAMS['LON_RANGE'][0] <= lon_sun <= SWS_PARAMS['LON_RANGE'][1]
                    in_lat = (SWS_PARAMS['LAT_N_RANGE'][0] <= lat_f <= SWS_PARAMS['LAT_N_RANGE'][1]) or \
                             (SWS_PARAMS['LAT_S_RANGE'][0] <= lat_f <= SWS_PARAMS['LAT_S_RANGE'][1])
                    if in_lon and in_lat:
                        # 表面密度がある程度ないと放出されないモデル
                        # ここでは簡単のためYield固定だが、実際は被覆率依存などを入れる場合がある
                        rate_sws[i, j] = (sw_flux * SWS_PARAMS['YIELD']) / SWS_PARAMS['REF_DENS']

                    # 総放出率
                    rate_total = rate_psd[i, j] + rate_td[i, j] + rate_sws[i, j]
                    if rate_total <= 0: continue

                    # ★ 平衡モード (Equilibrium Logic) ★
                    # タイムスケールが短い(すぐに放出される)場合、
                    # 入ってくる粒子数(flux_in)と出ていく粒子数が釣り合う密度(dens_eq)に強制設定する。
                    # これにより、極端に短いタイムステップ計算を回避し、計算を安定化させる。
                    timescale = 1.0 / rate_total
                    dens = surface_density[i, j]

                    if timescale <= MAIN_DT and t_curr > t_start_spinup:
                        # 前ステップでグリッドに戻ってきた粒子数(flux_in)を使用
                        flux_in = (prev_gained_grid[i, j] / cell_areas[j]) / MAIN_DT
                        dens_eq = flux_in / rate_total if rate_total > 0 else 0
                        na_eq_record[i, j] = dens_eq
                        dens = dens_eq  # 密度を平衡値に上書き

                    # 放出される原子数
                    n_avail = dens * cell_areas[j]
                    n_lost = min(n_avail, n_avail * rate_total * MAIN_DT)

                    loss_grid[i, j] = n_lost

                    # 各プロセスの内訳集計
                    atoms_psd_step += n_lost * (rate_psd[i, j] / rate_total)
                    atoms_td_step += n_lost * (rate_td[i, j] / rate_total)
                    atoms_sws_step += n_lost * (rate_sws[i, j] / rate_total)

            # スーパーパーティクル重み(weight)の決定
            w_psd = max(1.0, atoms_psd_step / TARGET_SPS['PSD'])
            w_td = max(1.0, atoms_td_step / TARGET_SPS['TD'])
            w_sws = max(1.0, atoms_sws_step / TARGET_SPS['SWS'])

            # 実際の粒子生成ループ
            for i in range(N_LON_FIXED):
                for j in range(N_LAT):
                    n_lost = loss_grid[i, j]
                    if n_lost <= 0: continue

                    # 割合に応じてプロセスを割り振り
                    r_p, r_t, r_s = rate_psd[i, j], rate_td[i, j], rate_sws[i, j]
                    tot = r_p + r_t + r_s

                    lon_f = (lon_edges[i] + lon_edges[i + 1]) / 2
                    lat_f = (lat_edges[j] + lat_edges[j + 1]) / 2
                    temp = calculate_surface_temperature_leblanc(lon_f, lat_f, AU, sub_lon)

                    params = [
                        ('PSD', n_lost * (r_p / tot), TEMP_PSD, w_psd),
                        ('TD', n_lost * (r_t / tot), temp, w_td),
                        ('SWS', n_lost * (r_s / tot), None, w_sws)
                    ]

                    for p_type, n_amount, T_or_none, w in params:
                        if n_amount <= 0: continue
                        num = int(n_amount / w)
                        # 端数処理 (確率的に+1個生成)
                        if np.random.random() < (n_amount / w - num): num += 1

                        for _ in range(num):
                            dt_init = MAIN_DT * np.random.random()
                            # 初速度の決定
                            if p_type == 'SWS':
                                E = sample_thompson_sigmund_energy(SWS_PARAMS['U_eV'])
                                spd = np.sqrt(2 * E * PHYSICAL_CONSTANTS['EV_TO_JOULE'] / PHYSICAL_CONSTANTS['MASS_NA'])
                            else:
                                spd = sample_speed_from_flux_distribution(PHYSICAL_CONSTANTS['MASS_NA'], T_or_none)

                            # 位置と速度ベクトル
                            lon_rot = lon_f - sub_lon
                            pos = lonlat_to_xyz(lon_rot, lat_f, PHYSICAL_CONSTANTS['RM'])
                            norm = pos / PHYSICAL_CONSTANTS['RM']
                            vel = spd * transform_local_to_world(sample_lambertian_direction_local(), norm)

                            new_particles.append({
                                'pos': pos, 'vel': vel, 'weight': w, 'dt_remaining': dt_init
                            })

            active_particles.extend(new_particles)

            # --- 2. 粒子移動 (Parallel Processing) ---
            tasks = []
            for p in active_particles:
                # dt_remainingがあればそれを使い、なければ全ステップ(MAIN_DT)進める
                dur = p.pop('dt_remaining', MAIN_DT)
                tasks.append({
                    'settings': settings, 'spec': spec_dict, 'particle_state': p,
                    'orbit': (TAA, AU, V_rad, V_tan, sub_lon),
                    'duration': dur
                })

            next_particles = []
            gained_grid = np.zeros_like(surface_density)  # 戻ってきた粒子数

            if tasks:
                # CPUコア数-1 で並列計算
                with Pool(cpu_count() - 1) as pool:
                    results = pool.map(simulate_particle_for_one_step, tasks)

                for res in results:
                    if res['status'] == 'alive':
                        next_particles.append(res['final_state'])
                    elif res['status'] == 'stuck':
                        # 表面に吸着した場合、そのグリッドに粒子(重み)を加算
                        pos = res['pos_at_impact']
                        w = res['weight']
                        ln = np.arctan2(pos[1], pos[0])
                        lt = np.arcsin(np.clip(pos[2] / np.linalg.norm(pos), -1, 1))
                        # 経度を固定座標系(Subsolar基準)に変換してインデックス特定
                        ln_fix = (ln + sub_lon + np.pi) % (2 * np.pi) - np.pi
                        ix = np.searchsorted(lon_edges, ln_fix) - 1
                        iy = np.searchsorted(lat_edges, lt) - 1
                        if 0 <= ix < N_LON_FIXED and 0 <= iy < N_LAT:
                            gained_grid[ix, iy] += w

            active_particles = next_particles
            prev_gained_grid = gained_grid.copy()

            # --- 3. 表面密度更新 ---
            loss_dens = loss_grid / cell_areas
            gain_dens = gained_grid / cell_areas

            # 拡散方程式の差分更新に相当 (New = Old + Gain - Loss)
            dens_next = surface_density + gain_dens - loss_dens

            # 平衡モードのグリッドは計算値を上書き
            mask_eq = ~np.isnan(na_eq_record)
            dens_next[mask_eq] = na_eq_record[mask_eq]

            surface_density = np.clip(dens_next, 0, None)

            # --- 4. 結果保存 (Snapshot) ---
            if prev_taa != -999:
                # 目標TAAをまたいだか判定
                passed = False
                for tgt in TARGET_TAA:
                    # 通常の通過 or 360->0度の通過
                    if (prev_taa < tgt <= TAA) or (prev_taa > 350 and TAA < 10 and tgt == 0):
                        passed = True
                        break

                # Spin-up期間が終わっている場合のみ保存
                if passed and t_curr >= t_start_run:
                    rel_h = (t_curr - t_start_run) / 3600.0
                    print(f" Saving TAA={TAA:.1f}, Time={rel_h:.1f}h, Particles={len(active_particles)}")

                    # 3D空間密度の集計 (Histogram)
                    dgrid = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION, GRID_RESOLUTION), dtype=np.float32)
                    gmin, gmax = -GRID_MAX_RM * PHYSICAL_CONSTANTS['RM'], GRID_MAX_RM * PHYSICAL_CONSTANTS['RM']
                    csize = (gmax - gmin) / GRID_RESOLUTION
                    cvol = csize ** 3  # セル体積

                    pos_arr = np.array([p['pos'] for p in active_particles])
                    weights_arr = np.array([p['weight'] for p in active_particles])

                    if len(pos_arr) > 0:
                        # numpyのヒストグラム機能で高速集計
                        H, _ = np.histogramdd(pos_arr, bins=(GRID_RESOLUTION, GRID_RESOLUTION, GRID_RESOLUTION),
                                              range=[(gmin, gmax), (gmin, gmax), (gmin, gmax)],
                                              weights=weights_arr)
                        dgrid = H.astype(np.float32) / cvol

                    # numpyバイナリ形式(.npy)で保存
                    fname_d = f"density_grid_t{int(rel_h):05d}_taa{int(round(TAA)):03d}.npy"
                    fname_s = f"surface_density_t{int(rel_h):05d}_taa{int(round(TAA)):03d}.npy"

                    np.save(os.path.join(target_output_dir, fname_d), dgrid)
                    np.save(os.path.join(target_output_dir, fname_s), surface_density)

            prev_taa = TAA
            t_curr += MAIN_DT
            pbar.update(1)

    print("Done. Simulation Completed.")


if __name__ == '__main__':
    # Windows環境等でのMultiprocessing対策
    sys.modules['__main__'].__spec__ = None
    main_snapshot_simulation()