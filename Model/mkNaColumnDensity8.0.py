# -*- coding: utf-8 -*-
"""
==============================================================================
概要
==============================================================================
このスクリプトは、水星のナトリウム大気のふるまいを、時間発展を考慮して
シミュレートする3次元モンテカルロ法に基づいたプログラムです。

Leblanc (2003) の論文に基づき、以下の特徴を持ちます。

1.  **動的表面密度**:
    - 水星表面を惑星固定座標系のグリッドで管理します。
    - 各セルのナトリウム密度は、放出によって減少し、吸着によって増加します。

2.  **複数の生成過程**:
    - 光刺激脱離 (PSD): 表面密度と太陽光に比例。
    - 熱脱離 (TD): 表面密度と表面温度に比例。
    - 微小隕石蒸発 (MMV): 表面密度に依存しない補給源。

==============================================================================
座標系
==============================================================================
このシミュレーションは、2つの座標系を併用します。

1.  **惑星固定座標系**:
    - 表面グリッド(surface_density_grid)、表面温度計算に使用。
    - 経度 0 は水星の特定の地点に固定。

2.  **水星中心・太陽固定回転座標系**:
    - 粒子の軌道追跡、空間密度グリッドの集計に使用。
    - +X 軸: 常に太陽の方向。
    - メインループで計算される `subsolar_lon_rad`（惑星固定座標系での
      太陽直下点経度）を用いて、2つの座標系をマッピングします。

==============================================================================
主な物理モデル (Leblanc 2003 準拠)
==============================================================================
1.  **粒子生成**:
    - PSD, TD: 表面のナトリウム貯蔵庫を消費します。
    - MMV: 外部からの補給源として機能します 。

2.  **初期速度 (フラックス分布)**:
    - PSD (T=1500K [cite: 269]), TD (T=表面温度 [cite: 246]), MMV (T=3000K [cite: 338])
      それぞれの温度におけるマクスウェル「フラックス」分布
      (f(E) ∝ E * exp(-E/kT)) に従う速度をサンプリングします。
    - 放出角度: ランバート（余弦則）分布。

3.  **軌道計算 (4次ルンゲ＝クッタ法)**:
    - (変更なし：水星重力、SRP、太陽重力、見かけの力)

4.  **消滅過程**:
    - 光電離: (変更なし)
    - 表面衝突: 表面に再衝突し吸着(stick)した場合、
                 衝突地点の惑星固定グリッドセルの密度を増加させます。
                 吸着しない場合は熱的に再放出されます。

==============================================================================
必要な外部ファイル
==============================================================================
1.  **orbit2025_v5.txt**: (変更なし)
2.  **SolarSpectrum_Na0.txt**: (変更なし)
"""

import numpy as np
import sys
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

# ==============================================================================
# 物理定数 (SI単位系)
# ==============================================================================
PHYSICAL_CONSTANTS = {
    'PI': np.pi,
    'AU': 1.496e11,  # [m]
    'MASS_NA': 22.98976928 * 1.66054e-27,  # [kg]
    'K_BOLTZMANN': 1.380649e-23,  # [J/K]
    'GM_MERCURY': 2.2032e13,  # [m^3/s^2]
    'RM': 2.440e6,  # [m]
    'C': 299792458.0,  # [m/s]
    'H': 6.62607015e-34,  # [J·s]
    'E_CHARGE': 1.602176634e-19,  # [C]
    'ME': 9.1093897e-31,  # [kg]
    'EPSILON_0': 8.854187817e-12,  # [F/m]
    'G': 6.6743e-11,  # [N m^2/kg^2]
    'MASS_SUN': 1.989e30,  # [kg]
    'EV_TO_JOULE': 1.602176634e-19,  # [J / eV]
}


# ==============================================================================
# 物理モデルに基づくヘルパー関数群
# ==============================================================================

def calculate_surface_temperature_leblanc(lon_fixed_rad, lat_rad, AU, subsolar_lon_rad):
    """
    水星表面の局所的な温度を計算します (Leblanc 2003,  モデル)。

    日照側では太陽天頂角(SZA)と太陽距離(AU)に依存し、
    夜側では最低温度(T_night)に固定されます。

    Args:
        lon_fixed_rad (float): 計算対象地点の経度 (惑星固定座標系) [rad]
        lat_rad (float): 計算対象地点の緯度 (惑星固定座標系) [rad]
        AU (float): 現在の太陽からの距離 [天文単位]
        subsolar_lon_rad (float): 太陽直下点の経度 (惑星固定座標系) [rad]

    Returns:
        float: 表面温度 [K]
    """
    T_night = 100.0  # 夜側の最低温度 [K] [cite: 219]

    # 太陽天頂角の余弦 (cos(SZA)) を計算
    cos_theta = np.cos(lat_rad) * np.cos(lon_fixed_rad - subsolar_lon_rad)

    if cos_theta <= 0:
        return T_night  # 夜側

    # 日照側の温度
    # T(lon, lat) = T0 + T1 * (cos(SZA))^0.25
    # T0 は近日点(perihelion)で 600K、遠日点(aphelion)で 475K
    # 水星の軌道: 0.307 AU (peri) to 0.467 AU (aph)
    # 簡単のため、AUに基づいてT0を線形補間する
    T0_peri = 600.0
    T0_aph = 475.0
    AU_peri = 0.307
    AU_aph = 0.467
    T0 = np.interp(AU, [AU_peri, AU_aph], [T0_peri, T0_aph])
    T1 = 100.0  # [cite: 217]

    return T0 + T1 * (cos_theta ** 0.25)


def calculate_sticking_probability(surface_temp_K):
    """
    表面温度に基づき、ナトリウム原子が表面に吸着する確率を計算します。
    (Leblanc 2003, [cite: 165-167, 169] モデル)

    Args:
        surface_temp_K (float): 衝突地点の表面温度 [K]

    Returns:
        float: 実効的な吸着確率 (0から1の範囲)
    """
    # 論文 [cite: 166] の実験データにフィットする定数
    # Stick=0.5 at T=250K, Stick=0.2 at T=500K
    A = 0.0804
    B = 458.0
    porosity = 0.8  # 表面の多孔性 [cite: 169]

    if surface_temp_K <= 0:
        return 1.0

    # 基本的な吸着確率 p = A * exp(B / T)
    p_stick = A * np.exp(B / surface_temp_K)

    # 多孔性を考慮した実効的な吸着確率 [cite: 167]
    p_stick_eff = p_stick / (1.0 - (1.0 - p_stick) * porosity)

    return min(p_stick_eff, 1.0)


def calculate_thermal_desorption_rate(surface_temp_K):
    """
    熱脱離の放出率 (1/s) を計算します。
    Rate = v * exp(-U / k_B * T_s)  (Leblanc 2003, )

    Args:
        surface_temp_K (float): 表面温度 [K]

    Returns:
        float: 放出率 [1/s] (この値を表面密度に乗算して使用する)
    """
    if surface_temp_K < 350.0:  # [cite: 241]
        return 0.0  # 低温ではほぼゼロ

    VIB_FREQ = 1e13  # 表面の振動数 v [1/s]
    BINDING_ENERGY_EV = 1.85  # 平均結合エネルギー U [eV] [cite: 238, 240]
    BINDING_ENERGY_J = BINDING_ENERGY_EV * PHYSICAL_CONSTANTS['EV_TO_JOULE']

    k_B = PHYSICAL_CONSTANTS['K_BOLTZMANN']

    exponent = -BINDING_ENERGY_J / (k_B * surface_temp_K)
    if exponent < -700:  # exp(-700) はほぼ 0
        return 0.0

    return VIB_FREQ * np.exp(exponent)


def calculate_mmv_flux(AU):
    """
    微小隕石蒸発 (MMV) によるナトリウムのフラックス [atoms/m^2/s] を計算します。
    これは表面密度に依存しない「補給源」です。
    (Leblanc 2003, [cite: 341-342] モデル)

    Args:
        AU (float): 現在の太陽距離 [AU]

    Returns:
        float: 放出フラックス [atoms/m^2/s]
    """
    # 論文 [cite: 342] によれば、近日点(0.307 AU)で 5e23 Na/s (惑星全体)
    TOTAL_FLUX_AT_PERI_NA_S = 5e23
    PERIHELION_AU = 0.307
    MERCURY_SURFACE_AREA_M2 = 4 * PHYSICAL_CONSTANTS['PI'] * (PHYSICAL_CONSTANTS['RM'] ** 2)

    # 近日点での平均フラックス [atoms/m^2/s]
    avg_flux_at_peri = TOTAL_FLUX_AT_PERI_NA_S / MERCURY_SURFACE_AREA_M2

    # 太陽距離に応じて R_Hel^(-1.9) でスケーリング [cite: 341]
    # R_Hel は AU 単位
    current_R_Hel = AU / PERIHELION_AU  # (間違い。AUはそのままR_Hel)
    # Flux(AU) = Flux(Peri) * (AU / Peri_AU)^(-1.9)
    scaling_factor = (AU / PERIHELION_AU) ** (-1.9)
    # (Leblanc 2003 [cite: 342] は 1/R_Hel^1.9 と記述しているので、
    #  Flux(AU) = C * (AU)^(-1.9) とすべき)
    # C = Flux(Peri) * (Peri_AU)^(1.9)
    C = avg_flux_at_peri * (PERIHELION_AU ** 1.9)

    return C * (AU ** (-1.9))


def sample_speed_from_flux_distribution(mass_kg, temp_k):
    """
    指定された温度の「マクスウェル・フラックス分布」に従う速さをサンプリングします。
    f(E) ∝ E * exp(-E / kT) [cite: 246]
    これは形状パラメータ k=2, スケールパラメータ θ=kT のガンマ分布です。

    Args:
        mass_kg (float): 粒子の質量 [kg]
        temp_k (float): 温度 [K]

    Returns:
        float: サンプリングされた速さ [m/s]
    """
    # ガンマ分布(形状=2, スケール=kT)からエネルギーEをサンプリング
    # np.random.gamma(shape, scale)
    kT = PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_k
    E = np.random.gamma(2.0, kT)

    # E = 0.5 * m * v^2  =>  v = sqrt(2E / m)
    return np.sqrt(2.0 * E / mass_kg)


def sample_lambertian_direction_local():
    """ (変更なし) """
    u1, u2 = np.random.random(2)
    phi = 2 * PHYSICAL_CONSTANTS['PI'] * u1
    cos_theta = np.sqrt(1 - u2)
    sin_theta = np.sqrt(u2)
    return np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])


def transform_local_to_world(local_vec, normal_vector):
    """ (変更なし) """
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


def get_orbital_params(time_sec, orbit_data, mercury_year_sec):
    """ (変更なし) """
    ROTATION_PERIOD_SEC = 58.646 * 24 * 3600
    current_time_in_orbit = time_sec % mercury_year_sec
    time_col = orbit_data[:, 2]
    taa = np.interp(current_time_in_orbit, time_col, orbit_data[:, 0])
    au = np.interp(current_time_in_orbit, time_col, orbit_data[:, 1])
    v_radial = np.interp(current_time_in_orbit, time_col, orbit_data[:, 3])
    v_tangential = np.interp(current_time_in_orbit, time_col, orbit_data[:, 4])
    # この subsolar_lon_rad は「惑星固定座標系」での太陽直下点経度
    subsolar_lon_rad = (2 * np.pi * time_sec / ROTATION_PERIOD_SEC) % (2 * np.pi)
    return taa, au, v_radial, v_tangential, subsolar_lon_rad


def lonlat_to_xyz(lon_rad, lat_rad, radius):
    """ (変更なし) """
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return np.array([x, y, z])


def xyz_to_lonlat_idx(pos_vec, lon_edges_fixed, lat_edges_fixed, N_LON_FIXED, N_LAT):
    """
    ワールド座標（回転座標系）のベクトルと太陽直下点経度から、
    惑星固定グリッドのインデックス (i_lon, i_lat) を計算します。

    Args:
        pos_vec (np.ndarray): 回転座標系での位置ベクトル [x, y, z]
        lon_edges_fixed (np.ndarray): 惑星固定経度の境界
        lat_edges_fixed (np.ndarray): 惑星固定緯度の境界
        subsolar_lon_rad (float): 惑星固定座標系での太陽直下点経度

    Returns:
        tuple: (i_lon, i_lat) (範囲外の場合は (-1, -1))
    """
    # 1. 回転座標系での経度・緯度を計算
    r = np.linalg.norm(pos_vec)
    if r == 0: return -1, -1
    lon_rot = np.arctan2(pos_vec[1], pos_vec[0])
    lat_rot = np.arcsin(np.clip(pos_vec[2] / r, -1.0, 1.0))

    # 2. 惑星固定座標系の経度に変換
    # lon_fixed = lon_rot + subsolar_lon_rad
    # (lon_fixed は -pi から 3pi の範囲になりうるので、-pi から pi にマッピング)
    # lon_fixed_mapped = (lon_fixed + np.pi) % (2 * np.pi) - np.pi
    #
    # (注意) この関数は「衝突判定」で使われる。
    # 衝突時の subsolar_lon_rad をワーカーから受け取る必要がある。
    # -> ワーカー関数 `simulate_particle_for_one_step` の引数を変更する
    #
    # (再考) ワーカーは回転座標系での位置だけを返し、
    # メインループが `subsolar_lon_rad` を使って変換する方が効率的。
    # この関数はメインループ内で使用する。

    # (i_lon, i_lat) を計算
    i_lon = np.searchsorted(lon_edges_fixed, lon_rot) - 1
    i_lat = np.searchsorted(lat_edges_fixed, lat_rot) - 1

    if 0 <= i_lon < N_LON_FIXED and 0 <= i_lat < N_LAT:
        return i_lon, i_lat
    else:
        return -1, -1


# ==============================================================================
# コア追跡関数 (並列処理の対象)
# ==============================================================================

def _calculate_acceleration(pos, vel, V_radial_ms, V_tangential_ms, AU, spec_data, settings):
    """ (変更なし) """
    x, y, z = pos
    r0 = AU * PHYSICAL_CONSTANTS['AU']
    velocity_for_doppler = vel[0] + V_radial_ms
    w_na_d2 = 589.1582e-9 * (1.0 + velocity_for_doppler / PHYSICAL_CONSTANTS['C'])
    w_na_d1 = 589.7558e-9 * (1.0 + velocity_for_doppler / PHYSICAL_CONSTANTS['C'])
    b = 0.0
    wl, gamma, sigma0_perdnu2, sigma0_perdnu1, JL = spec_data.values()
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
    if x < 0 and np.sqrt(y ** 2 + z ** 2) < PHYSICAL_CONSTANTS['RM']:
        b = 0.0
    accel_srp = np.array([-b, 0.0, 0.0])
    r_sq = np.sum(pos ** 2)
    accel_g = -PHYSICAL_CONSTANTS['GM_MERCURY'] * pos / (r_sq ** 1.5) if r_sq > 0 else np.array([0., 0., 0.])
    accel_sun = np.array([0.0, 0.0, 0.0])
    if settings.get('USE_SOLAR_GRAVITY', False):
        G = PHYSICAL_CONSTANTS['G']
        M_SUN = PHYSICAL_CONSTANTS['MASS_SUN']
        r_ps_vec = np.array([r0 - pos[0], -pos[1], -pos[2]])
        r_ps_mag_sq = np.sum(r_ps_vec ** 2)
        if r_ps_mag_sq > 0:
            accel_sun = (G * M_SUN * r_ps_vec) / (r_ps_mag_sq ** 1.5)
    accel_coriolis = np.array([0.0, 0.0, 0.0])
    accel_centrifugal = np.array([0.0, 0.0, 0.0])
    if settings.get('USE_CORIOLIS_FORCES', False):
        if r0 > 0:
            omega_val = V_tangential_ms / r0
            omega_sq = omega_val ** 2
            accel_centrifugal = np.array([
                #omega_sq * pos[0],
                omega_val ** 2 * (pos[0] - r0),
                omega_sq * pos[1],
                0.0])
            two_omega = 2 * omega_val
            accel_coriolis = np.array([two_omega * vel[1], -two_omega * vel[0], 0.0])
    return accel_srp + accel_g + accel_sun + accel_centrifugal + accel_coriolis


def simulate_particle_for_one_step(args):
    """
    一個のスーパーパーティクルを、指定された時間 (duration) だけ追跡します。

    【変更点】
    - 'status': 'stuck' の場合、消滅ではなく、衝突直前の位置ベクトルと
      重みを返します。
    """
    # ---------------------------------
    # 1. 引数の展開
    # ---------------------------------
    settings, spec_data = args['settings'], args['spec']
    TAA, AU, V_radial_ms, V_tangential_ms, subsolar_lon_rad_fixed = args['orbit']
    duration, DT = args['duration'], settings['DT']
    RM = PHYSICAL_CONSTANTS['RM']
    R_MAX = RM * settings['GRID_RADIUS_RM']
    pos, vel, weight = args['particle_state']['pos'], args['particle_state']['vel'], args['particle_state']['weight']

    tau_ionization = settings['T1AU'] * AU ** 2
    num_steps = int(duration / DT)

    # ---------------------------------
    # 2. 時間積分ループ (RK4)
    # ---------------------------------
    pos_at_start_of_step = pos.copy()  # 衝突判定用にステップ開始時の位置を保持

    for _ in range(num_steps):

        pos_at_start_of_step = pos.copy()  # ループの最初に更新

        # --- 2a. 光電離判定 ---
        if pos[0] > 0:
            if np.random.random() < (1.0 - np.exp(-DT / tau_ionization)):
                return {'status': 'ionized', 'final_state': None}

        # --- 2b. 4次ルンゲ＝クッタ法 (RK4) ---
        k1_vel = DT * _calculate_acceleration(pos, vel, V_radial_ms, V_tangential_ms, AU, spec_data, settings)
        k1_pos = DT * vel
        k2_vel = DT * _calculate_acceleration(pos + 0.5 * k1_pos, vel + 0.5 * k1_vel, V_radial_ms, V_tangential_ms, AU,
                                              spec_data, settings)
        k2_pos = DT * (vel + 0.5 * k1_vel)
        k3_vel = DT * _calculate_acceleration(pos + 0.5 * k2_pos, vel + 0.5 * k2_vel, V_radial_ms, V_tangential_ms, AU,
                                              spec_data, settings)
        k3_pos = DT * (vel + 0.5 * k2_vel)
        k4_vel = DT * _calculate_acceleration(pos + k3_pos, vel + k3_vel, V_radial_ms, V_tangential_ms, AU, spec_data,
                                              settings)
        k4_pos = DT * (vel + k3_vel)

        pos += (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos) / 6.0
        vel += (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel) / 6.0

        # --- 2c. 境界条件の判定 ---
        r_current = np.linalg.norm(pos)

        # (i) 脱出
        if r_current > R_MAX:
            return {'status': 'escaped', 'final_state': None}

        # (ii) 表面への衝突
        if r_current <= RM:
            # 衝突地点の惑星固定経度・緯度を計算
            # (ステップ開始時の位置 pos_at_start_of_step を使用)
            lon_rot = np.arctan2(pos_at_start_of_step[1], pos_at_start_of_step[0])
            lat_rot = np.arcsin(np.clip(pos_at_start_of_step[2] / np.linalg.norm(pos_at_start_of_step), -1.0, 1.0))

            # 惑星固定経度に変換
            lon_fixed = (lon_rot + subsolar_lon_rad_fixed)
            lon_fixed = (lon_fixed + PHYSICAL_CONSTANTS['PI']) % (2 * PHYSICAL_CONSTANTS['PI']) - PHYSICAL_CONSTANTS[
                'PI']
            lat_fixed = lat_rot

            # 衝突地点の表面温度を計算
            temp_at_impact = calculate_surface_temperature_leblanc(lon_fixed, lat_fixed, AU, subsolar_lon_rad_fixed)

            # (ii-a) 吸着
            if np.random.random() < calculate_sticking_probability(temp_at_impact):
                # ★変更点: 'stuck' の場合、衝突位置(回転座標系)と重みを返す
                return {'status': 'stuck', 'pos_at_impact': pos_at_start_of_step, 'weight': weight}

            # (ii-b) 熱的に再放出（バウンド）
            else:
                E_in = 0.5 * PHYSICAL_CONSTANTS['MASS_NA'] * np.sum(vel ** 2)
                E_T = PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_at_impact
                E_out = settings['BETA'] * E_T + (1.0 - settings['BETA']) * E_in
                v_out_speed = np.sqrt(2 * E_out / PHYSICAL_CONSTANTS['MASS_NA']) if E_out > 0 else 0.0

                impact_normal = pos_at_start_of_step / np.linalg.norm(pos_at_start_of_step)
                rebound_direction = transform_local_to_world(sample_lambertian_direction_local(), impact_normal)
                vel = v_out_speed * rebound_direction
                pos = (RM + 1.0) * impact_normal  # 粒子を表面のすぐ外側に戻す

                continue

    # ---------------------------------
    # 3. 生き残り
    # ---------------------------------
    return {'status': 'alive', 'final_state': {'pos': pos, 'vel': vel, 'weight': weight}}


# ==============================================================================
# メイン制御関数
# ==============================================================================

def main_snapshot_simulation():
    """
    シミュレーション全体を制御するメイン関数。
    （動的表面密度モデル）
    """
    start_time = time.time()

    # --- 1. シミュレーション設定 ---
    OUTPUT_DIRECTORY = r"./SimulationResult_DynamicSurface_202510"
    # 表面グリッド（惑星固定座標系）
    N_LON_FIXED, N_LAT = 72, 36  # 経度 72 (5度毎), 緯度 36 (5度毎)

    # Leblanc 2003 の初期値 (7.5e14 atoms/cm^2)
    INITIAL_SURFACE_DENSITY_PER_M2 = 7.5e14 * (100.0 ** 2)  # [atoms/m^2]

    SPIN_UP_YEARS = 3.0  # 動的平衡に達するため、長めのスピンアップ
    TIME_STEP_SEC = 1000
    TOTAL_SIM_YEARS = 1.0
    TARGET_TAA_DEGREES = np.arange(0, 360, 1)  # ★元のコードでは 5度毎 とありましたが、1度毎に戻しています
    ATOMS_PER_SUPERPARTICLE = 1e25  # 表面密度が濃いため、重みを増やす

    # --- 粒子生成モデル (Leblanc 2003 準拠) ---
    # 1AUでの紫外線光子フラックス [photons/m^2/s] (1.5e14 /cm^2)
    F_UV_1AU_PER_M2 = 1.5e14 * (100.0 ** 2)
    # 光刺激脱離の断面積 [m^2]
    Q_PSD_M2 = 1.4e-21 / (100.0 ** 2)
    # 速度分布用の温度 [K]
    TEMP_PSD = 1500.0  #
    TEMP_MMV = 3000.0  #
    # (TDの温度は局所表面温度 T_s を使用)

    # --- 出力グリッドの設定 ---
    GRID_RESOLUTION = 101
    GRID_MAX_RM = 5.0

    # --- 物理モデルのフラグ ---
    USE_SOLAR_GRAVITY = True
    USE_CORIOLIS_FORCES = True

    settings = {
        'BETA': 0.5,
        'T1AU': 168918.0,
        'DT': 1000.0,
        'GRID_RADIUS_RM': GRID_MAX_RM + 1.0,
        'USE_SOLAR_GRAVITY': USE_SOLAR_GRAVITY,
        'USE_CORIOLIS_FORCES': USE_CORIOLIS_FORCES
    }

    # 出力ディレクトリの準備
    run_name = f"DynamicGrid{N_LON_FIXED}x{N_LAT}_SP{ATOMS_PER_SUPERPARTICLE:.0e}_PSD_TD_MMV"
    target_output_dir = os.path.join(OUTPUT_DIRECTORY, run_name)
    os.makedirs(target_output_dir, exist_ok=True)
    print(f"結果は '{target_output_dir}' に保存されます。")
    print(f"--- 物理モデル設定 ---")
    print(f"Dynamic Surface Grid: {N_LON_FIXED}x{N_LAT}")
    print(f"Processes: PSD, Thermal Desorption, Micrometeoroid Vaporization")
    print(f"Solar Gravity: {USE_SOLAR_GRAVITY}")
    print(f"Coriolis/Centrifugal: {USE_CORIOLIS_FORCES}")
    print(f"----------------------")

    # --- 2. シミュレーションの初期化 ---
    MERCURY_YEAR_SEC = 87.97 * 24 * 3600

    # 表面グリッドの定義（惑星固定座標系）
    lon_edges_fixed = np.linspace(-np.pi, np.pi, N_LON_FIXED + 1)
    lat_edges_fixed = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    dlon_fixed = lon_edges_fixed[1] - lon_edges_fixed[0]
    # 各緯度帯のセル面積 [m^2] (N_LAT個の要素を持つ配列)
    cell_areas_m2 = (PHYSICAL_CONSTANTS['RM'] ** 2) * dlon_fixed * \
                    (np.sin(lat_edges_fixed[1:]) - np.sin(lat_edges_fixed[:-1]))

    # 動的表面密度グリッドの初期化
    surface_density_grid = np.full((N_LON_FIXED, N_LAT), INITIAL_SURFACE_DENSITY_PER_M2, dtype=np.float64)

    # --- 3. 外部ファイルの読み込み ---
    try:
        spec_data_np = np.loadtxt('SolarSpectrum_Na0.txt', usecols=(0, 3))
        orbit_file_name = 'orbit2025_v5.txt'
        orbit_data = np.loadtxt(orbit_file_name)
    except FileNotFoundError as e:
        print(f"エラー: データファイル '{e.filename}' が見つかりません。")
        sys.exit()

    if orbit_data.shape[1] < 5:
        print(f"エラー: '{orbit_file_name}' の列が不足しています。")
        sys.exit()

    # --- ★★★ 修正箇所 1: RUN開始時刻 (TAA=0) の調整 ★★★ ---
    taa_col = orbit_data[:, 0]
    time_col = orbit_data[:, 2]
    idx_perihelion = np.argmin(np.abs(taa_col))
    t_start_run = time_col[idx_perihelion]

    t_end_run = t_start_run + (TOTAL_SIM_YEARS * MERCURY_YEAR_SEC)
    t_start_spinup = t_start_run - (SPIN_UP_YEARS * MERCURY_YEAR_SEC)

    # メインループで回す時間ステップの配列
    time_steps = np.arange(t_start_spinup, t_end_run, TIME_STEP_SEC)

    print(f"--- 時間設定 (動的モデル) ---")
    print(f"軌道ファイル上のTAA=0 (近日点) 時刻: {t_start_run:.1f} s")
    print(f"スピンアップ開始時刻: {t_start_spinup:.1f} s ({-SPIN_UP_YEARS} 年前)")
    print(f"RUN開始時刻 (TAA=0): {t_start_run:.1f} s")
    print(f"RUN終了時刻: {t_end_run:.1f} s (+{TOTAL_SIM_YEARS} 年後)")
    print(f"------------------")
    # --- ★★★ 修正 1 ここまで ★★★ ---

    # (スペクトルデータの前処理)
    wl, gamma = spec_data_np[:, 0], spec_data_np[:, 1]
    if not np.all(np.diff(wl) > 0):
        sort_indices = np.argsort(wl)
        wl, gamma = wl[sort_indices], gamma[sort_indices]
    sigma_const = PHYSICAL_CONSTANTS['E_CHARGE'] ** 2 / (
            4 * PHYSICAL_CONSTANTS['ME'] * PHYSICAL_CONSTANTS['C'] * PHYSICAL_CONSTANTS['EPSILON_0'])
    spec_data_dict = {'wl': wl, 'gamma': gamma,
                      'sigma0_perdnu2': sigma_const * 0.641,
                      'sigma0_perdnu1': sigma_const * 0.320,
                      'JL': 5.18e14}

    # --- 4. メインループ (時間発展) ---
    active_particles = []
    previous_taa = -1
    target_taa_idx = 0

    with tqdm(total=len(time_steps), desc="Time Evolution") as pbar:
        # t_sec は t_start_spinup から t_end_run まで進む
        for t_sec in time_steps:
            # --- 4a. 現在時刻の軌道パラメータを取得 ---
            TAA, AU, V_radial_ms, V_tangential_ms, subsolar_lon_rad_fixed = get_orbital_params(
                t_sec, orbit_data, MERCURY_YEAR_SEC
            )

            # ★★★ 修正箇所 2: pbarの表示を t_start_run 基準に変更 ★★★
            run_phase = "Spin-up" if t_sec < t_start_run else "Run"
            pbar.set_description(f"[{run_phase}] TAA={TAA:.1f} | N_particles={len(active_particles)}")

            # --- 4b. 表面から新しい粒子を生成 (PSD, TD, MMV) ---
            # (変更なし)
            newly_launched_particles = []
            atoms_lost_grid = np.zeros_like(surface_density_grid)
            F_UV_current_per_m2 = F_UV_1AU_PER_M2 / (AU ** 2)
            flux_mmv_per_m2_s = calculate_mmv_flux(AU)

            for i_lon in range(N_LON_FIXED):
                for i_lat in range(N_LAT):
                    lon_fixed_rad = (lon_edges_fixed[i_lon] + lon_edges_fixed[i_lon + 1]) / 2
                    lat_rad = (lat_edges_fixed[i_lat] + lat_edges_fixed[i_lat + 1]) / 2
                    area_m2 = cell_areas_m2[i_lat]
                    current_density_per_m2 = surface_density_grid[i_lon, i_lat]
                    cos_Z = np.cos(lat_rad) * np.cos(lon_fixed_rad - subsolar_lon_rad_fixed)
                    temp_k = calculate_surface_temperature_leblanc(lon_fixed_rad, lat_rad, AU, subsolar_lon_rad_fixed)

                    n_atoms_psd, n_atoms_td, n_atoms_mmv = 0.0, 0.0, 0.0
                    n_atoms_mmv = flux_mmv_per_m2_s * area_m2 * TIME_STEP_SEC

                    if current_density_per_m2 > 0:
                        if cos_Z > 0:
                            rate_psd_per_s = F_UV_current_per_m2 * Q_PSD_M2 * cos_Z
                            n_atoms_psd = rate_psd_per_s * current_density_per_m2 * area_m2 * TIME_STEP_SEC
                        rate_td_per_s = calculate_thermal_desorption_rate(temp_k)
                        n_atoms_td = rate_td_per_s * current_density_per_m2 * area_m2 * TIME_STEP_SEC

                    procs = {
                        'PSD': {'n_atoms': n_atoms_psd, 'temp': TEMP_PSD},
                        'TD': {'n_atoms': n_atoms_td, 'temp': temp_k},
                        'MMV': {'n_atoms': n_atoms_mmv, 'temp': TEMP_MMV}
                    }

                    for proc_name, p in procs.items():
                        if p['n_atoms'] <= 0: continue
                        num_sps_float = p['n_atoms'] / ATOMS_PER_SUPERPARTICLE
                        num_sps_int = int(num_sps_float)
                        if np.random.random() < (num_sps_float - num_sps_int):
                            num_sps_int += 1
                        if num_sps_int == 0: continue

                        if proc_name != 'MMV':
                            atoms_lost_grid[i_lon, i_lat] += num_sps_int * ATOMS_PER_SUPERPARTICLE

                        for _ in range(num_sps_int):
                            lon_rot_rad = lon_fixed_rad - subsolar_lon_rad_fixed
                            lat_rot_rad = lat_rad
                            initial_pos_rot = lonlat_to_xyz(lon_rot_rad, lat_rot_rad, PHYSICAL_CONSTANTS['RM'])
                            surface_normal_rot = initial_pos_rot / PHYSICAL_CONSTANTS['RM']
                            speed = sample_speed_from_flux_distribution(PHYSICAL_CONSTANTS['MASS_NA'], p['temp'])
                            initial_vel_rot = speed * transform_local_to_world(sample_lambertian_direction_local(),
                                                                               surface_normal_rot)
                            newly_launched_particles.append({
                                'pos': initial_pos_rot,
                                'vel': initial_vel_rot,
                                'weight': ATOMS_PER_SUPERPARTICLE
                            })

            surface_density_grid -= atoms_lost_grid / cell_areas_m2
            np.clip(surface_density_grid, 0, None, out=surface_density_grid)
            active_particles.extend(newly_launched_particles)

            # --- 4c. 全ての粒子を1ステップ進め、結果を集計 ---
            # (変更なし)
            tasks = [{'settings': settings, 'spec': spec_data_dict, 'particle_state': p,
                      'orbit': (TAA, AU, V_radial_ms, V_tangential_ms, subsolar_lon_rad_fixed),
                      'duration': TIME_STEP_SEC} for p in
                     active_particles]

            next_active_particles = []
            atoms_gained_grid = np.zeros_like(surface_density_grid)

            if tasks:
                with Pool(processes=max(1, cpu_count() - 1)) as pool:
                    results = list(pool.imap(simulate_particle_for_one_step, tasks, chunksize=100))

                for res in results:
                    if res['status'] == 'alive':
                        next_active_particles.append(res['final_state'])
                    elif res['status'] == 'stuck':
                        pos_rot = res['pos_at_impact']
                        weight = res['weight']
                        lon_rot = np.arctan2(pos_rot[1], pos_rot[0])
                        lat_rot = np.arcsin(np.clip(pos_rot[2] / np.linalg.norm(pos_rot), -1.0, 1.0))
                        lon_fixed = (lon_rot + subsolar_lon_rad_fixed)
                        lon_fixed = (lon_fixed + PHYSICAL_CONSTANTS['PI']) % (2 * PHYSICAL_CONSTANTS['PI']) - \
                                    PHYSICAL_CONSTANTS['PI']
                        lat_fixed = lat_rot
                        i_lon = np.searchsorted(lon_edges_fixed, lon_fixed) - 1
                        i_lat = np.searchsorted(lat_edges_fixed, lat_fixed) - 1
                        if 0 <= i_lon < N_LON_FIXED and 0 <= i_lat < N_LAT:
                            atoms_gained_grid[i_lon, i_lat] += weight

            active_particles = next_active_particles
            surface_density_grid += atoms_gained_grid / cell_areas_m2

            # --- ★★★ 修正箇所 3: スナップショット保存判定 (バグ修正) ★★★ ---

            save_this_step = False  # 1. 毎ステップ必ず変数を初期化する

            if TAA < previous_taa:
                target_taa_idx = 0

            if target_taa_idx < len(TARGET_TAA_DEGREES):
                current_target_taa = TARGET_TAA_DEGREES[target_taa_idx]

                # 2. TAA=0 をまたぐ判定ロジックの修正
                is_crossing_zero = (current_target_taa == 0) and \
                                   ((TAA < previous_taa) or (TAA >= 0 and previous_taa < 0))

                # 3. 通常のターゲット(>0)の判定
                is_crossing_normal = (previous_taa < current_target_taa <= TAA)

                if is_crossing_normal or is_crossing_zero:
                    save_this_step = True
                    target_taa_idx += 1

            # --- 4e. 立方体グリッドに集計して保存 ---

            # ★★★ 修正箇所 4: 保存判定を t_sec >= t_start_run に変更 ★★★
            if save_this_step and t_sec >= t_start_run:
                pbar.write(f"\n>>> [Run] Saving grid snapshot at TAA={TAA:.1f} ({len(active_particles)} particles) <<<")

                # 3D空間グリッド
                density_grid = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION, GRID_RESOLUTION), dtype=np.float32)
                grid_min = -GRID_MAX_RM * PHYSICAL_CONSTANTS['RM']
                grid_max = GRID_MAX_RM * PHYSICAL_CONSTANTS['RM']
                cell_size = (grid_max - grid_min) / GRID_RESOLUTION
                cell_volume_m3 = cell_size ** 3

                for p in active_particles:
                    pos = p['pos']
                    ix = int((pos[0] - grid_min) / cell_size)
                    iy = int((pos[1] - grid_min) / cell_size)
                    iz = int((pos[2] - grid_min) / cell_size)
                    if 0 <= ix < GRID_RESOLUTION and 0 <= iy < GRID_RESOLUTION and 0 <= iz < GRID_RESOLUTION:
                        density_grid[ix, iy, iz] += p['weight']
                density_grid /= cell_volume_m3

                # ★★★ 修正箇所 5: 保存ファイル名の時刻を t_start_run からの相対時刻に変更 ★★★
                relative_time_sec = t_sec - t_start_run
                save_time_h = relative_time_sec / 3600

                # ファイル保存 (3D空間密度)
                filename = f"density_grid_t{int(save_time_h):05d}_taa{int(round(TAA)):03d}.npy"
                np.save(os.path.join(target_output_dir, filename), density_grid)

                # ファイル保存 (2D表面密度)
                filename_surf = f"surface_density_t{int(save_time_h):05d}_taa{int(round(TAA)):03d}.npy"
                np.save(os.path.join(target_output_dir, filename_surf), surface_density_grid)

            previous_taa = TAA
            pbar.update(1)

    end_time = time.time()
    print(f"\n★★★ シミュレーションが完了しました ★★★")
    print(f"総計算時間: {(end_time - start_time) / 3600:.2f} 時間")

    if __name__ == '__main__':
        # スクリプト実行時に、まず必須ファイルの存在を確認する
        print("必須ファイルを確認しています...")
        for f in ['orbit2025_v5.txt', 'SolarSpectrum_Na0.txt']:
            if not os.path.exists(f):
                print(f"エラー: 必須ファイル '{f}' が見つかりません。スクリプトと同じディレクトリに配置してください。")
                if f == 'orbit2025_v5.txt':
                    print("（orbit2025_v5.txt がない場合は、軌道生成スクリプトを先に実行してください）")
                sys.exit()  # ファイルがない場合は終了

        print("ファイルOK。シミュレーションを開始します。")

        # この呼び出しでシミュレーションが開始される
        main_snapshot_simulation()