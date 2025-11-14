# -*- coding: utf-8 -*-
"""
==============================================================================
概要
==============================================================================
このスクリプトは、水星のナトリウム大気のふるまいを、時間発展を考慮して
3次元モンテカルロ法に基づいてシミュレーションを行うことを目的にしたコードです。

Leblanc (2003) の論文に基づき、以下の特徴を持ちます。

1.  動的表面密度:
    - 水星表面を惑星固定座標系のグリッドで管理します。
    - 各セルのナトリウム密度は、放出によって減少し、吸着によって増加します。

2.  複数の生成過程:
    - 光刺激脱離 (PSD): 表面密度と太陽光に比例。
    - 熱脱離 (TD): 表面密度と表面温度に比例。
    - 微小隕石蒸発 (MIV/MMV): 表面密度に依存しない補給源。
    - 太陽風イオンスパッタリング (SWS): 表面密度に比例し、太陽風フラックスに依存。

==============================================================================
座標系
==============================================================================
このシミュレーションは、2つの座標系を併用します。

1.  惑星固定座標系（水星自転座標系）:
    - 表面グリッド(surface_density_grid)、表面温度計算に使用。
    - 経度 0 は水星の特定の地点に固定。

2.  水星中心・太陽固定回転座標系 (MSO: Mercury-Sun-Orbit):
    - 粒子の軌道追跡、空間密度グリッドの集計に使用。
    - +X方向: 常に太陽の方向。
    - -Y方向: 水星の公転の進行方向（軌道速度ベクトル方向）。
    - メインループで計算される `subsolar_lon_rad`（惑星固定座標系での
      太陽直下点経度）を用いて、2つの座標系をマッピングします。

==============================================================================
主な物理モデル (Leblanc 2003 準拠)
==============================================================================
1.  粒子生成:
    - PSD, TD, SWS: 表面のナトリウム貯蔵庫を消費します。
    - MMV: 外部からの補給源として機能します 。

2.  初期速度:
    - PSD (T=1500K), TD (T=表面温度), MMV (T=3000K)
      それぞれの温度におけるマクスウェル「フラックス」分布に従う速度。
    - SWS (U=0.27eV): Thompson-Sigmund 分布に従うエネルギー。
    - 放出角度: ランバート（余弦則）分布。

3.  軌道計算 (4次ルンゲ＝クッタ法):
    - 水星重力、太陽放射圧(SRP)、太陽重力、
      見かけの力（コリオリ力・遠心力）を考慮します。

4.  消滅過程:
    - 光電離: 太陽光による電離。
    - 表面衝突: 表面に衝突した際、温度と確率に基づき「吸着」または「反射」します。

==============================================================================
必要な外部ファイル
==============================================================================
1.  orbit2025_v5.txt:
    水星の軌道パラメータ（TAA, 日心距離, 視線速度など）が
    時刻歴で記録されたファイル。
2.  SolarSpectrum_Na0.txt:
    太陽放射圧計算用の波長ごとのgamma値（太陽スペクトルの情報）が
    記録されたファイル。
"""

import numpy as np
import sys
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
from numba import njit

# ==============================================================================
# 物理定数 (SI単位系)
# ==============================================================================
PHYSICAL_CONSTANTS = {
    'PI': np.pi,
    'AU': 1.496e11,  # 天文単位 [m]
    'MASS_NA': 22.98976928 * 1.66054e-27,  # ナトリウム原子質量 [kg]
    'K_BOLTZMANN': 1.380649e-23,  # ボルツマン定数 [J/K]
    'GM_MERCURY': 2.2032e13,  # 水星の万有引力定数と質山の積 [m^3/s^2]
    'RM': 2.440e6,  # 水星半径 [m]
    'C': 299792458.0,  # 光速 [m/s]
    'H': 6.62607015e-34,  # プランク定数 [J·s]
    'E_CHARGE': 1.602176634e-19,  # 電気素量 [C]
    'ME': 9.1093897e-31,  # 電子質量 [kg]
    'EPSILON_0': 8.854187817e-12,  # 真空の誘電率 [F/m]
    'G': 6.6743e-11,  # 万有引力定数 [N m^2/kg^2]
    'MASS_SUN': 1.989e30,  # 太陽質量 [kg]
    'EV_TO_JOULE': 1.602176634e-19,  # eV から Joule への換算係数 [J / eV]
}


# ==============================================================================
# 物理モデルに基づくヘルパー関数群
# ==============================================================================

def calculate_surface_temperature_leblanc(lon_fixed_rad, lat_rad, AU, subsolar_lon_rad):
    """
    惑星固定座標上の指定された地点の表面温度を計算します。(Leblanc 2003モデル)

    Args:
        lon_fixed_rad (float): 惑星固定座標系での経度 [rad]
        lat_rad (float): 惑星固定座標系での緯度 [rad]
        AU (float): 現在の日心距離 [AU]
        subsolar_lon_rad (float): 惑星固定座標系での太陽直下点経度 [rad]

    Returns:
        float: 表面温度 [K]
    """
    T_night = 100.0  # 夜側の最低温度
    # 太陽天頂角の余弦を計算
    cos_theta = np.cos(lat_rad) * np.cos(lon_fixed_rad - subsolar_lon_rad)

    # 夜側の場合
    if cos_theta <= 0:
        return T_night

    # 昼側の場合：日心距離に応じて太陽直下点温度を内挿
    T0_peri = 600.0  # 近日点での太陽直下点温度
    T0_aph = 475.0  # 遠日点での太陽直下点温度
    AU_peri = 0.307
    AU_aph = 0.467
    T0 = np.interp(AU, [AU_peri, AU_aph], [T0_peri, T0_aph])
    T1 = 100.0
    return T0 + T1 * (cos_theta ** 0.25)


def calculate_sticking_probability(surface_temp_K):
    """
    ナトリウム原子が表面に衝突した際の吸着確率（Sticking Probability）を計算します。

    Args:
        surface_temp_K (float): 衝突地点の表面温度 [K]

    Returns:
        float: 吸着確率 (0.0 から 1.0)
    """
    A = 0.0804
    B = 458.0
    porosity = 0.8  # 表面の多孔性
    if surface_temp_K <= 0:
        return 1.0
    p_stick = A * np.exp(B / surface_temp_K)
    # 多孔性を考慮した実効吸着確率
    p_stick_eff = p_stick / (1.0 - (1.0 - p_stick) * porosity)
    return min(p_stick_eff, 1.0)


def calculate_thermal_desorption_rate(surface_temp_K):
    """
    熱脱離 (Thermal Desorption) の発生率（原子1個あたり）を計算します。

    Args:
        surface_temp_K (float): 表面温度 [K]

    Returns:
        float: 熱脱離率 [1/s]
    """
    if surface_temp_K < 350.0:  # 低温では発生しないと仮定
        return 0.0
    VIB_FREQ = 1e13  # 表面原子の振動数 (論文パラメータ)
    BINDING_ENERGY_EV = 1.85  # 結合エネルギー [eV]
    BINDING_ENERGY_J = BINDING_ENERGY_EV * PHYSICAL_CONSTANTS['EV_TO_JOULE']
    k_B = PHYSICAL_CONSTANTS['K_BOLTZMANN']
    exponent = -BINDING_ENERGY_J / (k_B * surface_temp_K)
    if exponent < -700:  # np.expのアンダーフローを防ぐ
        return 0.0
    return VIB_FREQ * np.exp(exponent)


def calculate_mmv_flux(AU):
    """
    微小隕石蒸発 (Micrometeoroid Vaporization) による
    ナトリウム原子の平均放出フラックス（単位面積・単位時間あたり）を計算します。

    Args:
        AU (float): 現在の日心距離 [AU]

    Returns:
        float: MMVによるナトリウム放出フラックス [atoms/m^2/s]
    """
    TOTAL_FLUX_AT_PERI_NA_S = 5e23  # 近日点での惑星全体の総放出レート [atoms/s]
    PERIHELION_AU = 0.307
    MERCURY_SURFACE_AREA_M2 = 4 * PHYSICAL_CONSTANTS['PI'] * (PHYSICAL_CONSTANTS['RM'] ** 2)
    # 近日点での平均フラックス [atoms/m^2/s]
    avg_flux_at_peri = TOTAL_FLUX_AT_PERI_NA_S / MERCURY_SURFACE_AREA_M2
    # 日心距離依存性を考慮するための係数Cを計算 (Flux ∝ AU^-1.9)
    C = avg_flux_at_peri * (PERIHELION_AU ** 1.9)
    # 現在のAUにおけるフラックスを計算
    return C * (AU ** (-1.9))


def sample_speed_from_flux_distribution(mass_kg, temp_k):
    """
    指定された温度の「フラックス」マクスウェル分布に従う速度をサンプリングします。
    (通常の密度分布 v^2*exp(-E/kT) ではなく、フラックス分布 v^3*exp(-E/kT) に従う)

    これは、エネルギー E がガンマ分布 Gamma(shape=2, scale=kT) に従うことと等価です。

    Args:
        mass_kg (float): 粒子の質量 [kg]
        temp_k (float): 温度 [K]

    Returns:
        float: サンプリングされた速度 [m/s]
    """
    kT = PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_k
    # ガンマ分布 (shape=2.0, scale=kT) からエネルギーをサンプリング
    E = np.random.gamma(2.0, kT)
    # E = 0.5 * m * v^2 より速度を計算
    return np.sqrt(2.0 * E / mass_kg)


def sample_thompson_sigmund_energy(U_eV, E_max_eV=5.0):
    """
    Thompson-Sigmund 分布 f(E) ∝ E / (E + U)^3 に従うエネルギーを
    棄却サンプリング法 (Rejection Sampling) を用いて生成します。
    (SWS: 太陽風スパッタリング用)

    Args:
        U_eV (float): 表面束縛エネルギー [eV]
        E_max_eV (float): サンプリングするエネルギーの最大値 [eV] (論文に基づき~5eVで十分)

    Returns:
        float: サンプリングされたエネルギー [eV]
    """
    # 分布の最大値 f_max を計算 (f(E)は E = U/2 で最大値をとる)
    f_max = (U_eV / 2.0) / (U_eV / 2.0 + U_eV) ** 3

    while True:
        # エネルギーを 0 から E_max まで一様に試行
        E_try = np.random.uniform(0, E_max_eV)
        # 試行したエネルギーでの分布の値を計算
        f_E_try = E_try / (E_try + U_eV) ** 3
        # 0 から f_max までの一様乱数を生成
        y_try = np.random.uniform(0, f_max)
        # 棄却判定
        if y_try <= f_E_try:
            return E_try


def sample_lambertian_direction_local():
    """
    ランバート（余弦則）分布に従う放出方向ベクトルを、
    ローカル座標系（Z軸が法線方向）でサンプリングします。

    Returns:
        np.ndarray: ローカル座標系での方向ベクトル [x, y, z] (z >= 0)
    """
    u1, u2 = np.random.random(2)  # 0~1の一様乱数を2つ
    phi = 2 * PHYSICAL_CONSTANTS['PI'] * u1  # 方位角 (0 ~ 2pi)
    cos_theta = np.sqrt(1 - u2)  # 天頂角の余弦 (cosθ ∝ sqrt(1-u2))
    sin_theta = np.sqrt(u2)  # (sinθ ∝ sqrt(u2))
    return np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])


def transform_local_to_world(local_vec, normal_vector):
    """
    ローカル座標系（法線がZ軸）のベクトルを、
    ワールド座標系（指定された法線ベクトルが基準）に変換します。

    Args:
        local_vec (np.ndarray): ローカル座標系でのベクトル [x_local, y_local, z_local]
        normal_vector (np.ndarray): ワールド座標系での法線ベクトル [nx, ny, nz]

    Returns:
        np.ndarray: ワールド座標系でのベクトル [x_world, y_world, z_world]
    """
    # ローカルZ軸 = 法線ベクトル
    local_z_axis = normal_vector / np.linalg.norm(normal_vector)
    # ジンバルロックを避けるため、法線がワールドZ軸と(ほぼ)平行な場合は、
    # ワールドY軸を仮の「上」方向として使用する
    world_up = np.array([0., 0., 1.])
    if np.allclose(local_z_axis, world_up) or np.allclose(local_z_axis, -world_up):
        world_up = np.array([0., 1., 0.])
    # ローカルX軸 (ワールドの上方向とローカルZ軸の外積)
    local_x_axis = np.cross(world_up, local_z_axis)
    local_x_axis /= np.linalg.norm(local_x_axis)
    # ローカルY軸 (ローカルZ軸とローカルX軸の外積)
    local_y_axis = np.cross(local_z_axis, local_x_axis)
    # ローカル基底ベクトルを用いてワールド座標系ベクトルに変換
    return (local_vec[0] * local_x_axis +
            local_vec[1] * local_y_axis +
            local_vec[2] * local_z_axis)


def get_orbital_params(time_sec, orbit_data, mercury_year_sec):
    """
    指定された時刻における水星の軌道パラメータと太陽直下点経度を取得します。

    Args:
        time_sec (float): シミュレーション開始からの経過時間 [s]
        orbit_data (np.ndarray): 軌道データファイル（orbit2025_v5.txt）
        mercury_year_sec (float): 水星の1公転周期 [s]

    Returns:
        tuple: (taa, au, v_radial, v_tangential, subsolar_lon_rad)
    """
    # 水星の自転周期
    ROTATION_PERIOD_SEC = 58.646 * 24 * 3600
    # 軌道データは1公転分なので、時刻を公転周期で割った余りを計算
    current_time_in_orbit = time_sec % mercury_year_sec
    time_col = orbit_data[:, 2]
    # 軌道データから、現在の時刻に対応するパラメータを線形内挿で取得
    taa = np.interp(current_time_in_orbit, time_col, orbit_data[:, 0])
    au = np.interp(current_time_in_orbit, time_col, orbit_data[:, 1])
    v_radial = np.interp(current_time_in_orbit, time_col, orbit_data[:, 3])
    v_tangential = np.interp(current_time_in_orbit, time_col, orbit_data[:, 4])
    # 惑星固定座標系における太陽直下点経度を計算
    # (水星の自転に基づいて計算。t=0での初期位相は考慮が必要な場合がある)
    subsolar_lon_rad = (2 * np.pi * time_sec / ROTATION_PERIOD_SEC) % (2 * np.pi)
    return taa, au, v_radial, v_tangential, subsolar_lon_rad


def lonlat_to_xyz(lon_rad, lat_rad, radius):
    """
    球面座標（経度、緯度、半径）をデカルト座標（X, Y, Z）に変換します。

    Args:
        lon_rad (float): 経度 [rad]
        lat_rad (float): 緯度 [rad]
        radius (float): 半径 [m]

    Returns:
        np.ndarray: デカルト座標 [x, y, z]
    """
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return np.array([x, y, z])


def xyz_to_lonlat_idx(pos_vec, lon_edges_fixed, lat_edges_fixed, N_LON_FIXED, N_LAT):
    """
    デカルト座標（太陽固定回転座標系）を、
    惑星固定座標系のグリッドインデックス（経度、緯度）に変換します。
    ※この関数は現在、メインの処理では使用されていません。
      (メインループでは回転座標系での位置 -> 固定座標系経緯度 -> インデックス の順で計算)

    Args:
        pos_vec (np.ndarray): 位置ベクトル [x, y, z]
        lon_edges_fixed (np.ndarray): 固定座標系の経度グリッド境界
        lat_edges_fixed (np.ndarray): 固定座標系の緯度グリッド境界
        N_LON_FIXED (int): 経度グリッド数
        N_LAT (int): 緯度グリッド数

    Returns:
        tuple: (i_lon, i_lat) グリッドインデックス。範囲外の場合は (-1, -1)
    """
    r = np.linalg.norm(pos_vec)
    if r == 0: return -1, -1
    # 回転座標系での経緯度
    lon_rot = np.arctan2(pos_vec[1], pos_vec[0])
    lat_rot = np.arcsin(np.clip(pos_vec[2] / r, -1.0, 1.0))
    # グリッドインデックスを検索
    i_lon = np.searchsorted(lon_edges_fixed, lon_rot) - 1
    i_lat = np.searchsorted(lat_edges_fixed, lat_rot) - 1
    if 0 <= i_lon < N_LON_FIXED and 0 <= i_lat < N_LAT:
        return i_lon, i_lat
    else:
        return -1, -1


# ★ Numbaラッパー関数 (dictをNumbaに渡すため)
def _calculate_acceleration(pos, vel, V_radial_ms, V_tangential_ms, AU, spec_data, settings):
    """
    Pythonラッパー関数。
    dictを「荷ほどき」して、Numba化された _njit 関数に渡す。
    """
    # 1. spec_data (dict) からNumbaが分かる「配列」を取り出す
    wl, gamma, sigma0_perdnu2, sigma0_perdnu1, JL = spec_data['wl'], spec_data['gamma'], \
        spec_data['sigma0_perdnu2'], spec_data['sigma0_perdnu1'], \
        spec_data['JL']

    # 2. settings (dict) からNumbaが分かる「bool」を取り出す
    USE_SOLAR_GRAVITY = settings['USE_SOLAR_GRAVITY']
    USE_CORIOLIS_FORCES = settings['USE_CORIOLIS_FORCES']

    # 3. PHYSICAL_CONSTANTS (dict) からNumbaが分かる「float」を取り出す
    CONST_AU_m = PHYSICAL_CONSTANTS['AU']
    CONST_MASS_NA_kg = PHYSICAL_CONSTANTS['MASS_NA']
    CONST_C_ms = PHYSICAL_CONSTANTS['C']
    CONST_H_Js = PHYSICAL_CONSTANTS['H']
    CONST_GM_MERCURY_m3s2 = PHYSICAL_CONSTANTS['GM_MERCURY']
    CONST_RM_m = PHYSICAL_CONSTANTS['RM']
    CONST_G_mks = PHYSICAL_CONSTANTS['G']
    CONST_MASS_SUN_kg = PHYSICAL_CONSTANTS['MASS_SUN']

    # 4. Numba化された「本体」を呼び出す
    return _calculate_acceleration_njit(pos, vel, V_radial_ms, V_tangential_ms, AU,
                                        wl, gamma, sigma0_perdnu2, sigma0_perdnu1, JL,
                                        USE_SOLAR_GRAVITY, USE_CORIOLIS_FORCES,
                                        CONST_AU_m, CONST_MASS_NA_kg, CONST_C_ms, CONST_H_Js,
                                        CONST_GM_MERCURY_m3s2, CONST_RM_m, CONST_G_mks, CONST_MASS_SUN_kg)


@njit(cache=True)  # ★ Numba化する
def _calculate_acceleration_njit(pos, vel, V_radial_ms, V_tangential_ms, AU,
                                 # spec_data の中身 (配列とfloat)
                                 wl, gamma, sigma0_perdnu2, sigma0_perdnu1, JL,
                                 # settings の中身 (bool)
                                 USE_SOLAR_GRAVITY, USE_CORIOLIS_FORCES,
                                 # CONSTANTS の中身 (float)
                                 CONST_AU_m, CONST_MASS_NA_kg, CONST_C_ms, CONST_H_Js,
                                 CONST_GM_MERCURY_m3s2, CONST_RM_m, CONST_G_mks, CONST_MASS_SUN_kg
                                 ):
    """
    Numbaでコンパイルされる加速度計算の「本体」（dict非依存）
    """
    x, y, z = pos
    r0 = AU * CONST_AU_m  # ★定数に置き換え

    # --- 1. 太陽放射圧 (SRP) ---
    velocity_for_doppler = vel[0] + V_radial_ms
    w_na_d2 = 589.1582e-9 * (1.0 + velocity_for_doppler / CONST_C_ms)  # ★
    w_na_d1 = 589.7558e-9 * (1.0 + velocity_for_doppler / CONST_C_ms)  # ★

    b = 0.0
    # ★ spec_data (dict) ではなく、渡された配列(wl, gamma) を使う
    if (wl[0] * 1e-9 <= w_na_d2 < wl[-1] * 1e-9) and \
            (wl[0] * 1e-9 <= w_na_d1 < wl[-1] * 1e-9):
        gamma2 = np.interp(w_na_d2 * 1e9, wl, gamma)
        gamma1 = np.interp(w_na_d1 * 1e9, wl, gamma)

        F_lambda_1AU_m = JL * 1e4 * 1e9
        F_lambda_d1_at_Mercury = F_lambda_1AU_m / (AU ** 2) * gamma1
        F_nu_d1 = F_lambda_d1_at_Mercury * (w_na_d1 ** 2) / CONST_C_ms  # ★
        J1 = sigma0_perdnu1 * F_nu_d1
        F_lambda_d2_at_Mercury = F_lambda_1AU_m / (AU ** 2) * gamma2
        F_nu_d2 = F_lambda_d2_at_Mercury * (w_na_d2 ** 2) / CONST_C_ms  # ★
        J2 = sigma0_perdnu2 * F_nu_d2

        b = 1 / CONST_MASS_NA_kg * (  # ★
                (CONST_H_Js / w_na_d1) * J1 + (CONST_H_Js / w_na_d2) * J2)

    if x < 0 and np.sqrt(y ** 2 + z ** 2) < CONST_RM_m:  # ★
        b = 0.0
    accel_srp = np.array([-b, 0.0, 0.0])

    # --- 2. 水星重力 ---
    r_sq = np.sum(pos ** 2)
    accel_g = -CONST_GM_MERCURY_m3s2 * pos / (r_sq ** 1.5) if r_sq > 0 else np.array([0., 0., 0.])  # ★

    # --- 3. 太陽重力 (オプション) ---
    accel_sun = np.array([0.0, 0.0, 0.0])
    if USE_SOLAR_GRAVITY:  # ★
        G = CONST_G_mks  # ★
        M_SUN = CONST_MASS_SUN_kg  # ★
        r_ps_vec = np.array([r0 - pos[0], -pos[1], -pos[2]])
        r_ps_mag_sq = np.sum(r_ps_vec ** 2)
        if r_ps_mag_sq > 0:
            accel_sun = (G * M_SUN * r_ps_vec) / (r_ps_mag_sq ** 1.5)

    # --- 4. 見かけの力 (オプション) ---
    accel_coriolis = np.array([0.0, 0.0, 0.0])
    accel_centrifugal = np.array([0.0, 0.0, 0.0])
    if USE_CORIOLIS_FORCES:  # ★
        if r0 > 0:
            omega_val = V_tangential_ms / r0
            omega_sq = omega_val ** 2
            accel_centrifugal = np.array([
                omega_val ** 2 * (pos[0] - r0),
                omega_sq * pos[1],
                0.0])
            two_omega = 2 * omega_val
            accel_coriolis = np.array([two_omega * vel[1], -two_omega * vel[0], 0.0])

    return accel_srp + accel_g + accel_sun + accel_centrifugal + accel_coriolis


# ★ @njit を削除！ (dict を扱うため、Python関数として残す)
def simulate_particle_for_one_step(args):
    """
    1個のスーパーパーティクルを、指定された時間 (duration) だけ進める関数。
    (マルチプロセス処理用)

    Args:
        args (dict):
            'settings' (dict): シミュレーション設定
            'spec' (dict): スペクトルデータ
            'orbit' (tuple): 軌道パラメータ (TAA, AU, ...)
            'duration' (float): この関数で進める時間 [s] (★メインループのdt)
            'particle_state' (dict): 粒子の現在状態 {'pos', 'vel', 'weight'}

    Returns:
        dict:
            'status' (str): 'alive', 'ionized', 'escaped', 'stuck'
            'final_state' (dict or None): 'alive' の場合、次の状態
            'pos_at_impact' (np.ndarray or None): 'stuck' の場合、衝突位置
            'weight' (float or None): 'stuck' の場合、粒子の重み
    """
    settings, spec_data = args['settings'], args['spec']
    TAA, AU, V_radial_ms, V_tangential_ms, subsolar_lon_rad_fixed = args['orbit']

    # ★ DT = duration (メインループのタイムステップ) とする
    # duration, DT = args['duration'], settings['DT'] # ★ 変更前
    duration = args['duration']
    DT = duration  # ★ DT を duration と同一に設定

    RM = PHYSICAL_CONSTANTS['RM']
    R_MAX = RM * settings['GRID_RADIUS_RM']  # シミュレーション領域の最大半径
    pos, vel, weight = args['particle_state']['pos'], args['particle_state']['vel'], args['particle_state']['weight']

    # 光電離の寿命 (1AUでの値 * AU^2)
    tau_ionization = settings['T1AU'] * AU ** 2

    # ★ num_steps は 1 になる
    num_steps = int(duration / DT)  # 積分ステップ数 (int(1.0) = 1)
    if num_steps != 1:
        # 念のための安全確認 (ほぼ 1 になるはず)
        num_steps = 1

    pos_at_start_of_step = pos.copy()

    # 4次ルンゲ＝クッタ法 (RK4) で軌道を積分
    # ★ このループは1回だけ実行される
    for _ in range(num_steps):
        pos_at_start_of_step = pos.copy()

        # --- 光電離判定 ---
        # ★ DT (50秒など) 全体での電離確率を計算
        if pos[0] > 0:  # 昼側 (X > 0) にいる場合のみ
            if np.random.random() < (1.0 - np.exp(-DT / tau_ionization)):
                return {'status': 'ionized', 'final_state': None}

        # --- RK4 積分 ---
        # ★ DT が 50s など、非常に大きな値になる
        # ★ ここで呼ぶ _calculate_acceleration はラッパー版
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

        r_current = np.linalg.norm(pos)

        # --- 領域外（脱出）判定 ---
        if r_current > R_MAX:
            return {'status': 'escaped', 'final_state': None}

        # --- 表面衝突判定 ---
        if r_current <= RM:
            # 衝突位置（1ステップ前の位置）
            # (衝突判定を厳密にするには内挿が必要だが、ここでは1ステップ前の位置で代用)
            lon_rot = np.arctan2(pos_at_start_of_step[1], pos_at_start_of_step[0])
            lat_rot = np.arcsin(np.clip(pos_at_start_of_step[2] / np.linalg.norm(pos_at_start_of_step), -1.0, 1.0))

            # 衝突地点の惑星固定座標系での経緯度を計算
            lon_fixed = (lon_rot + subsolar_lon_rad_fixed)
            lon_fixed = (lon_fixed + PHYSICAL_CONSTANTS['PI']) % (2 * PHYSICAL_CONSTANTS['PI']) - PHYSICAL_CONSTANTS[
                'PI']
            lat_fixed = lat_rot

            # 衝突地点の表面温度を計算
            temp_at_impact = calculate_surface_temperature_leblanc(lon_fixed, lat_fixed, AU, subsolar_lon_rad_fixed)

            # 吸着確率に基づき、吸着 (stuck) か反射 (rebound) かを決定
            if np.random.random() < calculate_sticking_probability(temp_at_impact):
                # 吸着
                return {'status': 'stuck', 'pos_at_impact': pos_at_start_of_step, 'weight': weight}
            else:
                # 反射 (熱的な accomodation を考慮)
                E_in = 0.5 * PHYSICAL_CONSTANTS['MASS_NA'] * np.sum(vel ** 2)  # 入射エネルギー
                E_T = PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_at_impact  # 表面の熱エネルギー
                # 反射エネルギー (BETA: accomodation係数)
                E_out = settings['BETA'] * E_T + (1.0 - settings['BETA']) * E_in
                v_out_speed = np.sqrt(2 * E_out / PHYSICAL_CONSTANTS['MASS_NA']) if E_out > 0 else 0.0

                # 反射方向 (ランバート分布)
                impact_normal = pos_at_start_of_step / np.linalg.norm(pos_at_start_of_step)
                rebound_direction = transform_local_to_world(sample_lambertian_direction_local(), impact_normal)
                vel = v_out_speed * rebound_direction
                # 粒子を表面のわずかに上 (RM + 1.0m) に移動させて計算を続行
                pos = (RM + 1.0) * impact_normal
                continue  # 次のRK4ステップへ

    # duration の間、生き残った場合
    return {'status': 'alive', 'final_state': {'pos': pos, 'vel': vel, 'weight': weight}}


# ==============================================================================
# メイン制御関数
# ==============================================================================

def main_snapshot_simulation():
    """
    シミュレーション全体を制御するメイン関数。
    （動的表面密度モデル + 全放出プロセス）
    """
    start_time = time.time()

    # --- 1. シミュレーション設定 ---
    OUTPUT_DIRECTORY = r"./SimulationResult_202510"  # 出力先
    N_LON_FIXED, N_LAT = 72, 36  # 表面グリッドの解像度 (経度72, 緯度36)
    # 初期表面密度 [atoms/m^2]
    # (例: 7.5e14 Na/cm^2 の場合 -> 7.5e14 * (100 cm/m)^2 = 7.5e18 atoms/m^2)
    # (Leblanc 2003, ~4e12 Na/cm2, 0.0053 reservoir)
    INITIAL_SURFACE_DENSITY_PER_M2 = 7.5e14 * (100.0 ** 2) * 0.0053

    # --- ★動的タイムステップ設定 ---
    HOT_TEMP_THRESHOLD_K = 550.0  # [K] この温度以上を「高温時」とする
    HOT_TEMP_TIME_STEP_SEC = 500.0  # [s] 高温時の固定タイムステップ
    DEFAULT_MAX_TIME_STEP_SEC = 2000.0  # [s] 低温時の「最大」タイムステップ
    MIN_TIME_STEP_SEC = 0.1  # [s] 全体の「最小」タイムステップ
    SAFETY_FACTOR = 0.9  # タイムスケールに対する安全係数 (例: 10%)
    # [atoms/m^2] 枯渇したセルをタイムステップ計算から除外する閾値
    MIN_DENSITY_FOR_TIMESTEP = 1e10

    # --- ★低速粒子の追跡省略(バイパス)設定 ---
    LOW_SPEED_THRESHOLD_M_S = 900.0  # [m/s] この速度未満の粒子は追跡しない

    # シミュレーション時間設定
    SPIN_UP_YEARS = 1.0  # 表面密度を平衡状態にするためのスピンアップ期間 (水星年)
    # TIME_STEP_SEC = 1.0  # ★固定ステップは使用しない
    TOTAL_SIM_YEARS = 1.0  # 記録対象とするシミュレーション期間 (水星年)
    TARGET_TAA_DEGREES = np.arange(0, 360, 1)  # スナップショットを保存するTAA [度]

    # 可変重み法 (予算配分方式) の設定
    # 毎ステップで、各プロセスからこの数だけSPを生成することを目標とする
    # 実際の重み (weight) は (総放出原子数 / TARGET_SPS) で決定される
    TARGET_SPS_TD = 1000  # 熱脱離 (TD) の目標SP数/ステップ
    TARGET_SPS_PSD = 1000  # 光刺激脱離 (PSD) の目標SP数/ステップ
    TARGET_SPS_SWS = 1000  # 太陽風スパッタリング (SWS) の目標SP数/ステップ
    TARGET_SPS_MMV = 1000  # 微小隕石蒸発 (MMV) の目標SP数/ステップ

    # --- 粒子生成モデル (Leblanc 2003 準拠) ---
    # (PSD)
    F_UV_1AU_PER_M2 = 1.5e14 * (100.0 ** 2)  # 1AUでの光子フラックス [photons/m^2/s]
    Q_PSD_M2 = 1.0e-20 / (100.0 ** 2)  # 脱離断面積 [m^2]
    TEMP_PSD = 1500.0  # PSD粒子の初期温度 [K]
    # (MMV)
    TEMP_MMV = 3000.0  # MMV粒子の初期温度 [K]

    # --- SWS (CPS) 用のパラメータ辞書 ---
    SWS_PARAMS = {
        'SW_DENSITY_1AU': 10.0 * (100.0) ** 3,  # 1AUでの太陽風密度 [protons/m^3]
        'SW_VELOCITY': 400.0 * 1000.0,  # 太陽風速度 [m/s]
        'YIELD_EFF': 0.06,  # スパッタリング収率 (Na / proton)
        'U_eV': 0.27,  # Thompson-Sigmund 分布の束縛エネルギー [eV]
        'DENSITY_REF_M2': 7.5e14 * (100.0) ** 2,  # 参照表面密度 [atoms/m^2] (収率計算用)
        'LON_MIN_RAD': np.deg2rad(-40.0),  # SWS発生領域 (太陽直下点から)
        'LON_MAX_RAD': np.deg2rad(40.0),
        'LAT_N_MIN_RAD': np.deg2rad(30.0),
        'LAT_N_MAX_RAD': np.deg2rad(60.0),
        'LAT_S_MIN_RAD': np.deg2rad(-60.0),
        'LAT_S_MAX_RAD': np.deg2rad(-30.0),
    }

    # --- 出力グリッド（空間密度）の設定 ---
    GRID_RESOLUTION = 101  # 3Dグリッドの解像度 (x, y, z)
    GRID_MAX_RM = 5.0  # グリッドの最大範囲 (水星半径 RM の何倍か)

    # --- 物理モデルのフラグ ---
    USE_SOLAR_GRAVITY = True  # 太陽重力を考慮するか
    USE_CORIOLIS_FORCES = True  # 見かけの力（コリオリ力・遠心力）を考慮するか

    # シミュレーション設定を辞書にまとめる
    settings = {
        'BETA': 0.5,  # 表面反射時の熱 accomodation 係数
        'T1AU': 168918.0,  # 1AUでのナトリウム光電離寿命 [s]
        # 'DT': 1.0,  # ★ 軌道積分 (RK4) の時間ステップ [s] -> duration と同一にするため削除
        'GRID_RADIUS_RM': GRID_MAX_RM + 1.0,  # 粒子の脱出判定半径
        'USE_SOLAR_GRAVITY': USE_SOLAR_GRAVITY,
        'USE_CORIOLIS_FORCES': USE_CORIOLIS_FORCES
    }

    # --- 2. 出力ディレクトリとシミュレーションの初期化 ---
    run_name = f"DynamicGrid{N_LON_FIXED}x{N_LAT}_2.0"
    target_output_dir = os.path.join(OUTPUT_DIRECTORY, run_name)
    os.makedirs(target_output_dir, exist_ok=True)
    print(f"結果は '{target_output_dir}' に保存されます。")

    print(f"--- 物理モデル設定 (完全・可変重み方式) ---")
    print(f"Dynamic Surface Grid: {N_LON_FIXED}x{N_LAT}")
    print(f"Processes: PSD, Thermal Desorption, Micrometeoroid Vaporization, SWS")
    print(f"--- 目標SP数/ステップ (予算) ---")
    print(f"TD  (Variable): Target SPs/Step = {TARGET_SPS_TD}")
    print(f"PSD (Variable): Target SPs/Step = {TARGET_SPS_PSD}")
    print(f"SWS (Variable): Target SPs/Step = {TARGET_SPS_SWS}")
    print(f"MMV (Variable): Target SPs/Step = {TARGET_SPS_MMV}")
    print(f"----------------------")
    print(f"Solar Gravity: {USE_SOLAR_GRAVITY}")
    print(f"Coriolis/Centrifugal: {USE_CORIOLIS_FORCES}")
    print(f"----------------------")
    print(
        f"Dynamic Timestep: HOT_T={HOT_TEMP_THRESHOLD_K}K, HOT_DT={HOT_TEMP_TIME_STEP_SEC}s, MAX_DT={DEFAULT_MAX_TIME_STEP_SEC}s")
    print(f"Low Speed Bypass: v < {LOW_SPEED_THRESHOLD_M_S} m/s")
    print(f"★RK4 Integrator: 1 step per main timestep (DT=duration)")

    # 水星の1公転周期 [s]
    MERCURY_YEAR_SEC = 87.97 * 24 * 3600

    # 惑星固定座標系のグリッドを初期化
    lon_edges_fixed = np.linspace(-np.pi, np.pi, N_LON_FIXED + 1)
    lat_edges_fixed = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    dlon_fixed = lon_edges_fixed[1] - lon_edges_fixed[0]
    # 各緯度帯のセル面積 [m^2] を計算
    cell_areas_m2 = (PHYSICAL_CONSTANTS['RM'] ** 2) * dlon_fixed * \
                    (np.sin(lat_edges_fixed[1:]) - np.sin(lat_edges_fixed[:-1]))
    # 表面密度グリッドを初期値で埋める
    surface_density_grid = np.full((N_LON_FIXED, N_LAT), INITIAL_SURFACE_DENSITY_PER_M2, dtype=np.float64)

    # --- 3. 外部ファイルの読み込み ---
    try:
        spec_data_np = np.loadtxt('SolarSpectrum_Na0.txt', usecols=(0, 3))
        orbit_file_name = 'orbit2025_v5.txt'
        orbit_data = np.loadtxt(orbit_file_name)
    except FileNotFoundError as e:
        print(f"エラー: データファイル '{e.filename}' が見つかりません。");
        sys.exit()
    if orbit_data.shape[1] < 5:
        print(f"エラー: '{orbit_file_name}' の列が不足しています。");
        sys.exit()

    # --- 時間設定 ---
    # 軌道ファイルから TAA=0 (近日点) の時刻を探す
    taa_col = orbit_data[:, 0]
    time_col = orbit_data[:, 2]
    idx_perihelion = np.argmin(np.abs(taa_col))
    t_start_run = time_col[idx_perihelion]  # TAA=0 を RUN の開始時刻とする
    t_end_run = t_start_run + (TOTAL_SIM_YEARS * MERCURY_YEAR_SEC)
    t_start_spinup = t_start_run - (SPIN_UP_YEARS * MERCURY_YEAR_SEC)

    # ★シミュレーション全体の時間ステップは使用しない
    # time_steps = np.arange(t_start_spinup, t_end_run, TIME_STEP_SEC)
    print(f"--- 時間設定 (動的モデル) ---");
    print(f"軌道ファイル上のTAA=0 (近日点) 時刻: {t_start_run:.1f} s");
    print(f"スピンアップ開始時刻: {t_start_spinup:.1f} s ({-SPIN_UP_YEARS} 年前)");
    print(f"RUN開始時刻 (TAA=0): {t_start_run:.1f} s");
    print(f"RUN終了時刻: {t_end_run:.1f} s (+{TOTAL_SIM_YEARS} 年後)");
    print(f"------------------")

    # (スペクトルデータの前処理)
    sigma_const = PHYSICAL_CONSTANTS['E_CHARGE'] ** 2 / (
            4 * PHYSICAL_CONSTANTS['ME'] * PHYSICAL_CONSTANTS['C'] * PHYSICAL_CONSTANTS['EPSILON_0'])
    wl, gamma = spec_data_np[:, 0], spec_data_np[:, 1]
    # 波長が昇順でない場合はソートする
    if not np.all(np.diff(wl) > 0):
        sort_indices = np.argsort(wl)
        wl, gamma = wl[sort_indices], gamma[sort_indices]
    spec_data_dict = {'wl': wl, 'gamma': gamma,
                      'sigma0_perdnu2': sigma_const * 0.641,  # D2線
                      'sigma0_perdnu1': sigma_const * 0.320,  # D1線
                      'JL': 5.18e14}  # 太陽スペクトルの定数

    # --- 4. メインループ (時間発展) ---
    active_particles = []  # 現在シミュレーション空間にいる粒子リスト
    previous_taa = -1
    target_taa_idx = 0

    # 現在のステップの重み（初期値）
    current_step_weight_td = 1.0
    current_step_weight_psd = 1.0
    current_step_weight_sws = 1.0
    current_step_weight_mmv = 1.0

    # ★t_sec を初期化し、while ループに変更
    t_sec = t_start_spinup
    # ★pbar の total を総秒数に変更
    with tqdm(total=int(t_end_run - t_start_spinup), desc="Time Evolution") as pbar:
        # for t_sec in time_steps: # ★削除
        while t_sec < t_end_run:  # ★while ループに変更

            # --- 4a. 現在時刻の軌道パラメータを取得 ---
            TAA, AU, V_radial_ms, V_tangential_ms, subsolar_lon_rad_fixed = get_orbital_params(
                t_sec, orbit_data, MERCURY_YEAR_SEC
            )

            run_phase = "Spin-up" if t_sec < t_start_run else "Run"

            # --- 4b. (新設) 動的タイムステップの決定 ---

            # ★ 温度（subsolar_temp_k）による分岐を削除し、常に動的に計算する

            min_timescale = np.inf

            # レート計算に必要なパラメータを準備
            F_UV_current_per_m2 = F_UV_1AU_PER_M2 / (AU ** 2)
            flux_sw_1au = SWS_PARAMS['SW_DENSITY_1AU'] * SWS_PARAMS['SW_VELOCITY']
            current_flux_sw = flux_sw_1au / (AU ** 2)
            effective_flux_sw = current_flux_sw
            base_sputtering_rate_per_m2_s = effective_flux_sw * SWS_PARAMS['YIELD_EFF']
            DENSITY_REF_M2 = SWS_PARAMS['DENSITY_REF_M2']

            for i_lon in range(N_LON_FIXED):
                for i_lat in range(N_LAT):
                    # ★枯渇したセルはタイムステップ計算から除外
                    current_density_per_m2 = surface_density_grid[i_lon, i_lat]
                    if current_density_per_m2 < MIN_DENSITY_FOR_TIMESTEP:
                        continue

                    # このセルのパラメータ
                    lon_fixed_rad = (lon_edges_fixed[i_lon] + lon_edges_fixed[i_lon + 1]) / 2
                    lat_rad = (lat_edges_fixed[i_lat] + lat_edges_fixed[i_lat + 1]) / 2

                    # (1) PSD レート
                    rate_psd_per_s = 0.0
                    cos_Z = np.cos(lat_rad) * np.cos(lon_fixed_rad - subsolar_lon_rad_fixed)
                    if cos_Z > 0:
                        rate_psd_per_s = F_UV_current_per_m2 * Q_PSD_M2 * cos_Z

                    # (2) TD レート
                    temp_k = calculate_surface_temperature_leblanc(lon_fixed_rad, lat_rad, AU,
                                                                   subsolar_lon_rad_fixed)
                    rate_td_per_s = calculate_thermal_desorption_rate(temp_k)

                    # (3) SWS レート
                    rate_sws_per_s = 0.0
                    lon_sun_fixed_rad = (lon_fixed_rad - subsolar_lon_rad_fixed)
                    lon_sun_fixed_rad = (lon_sun_fixed_rad + PHYSICAL_CONSTANTS['PI']) % (
                            2 * PHYSICAL_CONSTANTS['PI']) - PHYSICAL_CONSTANTS['PI']
                    lat_sun_fixed_rad = lat_rad
                    is_in_lon_band = (SWS_PARAMS['LON_MIN_RAD'] <= lon_sun_fixed_rad <= SWS_PARAMS['LON_MAX_RAD'])
                    is_in_lat_n_band = (
                            SWS_PARAMS['LAT_N_MIN_RAD'] <= lat_sun_fixed_rad <= SWS_PARAMS['LAT_N_MAX_RAD'])
                    is_in_lat_s_band = (
                            SWS_PARAMS['LAT_S_MIN_RAD'] <= lat_sun_fixed_rad <= SWS_PARAMS['LAT_S_MAX_RAD'])
                    if is_in_lon_band and (is_in_lat_n_band or is_in_lat_s_band):
                        rate_sws_per_s = (base_sputtering_rate_per_m2_s / DENSITY_REF_M2)

                    # ★合計レートからタイムスケールを計算
                    total_rate_per_s = rate_psd_per_s + rate_td_per_s + rate_sws_per_s

                    if total_rate_per_s > 1e-20:  # ゼロ除算を避ける
                        timescale = 1.0 / total_rate_per_s
                        min_timescale = min(min_timescale, timescale)

            # --- ★ここからが新しいロジック ---
            # 1. 計算上のタイムステップを決定
            current_dt_main = SAFETY_FACTOR * min_timescale
            if np.isinf(current_dt_main):  # どのセルも枯渇しない場合
                current_dt_main = DEFAULT_MAX_TIME_STEP_SEC

            # 2. ★ご要望のロジックを適用: 「50s以下だったら50sにする」
            #    (HOT_TEMP_TIME_STEP_SEC = 50.0s が最小値となる)
            current_dt_main = max(current_dt_main, HOT_TEMP_TIME_STEP_SEC)

            # 3. ★最終的な最大値(2000s)でクリップする
            current_dt_main = min(current_dt_main, DEFAULT_MAX_TIME_STEP_SEC)

            # --- 4c. (旧 4b.) 表面から新しい粒子を生成 ---

            pbar.set_description(
                f"[{run_phase}] DT={current_dt_main:.2f}s | TAA={TAA:.1f} | N_act={len(active_particles)} | "
                f"W_TD={current_step_weight_td:.1e} W_PSD={current_step_weight_psd:.1e} "
                f"W_SWS={current_step_weight_sws:.1e} W_MMV={current_step_weight_mmv:.1e}"
            )

            # (A) 事前ループ: 惑星全体の「総放出原子数」をプロセスごとに計算する
            total_atoms_td_this_step = 0.0
            total_atoms_psd_this_step = 0.0
            total_atoms_sws_this_step = 0.0
            n_atoms_mmv = 0.0

            # 各プロセスで「失われる原子数」ではなく「レート[1/s]」を記録する配列に変更
            rate_psd_grid_s = np.zeros_like(surface_density_grid)
            rate_td_grid_s = np.zeros_like(surface_density_grid)
            rate_sws_grid_s = np.zeros_like(surface_density_grid)
            # このステップで失われる「原子総数」を記録する配列
            total_atoms_lost_grid = np.zeros_like(surface_density_grid)

            # (A-1) MMV の総原子数を計算 (表面密度に依存しない)
            flux_mmv_per_m2_s = calculate_mmv_flux(AU)
            MERCURY_SURFACE_AREA_M2 = 4 * PHYSICAL_CONSTANTS['PI'] * (PHYSICAL_CONSTANTS['RM'] ** 2)
            # TIME_STEP_SEC を current_dt_main に変更
            n_atoms_mmv = flux_mmv_per_m2_s * MERCURY_SURFACE_AREA_M2 * current_dt_main

            # (A-2) PSD, TD, SWS のレートを計算 (グリッドをループ)
            F_UV_current_per_m2 = F_UV_1AU_PER_M2 / (AU ** 2)
            flux_sw_1au = SWS_PARAMS['SW_DENSITY_1AU'] * SWS_PARAMS['SW_VELOCITY']
            current_flux_sw = flux_sw_1au / (AU ** 2)
            # (ここでは時間変動は考慮しない)
            effective_flux_sw = current_flux_sw

            base_sputtering_rate_per_m2_s = effective_flux_sw * SWS_PARAMS['YIELD_EFF']
            DENSITY_REF_M2 = SWS_PARAMS['DENSITY_REF_M2']

            for i_lon in range(N_LON_FIXED):
                for i_lat in range(N_LAT):
                    current_density_per_m2 = surface_density_grid[i_lon, i_lat]
                    if current_density_per_m2 <= 0:
                        continue

                    lon_fixed_rad = (lon_edges_fixed[i_lon] + lon_edges_fixed[i_lon + 1]) / 2
                    lat_rad = (lat_edges_fixed[i_lat] + lat_edges_fixed[i_lat + 1]) / 2
                    area_m2 = cell_areas_m2[i_lat]

                    # このセルで利用可能な原子の総数
                    atoms_available_in_cell = current_density_per_m2 * area_m2

                    # 太陽天頂角の余弦
                    cos_Z = np.cos(lat_rad) * np.cos(lon_fixed_rad - subsolar_lon_rad_fixed)

                    # --- レートの計算 ---
                    rate_psd_per_s = 0.0
                    rate_td_per_s = 0.0
                    rate_sws_per_s = 0.0

                    # (1) PSD
                    if cos_Z > 0:  # 昼側のみ
                        # PSDの放出率 (原子1個あたり) [1/s]
                        rate_psd_per_s = F_UV_current_per_m2 * Q_PSD_M2 * cos_Z
                        rate_psd_grid_s[i_lon, i_lat] = rate_psd_per_s  # レートを保存

                    # (2) TD
                    temp_k = calculate_surface_temperature_leblanc(lon_fixed_rad, lat_rad, AU, subsolar_lon_rad_fixed)
                    rate_td_per_s = calculate_thermal_desorption_rate(temp_k)  # [1/s]

                    if rate_td_per_s > 0:
                        rate_td_grid_s[i_lon, i_lat] = rate_td_per_s  # レートを保存

                    # (3) SWS
                    # SWSの発生領域は「太陽固定座標系」で定義されるため、座標変換が必要
                    lon_sun_fixed_rad = (lon_fixed_rad - subsolar_lon_rad_fixed)
                    lon_sun_fixed_rad = (lon_sun_fixed_rad + PHYSICAL_CONSTANTS['PI']) % (
                            2 * PHYSICAL_CONSTANTS['PI']) - PHYSICAL_CONSTANTS['PI']
                    lat_sun_fixed_rad = lat_rad

                    is_in_lon_band = (SWS_PARAMS['LON_MIN_RAD'] <= lon_sun_fixed_rad <= SWS_PARAMS['LON_MAX_RAD'])
                    is_in_lat_n_band = (SWS_PARAMS['LAT_N_MIN_RAD'] <= lat_sun_fixed_rad <= SWS_PARAMS['LAT_N_MAX_RAD'])
                    is_in_lat_s_band = (SWS_PARAMS['LAT_S_MIN_RAD'] <= lat_sun_fixed_rad <= SWS_PARAMS['LAT_S_MAX_RAD'])

                    if is_in_lon_band and (is_in_lat_n_band or is_in_lat_s_band):
                        rate_sws_per_s = (base_sputtering_rate_per_m2_s / DENSITY_REF_M2)
                        rate_sws_grid_s[i_lon, i_lat] = rate_sws_per_s  # レートを保存

                    # --- 安定な枯渇計算 (最重要) ---
                    # 3つのレートを合計
                    total_rate_per_s = rate_psd_per_s + rate_td_per_s + rate_sws_per_s

                    if total_rate_per_s > 0:
                        # 合計レートと dt から、失われる原子の「総数」を安定に計算
                        # 1. 線形近似で枯渇量を計算
                        n_atoms_total_lost_linear = atoms_available_in_cell * total_rate_per_s * current_dt_main

                        # 2. 利用可能な原子数でクリップ（上限設定）
                        n_atoms_total_lost = min(n_atoms_total_lost_linear, atoms_available_in_cell)

                        total_atoms_lost_grid[i_lon, i_lat] = n_atoms_total_lost  # 総枯渇量を保存

                        # 各プロセスの「総放出原子数」は、総枯渇量をレート比で按分して求める
                        # (これは重み計算 W_TD のために必要)
                        if rate_psd_per_s > 0:
                            total_atoms_psd_this_step += n_atoms_total_lost * (rate_psd_per_s / total_rate_per_s)
                        if rate_td_per_s > 0:
                            total_atoms_td_this_step += n_atoms_total_lost * (rate_td_per_s / total_rate_per_s)
                        if rate_sws_per_s > 0:
                            total_atoms_sws_this_step += n_atoms_total_lost * (rate_sws_per_s / total_rate_per_s)

            # (B) 重みの決定 (全プロセスを可変に)
            # 重み = (このステップで放出される総原子数) / (生成したいSP数)
            current_step_weight_td = 1.0
            if total_atoms_td_this_step > 0 and TARGET_SPS_TD > 0:
                current_step_weight_td = total_atoms_td_this_step / TARGET_SPS_TD

            current_step_weight_psd = 1.0
            if total_atoms_psd_this_step > 0 and TARGET_SPS_PSD > 0:
                current_step_weight_psd = total_atoms_psd_this_step / TARGET_SPS_PSD

            current_step_weight_sws = 1.0
            if total_atoms_sws_this_step > 0 and TARGET_SPS_SWS > 0:
                current_step_weight_sws = total_atoms_sws_this_step / TARGET_SPS_SWS

            current_step_weight_mmv = 1.0
            if n_atoms_mmv > 0 and TARGET_SPS_MMV > 0:
                current_step_weight_mmv = n_atoms_mmv / TARGET_SPS_MMV

            # (C) メインループ: 粒子を生成
            newly_launched_particles = []
            # ★吸着グリッドをここで初期化 (4dから移動)
            atoms_gained_grid = np.zeros_like(surface_density_grid)

            # (C-1) MMV の粒子を生成 (可変重み)
            if n_atoms_mmv > 0 and TARGET_SPS_MMV > 0:
                num_sps_float_mmv = TARGET_SPS_MMV
                num_sps_int_mmv = int(num_sps_float_mmv)
                # 確率的に端数を処理
                if np.random.random() < (num_sps_float_mmv - num_sps_int_mmv):
                    num_sps_int_mmv += 1

                if num_sps_int_mmv > 0:
                    M_rejection = 4.0 / 3.0  # 棄却サンプリング用
                    for _ in range(num_sps_int_mmv):
                        # MMVの放出位置は太陽方向に対して非一様 (棄却法)
                        while True:
                            lon_rot_rad = np.random.uniform(-np.pi, np.pi)
                            prob_accept = (1.0 - (1.0 / 3.0) * np.sin(lon_rot_rad)) / M_rejection
                            if np.random.random() < prob_accept:
                                break
                        lat_rot_rad = np.arcsin(np.random.uniform(-1.0, 1.0))

                        # (回転座標系での)初期位置と速度を計算
                        initial_pos_rot = lonlat_to_xyz(lon_rot_rad, lat_rot_rad, PHYSICAL_CONSTANTS['RM'])
                        surface_normal_rot = initial_pos_rot / PHYSICAL_CONSTANTS['RM']
                        speed = sample_speed_from_flux_distribution(PHYSICAL_CONSTANTS['MASS_NA'], TEMP_MMV)
                        initial_vel_rot = speed * transform_local_to_world(sample_lambertian_direction_local(),
                                                                           surface_normal_rot)

                        # --- ★(新設) 低速粒子の追跡省略(バイパス)判定 ---
                        # MMV粒子は高速だが、念のためここにも判定を入れる
                        if speed < LOW_SPEED_THRESHOLD_M_S:
                            # MMVは特定のセルから出ないため、i_lon, i_lat が未定義。
                            # 本来は衝突地点を計算すべきだが、MMVはほぼ高速なので
                            # ここでは単純に追跡をスキップする (吸着にも加算しない)
                            continue
                        # --- ★バイパス判定 終了 ---

                        newly_launched_particles.append({
                            'pos': initial_pos_rot, 'vel': initial_vel_rot,
                            'weight': current_step_weight_mmv
                        })

            # (C-2) PSD, TD, SWS の粒子を生成 (グリッドをループ)
            for i_lon in range(N_LON_FIXED):
                for i_lat in range(N_LAT):

                    # このセルで失われる原子の総数
                    total_atoms_to_lose = total_atoms_lost_grid[i_lon, i_lat]
                    if total_atoms_to_lose <= 0:
                        continue

                    # このセルの各レートを取得
                    rate_psd = rate_psd_grid_s[i_lon, i_lat]
                    rate_td = rate_td_grid_s[i_lon, i_lat]
                    rate_sws = rate_sws_grid_s[i_lon, i_lat]
                    total_rate = rate_psd + rate_td + rate_sws

                    if total_rate <= 0: continue

                    # 各プロセスが寄与する原子数 (総枯渇量をレート比で按分)
                    n_atoms_psd = total_atoms_to_lose * (rate_psd / total_rate)
                    n_atoms_td = total_atoms_to_lose * (rate_td / total_rate)
                    n_atoms_sws = total_atoms_to_lose * (rate_sws / total_rate)

                    # このセルのパラメータ
                    lon_fixed_rad = (lon_edges_fixed[i_lon] + lon_edges_fixed[i_lon + 1]) / 2
                    lat_rad = (lat_edges_fixed[i_lat] + lat_edges_fixed[i_lat + 1]) / 2
                    temp_k = calculate_surface_temperature_leblanc(lon_fixed_rad, lat_rad, AU,
                                                                   subsolar_lon_rad_fixed)

                    # 各プロセスをループ処理
                    procs = {
                        'PSD': {'n_atoms': n_atoms_psd, 'temp': TEMP_PSD, 'U_eV': None,
                                'weight': current_step_weight_psd},
                        'TD': {'n_atoms': n_atoms_td, 'temp': temp_k, 'U_eV': None, 'weight': current_step_weight_td},
                        'SWS': {'n_atoms': n_atoms_sws, 'temp': None, 'U_eV': SWS_PARAMS['U_eV'],
                                'weight': current_step_weight_sws}
                    }

                    for proc_name, p in procs.items():
                        if p['n_atoms'] <= 0 or p['weight'] <= 0: continue

                        weight_to_use = p['weight']

                        # 生成すべきSP数を計算
                        # N_SP = (このセルで放出される原子数) / (SP 1個の重み)
                        if np.isinf(weight_to_use):
                            num_sps_float = 0.0
                        else:
                            num_sps_float = p['n_atoms'] / weight_to_use
                        if np.isnan(num_sps_float):
                            num_sps_float = 0.0

                        # 確率的に端数を処理
                        num_sps_int = int(num_sps_float)
                        if np.random.random() < (num_sps_float - num_sps_int):
                            num_sps_int += 1
                        if num_sps_int == 0: continue

                        # 枯渇量の計算はここでは不要 (既に total_atoms_lost_grid で確定済)

                        # SPを生成
                        for _ in range(num_sps_int):
                            # 初期速度を決定
                            if proc_name == 'SWS':
                                energy_eV = sample_thompson_sigmund_energy(p['U_eV'])
                                energy_J = energy_eV * PHYSICAL_CONSTANTS['EV_TO_JOULE']
                                speed = np.sqrt(2.0 * energy_J / PHYSICAL_CONSTANTS['MASS_NA'])
                            else:
                                speed = sample_speed_from_flux_distribution(PHYSICAL_CONSTANTS['MASS_NA'],
                                                                            p['temp'])

                            # --- ★(新設) 低速粒子の追跡省略(バイパス)判定 ---
                            if speed < LOW_SPEED_THRESHOLD_M_S:
                                # この粒子は追跡しない (局所平衡とみなす)
                                # 粒子分の重み(原子数)を、吸着グリッドに直接加算する。
                                # (放出されたセルにそのまま戻ると仮定)
                                atoms_gained_grid[i_lon, i_lat] += weight_to_use

                                # 粒子を newly_launched_particles に追加せず、
                                # このSPの生成処理を終了する。
                                continue
                            # --- ★バイパス判定 終了 ---

                            # (回転座標系での)初期位置と速度を計算
                            lon_rot_rad = lon_fixed_rad - subsolar_lon_rad_fixed
                            lat_rot_rad = lat_rad

                            initial_pos_rot = lonlat_to_xyz(lon_rot_rad, lat_rot_rad, PHYSICAL_CONSTANTS['RM'])
                            surface_normal_rot = initial_pos_rot / PHYSICAL_CONSTANTS['RM']
                            initial_vel_rot = speed * transform_local_to_world(sample_lambertian_direction_local(),
                                                                               surface_normal_rot)
                            newly_launched_particles.append({
                                'pos': initial_pos_rot, 'vel': initial_vel_rot,
                                'weight': weight_to_use
                            })

            # --- 4c 終了: 表面密度（枯渇）の更新 ---
            # (atoms_lost_grid は原子数, cell_areas_m2 は面積 [m^2])
            # total_atoms_lost_grid を使って枯渇させる
            surface_density_grid -= total_atoms_lost_grid / cell_areas_m2
            # 密度が負にならないようにクリップ
            np.clip(surface_density_grid, 0, None, out=surface_density_grid)

            # 新しく生成された粒子をアクティブリストに追加
            active_particles.extend(newly_launched_particles)

            # --- 4d. (旧 4c.) 全ての粒子を1ステップ進め、結果を集計 ---
            # マルチプロセス用のタスクリストを作成
            tasks = [{'settings': settings, 'spec': spec_data_dict, 'particle_state': p,
                      'orbit': (TAA, AU, V_radial_ms, V_tangential_ms, subsolar_lon_rad_fixed),
                      'duration': current_dt_main} for p in  # current_dt_main に変更
                     active_particles]

            next_active_particles = []
            # ★atoms_gained_grid の初期化は 4c に移動したので、ここでは行わない
            # atoms_gained_grid = np.zeros_like(surface_density_grid) # ★削除

            if tasks:
                # プロセスプールを作成して並列実行
                with Pool(processes=max(1, cpu_count() - 1)) as pool:
                    results = list(pool.imap(simulate_particle_for_one_step, tasks, chunksize=100))

                # 結果を集計
                for res in results:
                    if res['status'] == 'alive':
                        # 生き残った粒子
                        next_active_particles.append(res['final_state'])
                    elif res['status'] == 'stuck':
                        # 表面に吸着した粒子
                        pos_rot = res['pos_at_impact']
                        weight = res['weight']

                        # 衝突位置（回転座標系）
                        lon_rot = np.arctan2(pos_rot[1], pos_rot[0])
                        lat_rot = np.arcsin(np.clip(pos_rot[2] / np.linalg.norm(pos_rot), -1.0, 1.0))

                        # 衝突位置（固定座標系）
                        lon_fixed = (lon_rot + subsolar_lon_rad_fixed)
                        lon_fixed = (lon_fixed + PHYSICAL_CONSTANTS['PI']) % (2 * PHYSICAL_CONSTANTS['PI']) - \
                                    PHYSICAL_CONSTANTS['PI']
                        lat_fixed = lat_rot

                        # 対応するグリッドインデックスを検索
                        i_lon = np.searchsorted(lon_edges_fixed, lon_fixed) - 1
                        i_lat = np.searchsorted(lat_edges_fixed, lat_fixed) - 1
                        if 0 <= i_lon < N_LON_FIXED and 0 <= i_lat < N_LAT:
                            # 吸着原子数を加算
                            atoms_gained_grid[i_lon, i_lat] += weight
                    # 'ionized' と 'escaped' は何もしない (リストから消える)

            active_particles = next_active_particles
            # 表面密度（吸着）の更新
            # ★ここで加算される atoms_gained_grid には、
            # ★ 4c でバイパスした低速粒子 と 4d で追跡して衝突した高速粒子 の両方が含まれる
            surface_density_grid += atoms_gained_grid / cell_areas_m2

            # --- 4e. (旧 4d.) スナップショット保存判定 ---
            save_this_step = False
            if TAA < previous_taa:  # TAAが360->0になった（周回した）
                target_taa_idx = 0

            if target_taa_idx < len(TARGET_TAA_DEGREES):
                current_target_taa = TARGET_TAA_DEGREES[target_taa_idx]
                # TAA=0 をまたぐ場合の処理
                is_crossing_zero = (current_target_taa == 0) and \
                                   ((TAA < previous_taa) or (TAA >= 0 and previous_taa < 0))
                # 通常のTAAをまたぐ処理
                is_crossing_normal = (previous_taa < current_target_taa <= TAA)

                if is_crossing_normal or is_crossing_zero:
                    save_this_step = True
                    target_taa_idx += 1

            # --- 4f. (旧 4e.) 立方体グリッドに集計して保存 ---
            if save_this_step and t_sec >= t_start_run:  # Runフェーズのみ保存
                pbar.write(f"\n>>> [Run] Saving grid snapshot at TAA={TAA:.1f} ({len(active_particles)} particles) <<<")

                # 空間密度グリッドを初期化
                density_grid = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION, GRID_RESOLUTION), dtype=np.float32)
                grid_min = -GRID_MAX_RM * PHYSICAL_CONSTANTS['RM']
                grid_max = GRID_MAX_RM * PHYSICAL_CONSTANTS['RM']
                cell_size = (grid_max - grid_min) / GRID_RESOLUTION
                cell_volume_m3 = cell_size ** 3

                # アクティブな粒子をグリッドに集計
                for p in active_particles:
                    pos = p['pos']
                    ix = int((pos[0] - grid_min) / cell_size)
                    iy = int((pos[1] - grid_min) / cell_size)
                    iz = int((pos[2] - grid_min) / cell_size)
                    if 0 <= ix < GRID_RESOLUTION and 0 <= iy < GRID_RESOLUTION and 0 <= iz < GRID_RESOLUTION:
                        density_grid[ix, iy, iz] += p['weight']

                # SPの重みを実密度に変換 (個数 / 体積)
                density_grid /= cell_volume_m3

                relative_time_sec = t_sec - t_start_run
                save_time_h = relative_time_sec / 3600  # [hour]

                # 空間密度グリッドを保存
                filename = f"density_grid_t{int(save_time_h):05d}_taa{int(round(TAA)):03d}.npy"
                np.save(os.path.join(target_output_dir, filename), density_grid)

                # 表面密度グリッドも保存
                filename_surf = f"surface_density_t{int(save_time_h):05d}_taa{int(round(TAA)):03d}.npy"
                np.save(os.path.join(target_output_dir, filename_surf), surface_density_grid)

            previous_taa = TAA

            # --- ★ループの最後で時間を更新 ---
            pbar.update(current_dt_main)  # pbar を進んだ時間だけ更新
            t_sec += current_dt_main  # シミュレーション時刻を進める
            # pbar.update(1) <- ★削除

    end_time = time.time()
    print(f"\n★★★ シミュレーションが完了しました ★★★")
    print(f"総計算時間: {(end_time - start_time) / 3600:.2f} 時間")


if __name__ == '__main__':
    print("必須ファイルを確認しています...")
    for f in ['orbit2025_v5.txt', 'SolarSpectrum_Na0.txt']:
        if not os.path.exists(f):
            print(f"エラー: 必須ファイル '{f}' が見つかりません。スクリプトと同じディレクトリに配置してください。")
            if f == 'orbit2025_v5.txt':
                print("（orbit2025_v5.txt がない場合は、軌道生成スクリトを先に実行してください）")
            sys.exit()
    print("ファイルOK。シミュレーションを開始します。")
    main_snapshot_simulation()