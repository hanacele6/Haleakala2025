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
1.  orbit2025_v5.txt: (★ v6.txt -> orbit2025.txt に合わせる)
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
    if surface_temp_K < 10.0:  # 低温では発生しないと仮定
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
    ★ 修正: 軌道ファイルから正しい太陽直下点経度を読み込む
    """
    # 水星の自転周期
    # ROTATION_PERIOD_SEC = 58.646 * 24 * 3600 # ★ もう不要

    # 軌道データは1公転分なので、時刻を公転周期で割った余りを計算
    current_time_in_orbit = time_sec % mercury_year_sec
    time_col = orbit_data[:, 2]  # 3列目 (インデックス 2) が時刻

    # 軌道データから、現在の時刻に対応するパラメータを線形内挿で取得
    taa = np.interp(current_time_in_orbit, time_col, orbit_data[:, 0])  # 1列目 (TAA)
    au = np.interp(current_time_in_orbit, time_col, orbit_data[:, 1])  # 2列目 (AU)
    v_radial = np.interp(current_time_in_orbit, time_col, orbit_data[:, 3])  # 4列目 (V_radial)
    v_tangential = np.interp(current_time_in_orbit, time_col, orbit_data[:, 4])  # 5列目 (V_tangential)

    # ★★★【修正箇所】★★★
    # 6列目 (インデックス 5) の「太陽直下点経度 [deg]」を読み込む
    SUBSOLAR_LON_COL_IDX = 5

    subsolar_lon_deg_fixed = np.interp(
        current_time_in_orbit,
        time_col,
        orbit_data[:, SUBSOLAR_LON_COL_IDX]
    )

    # [deg] から [rad] に変換して戻り値とする
    subsolar_lon_rad = np.deg2rad(subsolar_lon_deg_fixed)

    # ★★★【削除】★★★
    # 以前の不正確な計算は削除します
    # subsolar_lon_rad = (-2 * np.pi * time_sec / ROTATION_PERIOD_SEC) % (2 * np.pi)

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


def _calculate_acceleration(pos, vel, V_radial_ms, V_tangential_ms, AU, spec_data, settings):
    """
    指定された位置と速度における粒子（ナトリウム原子）の加速度を計算します。
    (太陽固定回転座標系で計算)

    考慮する力:
    1. 太陽放射圧 (SRP) - ドップラーシフト考慮
    2. 水星重力
    3. 太陽重力 (オプション)
    4. 見かけの力（コリオリ力、遠心力） (オプション)

    Args:
        pos (np.ndarray): 位置ベクトル [x, y, z] [m]
        vel (np.ndarray): 速度ベクトル [vx, vy, vz] [m/s]
        V_radial_ms (float): 水星の視線速度 (太陽に近づく向きが正) [m/s]
        V_tangential_ms (float): 水星の公転速度 [m/s]
        AU (float): 日心距離 [AU]
        spec_data (dict): スペクトルデータ
        settings (dict): シミュレーション設定

    Returns:
        np.ndarray: 加速度ベクトル [ax, ay, az] [m/s^2]
    """
    x, y, z = pos
    r0 = AU * PHYSICAL_CONSTANTS['AU']  # 水星-太陽間距離 [m]

    # --- 1. 太陽放射圧 (SRP) ---
    # 粒子の視線速度 (水星の視線速度 + 水星から見た粒子のX軸速度)
    velocity_for_doppler = vel[0] + V_radial_ms
    # ドップラーシフト後のD2, D1線の波長
    w_na_d2 = 589.1582e-9 * (1.0 + velocity_for_doppler / PHYSICAL_CONSTANTS['C'])
    w_na_d1 = 589.7558e-9 * (1.0 + velocity_for_doppler / PHYSICAL_CONSTANTS['C'])

    b = 0.0  # 加速度 [m/s^2]
    wl, gamma, sigma0_perdnu2, sigma0_perdnu1, JL = spec_data.values()

    # 波長がスペクトルデータの範囲内か確認
    if (wl[0] * 1e-9 <= w_na_d2 < wl[-1] * 1e-9) and \
            (wl[0] * 1e-9 <= w_na_d1 < wl[-1] * 1e-9):
        # 対応するgamma値（太陽スペクトルのフラックス補正係数）を内挿
        gamma2 = np.interp(w_na_d2 * 1e9, wl, gamma)
        gamma1 = np.interp(w_na_d1 * 1e9, wl, gamma)

        # 放射圧の計算 (Leblanc 2003 等の方式)
        F_lambda_1AU_m = JL * 1e4 * 1e9
        F_lambda_d1_at_Mercury = F_lambda_1AU_m / (AU ** 2) * gamma1
        F_nu_d1 = F_lambda_d1_at_Mercury * (w_na_d1 ** 2) / PHYSICAL_CONSTANTS['C']
        J1 = sigma0_perdnu1 * F_nu_d1
        F_lambda_d2_at_Mercury = F_lambda_1AU_m / (AU ** 2) * gamma2
        F_nu_d2 = F_lambda_d2_at_Mercury * (w_na_d2 ** 2) / PHYSICAL_CONSTANTS['C']
        J2 = sigma0_perdnu2 * F_nu_d2

        b = 1 / PHYSICAL_CONSTANTS['MASS_NA'] * (
                (PHYSICAL_CONSTANTS['H'] / w_na_d1) * J1 + (PHYSICAL_CONSTANTS['H'] / w_na_d2) * J2)

    # 水星の影に入った場合 (X < 0 かつ YZ平面での半径がRM未満)
    if x < 0 and np.sqrt(y ** 2 + z ** 2) < PHYSICAL_CONSTANTS['RM']:
        b = 0.0  # 放射圧は 0

    accel_srp = np.array([-b, 0.0, 0.0])  # SRPは常に-X方向

    # --- 2. 水星重力 ---
    r_sq = np.sum(pos ** 2)
    accel_g = -PHYSICAL_CONSTANTS['GM_MERCURY'] * pos / (r_sq ** 1.5) if r_sq > 0 else np.array([0., 0., 0.])

    # --- 3. 太陽重力 (オプション) ---
    accel_sun = np.array([0.0, 0.0, 0.0])
    if settings.get('USE_SOLAR_GRAVITY', False):
        G = PHYSICAL_CONSTANTS['G']
        M_SUN = PHYSICAL_CONSTANTS['MASS_SUN']
        # 粒子から太陽へのベクトル (太陽は [r0, 0, 0] にある)
        r_ps_vec = np.array([r0 - pos[0], -pos[1], -pos[2]])
        r_ps_mag_sq = np.sum(r_ps_vec ** 2)
        if r_ps_mag_sq > 0:
            accel_sun = (G * M_SUN * r_ps_vec) / (r_ps_mag_sq ** 1.5)

    # --- 4. 見かけの力 (オプション) ---
    accel_coriolis = np.array([0.0, 0.0, 0.0])
    accel_centrifugal = np.array([0.0, 0.0, 0.0])
    if settings.get('USE_CORIOLIS_FORCES', False):
        if r0 > 0:
            # 座標系の回転角速度 (Z軸まわり)
            omega_val = V_tangential_ms / r0  # [rad/s]
            omega_sq = omega_val ** 2

            # 遠心力加速度: a_cen = -ω x (ω x r')
            # r' = [x-r0, y, z] (回転中心(太陽)から粒子へのベクトル)
            # ω = [0, 0, ω]
            # このシミュレーションでは、太陽固定座標系（原点が水星）で運動方程式を
            # 解いているため、太陽の重力(accel_sun)と、水星の公転運動による
            # 遠心力(accel_centrifugal)の「合力」を計算する必要がある。
            # (※ Leblanc 2003 の A_cf の式に基づき実装)
            accel_centrifugal = np.array([
                omega_val ** 2 * (pos[0] - r0),
                omega_sq * pos[1],
                0.0])

            # コリオリ力加速度: a_cor = -2ω x v
            two_omega = 2 * omega_val
            accel_coriolis = np.array([two_omega * vel[1], -two_omega * vel[0], 0.0])

    # 全ての加速度を合計
    return accel_srp + accel_g + accel_sun + accel_centrifugal + accel_coriolis


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

    # --- ★ 修正点 1 (要件7) ---
    # メインループのステップ幅 (duration) と RK4積分ステップ (DT) は
    # 同じ値 (例: 500s) に設定されているはず
    duration = args['duration']
    DT = settings['DT']  # (例: 500s)
    # --- ★ 修正ここまで ---

    RM = PHYSICAL_CONSTANTS['RM']
    R_MAX = RM * settings['GRID_RADIUS_RM']  # シミュレーション領域の最大半径
    pos, vel, weight = args['particle_state']['pos'], args['particle_state']['vel'], args['particle_state']['weight']

    # 光電離の寿命 (1AUでの値 * AU^2)
    tau_ionization = settings['T1AU'] * AU ** 2

    # --- ★ 修正点 2 (要件7) ---
    # 積分ステップ数を duration / DT で計算 (例: 500s / 500s = 1)
    num_steps = int(duration / DT)
    if num_steps < 1:  # duration が DT より小さい場合 (通常は発生しない)
        num_steps = 1
        DT = duration  # DT 自体を duration に合わせる
    # --- ★ 修正ここまで ---

    pos_at_start_of_step = pos.copy()

    # 4次ルンゲ＝クッタ法 (RK4) で軌道を積分
    # ★ このループが num_steps 回 (例: 1回) 実行される
    for _ in range(num_steps):
        pos_at_start_of_step = pos.copy()

        # --- 光電離判定 ---
        # ★ DT (500秒など) 全体での電離確率を計算
        if pos[0] > 0:  # 昼側 (X > 0) にいる場合のみ
            if np.random.random() < (1.0 - np.exp(-DT / tau_ionization)):
                return {'status': 'ionized', 'final_state': None}

        # --- RK4 積分 ---
        # ★ DT が 500s などの値になる
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


def main_snapshot_simulation():
    """
    シミュレーション全体を制御するメイン関数。
    （動的表面密度モデル + 全放出プロセス）
    ★修正点: 平衡モード(na_eq)を使用したセルは、次のステップの表面密度を
             強制的に na_eq で上書きし、在庫の持ち越し（ゾンビ在庫）を防ぐ。
    """
    start_time = time.time()

    # --- 1. シミュレーション設定 ---
    OUTPUT_DIRECTORY = r"./SimulationResult_202510"
    N_LON_FIXED, N_LAT = 72, 36
    INITIAL_SURFACE_DENSITY_PER_M2 = 7.5e14 * (100.0 ** 2) * 0.0053

    # ★要件1, 7: 固定タイムステップ [s]
    MAIN_TIME_STEP_SEC = 500.0

    # [atoms/m^2] 枯渇したセルをタイムステップ計算から除外する閾値
    MIN_DENSITY_FOR_TIMESTEP = 1e10

    # --- ★修正: 低速粒子の追跡省略設定 (0.0にして全追跡) ---
    LOW_SPEED_THRESHOLD_M_S = 0.0

    # シミュレーション時間設定
    SPIN_UP_YEARS = 1.0
    TOTAL_SIM_YEARS = 1.0
    TARGET_TAA_DEGREES = np.arange(0, 360, 1)

    # 可変重み法 (予算配分方式) の設定
    TARGET_SPS_TD = 1000
    TARGET_SPS_PSD = 1000
    TARGET_SPS_SWS = 1000
    TARGET_SPS_MMV = 1000

    # --- 粒子生成モデル ---
    F_UV_1AU_PER_M2 = 1.5e14 * (100.0 ** 2)
    Q_PSD_M2 = 2.0e-20 / (100.0 ** 2)
    TEMP_PSD = 1500.0
    TEMP_MMV = 3000.0

    # --- SWS (CPS) ---
    SWS_PARAMS = {
        'SW_DENSITY_1AU': 10.0 * (100.0) ** 3,
        'SW_VELOCITY': 400.0 * 1000.0,
        'YIELD_EFF': 0.06,
        'U_eV': 0.27,
        'DENSITY_REF_M2': 7.5e14 * (100.0) ** 2,
        'LON_MIN_RAD': np.deg2rad(-40.0),
        'LON_MAX_RAD': np.deg2rad(40.0),
        'LAT_N_MIN_RAD': np.deg2rad(30.0),
        'LAT_N_MAX_RAD': np.deg2rad(60.0),
        'LAT_S_MIN_RAD': np.deg2rad(-60.0),
        'LAT_S_MAX_RAD': np.deg2rad(-30.0),
    }

    # --- 出力グリッド ---
    GRID_RESOLUTION = 101
    GRID_MAX_RM = 5.0
    USE_SOLAR_GRAVITY = True
    USE_CORIOLIS_FORCES = True

    settings = {
        'BETA': 0.5,
        'T1AU': 168918.0,
        'DT': MAIN_TIME_STEP_SEC,
        'GRID_RADIUS_RM': GRID_MAX_RM + 1.0,
        'USE_SOLAR_GRAVITY': USE_SOLAR_GRAVITY,
        'USE_CORIOLIS_FORCES': USE_CORIOLIS_FORCES
    }

    # --- 2. 初期化 ---
    run_name = f"DynamicGrid{N_LON_FIXED}x{N_LAT}_8.0"
    target_output_dir = os.path.join(OUTPUT_DIRECTORY, run_name)
    os.makedirs(target_output_dir, exist_ok=True)
    print(f"結果は '{target_output_dir}' に保存されます。")

    MERCURY_YEAR_SEC = 87.97 * 24 * 3600

    # グリッド初期化
    lon_edges_fixed = np.linspace(-np.pi, np.pi, N_LON_FIXED + 1)
    lat_edges_fixed = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    dlon_fixed = lon_edges_fixed[1] - lon_edges_fixed[0]
    cell_areas_m2 = (PHYSICAL_CONSTANTS['RM'] ** 2) * dlon_fixed * \
                    (np.sin(lat_edges_fixed[1:]) - np.sin(lat_edges_fixed[:-1]))
    surface_density_grid = np.full((N_LON_FIXED, N_LAT), INITIAL_SURFACE_DENSITY_PER_M2, dtype=np.float64)

    # --- 3. ファイル読み込み ---
    try:
        spec_data_np = np.loadtxt('SolarSpectrum_Na0.txt', usecols=(0, 3))
        orbit_file_name = 'orbit2025_v6.txt'
        orbit_data = np.loadtxt(orbit_file_name)
    except FileNotFoundError as e:
        print(f"エラー: データファイル '{e.filename}' が見つかりません。");
        sys.exit()

    # 時間設定
    taa_col = orbit_data[:, 0]
    time_col = orbit_data[:, 2]
    idx_perihelion = np.argmin(np.abs(taa_col))
    t_start_run = time_col[idx_perihelion]
    t_end_run = t_start_run + (TOTAL_SIM_YEARS * MERCURY_YEAR_SEC)
    t_start_spinup = t_start_run - (SPIN_UP_YEARS * MERCURY_YEAR_SEC)

    # スペクトル
    sigma_const = PHYSICAL_CONSTANTS['E_CHARGE'] ** 2 / (
            4 * PHYSICAL_CONSTANTS['ME'] * PHYSICAL_CONSTANTS['C'] * PHYSICAL_CONSTANTS['EPSILON_0'])
    wl, gamma = spec_data_np[:, 0], spec_data_np[:, 1]
    if not np.all(np.diff(wl) > 0):
        sort_indices = np.argsort(wl)
        wl, gamma = wl[sort_indices], gamma[sort_indices]
    spec_data_dict = {'wl': wl, 'gamma': gamma,
                      'sigma0_perdnu2': sigma_const * 0.641,
                      'sigma0_perdnu1': sigma_const * 0.320,
                      'JL': 5.18e14}

    # --- 4. メインループ ---
    active_particles = []
    previous_taa = -1
    target_taa_idx = 0

    current_step_weight_td = 1.0
    current_step_weight_psd = 1.0
    current_step_weight_sws = 1.0
    current_step_weight_mmv = 1.0

    previous_atoms_gained_grid = np.zeros_like(surface_density_grid)

    t_sec = t_start_spinup

    with tqdm(total=int(t_end_run - t_start_spinup), desc="Time Evolution") as pbar:
        while t_sec < t_end_run:

            current_dt_main = MAIN_TIME_STEP_SEC
            TAA, AU, V_radial_ms, V_tangential_ms, subsolar_lon_rad_fixed = get_orbital_params(
                t_sec, orbit_data, MERCURY_YEAR_SEC
            )
            run_phase = "Spin-up" if t_sec < t_start_run else "Run"

            pbar.set_description(f"[{run_phase}] TAA={TAA:.1f} | N_act={len(active_particles)}")

            # 統計用変数リセット
            total_atoms_td_this_step = 0.0
            total_atoms_psd_this_step = 0.0
            total_atoms_sws_this_step = 0.0

            rate_psd_grid_s = np.zeros_like(surface_density_grid)
            rate_td_grid_s = np.zeros_like(surface_density_grid)
            rate_sws_grid_s = np.zeros_like(surface_density_grid)
            total_atoms_lost_grid = np.zeros_like(surface_density_grid)

            # ★★★ 追加: 平衡密度を記録するグリッド (初期値は NaN) ★★★
            na_eq_record_grid = np.full((N_LON_FIXED, N_LAT), np.nan)

            # MMV計算
            flux_mmv_per_m2_s = calculate_mmv_flux(AU)
            MERCURY_SURFACE_AREA_M2 = 4 * PHYSICAL_CONSTANTS['PI'] * (PHYSICAL_CONSTANTS['RM'] ** 2)
            n_atoms_mmv = flux_mmv_per_m2_s * MERCURY_SURFACE_AREA_M2 * current_dt_main

            # 共通定数
            F_UV_current_per_m2 = F_UV_1AU_PER_M2 / (AU ** 2)
            flux_sw_1au = SWS_PARAMS['SW_DENSITY_1AU'] * SWS_PARAMS['SW_VELOCITY']
            base_sputtering_rate_per_m2_s = (flux_sw_1au / (AU ** 2)) * SWS_PARAMS['YIELD_EFF']
            DENSITY_REF_M2 = SWS_PARAMS['DENSITY_REF_M2']

            # --- グリッドループ ---
            for i_lon in range(N_LON_FIXED):
                for i_lat in range(N_LAT):

                    # 座標
                    lon_fixed_rad = (lon_edges_fixed[i_lon] + lon_edges_fixed[i_lon + 1]) / 2
                    lat_rad = (lat_edges_fixed[i_lat] + lat_edges_fixed[i_lat + 1]) / 2

                    # (1) PSD Rate
                    rate_psd_per_s = 0.0
                    cos_Z = np.cos(lat_rad) * np.cos(lon_fixed_rad - subsolar_lon_rad_fixed)
                    if cos_Z > 0:
                        rate_psd_per_s = F_UV_current_per_m2 * Q_PSD_M2 * cos_Z
                        rate_psd_grid_s[i_lon, i_lat] = rate_psd_per_s

                    # (2) TD Rate (変更なし: 1.5eV, no cutoff で計算済みと仮定)
                    rate_td_per_s = 0.0
                    temp_k = calculate_surface_temperature_leblanc(lon_fixed_rad, lat_rad, AU, subsolar_lon_rad_fixed)
                    rate_td_per_s = calculate_thermal_desorption_rate(temp_k)
                    if rate_td_per_s > 0:
                        rate_td_grid_s[i_lon, i_lat] = rate_td_per_s

                    # (3) SWS Rate
                    rate_sws_per_s = 0.0
                    lon_sun_fixed_rad = (lon_fixed_rad - subsolar_lon_rad_fixed + PHYSICAL_CONSTANTS['PI']) % (
                            2 * PHYSICAL_CONSTANTS['PI']) - PHYSICAL_CONSTANTS['PI']
                    is_in_lon = (SWS_PARAMS['LON_MIN_RAD'] <= lon_sun_fixed_rad <= SWS_PARAMS['LON_MAX_RAD'])
                    is_in_lat_n = (SWS_PARAMS['LAT_N_MIN_RAD'] <= lat_rad <= SWS_PARAMS['LAT_N_MAX_RAD'])
                    is_in_lat_s = (SWS_PARAMS['LAT_S_MIN_RAD'] <= lat_rad <= SWS_PARAMS['LAT_S_MAX_RAD'])
                    if is_in_lon and (is_in_lat_n or is_in_lat_s):
                        rate_sws_per_s = (base_sputtering_rate_per_m2_s / DENSITY_REF_M2)
                        rate_sws_grid_s[i_lon, i_lat] = rate_sws_per_s

                    total_rate_per_s = rate_psd_per_s + rate_td_per_s + rate_sws_per_s

                    # --- タイムスケール判定 ---
                    local_timescale = np.inf
                    if total_rate_per_s > 1e-30:
                        local_timescale = 1.0 / total_rate_per_s

                    current_density_per_m2 = 0.0

                    # ★条件変更なし: 500s のまま
                    if local_timescale <= 500.0 and t_sec > t_start_spinup:
                        # 平衡モード
                        gain_flux = (previous_atoms_gained_grid[i_lon, i_lat] / cell_areas_m2[i_lat]) / current_dt_main
                        if total_rate_per_s > 0:
                            current_density_per_m2 = gain_flux / total_rate_per_s

                            # ★★★ 重要: この平衡密度を記録しておく (後で上書きに使用) ★★★
                            na_eq_record_grid[i_lon, i_lat] = current_density_per_m2
                        else:
                            current_density_per_m2 = 0.0
                    else:
                        # 時間発展モード
                        current_density_per_m2 = surface_density_grid[i_lon, i_lat]

                    if current_density_per_m2 <= 0:
                        continue

                    # --- 枯渇量計算 (変更なし: 線形近似) ---
                    atoms_available_in_cell = current_density_per_m2 * cell_areas_m2[i_lat]

                    if total_rate_per_s > 0:
                        # 線形近似
                        n_lost_linear = atoms_available_in_cell * total_rate_per_s * current_dt_main
                        n_lost = min(n_lost_linear, atoms_available_in_cell)

                        total_atoms_lost_grid[i_lon, i_lat] = n_lost

                        if rate_psd_per_s > 0:
                            total_atoms_psd_this_step += n_lost * (rate_psd_per_s / total_rate_per_s)
                        if rate_td_per_s > 0:
                            total_atoms_td_this_step += n_lost * (rate_td_per_s / total_rate_per_s)
                        if rate_sws_per_s > 0:
                            total_atoms_sws_this_step += n_lost * (rate_sws_per_s / total_rate_per_s)

            # (B) 重み決定 (省略: 変更なし)
            if total_atoms_td_this_step > 0:
                current_step_weight_td = total_atoms_td_this_step / TARGET_SPS_TD
            else:
                current_step_weight_td = 1.0
            if total_atoms_psd_this_step > 0:
                current_step_weight_psd = total_atoms_psd_this_step / TARGET_SPS_PSD
            else:
                current_step_weight_psd = 1.0
            if total_atoms_sws_this_step > 0:
                current_step_weight_sws = total_atoms_sws_this_step / TARGET_SPS_SWS
            else:
                current_step_weight_sws = 1.0
            if n_atoms_mmv > 0:
                current_step_weight_mmv = n_atoms_mmv / TARGET_SPS_MMV
            else:
                current_step_weight_mmv = 1.0

            # (C) 粒子生成ループ (省略: 変更なし)
            newly_launched_particles = []
            atoms_gained_grid = np.zeros_like(surface_density_grid)

            # (C-1) MMV
            if n_atoms_mmv > 0 and TARGET_SPS_MMV > 0:
                num_sps_float = TARGET_SPS_MMV
                num_sps_int = int(num_sps_float)
                if np.random.random() < (num_sps_float - num_sps_int): num_sps_int += 1
                if num_sps_int > 0:
                    M_rejection = 4.0 / 3.0
                    for _ in range(num_sps_int):
                        while True:
                            lon_rot = np.random.uniform(-np.pi, np.pi)
                            if np.random.random() < (1.0 - (1.0 / 3.0) * np.sin(lon_rot)) / M_rejection: break
                        lat_rot = np.arcsin(np.random.uniform(-1.0, 1.0))
                        pos_rot = lonlat_to_xyz(lon_rot, lat_rot, PHYSICAL_CONSTANTS['RM'])
                        norm = pos_rot / PHYSICAL_CONSTANTS['RM']
                        speed = sample_speed_from_flux_distribution(PHYSICAL_CONSTANTS['MASS_NA'], TEMP_MMV)
                        vel_rot = speed * transform_local_to_world(sample_lambertian_direction_local(), norm)
                        if speed < LOW_SPEED_THRESHOLD_M_S: continue
                        newly_launched_particles.append(
                            {'pos': pos_rot, 'vel': vel_rot, 'weight': current_step_weight_mmv})

            # (C-2) PSD, TD, SWS
            for i_lon in range(N_LON_FIXED):
                for i_lat in range(N_LAT):
                    n_total_lose = total_atoms_lost_grid[i_lon, i_lat]
                    if n_total_lose <= 0: continue
                    r_psd = rate_psd_grid_s[i_lon, i_lat]
                    r_td = rate_td_grid_s[i_lon, i_lat]
                    r_sws = rate_sws_grid_s[i_lon, i_lat]
                    total_r = r_psd + r_td + r_sws
                    if total_r <= 0: continue
                    lon_f = (lon_edges_fixed[i_lon] + lon_edges_fixed[i_lon + 1]) / 2
                    lat_f = (lat_edges_fixed[i_lat] + lat_edges_fixed[i_lat + 1]) / 2
                    temp_k = calculate_surface_temperature_leblanc(lon_f, lat_f, AU, subsolar_lon_rad_fixed)
                    procs = [('PSD', n_total_lose * (r_psd / total_r), TEMP_PSD, None, current_step_weight_psd),
                             ('TD', n_total_lose * (r_td / total_r), temp_k, None, current_step_weight_td),
                             ('SWS', n_total_lose * (r_sws / total_r), None, SWS_PARAMS['U_eV'],
                              current_step_weight_sws)]
                    for pname, n_atoms, T, U, w in procs:
                        if n_atoms <= 0 or w <= 0: continue
                        num_sps = int(n_atoms / w)
                        if np.random.random() < (n_atoms / w - num_sps): num_sps += 1
                        for _ in range(num_sps):
                            if pname == 'SWS':
                                E_ev = sample_thompson_sigmund_energy(U)
                                spd = np.sqrt(
                                    2.0 * E_ev * PHYSICAL_CONSTANTS['EV_TO_JOULE'] / PHYSICAL_CONSTANTS['MASS_NA'])
                            else:
                                spd = sample_speed_from_flux_distribution(PHYSICAL_CONSTANTS['MASS_NA'], T)
                            if spd < LOW_SPEED_THRESHOLD_M_S:
                                atoms_gained_grid[i_lon, i_lat] += w
                                continue
                            lon_rot = lon_f - subsolar_lon_rad_fixed
                            pos_rot = lonlat_to_xyz(lon_rot, lat_f, PHYSICAL_CONSTANTS['RM'])
                            norm = pos_rot / PHYSICAL_CONSTANTS['RM']
                            vel_rot = spd * transform_local_to_world(sample_lambertian_direction_local(), norm)
                            newly_launched_particles.append({'pos': pos_rot, 'vel': vel_rot, 'weight': w})

            active_particles.extend(newly_launched_particles)

            # (D) 粒子追跡 (省略: 変更なし)
            tasks = [{'settings': settings, 'spec': spec_data_dict, 'particle_state': p,
                      'orbit': (TAA, AU, V_radial_ms, V_tangential_ms, subsolar_lon_rad_fixed),
                      'duration': current_dt_main} for p in active_particles]
            next_active_particles = []
            if tasks:
                with Pool(processes=max(1, cpu_count() - 1)) as pool:
                    results = list(pool.imap(simulate_particle_for_one_step, tasks, chunksize=100))
                for res in results:
                    if res['status'] == 'alive':
                        next_active_particles.append(res['final_state'])
                    elif res['status'] == 'stuck':
                        pos = res['pos_at_impact']
                        lon = np.arctan2(pos[1], pos[0])
                        lat = np.arcsin(np.clip(pos[2] / np.linalg.norm(pos), -1, 1))
                        lon_fix = (lon + subsolar_lon_rad_fixed + np.pi) % (2 * np.pi) - np.pi
                        ix = np.searchsorted(lon_edges_fixed, lon_fix) - 1
                        iy = np.searchsorted(lat_edges_fixed, lat) - 1
                        if 0 <= ix < N_LON_FIXED and 0 <= iy < N_LAT:
                            atoms_gained_grid[ix, iy] += res['weight']
            active_particles = next_active_particles

            # --- 4e. 表面密度更新 (★ここを修正) ---

            # 1. まずは通常通りオイラー法で更新
            na_t = surface_density_grid
            gain = atoms_gained_grid / cell_areas_m2
            loss = total_atoms_lost_grid / cell_areas_m2
            na_next = na_t + gain - loss

            # 2. ★★★ ゾンビ在庫対策: 平衡モードだったセルを強制上書き ★★★
            # na_eq_record_grid が NaN でない場所 = 平衡計算が行われた場所
            mask_eq = ~np.isnan(na_eq_record_grid)

            # その場所だけ、計算結果を無視して平衡密度(na_eq)を代入する
            na_next[mask_eq] = na_eq_record_grid[mask_eq]

            # 3. クリップして更新
            surface_density_grid = np.clip(na_next, 0, None)

            # 次のステップ用に保存
            previous_atoms_gained_grid = atoms_gained_grid.copy()

            # (F) 保存判定 (省略: 変更なし)
            if TAA < previous_taa: target_taa_idx = 0
            if target_taa_idx < len(TARGET_TAA_DEGREES):
                tgt = TARGET_TAA_DEGREES[target_taa_idx]
                if (previous_taa < tgt <= TAA) or (
                        (tgt == 0) and (TAA < previous_taa or (TAA >= 0 and previous_taa < 0))):
                    if t_sec >= t_start_run:
                        pbar.write(f"\n>>> [Run] Saving at TAA={TAA:.1f} ({len(active_particles)} particles) <<<")
                        dgrid = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION, GRID_RESOLUTION), dtype=np.float32)
                        gmin, gmax = -GRID_MAX_RM * PHYSICAL_CONSTANTS['RM'], GRID_MAX_RM * PHYSICAL_CONSTANTS['RM']
                        csize = (gmax - gmin) / GRID_RESOLUTION
                        cvol = csize ** 3
                        for p in active_particles:
                            pos = p['pos']
                            ix = int((pos[0] - gmin) / csize)
                            iy = int((pos[1] - gmin) / csize)
                            iz = int((pos[2] - gmin) / csize)
                            if 0 <= ix < GRID_RESOLUTION and 0 <= iy < GRID_RESOLUTION and 0 <= iz < GRID_RESOLUTION:
                                dgrid[ix, iy, iz] += p['weight']
                        dgrid /= cvol
                        rel_t = t_sec - t_start_run
                        fname = f"density_grid_t{int(rel_t / 3600):05d}_taa{int(round(TAA)):03d}.npy"
                        np.save(os.path.join(target_output_dir, fname), dgrid)
                        fname_s = f"surface_density_t{int(rel_t / 3600):05d}_taa{int(round(TAA)):03d}.npy"
                        np.save(os.path.join(target_output_dir, fname_s), surface_density_grid)
                    target_taa_idx += 1
            previous_taa = TAA

            pbar.update(current_dt_main)
            t_sec += current_dt_main

    print(f"Finished. Total Time: {(time.time() - start_time) / 3600:.2f}h")

if __name__ == '__main__':
    # 実行前に設定やファイルを確認
    print("シミュレーションを開始します...")
    main_snapshot_simulation()