# -*- coding: utf-8 -*-
"""
水星ナトリウム大気 3次元時間発展モンテカルロシミュレーションコード
(熱脱離 - Thermal Desorption - 専用バージョン)

==============================================================================
概要
==============================================================================
このスクリプトは、水星のナトリウム大気のふるまいを、時間発展を考慮して
シミュレートする3次元モンテカルロ法に基づいたプログラムです。

Leblanc (2003) に基づき、粒子生成源として **熱脱離 (TD)** のみを考慮します。

表面からのナトリウム原子の生成、宇宙空間での運動、そして消滅過程を追跡します。

==============================================================================
座標系
==============================================================================
このシミュレーションは、「水星中心・太陽固定回転座標系」を採用しています。

-   **原点 (0, 0, 0)**: 水星の中心
-   **+X 軸**: 常に太陽の方向を指します (Sun-Mercury line)
-   **-Y 軸**: 水星の公転軌道面に含まれ、公転の進行方向を指します
-   **+Z 軸**: 軌道面に垂直な方向（公転の角運動量ベクトル方向）

==============================================================================
主な物理モデル (TD版)
==============================================================================
1.  **粒子生成 (Thermal Desorption, TD)**:
    -   局所的な表面温度 (Ts) とナトリウムの結合エネルギー (U) に基づき、
        アレニウスの式 (ν * exp(-U / kBTs)) に従って全表面から粒子が
        生成されます (事実上、高温の日照側が支配的)。
    -   本バージョンでは、表面密度は無限供給源（常に一定）と仮定しています。

2.  **初期速度**:
    -   放出される原子の速さ: 局所表面温度(Ts)における
        マクスウェル分布に従います。
    -   放出角度: 表面の法線方向を基準としたランバート（余弦則）分布に従います。

3.  **軌道計算 (4次ルンゲ＝クッタ法)**:
    -   粒子にかかる力として以下を考慮します。
        1.  水星の重力 (中心力)
        2.  太陽光の放射圧 (SRP, -X方向の力, ドップラーシフト考慮)
        3.  太陽の重力 (潮汐力として作用)
        4.  遠心力 (回転座標系による見かけの力)
        5.  コリオリ力 (回転座標系による見かけの力)
    -   軌道積分には4次のルンゲ＝クッタ（RK4）法を使用します。

4.  **消滅過程**:
    -   光電離: 太陽光に照らされている領域（+X側）を飛行する粒子は、
                 確率的にイオン化され、シミュレーションから除去されます。
    -   表面衝突: 表面に再衝突した粒子は、衝突地点の局所表面温度に応じた
                 吸着確率(sticking probability)で吸着（消滅）します。
                 吸着しなかった場合は、エネルギーを交換して熱的に再放出されます。
"""

import numpy as np
import sys
import os
from multiprocessing import Pool, cpu_count  # 並列処理のため
from tqdm import tqdm  # 進捗バー表示のため
import time

# ==============================================================================
# 物理定数 (SI単位系)
# ==============================================================================
# シミュレーション全体で使用される物理定数を辞書として一元管理します。
PHYSICAL_CONSTANTS = {
    'PI': np.pi,  # 円周率
    'AU': 1.496e11,  # 天文単位 (Astronomical Unit) [m]
    'MASS_NA': 22.98976928 * 1.66054e-27,  # ナトリウム原子の質量 [kg]
    'K_BOLTZMANN': 1.380649e-23,  # ボルツマン定数 [J/K]
    'GM_MERCURY': 2.2032e13,  # 水星の重力定数 G * M_Mercury [m^3/s^2]
    'RM': 2.440e6,  # 水星の半径 (Radius of Mercury) [m]
    'C': 299792458.0,  # 光速 [m/s]
    'H': 6.62607015e-34,  # プランク定数 [J·s]
    'E_CHARGE': 1.602176634e-19,  # 電気素量 [C]
    'ME': 9.1093897e-31,  # 電子の質量 [kg]
    'EPSILON_0': 8.854187817e-12,  # 真空の誘電率 [F/m]
    'G': 6.6743e-11,  # 万有引力定数 [N m^2/kg^2]
    'MASS_SUN': 1.989e30,  # 太陽の質量 [kg]
}

# --- 変更 (TD) ---
# 熱脱離 (TD) モデル用の物理定数を追加 (Leblanc 2003, Sec 2.1)
# 振動周波数 [1/s] (v in [237])
NU_VIBRATIONAL = 1.0e13
# 結合エネルギー [eV] (U=1.85 eV in [238])
E_BINDING_EV = 1.85
# 結合エネルギー [J]
E_BINDING_J = E_BINDING_EV * PHYSICAL_CONSTANTS['E_CHARGE']


# --- 変更 (TD) ここまで ---


# ==============================================================================
# 物理モデルに基づくヘルパー関数群
# ==============================================================================

def calculate_surface_temperature(lon_rad, lat_rad, AU, subsolar_lon_rad):
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
    表面温度に基づき、ナトリウム原子が表面に吸着する確率（付着確率）を計算します。

    温度が低いほど吸着しやすくなります (指数関数的に増加)。
    表面の多孔性(porosity)も考慮されます。

    Args:
        surface_temp_K (float): 衝突地点の表面温度 [K]

    Returns:
        float: 実効的な吸着確率 (0から1の範囲)
    """
    A = 0.08
    B = 458.0
    porosity = 0.8  # 表面の多孔性 (0-1)

    if surface_temp_K <= 0:
        return 1.0  # 物理的にありえないが、安全のため

    # 基本的な吸着確率 p = A * exp(B / T)
    p_stick = A * np.exp(B / surface_temp_K)

    # 多孔性を考慮した実効的な吸着確率
    p_stick_eff = p_stick / (1.0 - (1.0 - p_stick) * porosity)

    # 確率は1を超えることはない
    return min(p_stick_eff, 1.0)


def sample_maxwellian_speed(mass_kg, temp_k):
    """
    指定された温度の3次元マクスウェル分布に従う速さ（スカラー値）を
    サンプリングします。

    Args:
        mass_kg (float): 粒子の質量 [kg]
        temp_k (float): 温度 [K]

    Returns:
        float: サンプリングされた速さ [m/s]
    """
    # マクスウェル分布のスケールパラメータ (最確速とは異なる)
    # a = sqrt(kT/m)
    scale_param = np.sqrt(PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_k / mass_kg)

    # 3次元の正規分布から速度ベクトル(vx, vy, vz)を生成し、
    # その大きさ（速さ）を返すことでマクスウェル分布からのサンプリングと等価になる
    vx, vy, vz = np.random.normal(0, scale_param, 3)
    return np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)


def sample_lambertian_direction_local():
    """
    ランバート（余弦則）分布に従う方向ベクトルをローカル座標系で生成します。

    ローカル座標系では、Z軸が表面の法線方向（天頂方向）に対応します。
    cos(theta) に比例して方向が選ばれるため、法線に近い方向ほど
    選ばれやすくなります。

    Returns:
        np.ndarray: 3次元の方向ベクトル (ローカル座標系, 正規化済み) [x, y, z]
    """
    # 逆関数サンプリング法を使用
    u1, u2 = np.random.random(2)  # [0, 1) の一様乱数を2つ

    # 方位角 phi = 2 * pi * u1
    phi = 2 * PHYSICAL_CONSTANTS['PI'] * u1

    # 天頂角 theta = acos(sqrt(1 - u2))
    cos_theta = np.sqrt(1 - u2)
    sin_theta = np.sqrt(u2)  # sin(acos(sqrt(1-u2))) = sqrt(1 - (1-u2)) = sqrt(u2)

    # ローカル座標系 (x, y, z)
    return np.array([sin_theta * np.cos(phi),  # x = sin(theta)cos(phi)
                     sin_theta * np.sin(phi),  # y = sin(theta)sin(phi)
                     cos_theta])  # z = cos(theta)


def transform_local_to_world(local_vec, normal_vector):
    """
    ローカル座標系（法線がZ軸）のベクトルをワールド座標系に変換します。

    Args:
        local_vec (np.ndarray): ローカル座標系でのベクトル [x, y, z]
        normal_vector (np.ndarray): ワールド座標系での法線ベクトル
                                   （これがローカルZ軸に相当）

    Returns:
        np.ndarray: ワールド座標系に変換されたベクトル
    """
    # 1. ローカルZ軸（法線ベクトル）を定義
    local_z_axis = normal_vector / np.linalg.norm(normal_vector)

    # 2. ローカルX軸、Y軸を計算
    #    ワールド座標系のUPベクトル（[0,0,1] or [0,1,0]）と
    #    法線ベクトル(local_z_axis)の外積でローカルX軸を定義する

    # 任意のワールドUPベクトル（ただし法線と平行でないもの）
    world_up = np.array([0., 0., 1.])
    if np.allclose(local_z_axis, world_up) or np.allclose(local_z_axis, -world_up):
        # 法線がZ軸と平行な場合（極点）、UPベクトルをY軸に変更
        world_up = np.array([0., 1., 0.])

    # ローカルX軸 = UP x Z (外積)
    local_x_axis = np.cross(world_up, local_z_axis)
    local_x_axis /= np.linalg.norm(local_x_axis)

    # ローカルY軸 = Z x X (外積)
    local_y_axis = np.cross(local_z_axis, local_x_axis)
    # (local_x, local_y, local_z は直交基底をなす)

    # 3. ローカルベクトルをワールド座標系の基底に射影して合成
    return (local_vec[0] * local_x_axis +
            local_vec[1] * local_y_axis +
            local_vec[2] * local_z_axis)


def get_orbital_params(time_sec, orbit_data, mercury_year_sec):
    """
    指定された時刻における水星の軌道パラメータと太陽直下点経度を取得します。

    orbit2025_v5.txt のデータ構造に基づいて、Time列を基準に線形補間します。

    (注: この座標系では subsolar_lon_rad は物理的に意味を持ちませんが、
     元コードの互換性のために残されています)

    orbit_data の列定義 (orbit2025_v5.txt):
    - col 0: TAA[deg]
    - col 1: AU[-]
    - col 2: Time[s]  <- 補間のX軸として使用
    - col 3: V_radial_ms[m/s] (視線速度, 太陽に近づく方向が正)
    - col 4: V_tangential_ms[m/s] (接線速度, 公転進行方向が正)

    Args:
        time_sec (float): シミュレーション開始からの総経過時間 [s]
        orbit_data (np.ndarray): np.loadtxt('orbit2025_v5.txt') で読み込んだデータ
        mercury_year_sec (float): 水星の1公転周期の秒数 [s]

    Returns:
        tuple: 5つの値のタプル
            - taa (float): 真近点離角 [deg]
            - au (float): 太陽距離 [AU]
            - v_radial (float): 太陽への視線速度 [m/s]
            - v_tangential (float): 公転の接線速度 [m/s]
            - subsolar_lon_rad (float): 太陽直下点の経度 [rad] (※ダミー値)
    """
    ROTATION_PERIOD_SEC = 58.646 * 24 * 3600  # 水星の自転周期 [s]

    # シミュレーション時間を公転周期で割った余りを求め、
    # 軌道ファイルの時間範囲 (0 ~ T_sec) にマッピングする
    current_time_in_orbit = time_sec % mercury_year_sec

    # 補間の基準となる時間軸（ファイルの3列目、インデックス2）
    time_col = orbit_data[:, 2]

    # np.interp(x, xp, fp)
    # x: 補間したい点のX座標 (current_time_in_orbit)
    # xp: データのX座標の配列 (time_col)
    # fp: データのY座標の配列 (orbit_data[:, N])

    # 線形補間
    taa = np.interp(current_time_in_orbit, time_col, orbit_data[:, 0])  # TAA (Col 0)
    au = np.interp(current_time_in_orbit, time_col, orbit_data[:, 1])  # AU (Col 1)
    v_radial = np.interp(current_time_in_orbit, time_col, orbit_data[:, 3])  # V_radial (Col 3)
    v_tangential = np.interp(current_time_in_orbit, time_col, orbit_data[:, 4])  # V_tangential (Col 4)

    # (※注)
    # 太陽固定座標系では、太陽直下点経度は常に 0.0 です。
    # この関数で計算される `subsolar_lon_rad` は、水星の「自転」に基づいた
    # 惑星固定座標系での太陽直下点経度であり、
    # このシミュレーションの他の部分（粒子生成、表面衝突）では
    # *使用されません*（常に 0.0 がハードコードされています）。
    # 元コードのロジックを維持するため、計算自体は残しています。
    subsolar_lon_rad = (2 * np.pi * time_sec / ROTATION_PERIOD_SEC) % (2 * np.pi)

    return taa, au, v_radial, v_tangential, subsolar_lon_rad


def lonlat_to_xyz(lon_rad, lat_rad, radius):
    """
    惑星中心の経度・緯度 [rad] を三次元直交座標 [m] に変換します。
    (数学の標準的な球面座標系定義)
    - 経度 0 (lon=0): +X 軸上 (太陽方向)
    - 緯度 0 (lat=0): XY 平面上
    - 緯度 +pi/2 (lat=pi/2): +Z 軸上

    Args:
        lon_rad (float): 経度 [rad]
        lat_rad (float): 緯度 [rad]
        radius (float): 半径 [m]

    Returns:
        np.ndarray: 3次元座標 [x, y, z] [m]
    """
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return np.array([x, y, z])


# ==============================================================================
# コア追跡関数 (並列処理の対象)
# ==============================================================================

def _calculate_acceleration(pos, vel, V_radial_ms, V_tangential_ms, AU, spec_data, settings):
    """
    【シミュレーションの核】
    粒子にかかる総加速度（重力＋放射圧＋見かけの力）を計算します。

    座標系: 水星中心・太陽固定回転座標系
    - 太陽は +X 方向 (距離 r0)
    - 公転は +Y 方向
    - 回転軸は +Z 方向 (角速度 ω = V_tan / r0)

    Args:
        pos (np.ndarray): 粒子の位置ベクトル [x, y, z] [m]
        vel (np.ndarray): 粒子の速度ベクトル [vx, vy, vz] [m/s]
        V_radial_ms (float): 水星の太陽への視線速度 [m/s] (近づく方向が正)
        V_tangential_ms (float): 水星の公転接線速度 [m/s] (+Y方向が正)
        AU (float): 太陽距離 [天文単位]
        spec_data (dict): 太陽スペクトルデータ
        settings (dict): シミュレーション設定フラグ

    Returns:
        np.ndarray: 総加速度ベクトル [ax, ay, az] [m/s^2]
    """
    x, y, z = pos
    r0 = AU * PHYSICAL_CONSTANTS['AU']  # 水星-太陽間距離 [m]

    # --- 1. 太陽放射圧 (Solar Radiation Pressure, SRP) ---

    # ドップラーシフト計算用の相対速度
    # 太陽は+X方向におり、光は-X方向に進んでくる。
    # 粒子のX方向速度(vel[0])と水星のX方向視線速度(V_radial_ms)を考慮。
    # V_radial_ms は「近づく速度」が正。
    # 粒子から見た太陽の相対速度 = 粒子の速度 - 水星の速度
    # 太陽に近づく方向を正とすると、(vel[0]) - (-V_radial_ms) = vel[0] + V_radial_ms
    velocity_for_doppler = vel[0] + V_radial_ms

    # ドップラーシフト後の波長を計算
    w_na_d2 = 589.1582e-9 * (1.0 + velocity_for_doppler / PHYSICAL_CONSTANTS['C'])
    w_na_d1 = 589.7558e-9 * (1.0 + velocity_for_doppler / PHYSICAL_CONSTANTS['C'])

    b = 0.0  # 放射圧による加速度 [m/s^2]
    wl, gamma, sigma0_perdnu2, sigma0_perdnu1, JL = spec_data.values()

    # スペクトルデータの範囲内かチェック
    if (wl[0] * 1e-9 <= w_na_d2 < wl[-1] * 1e-9) and \
            (wl[0] * 1e-9 <= w_na_d1 < wl[-1] * 1e-9):
        # ドップラーシフト後の波長に対応する太陽フラックスを補間
        gamma2 = np.interp(w_na_d2 * 1e9, wl, gamma)
        gamma1 = np.interp(w_na_d1 * 1e9, wl, gamma)

        # フラックス計算 (詳細は元コードの物理に基づく)
        F_lambda_1AU_m = JL * 1e4 * 1e9
        F_lambda_d1_at_Mercury = F_lambda_1AU_m / (AU ** 2) * gamma1
        F_nu_d1 = F_lambda_d1_at_Mercury * (w_na_d1 ** 2) / PHYSICAL_CONSTANTS['C']
        J1 = sigma0_perdnu1 * F_nu_d1
        F_lambda_d2_at_Mercury = F_lambda_1AU_m / (AU ** 2) * gamma2
        F_nu_d2 = F_lambda_d2_at_Mercury * (w_na_d2 ** 2) / PHYSICAL_CONSTANTS['C']
        J2 = sigma0_perdnu2 * F_nu_d2

        # 加速度 b = (1/m) * (運動量変化) = (1/m) * (J1*h*nu1 + J2*h*nu2)
        b = 1 / PHYSICAL_CONSTANTS['MASS_NA'] * (
                (PHYSICAL_CONSTANTS['H'] / w_na_d1) * J1 + (PHYSICAL_CONSTANTS['H'] / w_na_d2) * J2)

    # 水星の影 (太陽は+X方向なので、影は-X方向)
    if x < 0 and np.sqrt(y ** 2 + z ** 2) < PHYSICAL_CONSTANTS['RM']:
        b = 0.0  # 影の中では放射圧は0

    # 放射圧は太陽光が来る方向（-X）へ粒子を押す
    accel_srp = np.array([-b, 0.0, 0.0])

    # --- 2. 水星の重力 ---
    r_sq = np.sum(pos ** 2)
    accel_g = -PHYSICAL_CONSTANTS['GM_MERCURY'] * pos / (r_sq ** 1.5) if r_sq > 0 else np.array([0., 0., 0.])

    # --- 3. 太陽の重力 ---
    accel_sun = np.array([0.0, 0.0, 0.0])
    if settings.get('USE_SOLAR_GRAVITY', False):
        G = PHYSICAL_CONSTANTS['G']
        M_SUN = PHYSICAL_CONSTANTS['MASS_SUN']
        # 太陽の位置ベクトル: R_sun = [r0, 0, 0]
        # 粒子から太陽へのベクトル: r_ps = R_sun - pos
        r_ps_vec = np.array([r0 - pos[0], -pos[1], -pos[2]])
        r_ps_mag_sq = np.sum(r_ps_vec ** 2)
        if r_ps_mag_sq > 0:
            # 万有引力の法則 F = G * M_sun * m_Na / |r_ps|^2
            # 加速度 a = F / m_Na = G * M_sun * r_ps_vec / |r_ps|^3
            accel_sun = (G * M_SUN * r_ps_vec) / (r_ps_mag_sq ** 1.5)

    # --- 4. 見かけの力 (コリオリ力・遠心力) ---
    accel_coriolis = np.array([0.0, 0.0, 0.0])
    accel_centrifugal = np.array([0.0, 0.0, 0.0])
    if settings.get('USE_CORIOLIS_FORCES', False):
        if r0 > 0:
            # 公転の角速度ベクトル ω = [0, 0, ω_val]
            # ω_val = V_tan / r0
            omega_val = V_tangential_ms / r0
            omega_sq = omega_val ** 2  # ω^2

            # (a) 遠心力: a_cen = -ω x (ω x r)
            # a_cen = [ω^2*x, ω^2*y, 0]
            accel_centrifugal = np.array([
                # omega_sq * pos[0],
                omega_val ** 2 * (pos[0] - r0),
                omega_sq * pos[1],
                0.0
            ])

            # (b) コリオリ力: a_cor = -2 * (ω x v)
            # ω x v = [-ω*vy, ω*vx, 0]
            # a_cor = [2*ω*vy, -2*ω*vx, 0]
            two_omega = 2 * omega_val
            accel_coriolis = np.array([
                two_omega * vel[1],  # +2ω * v_y
                -two_omega * vel[0],  # -2ω * v_x
                0.0
            ])

    # --- 総加速度 ---
    # 全ての力を合計して返す
    return accel_srp + accel_g + accel_sun + accel_centrifugal + accel_coriolis


def simulate_particle_for_one_step(args):
    """
    一個のスーパーパーティクルを、指定された時間 (duration) だけ追跡します。
    この関数は `multiprocessing.Pool` によって並列処理される
    ワーカー関数です。

    Args:
        args (dict): シミュレーションに必要な全てのパラメータを含む辞書。
            - 'settings': 共通設定の辞書
            - 'spec': スペクトルデータの辞書
            - 'orbit': 現在時刻の軌道パラメータ (5-tuple)
            - 'duration': この粒子を追跡する時間 [s] (メインループのTIME_STEP_SEC)
            - 'particle_state': 追跡対象の粒子の状態 {'pos', 'vel', 'weight'}

    Returns:
        dict: 粒子の最終状態を示す辞書。
            - 'status' (str): 粒子の結末 ('alive', 'ionized', 'escaped', 'stuck')
            - 'final_state' (dict or None): 'alive' の場合のみ、
                                            粒子の最終状態 {'pos', 'vel', 'weight'}
    """
    # ---------------------------------
    # 1. 引数の展開
    # ---------------------------------
    settings, spec_data = args['settings'], args['spec']
    TAA, AU, V_radial_ms, V_tangential_ms, subsolar_lon_rad = args['orbit']
    duration, DT = args['duration'], settings['DT']  # この関数の実行時間, 積分ステップ
    RM = PHYSICAL_CONSTANTS['RM']
    R_MAX = RM * settings['GRID_RADIUS_RM']  # シミュレーション空間の最大半径
    pos, vel, weight = args['particle_state']['pos'], args['particle_state']['vel'], args['particle_state']['weight']

    # 1AUでの電離寿命(T1AU)を、現在の太陽距離(AU)にスケーリング
    # 太陽フラックスは 1/AU^2 に比例するので、寿命は AU^2 に比例する
    tau_ionization = settings['T1AU'] * AU ** 2

    num_steps = int(duration / DT)  # このワーカーで実行する積分ステップ数

    # ---------------------------------
    # 2. 時間積分ループ (RK4)
    # ---------------------------------
    for _ in range(num_steps):

        # --- 2a. 光電離判定 ---
        # 太陽光が当たっている側 (+X側) にいるか？
        if pos[0] > 0:
            # この時間ステップDTの間に電離する確率 P = 1 - exp(-DT / tau)
            if np.random.random() < (1.0 - np.exp(-DT / tau_ionization)):
                return {'status': 'ionized', 'final_state': None}  # 消滅

        # --- 2b. 4次ルンゲ＝クッタ法 (RK4) による軌道積分 ---
        pos_prev = pos.copy()  # 衝突判定のために1ステップ前の位置を保持

        # k1 (現時点での加速度)
        k1_vel = DT * _calculate_acceleration(pos, vel, V_radial_ms, V_tangential_ms, AU, spec_data, settings)
        k1_pos = DT * vel
        # k2 (0.5*DT後の中間点での加速度)
        k2_vel = DT * _calculate_acceleration(pos + 0.5 * k1_pos, vel + 0.5 * k1_vel, V_radial_ms, V_tangential_ms, AU,
                                              spec_data, settings)
        k2_pos = DT * (vel + 0.5 * k1_vel)
        # k3 (0.5*DT後の中間点での加速度, k2の傾きを使用)
        k3_vel = DT * _calculate_acceleration(pos + 0.5 * k2_pos, vel + 0.5 * k2_vel, V_radial_ms, V_tangential_ms, AU,
                                              spec_data, settings)
        k3_pos = DT * (vel + 0.5 * k2_vel)
        # k4 (DT後の終点での加速度, k3の傾きを使用)
        k4_vel = DT * _calculate_acceleration(pos + k3_pos, vel + k3_vel, V_radial_ms, V_tangential_ms, AU, spec_data,
                                              settings)
        k4_pos = DT * (vel + k3_vel)

        # 最終的な位置と速度の更新 (k1, k2, k3, k4 の重み付き平均)
        pos += (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos) / 6.0
        vel += (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel) / 6.0

        # --- 2c. 境界条件の判定 ---
        r_current = np.linalg.norm(pos)  # 現在の水星中心からの距離

        # (i) シミュレーション領域外への脱出
        if r_current > R_MAX:
            return {'status': 'escaped', 'final_state': None}  # 消滅

        # (ii) 表面への衝突
        if r_current <= RM:
            # 衝突地点の経度・緯度を計算 (1ステップ前の位置 pos_prev を使用)
            impact_lon = np.arctan2(pos_prev[1], pos_prev[0])  # (y, x)
            impact_lat = np.arcsin(np.clip(pos_prev[2] / np.linalg.norm(pos_prev), -1.0, 1.0))

            # 衝突地点の表面温度を計算
            # (太陽固定座標系なので、太陽直下点経度は 0.0)
            temp_at_impact = calculate_surface_temperature(impact_lon, impact_lat, AU, 0.0)

            # (ii-a) 吸着して消滅
            if np.random.random() < calculate_sticking_probability(temp_at_impact):
                return {'status': 'stuck', 'final_state': None}  # 消滅

            # (ii-b) 熱的に再放出（バウンド）
            else:
                # 入射エネルギー
                E_in = 0.5 * PHYSICAL_CONSTANTS['MASS_NA'] * np.sum(vel ** 2)
                # 表面の熱エネルギー
                E_T = PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_at_impact

                # 熱緩和係数(BETA)に基づいて衝突後のエネルギーを決定
                E_out = settings['BETA'] * E_T + (1.0 - settings['BETA']) * E_in
                v_out_speed = np.sqrt(2 * E_out / PHYSICAL_CONSTANTS['MASS_NA']) if E_out > 0 else 0.0

                # 放出方向を計算
                impact_normal = pos_prev / np.linalg.norm(pos_prev)  # 衝突地点の法線
                rebound_direction = transform_local_to_world(sample_lambertian_direction_local(), impact_normal)

                # 速度と位置を更新
                vel = v_out_speed * rebound_direction
                pos = RM * impact_normal  # 粒子を表面（半径RM）に戻す

                continue  # 次の積分ステップへ

    # ---------------------------------
    # 3. 生き残り
    # ---------------------------------
    # duration (e.g., 1000秒) の間、消滅せずに生き延びた場合
    return {'status': 'alive', 'final_state': {'pos': pos, 'vel': vel, 'weight': weight}}


# ==============================================================================
# メイン制御関数
# ==============================================================================

def main_snapshot_simulation():
    """
    シミュレーション全体を制御するメイン関数。
    """
    start_time = time.time()  # ベンチマーク用

    # --- 1. シミュレーション設定 ---

    # 出力ディレクトリ
    OUTPUT_DIRECTORY = r"./SimulationResult_202510"
    # 表面グリッドの分割数（粒子生成用）
    N_LON, N_LAT = 48, 24

    # --- 変更 (TD) ---
    # 表面密度 [atoms/m^2] (無限供給源モデル)
    # Leblanc 2003 [138] に基づく:
    # 表面総原子密度 7.5e14 atoms/cm^2 = 7.5e18 atoms/m^2
    # Na含有率 c_Na = 0.0053
    CONSTANT_SURFACE_DENSITY = (7.5e18) * 0.0053  # 約 3.975e16 [atoms/m^2]
    # --- 変更 (TD) ここまで ---

    # スピンアップ（助走）期間 [水星年]
    SPIN_UP_YEARS = 0.1
    # メインループの時間刻み [s]
    TIME_STEP_SEC = 1000
    # 総シミュレーション時間 [水星年]
    TOTAL_SIM_YEARS = 1.0
    # スナップショットを保存するTAA [度] (0, 1, 2, ..., 359度)
    TARGET_TAA_DEGREES = np.arange(0, 360, 1)
    # 超粒子1個が表す現実の原子数
    ATOMS_PER_SUPERPARTICLE = 1e27

    # --- 削除 (PSD) ---
    # F_UV_1AU_PER_M2, Q_PSD_M2, SOURCE_TEMPERATURE は TD には不要
    # --- 削除 (PSD) ここまで ---

    # --- 出力グリッドの設定 ---
    # 立方体グリッドの解像度（各軸のセル数, 奇数推奨）
    GRID_RESOLUTION = 101
    # グリッドの最大範囲（水星半径単位, 例: 5.0 -> -5RMから+5RMまで）
    GRID_MAX_RM = 5.0

    # --- 物理モデルのフラグ ---
    USE_SOLAR_GRAVITY = True  # 太陽の重力を考慮するか
    USE_CORIOLIS_FORCES = True  # コリオリ力・遠心力を考慮するか

    # --- その他の設定 (settings辞書にまとめる) ---
    settings = {
        'BETA': 0.5,  # 表面衝突時の熱緩和係数 β
        'T1AU': 168918.0,  # 1AUでの光電離寿命 [s]
        'DT': 1000.0,  # 軌道積分の時間刻み [s] (TIME_STEP_SECと一致させる)
        'N_LON': N_LON, 'N_LAT': N_LAT,
        # 粒子が脱出する半径（グリッドより少し大きく）
        'GRID_RADIUS_RM': GRID_MAX_RM + 1.0,
        'USE_SOLAR_GRAVITY': USE_SOLAR_GRAVITY,
        'USE_CORIOLIS_FORCES': USE_CORIOLIS_FORCES
    }

    # 出力ディレクトリの準備
    # --- 変更 (TD) ---
    run_name = f"Grid{GRID_RESOLUTION}_Range{int(GRID_MAX_RM)}RM_SP{ATOMS_PER_SUPERPARTICLE:.0e}_TD"
    # --- 変更 (TD) ここまで ---

    target_output_dir = os.path.join(OUTPUT_DIRECTORY, run_name)
    os.makedirs(target_output_dir, exist_ok=True)
    print(f"結果は '{target_output_dir}' に保存されます。")
    print(f"--- 物理モデル設定 (TD Only) ---")
    print(f"Binding Energy: {E_BINDING_EV} eV")
    print(f"Vib. Frequency: {NU_VIBRATIONAL:.1e} s^-1")
    print(f"Na Surface Density: {CONSTANT_SURFACE_DENSITY:.3e} atoms/m^2")
    print(f"Solar Gravity: {USE_SOLAR_GRAVITY}")
    print(f"Coriolis/Centrifugal: {USE_CORIOLIS_FORCES}")
    print(f"--------------------------------")

    # --- 2. シミュレーションの初期化 ---

    # 水星の1年の秒数
    MERCURY_YEAR_SEC = 87.97 * 24 * 3600

    # 表面グリッドの定義（粒子生成用）
    lon_edges = np.linspace(-np.pi, np.pi, N_LON + 1)  # 経度境界
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)  # 緯度境界
    dlon = lon_edges[1] - lon_edges[0]  # 経度幅 [rad]
    # 各緯度帯のセル面積 [m^2]
    cell_areas_m2 = (PHYSICAL_CONSTANTS['RM'] ** 2) * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))

    # --- 3. 外部ファイルの読み込み ---
    try:
        spec_data_np = np.loadtxt('SolarSpectrum_Na0.txt', usecols=(0, 3))
        orbit_file_name = 'orbit2025_v5.txt'
        orbit_data = np.loadtxt(orbit_file_name)

    except FileNotFoundError as e:
        print(f"エラー: データファイル '{e.filename}' が見つかりません。");
        sys.exit()

    # 軌道ファイルの列数チェック
    if orbit_data.shape[1] < 5:
        print(f"エラー: '{orbit_file_name}' の列が不足しています。")
        print(f"TAA, AU, Time, V_radial, V_tangential の5列が必要です。")
        sys.exit()

    # --- 軌道ファイルの TAA=0 (近日点) に最も近い時刻 [s] を見つける ---
    taa_col = orbit_data[:, 0]
    time_col = orbit_data[:, 2]

    # TAAが最小値(ほぼ0度)をとるインデックスを見つける
    idx_perihelion = np.argmin(np.abs(taa_col))
    # その時刻 [s] を「RUN開始の基準時刻」とする
    t_start_run = time_col[idx_perihelion]

    # 本番の終了時刻 [s]
    t_end_run = t_start_run + (TOTAL_SIM_YEARS * MERCURY_YEAR_SEC)

    # スピンアップの開始時刻 [s]
    # (RUN開始時刻から、指定されたスピンアップ年数だけ遡る)
    t_start_spinup = t_start_run - (SPIN_UP_YEARS * MERCURY_YEAR_SEC)

    # メインループで回す時間ステップの配列
    time_steps = np.arange(t_start_spinup, t_end_run, TIME_STEP_SEC)

    print(f"--- 時間設定 ---")
    print(f"軌道ファイル上のTAA=0 (近日点) 時刻: {t_start_run:.1f} s")
    print(f"スピンアップ開始時刻: {t_start_spinup:.1f} s ({-SPIN_UP_YEARS} 年前)")
    print(f"RUN開始時刻 (TAA=0): {t_start_run:.1f} s")
    print(f"RUN終了時刻: {t_end_run:.1f} s (+{TOTAL_SIM_YEARS} 年後)")
    print(f"------------------")

    # スペクトルデータの前処理 (波長でソートされていることを保証)
    wl, gamma = spec_data_np[:, 0], spec_data_np[:, 1]
    if not np.all(np.diff(wl) > 0):  # もしソートされていなければ
        sort_indices = np.argsort(wl)
        wl, gamma = wl[sort_indices], gamma[sort_indices]

    # 放射圧計算用の定数を事前に計算し、辞書にまとめる
    sigma_const = PHYSICAL_CONSTANTS['E_CHARGE'] ** 2 / (
            4 * PHYSICAL_CONSTANTS['ME'] * PHYSICAL_CONSTANTS['C'] * PHYSICAL_CONSTANTS['EPSILON_0'])
    spec_data_dict = {'wl': wl, 'gamma': gamma,
                      'sigma0_perdnu2': sigma_const * 0.641,
                      'sigma0_perdnu1': sigma_const * 0.320,
                      'JL': 5.18e14}

    # --- 4. メインループ (時間発展) ---

    active_particles = []  # 現在シミュレーション空間にいる全粒子リスト
    previous_taa = -1  # TAAが360->0に戻ったことを検出するため
    target_taa_idx = 0  # 次に保存すべきTAAのインデックス

    # tqdmで進捗バーを表示
    with tqdm(total=len(time_steps), desc="Time Evolution") as pbar:
        # t_sec は t_start_spinup から t_end_run まで進む
        for t_sec in time_steps:

            # --- 4a. 現在時刻の軌道パラメータを取得 ---
            TAA, AU, V_radial_ms, V_tangential_ms, subsolar_lon_rad_dummy = get_orbital_params(
                t_sec, orbit_data, MERCURY_YEAR_SEC
            )

            run_phase = "Spin-up" if t_sec < t_start_run else "Run"
            pbar.set_description(f"[{run_phase}] TAA={TAA:.1f} | N_particles={len(active_particles)}")

            # --- 4b. 表面から新しい粒子を生成 (TD: 熱脱離) ---
            newly_launched_particles = []

            # ボルツマン定数を事前に取得
            k_B = PHYSICAL_CONSTANTS['K_BOLTZMANN']

            for i_lon in range(N_LON):
                for i_lat in range(N_LAT):
                    lon_center_rad = (lon_edges[i_lon] + lon_edges[i_lon + 1]) / 2
                    lat_center_rad = (lat_edges[i_lat] + lat_edges[i_lat + 1]) / 2

                    # 太陽固定座標系なので、太陽直下点経度は常に 0.0
                    T_s = calculate_surface_temperature(lon_center_rad, lat_center_rad, AU, 0.0)

                    # 最低温度(100K)では脱離率はほぼ0なので、計算をスキップ
                    if T_s <= 100.0:
                        continue

                    # 熱脱離の頻度 f_TD = ν * exp(-U / (kB * T_s)) [1/s]
                    exponent = -E_BINDING_J / (k_B * T_s)
                    # exp(-200) など、極小値はオーバーフロー/アンダーフローせず 0.0 になる
                    if exponent < -700:  # approx exp(-700) is ~1e-304
                        f_TD = 0.0
                    else:
                        f_TD = NU_VIBRATIONAL * np.exp(exponent)

                    if f_TD == 0.0:
                        continue

                    # 脱離率 R_TD = f_TD * N_Na [atoms/m^2/s]
                    desorption_rate_per_m2_s = f_TD * CONSTANT_SURFACE_DENSITY

                    # このタイムステップで、このセルから生成される原子数
                    n_atoms_to_desorb = desorption_rate_per_m2_s * cell_areas_m2[i_lat] * TIME_STEP_SEC

                    if n_atoms_to_desorb <= 0:
                        continue

                    # 生成すべき超粒子(SP)の数を計算 (確率的)
                    num_sps_to_launch_float = n_atoms_to_desorb / ATOMS_PER_SUPERPARTICLE
                    num_to_launch_int = int(num_sps_to_launch_float)
                    if np.random.random() < (num_sps_to_launch_float - num_to_launch_int):
                        num_to_launch_int += 1

                    if num_to_launch_int == 0:
                        continue

                    # 計算された数の超粒子を生成
                    for _ in range(num_to_launch_int):
                        # セル内でランダムな位置を経度・緯度で決定
                        # (緯度は面積が均等になるよう sin(lat) でランダム化)
                        random_lon_rad = np.random.uniform(lon_edges[i_lon], lon_edges[i_lon + 1])
                        sin_lat_min, sin_lat_max = np.sin(lat_edges[i_lat]), np.sin(lat_edges[i_lat + 1])
                        random_lat_rad = np.arcsin(np.random.uniform(sin_lat_min, sin_lat_max))

                        # 初期位置 (ワールド座標系)
                        initial_pos = lonlat_to_xyz(random_lon_rad, random_lat_rad, PHYSICAL_CONSTANTS['RM'])
                        surface_normal = initial_pos / np.linalg.norm(initial_pos)

                        # --- 変更 (TD) ---
                        # 初期速度: 局所表面温度 T_s でのマクスウェル分布に従う
                        speed = sample_maxwellian_speed(PHYSICAL_CONSTANTS['MASS_NA'], T_s)
                        # --- 変更 (TD) ここまで ---

                        # 初期速度ベクトル (ローカル座標系 -> ワールド座標系)
                        initial_vel = speed * transform_local_to_world(sample_lambertian_direction_local(),
                                                                       surface_normal)

                        # 新しい粒子をリストに追加
                        newly_launched_particles.append({
                            'pos': initial_pos,
                            'vel': initial_vel,
                            'weight': ATOMS_PER_SUPERPARTICLE
                        })

            # アクティブな粒子リストに、新しく生成された粒子を追加
            active_particles.extend(newly_launched_particles)

            # --- 4c. 全ての粒子を1ステップ進める (並列処理) ---
            tasks = [{'settings': settings, 'spec': spec_data_dict, 'particle_state': p,
                      'orbit': (TAA, AU, V_radial_ms, V_tangential_ms, subsolar_lon_rad_dummy),
                      'duration': TIME_STEP_SEC} for p in
                     active_particles]

            next_active_particles = []
            if tasks:
                # CPUコア数-1 を使って並列処理プールを作成
                with Pool(processes=max(1, cpu_count() - 1)) as pool:
                    # imap で非同期にタスクを実行し、結果をリストに集める
                    # chunksizeを調整してパフォーマンスを最適化 (e.g., 100)
                    results = list(pool.imap(simulate_particle_for_one_step, tasks, chunksize=100))

                # 'alive' (生存) だった粒子の最終状態のみを次ステップのリストに追加
                for res in results:
                    if res['status'] == 'alive':
                        next_active_particles.append(res['final_state'])

            active_particles = next_active_particles

            # --- 4d. スナップショット保存判定 (バグ修正済み) ---
            save_this_step = False  # 1. 毎ステップ必ず変数を初期化する

            # TAAが 359 -> 0 のようにリセットされた場合、ターゲットをリセット
            if TAA < previous_taa:
                target_taa_idx = 0

            # まだ保存すべきTAAが残っているか？
            if target_taa_idx < len(TARGET_TAA_DEGREES):
                current_target_taa = TARGET_TAA_DEGREES[target_taa_idx]

                # 2. TAA=0 をまたぐ判定ロジックの修正
                is_crossing_zero = (current_target_taa == 0) and \
                                   ((TAA < previous_taa) or (TAA >= 0 and previous_taa < 0))

                # 3. 通常のターゲット(>0)の判定
                is_crossing_normal = (previous_taa < current_target_taa <= TAA)

                if is_crossing_normal or is_crossing_zero:
                    save_this_step = True
                    target_taa_idx += 1  # 次のターゲットへ

            # --- 4e. 立方体グリッドに集計して保存 ---
            # スピンアップ期間が終了し (t_sec >= t_start_run)、
            # かつ、保存ターゲットTAAを通過した場合
            if save_this_step and t_sec >= t_start_run:

                pbar.write(f"\n>>> [Run] Saving grid snapshot at TAA={TAA:.1f} ({len(active_particles)} particles) <<<")

                # 3Dグリッドをゼロで初期化
                density_grid = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION, GRID_RESOLUTION), dtype=np.float32)

                # グリッドの物理的範囲
                grid_min = -GRID_MAX_RM * PHYSICAL_CONSTANTS['RM']
                grid_max = GRID_MAX_RM * PHYSICAL_CONSTANTS['RM']
                cell_size = (grid_max - grid_min) / GRID_RESOLUTION
                cell_volume_m3 = cell_size ** 3

                # 粒子をグリッドに割り当てる (ヒストグラム作成)
                for p in active_particles:
                    pos = p['pos']
                    ix = int((pos[0] - grid_min) / cell_size)
                    iy = int((pos[1] - grid_min) / cell_size)
                    iz = int((pos[2] - grid_min) / cell_size)
                    # 粒子がグリッドの範囲内か確認
                    if 0 <= ix < GRID_RESOLUTION and 0 <= iy < GRID_RESOLUTION and 0 <= iz < GRID_RESOLUTION:
                        density_grid[ix, iy, iz] += p['weight']

                # グリッドセルの値(原子数)を体積で割って、数密度 [atoms/m^3] に変換
                density_grid /= cell_volume_m3

                # ファイルに保存
                relative_time_sec = t_sec - t_start_run
                save_time_h = relative_time_sec / 3600
                filename = f"density_grid_t{int(save_time_h):05d}_taa{int(round(TAA)):03d}.npy"
                np.save(os.path.join(target_output_dir, filename), density_grid)

            previous_taa = TAA  # TAAを更新
            pbar.update(1)  # 進捗バーを1つ進める

    # --- 5. 終了 ---
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