"""
水星外気圏 3D粒子軌道シミュレーション (ver 2025.10.21)

概要:
このスクリプトは、水星の表面から放出された粒子（ナトリウム原子を想定）の
3次元軌道を計算し、Plotlyを使用して可視化します。
4次のルンゲ＝クッタ（RK4）法を用いて、粒子の運動方程式を数値的に積分します。

主な機能:
- ★★★ 変更: 外部ファイル('orbit2025_v5.txt')から「指定したTAA」に
-   最も近い軌道パラメータ（太陽距離、視線速度、接線速度）を読み込みます。
- 以下の物理モデル（加速度）の影響をフラグでON/OFF切り替え可能です。
  - USE_GRAVITY: 水星の重力
  - USE_SOLAR_GRAVITY: 太陽の重力
  - USE_RADIATION_PRESSURE: 太陽放射圧 (ドップラーシフト考慮)
  - USE_CORIOLIS_FORCES: 回転座標系における見かけの力（遠心力・コリオリ力）
- 粒子は水星の昼側表面のグリッド中心から、法線方向に固定速度(1000 m/s)で放出されます。
- 計算された全粒子の軌道を3Dグラフで表示します。

座標系:
- 原点: 水星の中心
- +X軸: 水星から太陽へ向かう方向
- +Z軸: 水星の公転軌道面に対して北向き（水星の自転軸と一致すると仮定）
- +Y軸: 右手系を完成させる方向（+X, +Yが公転軌道面を張る。夕方（Dusk）側）
-
- ※この座標系は水星と共に太陽の周りを公転し、かつ水星-太陽ラインが
-   常にX軸になるように回転する「回転座標系」です。
"""

import numpy as np
from scipy.stats import maxwell
import plotly.graph_objects as go
import plotly.io as pio
import sys
from typing import Dict, Any, List, Tuple, Union  # 型ヒント用のモジュール

# --- 表示設定 ---
# Plotlyのグラフをデフォルトでブラウザで開くように設定
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

# --- シミュレーション設定 (グローバル変数) ---
TEMPERATURE = 1500.0  # (使用されなくなりました) 粒子の放出温度 [K]
FLIGHT_DURATION = 5000  # 粒子の最大飛行（追跡）時間 [s]
DT = 10  # 時間ステップ [s]

# --- 物理モデルのトグルスイッチ ---
USE_GRAVITY = False  # Trueにすると水星の重力を考慮
USE_RADIATION_PRESSURE = False  # Trueにすると太陽放射圧を考慮
USE_SOLAR_GRAVITY = False  # Trueにすると太陽の重力を考慮
USE_CORIOLIS_FORCES = True  # Trueにするとコリオリ力・遠心力を考慮

# ★★★ 追加: 読み込みたいTAAを指定 ★★★
TARGET_TAA = 0.0  # [度] 'orbit2025_v5.txt' から読み込むTAAの値を指定
                  # (例: 0.0, 30.0, 90.0, 180.0 など)

# --- 発生源モデルのパラメータ ---
# (使用されなくなりました)
# ATOMS_PER_SUPERPARTICLE = 1e24
# F_UV_1AU = 1.5e14 * (100) ** 2
# Q_PSD = 2.0e-20 / (100) ** 2
# SIGMA_NA_INITIAL = 1.5e23 / (1e3) ** 2

# 粒子放出位置を決定するためのグリッド設定
N_LAT = 12  # 緯度（Latitude）方向の分割数
N_LON = 24  # 経度（Longitude）方向の分割数

# ==============================================================================
# 外部ファイルから軌道情報とスペクトルデータを読み込む
# ==============================================================================

# --- グローバル変数として軌道情報を格納 ---
# これらはシミュレーション全体（特に加速度計算）で参照されます
r0: float = 0.0  # 水星-太陽間 距離 [m]
AU_val: float = 0.0  # 水星-太陽間 距離 [AU]
V_MERCURY_RADIAL: float = 0.0  # 水星の視線速度 [m/s] (放射圧のドップラーシフト計算用)
V_MERCURY_TANGENTIAL: float = 0.0  # 水星の接線速度 [m/s] (回転座標系の角速度計算用)
TAA: float = 0.0  # 真近点離角 (True Anomaly Angle) [度]

try:
    # ★★★ 変更: NumPyでファイル全体を読み込む ★★★
    filename = 'orbit2025_v5.txt'
    # skiprows=1 でヘッダーを読み飛ばす
    orbit_data = np.loadtxt(filename, skiprows=1)

    # 読み込んだデータが1行だけの場合、エラーにならないよう整形
    if orbit_data.ndim == 1:
        orbit_data = orbit_data.reshape(1, -1)
    if orbit_data.shape[0] == 0:
        raise ValueError(f"{filename} にはデータが含まれていません。")

    # 1列目 (TAA) を取得
    taa_column = orbit_data[:, 0]

    # ★★★ 変更: 指定された TARGET_TAA に最も近い行を探す ★★★
    # TAAの差の絶対値が最小になる行のインデックスを取得
    target_index = np.argmin(np.abs(taa_column - TARGET_TAA))

    # 最も近い行のデータを取得
    # 0:TAA, 1:AU, 2:Time, 3:V_radial, 4:V_tangential
    selected_row = orbit_data[target_index]
    TAA, AU_val, _, V_radial_ms, V_tangential_ms = selected_row

    # ★★★ 変更なし: グローバル変数への格納 ★★★
    r0 = AU_val * AU
    V_MERCURY_RADIAL = V_radial_ms
    V_MERCURY_TANGENTIAL = V_tangential_ms

    print(f"軌道情報ファイル '{filename}' を読み込みました。")
    # ★★★ 変更: どのTAAが選ばれたかを表示 ★★★
    print(f"-> 目標 TAA {TARGET_TAA:.1f} 度に最も近い TAA = {TAA:.1f} 度の軌道情報を使用します。")
    print(f"-> 太陽距離 = {AU_val:.3f} AU")
    print(f"-> V_radial = {V_MERCURY_RADIAL:.1f} m/s, V_tangential = {V_MERCURY_TANGENTIAL:.1f} m/s")

except FileNotFoundError:
    # (エラーハンドリングは変更なし)
    print(f"エラー: '{filename}' が見つかりません。", file=sys.stderr)
    print("直前のスクリプトを実行して、'orbit2025_v5.txt' を生成してください。", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"軌道情報ファイルの読み込み中にエラーが発生しました: {e}", file=sys.stderr)
    sys.exit(1)


# --- 太陽スペクトルデータの読み込み ---
spec_data_dict: Dict[str, Any] = {}
if USE_RADIATION_PRESSURE:
    try:
        # 'SolarSpectrum_Na0.txt' から2列 (波長[A], ガンマ値) を読み込む
        # 単位: 波長[A], gamma[ (s^-1) / (W m^-2 Hz^-1) ] (要確認)
        spec_data_np = np.loadtxt('SolarSpectrum_Na0.txt', usecols=(0, 3))
        wl_angstrom = spec_data_np[:, 0]  # 波長 [A]
        gamma = spec_data_np[:, 1]  # ガンマ値

        # 波長が昇順にソートされていることを保証 (np.interp のため)
        if not np.all(np.diff(wl_angstrom) > 0):
            sort_indices = np.argsort(wl_angstrom)
            wl_angstrom, gamma = wl_angstrom[sort_indices], gamma[sort_indices]

        # 散乱断面積の計算に必要な定数
        sigma_const = PHYSICAL_CONSTANTS['E_CHARGE'] ** 2 / (
                4 * PHYSICAL_CONSTANTS['ME'] * PHYSICAL_CONSTANTS['C'] * PHYSICAL_CONSTANTS['EPSILON_0'])

        # 放射圧計算に必要なデータを辞書にまとめる
        spec_data_dict = {
            'wl': wl_angstrom,  # 波長 [A]
            'gamma': gamma,  # ガンマ値
            'sigma0_perdnu1': sigma_const * 0.320,  # D1線用
            'sigma0_perdnu2': sigma_const * 0.641,  # D2線用
            'JL': 5.18e14 * 1e4  # [photons s^-1 m^-2 A^-1] at 1 AU (要確認)
        }
        print("太陽スペクトルデータを正常に読み込みました。")
    except FileNotFoundError:
        print("エラー: 'SolarSpectrum_Na0.txt' が見つかりません。", file=sys.stderr)
        print("-> 放射圧は計算されません (USE_RADIATION_PRESSURE = False に設定)")
        USE_RADIATION_PRESSURE = False


# ==============================================================================
# 加速度計算の関数
# ==============================================================================

def calculate_radiation_acceleration(
        pos: np.ndarray,
        vel: np.ndarray,
        spec_data: Dict[str, Any],
        sun_distance_au: float,
        orbital_radial_velocity_ms: float
) -> np.ndarray:
    """
    太陽放射圧による加速度を計算する。

    粒子の速度と水星の公転速度によるドップラーシフトを考慮し、
    D1線とD2線の両方からの寄与を合計して加速度ベクトルを返す。

    Args:
        pos (np.ndarray): 粒子の位置ベクトル [x, y, z] [m]
        vel (np.ndarray): 粒子の速度ベクトル [vx, vy, vz] [m/s]
        spec_data (Dict[str, Any]): 太陽スペクトルデータと物理定数を含む辞書
        sun_distance_au (float): 水星と太陽の距離 [AU]
        orbital_radial_velocity_ms (float): 水星の太陽に対する視線速度 [m/s]

    Returns:
        np.ndarray: 放射圧による加速度ベクトル [ax, ay, az] [m/s^2]

    NOTE/FIXME:
        現在の実装では `return np.array([-b, 0.0, 0.0])` となっており、
        -X方向（太陽方向）への力を計算しています。
        物理的には放射圧は反太陽方向（+X方向）へ働くため、
        `[b, 0.0, 0.0]` が正しい可能性があります。
        （あるいは、座標系の定義が +X=反太陽方向 であるか、
          計算途中の符号が反転している可能性があります）
        ※ただし、`get_total_acceleration` の太陽重力計算では
          +X=太陽方向 と仮定されているため、不整合の可能性があります。
    """
    x, y, z = pos

    # 水星の影に入っているか判定 (X < 0 かつ 軌道円筒の内側)
    if x < 0 and np.sqrt(y ** 2 + z ** 2) < PHYSICAL_CONSTANTS['RADIUS_MERCURY']:
        return np.array([0.0, 0.0, 0.0])  # 影の中では放射圧ゼロ

    # --- ドップラーシフトの計算 ---
    # 座標系ではX軸が太陽方向のため、粒子の視線速度 = X方向の速度
    # 粒子自身の速度(vel[0])と、水星の公転による視線速度(orbital_radial_velocity_ms)を合算
    velocity_for_doppler = vel[0] + orbital_radial_velocity_ms

    # ドップラーシフト後のD2, D1線の波長 [m]
    w_na_d2 = 589.1582e-9 * (1.0 - velocity_for_doppler / PHYSICAL_CONSTANTS['C'])
    w_na_d1 = 589.7558e-9 * (1.0 - velocity_for_doppler / PHYSICAL_CONSTANTS['C'])

    # スペクトルデータを辞書から展開
    wl, gamma, sigma0_perdnu1, sigma0_perdnu2, JL = spec_data.values()

    # シフト後の波長が、読み込んだスペクトルデータの範囲外なら放射圧ゼロ
    # (wlの単位は [A] なので、比較のために 1e9 を掛けて [nm] -> [A] * 1e-1 に...
    #  いや、wl[A] * 1e-10 = [m] vs w_na_d2 [m] が正しい。コードは wl * 1e-9。単位系要確認)
    # ※元のコードのロジック `wl[0] * 1e-9 <= w_na_d2 < wl[-1] * 1e-9` を尊重
    if not (wl[0] * 1e-9 <= w_na_d2 < wl[-1] * 1e-9 and wl[0] * 1e-9 <= w_na_d1 < wl[-1] * 1e-9):
        return np.array([0.0, 0.0, 0.0])

    # --- 放射圧の計算 ---
    # シフト後の波長におけるガンマ値を線形補間(interp)で求める
    # (w_na_d* * 1e9 は [m] -> [nm*10]？ 単位系要確認。元のコードのロジックを尊重)
    gamma2 = np.interp(w_na_d2 * 1e9, wl, gamma)
    gamma1 = np.interp(w_na_d1 * 1e9, wl, gamma)

    F_lambda_1AU_m = JL * 1e9  # 単位変換 (要確認)

    # D1線
    F_lambda_d1 = F_lambda_1AU_m / (sun_distance_au ** 2) * gamma1  # 距離の2乗で減衰
    F_nu_d1 = F_lambda_d1 * (w_na_d1 ** 2) / PHYSICAL_CONSTANTS['C']  # F_lambda -> F_nu 変換
    J1 = sigma0_perdnu1 * F_nu_d1

    # D2線
    F_lambda_d2 = F_lambda_1AU_m / (sun_distance_au ** 2) * gamma2
    F_nu_d2 = F_lambda_d2 * (w_na_d2 ** 2) / PHYSICAL_CONSTANTS['C']
    J2 = sigma0_perdnu2 * F_nu_d2

    # 合計の加速度 ( F = dp/dt = (h/λ) * J ) -> ( a = F/m )
    b = (1 / PHYSICAL_CONSTANTS['MASS_NA']) * (
            (PHYSICAL_CONSTANTS['H'] / w_na_d1) * J1 + (PHYSICAL_CONSTANTS['H'] / w_na_d2) * J2)

    # X軸（太陽-水星ライン）に沿った加速度として返す
    return np.array([-b, 0.0, 0.0])


def get_total_acceleration(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """
    指定された位置と速度における総加速度（重力＋放射圧＋見かけの力）を計算する。

    `USE_...` フラグに基づいて、計算に含める力を決定する。

    Args:
        pos (np.ndarray): 粒子の位置ベクトル [x, y, z] [m]
        vel (np.ndarray): 粒子の速度ベクトル [vx, vy, vz] [m/s]

    Returns:
        np.ndarray: 合計の加速度ベクトル [ax, ay, az] [m/s^2]
    """
    accel = np.array([0.0, 0.0, 0.0])  # 合計加速度ベクトルを初期化
    G = PHYSICAL_CONSTANTS['G']

    # 1. 水星の重力 (中心力)
    if USE_GRAVITY:
        r_sq = np.sum(pos ** 2)  # 原点（水星中心）からの距離の2乗
        if r_sq > 0:
            # a = -GM * r_vec / |r|^3
            grav_accel = -G * PHYSICAL_CONSTANTS['MASS_MERCURY'] * pos / (r_sq ** 1.5)
            accel += grav_accel

    # 2. 太陽の重力
    if USE_SOLAR_GRAVITY:
        x, y, z = pos
        M_SUN = PHYSICAL_CONSTANTS['MASS_SUN']
        # 太陽の位置を [r0, 0, 0] と仮定
        # 粒子から太陽へ向かうベクトル r_ps = [r0-x, -y, -z]
        r_ps_vec = np.array([r0 - x, -y, -z])
        r_ps_mag_sq = np.sum(r_ps_vec ** 2)
        if r_ps_mag_sq > 0:
            # a = G * M_sun * r_ps_vec / |r_ps|^3 (太陽への引力)
            r_ps_mag = np.sqrt(r_ps_mag_sq)
            accel_sun = (G * M_SUN * r_ps_vec) / (r_ps_mag ** 3)
            accel += accel_sun

    # 3. 太陽放射圧
    if USE_RADIATION_PRESSURE:
        # 放射圧の計算には水星の視線速度 (V_MERCURY_RADIAL) が必要
        rad_accel = calculate_radiation_acceleration(
            pos, vel, spec_data_dict, AU_val, V_MERCURY_RADIAL
        )
        accel += rad_accel

    # 4. 回転座標系の「見かけの力」
    if USE_CORIOLIS_FORCES:
        # 角速度ωの計算 [rad/s]
        # 水星の公転運動 (v_t = ω * r) から ω = v_t / r を求める
        # v_t = V_MERCURY_TANGENTIAL (接線速度), r = r0 (太陽距離)
        omega_val = V_MERCURY_TANGENTIAL / r0

        # 水星（原点）が太陽に引かれる重力と釣り合う、反太陽方向（-X）の力
        # a_trans = - (G * M_SUN / r0^2) = - (ω^2 * r0) = - (V_t^2 / r0)
        # omega_val = V_t / r0 なので、 -omega_val^2 * r0 ではないことに注意
        # 正しくは a_trans = - (V_MERCURY_TANGENTIAL**2 / r0)

        # V_MERCURY_TANGENTIAL は V_t なので、 V_t^2 / r0 が正しい加速度
        #accel_translational = np.array([
        #    -(V_MERCURY_TANGENTIAL ** 2) / r0,
        #    0.0,
        #    0.0
        #])
        #accel += accel_translational

        # Z軸周りの回転 (ω = [0, 0, ω]) を仮定
        # r = [x, y, z], v = [vx, vy, vz]

        # (4a) 遠心力: a_cen = -ω x (ω x r)
        # ω x r = [-ω*y, ω*x, 0]
        # -ω x (ω x r) = -[0, 0, ω] x [-ω*y, ω*x, 0]
        #             = -[ -ω*(ω*x), -ω*(-ω*y), 0 ]
        #             = [ω^2 * x, ω^2 * y, 0]
        #accel_centrifugal = np.array([
        #    omega_val ** 2 * pos[0],  # ω^2 * x
        #    omega_val ** 2 * pos[1],  # ω^2 * y
        #    0.0
        #])
        #accel += accel_centrifugal
        accel_combined_centrifugal = np.array([
            omega_val ** 2 * (pos[0] - r0),  # X成分を (pos[0] - r0) で計算
            omega_val ** 2 * pos[1],  # Y成分は pos[1] のまま
            0.0
        ])
        accel += accel_combined_centrifugal

        # (4b) コリオリ力: a_cor = -2 * (ω x v)
        # ω x v = [-ω*vy, ω*vx, 0]
        # -2 * (ω x v) = -2 * [-ω*vy, ω*vx, 0]
        #              = [2*ω*vy, -2*ω*vx, 0]
        accel_coriolis = np.array([
            2 * omega_val * (vel[1] - V_MERCURY_TANGENTIAL),  # +2ω * v_y
            -2 * omega_val * vel[0],  # -2ω * v_x
            0.0
        ])
        accel += accel_coriolis

    return accel


# ==============================================================================
# サンプリング関数 (現在は不使用)
# ==============================================================================
# (参考: 以前のランダムサンプリング関数)

def sample_emission_angle_lambertian() -> Tuple[float, float, float]:
    """
    (不使用) ランバート（余弦）則に従う放出角度をサンプリングする。

    Returns:
        Tuple[float, float, float]: (sin_theta, cos_theta, phi)
    """
    u1, u2 = np.random.random(2)
    cos_theta = np.sqrt(1 - u2)  # cos(θ) ∝ sqrt(1-u2) -> 確率密度 P(θ) ∝ sin(θ)cos(θ)
    sin_theta = np.sqrt(u2)
    phi = 2 * PI * u1  # φ は 0〜2π で一様
    return sin_theta, cos_theta, phi


def sample_maxwellian_speed(mass_kg: float, temp_k: float) -> float:
    """
    (不使用) マクスウェル分布に従う速度をサンプリングする。

    Args:
        mass_kg (float): 粒子の質量 [kg]
        temp_k (float): 温度 [K]

    Returns:
        float: サンプリングされた速度 [m/s]
    """
    # マクスウェル分布のスケールパラメータ a = sqrt(kT/m)
    scale_param = np.sqrt(PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_k / mass_kg)
    return maxwell.rvs(scale=scale_param)


# ==============================================================================
# 粒子放出位置の生成 (MODIFIED)
# ==============================================================================
print("--- MODIFIED: グリッドセル中心から法線方向に粒子を生成します ---")

# 粒子の初期位置 [m] を格納するリスト
initial_positions: List[np.ndarray] = []

# 緯度(lat)と経度(lon)のグリッドを作成
# 緯度: -π/2 (南極) から +π/2 (北極) まで
lat_rad = np.linspace(-PI / 2, PI / 2, N_LAT)
# 経度: -π (反太陽側) から +π (反太陽側) まで
# lon=0 が太陽直下点 (+X軸)
lon_rad = np.linspace(-PI, PI, N_LON)

# (グリッド間隔：デバッグ用)
# dlat = lat_rad[1] - lat_rad[0]
# dlon = lon_rad[1] - lon_rad[0]

for i in range(N_LAT):
    for j in range(N_LON):
        lat_center, lon_center = lat_rad[i], lon_rad[j]

        # 太陽天頂角(SZA)の余弦(cosine)を計算
        # 座標系: +X=太陽方向, +Z=北極
        # SZA=0 (太陽直下点) は (lat=0, lon=0)
        # cos(SZA) = (太陽ベクトル)・(法線ベクトル)
        # 太陽ベクトル = [1, 0, 0] (方向のみ)
        # 法線ベクトル = [cos(lat)cos(lon), cos(lat)sin(lon), sin(lat)]
        # -> cos(SZA) = cos(lat)cos(lon)
        cos_Z = np.cos(lat_center) * np.cos(lon_center)

        # 太陽が当たっているセル(昼側)かどうかの判定 (cos(SZA) > 0)
        if cos_Z <= 0:
            continue  # 夜側のセルからは放出しない

        # --- MODIFIED: ---
        # 以前のランダムな点ではなく、グリッド中心を放出点とする
        # また、セルごとに1つの粒子のみを生成する
        radius_m = PHYSICAL_CONSTANTS['RADIUS_MERCURY']

        # 球面座標 [r, lat, lon] -> 直交座標 [x, y, z]
        # x = r * cos(lat) * cos(lon)
        # y = r * cos(lat) * sin(lon)
        # z = r * sin(lat)
        x = radius_m * np.cos(lat_center) * np.cos(lon_center)
        y = radius_m * np.cos(lat_center) * np.sin(lon_center)
        z = radius_m * np.sin(lat_center)

        initial_positions.append(np.array([x, y, z]))
        # --- END MODIFIED ---

print(f"合計 {len(initial_positions)} 個のスーパーパーティクルを生成しました。")
print("各粒子の軌道計算を開始します...")

# ==============================================================================
# メインシミュレーション
# ==============================================================================

# 全粒子の軌道データ（位置ベクトルのリスト）を格納するリスト
all_trajectories: List[List[np.ndarray]] = []

# 生成された初期位置それぞれに対して軌道計算を実行
for initial_pos in initial_positions:

    # 位置ベクトル `pos` と速度ベクトル `vel` を初期化
    pos = initial_pos.copy()  # [m]

    # --- MODIFIED: 速度と角度を固定 ---
    # 1. 速度(速さ)を 1000 m/s に固定
    speed = 1000.0  # [m/s]

    # 2. 角度を法線ベクトル方向（中心から外向き）に固定
    #    法線ベクトル = 位置ベクトル / |位置ベクトル|
    #    (原点が水星中心なので、位置ベクトル = 法線ベクトル方向)
    local_z_axis = pos / np.linalg.norm(pos)
    vel = speed * local_z_axis  # 速度ベクトル [m/s]
    # --- END MODIFIED ---

    # この粒子の軌道（位置ベクトルのリスト）を格納するリスト
    particle_trajectory: List[np.ndarray] = []

    # 時間 0 から FLIGHT_DURATION まで、DT 刻みでループ
    for t in np.arange(0, FLIGHT_DURATION, DT):

        # 粒子が水星表面より内側に入ったら、衝突（吸着）とみなす
        # t > 0 は、t=0 (放出時) に判定しないためのおまじない
        if np.sum(pos ** 2) < PHYSICAL_CONSTANTS['RADIUS_MERCURY'] ** 2 and t > 0:
            break  # この粒子の追跡を終了

        # 現在の位置を軌道リストに追加
        particle_trajectory.append(pos.copy())

        # --- 4次のルンゲ＝クッタ（RK4）法による数値積分 ---
        # v(t+dt) = v(t) + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
        # r(t+dt) = r(t) + (k1_r + 2*k2_r + 2*k3_r + k4_r) / 6

        # k1 (時刻 t, 位置 pos, 速度 vel での加速度と速度)
        # a1 = a(t, pos, vel)
        a1 = get_total_acceleration(pos, vel)
        k1_vel = DT * a1
        k1_pos = DT * vel

        # k2 (時刻 t+DT/2, 位置 pos+k1_r/2, 速度 vel+k1_v/2 での加速度と速度)
        # a2 = a(t+DT/2, pos + k1_pos/2, vel + k1_vel/2)
        a2 = get_total_acceleration(pos + 0.5 * k1_pos, vel + 0.5 * k1_vel)
        k2_vel = DT * a2
        k2_pos = DT * (vel + 0.5 * k1_vel)

        # k3 (時刻 t+DT/2, 位置 pos+k2_r/2, 速度 vel+k2_v/2 での加速度と速度)
        # a3 = a(t+DT/2, pos + k2_pos/2, vel + k2_vel/2)
        a3 = get_total_acceleration(pos + 0.5 * k2_pos, vel + 0.5 * k2_vel)
        k3_vel = DT * a3
        k3_pos = DT * (vel + 0.5 * k2_vel)

        # k4 (時刻 t+DT, 位置 pos+k3_r, 速度 vel+k3_v での加速度と速度)
        # a4 = a(t+DT, pos + k3_pos, vel + k3_vel)
        a4 = get_total_acceleration(pos + k3_pos, vel + k3_vel)
        k4_vel = DT * a4
        k4_pos = DT * (vel + k3_vel)

        # 最終的な位置と速度の更新
        pos += (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos) / 6.0
        vel += (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel) / 6.0
        # --- RK4 終了 ---

    # この粒子の全軌道データを、全軌道リストに追加
    all_trajectories.append(particle_trajectory)

print("軌道計算が完了しました。可視化処理を開始します。")

# ==============================================================================
# Plotlyによる可視化
# ==============================================================================

# グラフの単位を [m] ではなく [R_M] (水星半径) にするためのスケール
radius = PHYSICAL_CONSTANTS['RADIUS_MERCURY']

# --- 1. 水星の球体メッシュを作成 ---
# パラメトリック方程式 (球面座標) で球を表現
u = np.linspace(0, 2 * np.pi, 100)  # 経度方向
v = np.linspace(0, np.pi, 100)  # 緯度方向
x_sphere_m = radius * np.outer(np.cos(u), np.sin(v))
y_sphere_m = radius * np.outer(np.sin(u), np.sin(v))
z_sphere_m = radius * np.outer(np.ones(np.size(u)), np.cos(v))

# 球体のトレース (go.Surface) を作成
sphere_trace = go.Surface(
    x=x_sphere_m / radius,  # X座標 [R_M]
    y=y_sphere_m / radius,  # Y座標 [R_M]
    z=z_sphere_m / radius,  # Z座標 [R_M]
    colorscale='Greys',  # 陰影が分かりやすいように灰色
    opacity=0.8,  # 80%の不透明度
    showscale=False,  # カラーバーは非表示
    name='水星',

    # --- ライティング（陰影）設定 ---
    # 太陽がX軸正方向にあるため、X軸の正の無限遠から光を当てる
    lightposition=dict(x=10000, y=0, z=0),
    lighting=dict(
        # ambient: 環境光（影の部分の明るさ）を減らし、コントラストを強める
        ambient=0.2,
        # diffuse: 拡散光（直接光の強さ）を強める
        diffuse=1.0,
        # specular: 鏡面反射 (ハイライトの強さ)
        specular=0.0
    )
)

# --- 2. 粒子の軌道データ（ライン）を作成 ---
# Plotlyで複数の線を別々に描画するため、
# [x1, y1, z1, None, x2, y2, z2, None, ...] の形式のリストを作成する
lines_x_m: List[Union[float, None]] = []
lines_y_m: List[Union[float, None]] = []
lines_z_m: List[Union[float, None]] = []

for trajectory in all_trajectories:
    if not trajectory: continue  # 軌道データが空の場合はスキップ

    # trajectory は [(x,y,z), (x,y,z), ...] というnp.ndarrayのリスト
    # zip(*trajectory) を使うと、[(x,x,x,...), (y,y,y,...), (z,z,z,...)] に転置できる
    x, y, z = zip(*trajectory)

    lines_x_m.extend(x)  # x座標のリストを追加
    lines_y_m.extend(y)  # y座標のリストを追加
    lines_z_m.extend(z)  # z座標のリストを追加

    # 軌道の終わりに「None」を追加することで、Plotly上で線が途切れる
    lines_x_m.append(None)
    lines_y_m.append(None)
    lines_z_m.append(None)

# 座標を [m] -> [R_M] にスケーリング
lines_x_scaled = [val / radius if val is not None else None for val in lines_x_m]
lines_y_scaled = [val / radius if val is not None else None for val in lines_y_m]
lines_z_scaled = [val / radius if val is not None else None for val in lines_z_m]

# 軌道のトレース (go.Scatter3d) を作成
lines_trace = go.Scatter3d(
    x=lines_x_scaled, y=lines_y_scaled, z=lines_z_scaled,
    mode='lines',
    line=dict(color='orange', width=2),  # 線の色と太さ
    name='粒子軌道'
)

# --- 3. グラフの作成と表示 ---
fig = go.Figure(data=[sphere_trace, lines_trace])

# レイアウトの更新
fig.update_layout(
    title=f'水星粒子軌道 (TAA={TAA:.1f}°, Gravity:{USE_GRAVITY}, Rad.Pressure:{USE_RADIATION_PRESSURE}, Solar.Grav:{USE_SOLAR_GRAVITY}, Coriolis:{USE_CORIOLIS_FORCES})',
    scene=dict(
        xaxis_title='X [R_M]',
        yaxis_title='Y [R_M]',
        zaxis_title='Z [R_M]',
        # 'data' を指定することで、X,Y,Z軸のスケールを
        # データの範囲に基づいて1:1:1に保つ（球が歪まない）
        aspectmode='data'
    ),
    legend=dict(x=0, y=1)  # 凡例を左上に表示
)

# グラフをブラウザで表示
fig.show()