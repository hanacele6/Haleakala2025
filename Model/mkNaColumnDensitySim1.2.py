import numpy as np
import sys
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import matplotlib.pyplot as plt


# --- 物理モデルに基づくヘルパー関数 ---

def calculate_surface_temperature(x, y, AU):
    """
    論文の式(1)に基づき、衝突地点の表面温度[K]を計算する。
    Args:
        x (float): 衝突地点のX座標 [km] (太陽方向が+X)
        y (float): 衝突地点のY座標 [km]
        AU (float): 現在の水星-太陽間の距離 [au]
    Returns:
        float: 表面温度 [K]
    """
    # 論文で使用されている定数
    T0 = 100.0  # 夜側の温度 [K]
    T1 = 600.0  # 日側の温度スケーリング係数 [K] [cite: 172]

    # 粒子が夜側 (x <= 0) に衝突した場合
    if x <= 0:
        return T0

    # 日側の場合、太陽直下点からの角度の余弦を計算
    # 2Dモデルなので、cos(φ)cos(λ)は単純にcos(theta)で近似
    cos_theta = x / np.sqrt(x ** 2 + y ** 2)
    if cos_theta < 0:  # 念のための安全策
        return T0

    # 論文の式(1)を計算
    # T_s = T_0 + T_1 * (cos_theta)^(1/4) * (0.306 au / r_0)^2
    temp = T0 + T1 * (cos_theta ** 0.25) * ((0.306 / AU) ** 2)  #
    return temp


def calculate_sticking_probability(surface_temp_K):
    """
    論文の式(12)と式(13)に基づき、吸着率を計算する。
    Args:
        surface_temp_K (float): 表面温度 [K]
    Returns:
        float: 実効的な吸着確率
    """
    # 論文で使用されている定数
    A = 0.08  # [cite: 323]
    B = 458.0  # [K] [cite: 323]
    porosity = 0.8  # [cite: 374]

    # 式(12): 温度に依存する基本的な吸着率
    p_stick = A * np.exp(B / surface_temp_K)  # [cite: 321]

    # 式(13): 多孔性を考慮した実効的な吸着率
    p_stick_eff = p_stick / (1.0 - (1.0 - p_stick) * porosity)  # [cite: 359]

    return p_stick_eff


# --- 新しい関数: 物理モデルに基づくサンプリング ---

def sample_weibull_speed(mass_kg,
                         U_ev=0.05,
                         beta_shape=0.7):
    """ワイブル分布からエネルギーをサンプリングし、速度[km/s]を返す。"""
    E_CHARGE_SI = 1.602176634e-19
    p = np.random.random()
    E_ev = U_ev * (p ** (-1.0 / (beta_shape + 1.0)) - 1.0)
    E_joule = E_ev * E_CHARGE_SI
    v_ms = np.sqrt(2 * E_joule / mass_kg)
    return v_ms / 1000.0


def sample_cosine_angle():
    """cosine則に従う角度 [-pi/2, pi/2] をサンプリングして返す。"""
    p = np.random.random()
    return np.arcsin(2 * p - 1)


# --- シミュレーションのコア関数 (改造版) ---

def simulate_single_particle_for_density(args):
    """
    1個の原子の軌道をシミュレートし、「滞在時間」を記録した2Dグリッドを返す。
    """
    # --- 引数を受け取る ---
    constants, settings, spec_data = args['consts'], args['settings'], args['spec']
    TAA, AU, lon, lat, Vms_ms = args['orbit']
    grid_params = args['grid_params']

    # --- 定数と設定を展開 ---
    C, PI, H, MASS_NA, RM, GM_MERCURY, K_BOLTZMANN = constants.values()
    GRAVITY_ENABLED, BETA, T1AU, DT = settings.values()  # STICKING_COEFFICIENTを削除
    wl, gamma, sigma0_perdnu2, sigma0_perdnu1, JL = spec_data.values()

    # --- グリッドパラメータを展開 ---
    GRID_SIZE, GRID_MAX_R = grid_params['size'], grid_params['max_r']
    CELL_SIZE = 2 * GRID_MAX_R / GRID_SIZE

    local_density_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    # --- 放出位置と速度の初期化 ---
    p = np.random.random()
    source_angle_rad = np.arcsin(2 * p - 1)
    x = RM * np.cos(source_angle_rad)
    y = RM * np.sin(source_angle_rad)
    ejection_speed = sample_weibull_speed(MASS_NA)
    surface_normal_angle = np.arctan2(y, x)
    random_offset_angle = sample_cosine_angle()
    ejection_angle_rad = surface_normal_angle + random_offset_angle
    vx_ms = ejection_speed * np.cos(ejection_angle_rad)
    vy_ms = ejection_speed * np.sin(ejection_angle_rad)

    # --- シミュレーションループ ---
    Vms = Vms_ms / 1000.0
    tau = T1AU * AU ** 2
    itmax = int(tau * 5.0 / DT + 0.5)

    for it in range(itmax):
        # (中略: 放射圧や重力の計算は変更なし)
        velocity_for_doppler = vx_ms + Vms
        w_na_d2 = 589.1582 * (1.0 - velocity_for_doppler / C)
        w_na_d1 = 589.7558 * (1.0 - velocity_for_doppler / C)
        if not (wl[0] <= w_na_d2 < wl[-1] and wl[0] <= w_na_d1 < wl[-1]):
            break
        gamma2 = np.interp(w_na_d2, wl, gamma)
        gamma1 = np.interp(w_na_d1, wl, gamma)
        m_na_wl = (w_na_d2 + w_na_d1) / 2.0
        jl_nu = JL * 1e9 * ((m_na_wl * 1e-9) ** 2 / (C * 1e3))
        J2 = sigma0_perdnu2 * jl_nu / AU ** 2 * gamma2
        J1 = sigma0_perdnu1 * jl_nu / AU ** 2 * gamma1
        b = (H / MASS_NA) * (J1 / (w_na_d1 * 1e-7) + J2 / (w_na_d2 * 1e-7))
        if x < 0 and np.sqrt(y ** 2) < RM:
            b = 0.0
        Nad = np.exp(-DT * it / tau)
        accel_gx, accel_gy = 0.0, 0.0
        if GRAVITY_ENABLED:
            r_sq_grav = x ** 2 + y ** 2
            if r_sq_grav > 0:
                r_grav = np.sqrt(r_sq_grav)
                grav_accel_total = GM_MERCURY / r_sq_grav
                accel_gx = -grav_accel_total * (x / r_grav)
                accel_gy = -grav_accel_total * (y / r_grav)
        vx_ms_prev, vy_ms_prev = vx_ms, vy_ms
        accel_srp_x = b / 100.0 / 1000.0
        total_accel_x = accel_srp_x + accel_gx
        total_accel_y = accel_gy
        vx_ms += total_accel_x * DT
        vy_ms += total_accel_y * DT
        x_prev, y_prev = x, y
        x += ((vx_ms_prev + vx_ms) / 2.0) * DT
        y += ((vy_ms_prev + vy_ms) / 2.0) * DT

        # グリッドへの滞在時間記録
        ix = int((x + GRID_MAX_R) / CELL_SIZE)
        iy = int((y + GRID_MAX_R) / CELL_SIZE)
        if 0 <= ix < GRID_SIZE and 0 <= iy < GRID_SIZE:
            local_density_grid[iy, ix] += Nad * DT

        # --- ★★★ 変更点: 地表衝突時の処理 ★★★ ---
        r_current = np.sqrt(x ** 2 + y ** 2)
        if r_current <= RM:
            # 1. 衝突地点の局所的な表面温度を計算
            temp_at_impact = calculate_surface_temperature(x_prev, y_prev, AU)

            # 2. その温度における吸着確率を計算
            stick_prob = calculate_sticking_probability(temp_at_impact)

            # 3. 吸着判定 (確率に基づいてbreak)
            if np.random.random() < stick_prob:
                break

            # 4. 反射する場合: 局所温度を使って反射エネルギーを計算
            v_in_sq = vx_ms_prev ** 2 + vy_ms_prev ** 2
            E_in = 0.5 * MASS_NA * (v_in_sq * 1e6)
            E_T = K_BOLTZMANN * temp_at_impact  # ★ 局所温度を使用
            E_out = BETA * E_T + (1.0 - BETA) * E_in
            v_out_sq = E_out / (0.5 * MASS_NA) / 1e6
            v_out_speed = np.sqrt(v_out_sq) if v_out_sq > 0 else 0.0

            surface_angle = np.arctan2(y_prev, x_prev)
            # Lambert則に従うランダムな角度で反射
            rebound_angle_rad = surface_angle + np.random.uniform(-PI / 2.0, PI / 2.0)

            vx_ms = v_out_speed * np.cos(rebound_angle_rad)
            vy_ms = v_out_speed * np.sin(rebound_angle_rad)
            x = RM * np.cos(surface_angle)
            y = RM * np.sin(surface_angle)

    return local_density_grid


def main():
    """メインの制御関数"""
    # --- 設定項目 ---
    OUTPUT_DIRECTORY = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult"
    TARGET_TAA_LINE_NUMBER = 0
    GRID_SIZE = 51
    GRID_RADIUS_RM = 5.0
    TOTAL_SOURCE_FLUX = 1.0e25  # [atoms/s]
    N_PARTICLES = 100000

    settings = {
        'GRAVITY_ENABLED': True,
        'BETA': 0.5,
        'T1AU': 61728.4,
        'DT': 10.0,
        # 'STICKING_COEFFICIENT': 0.15 # ★ 固定値を削除
    }
    # --- 設定はここまで ---

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    grid_params = {'size': GRID_SIZE, 'max_r': 2439.7 * GRID_RADIUS_RM}
    constants = {
        'C': 299792.458, 'PI': np.pi, 'H': 6.626068e-34 * 1e7,
        'MASS_NA': 22.98976928 * 1.66054e-27, 'RM': 2439.7,
        'GM_MERCURY': 2.2032e4, 'K_BOLTZMANN': 1.380649e-23
    }

    try:
        spec_data_np = np.loadtxt('SolarSpectrum_Na0.txt', usecols=(0, 3))
        wl, gamma = spec_data_np[:, 0], spec_data_np[:, 1]
        orbit_lines = open('orbit360.txt', 'r').readlines()
    except FileNotFoundError as e:
        print(f"エラー: データファイルが見つかりません - {e}")
        sys.exit()

    if not np.all(np.diff(wl) > 0):
        sort_indices = np.argsort(wl)
        wl, gamma = wl[sort_indices], gamma[sort_indices]

    ME, E_CHARGE = 9.1093897e-31, 1.60217733e-19
    sigma_const = constants['PI'] * E_CHARGE ** 2 / (ME * constants['C'] * 1e3)
    spec_data_dict = {
        'wl': wl, 'gamma': gamma, 'sigma0_perdnu2': sigma_const * 0.641,
        'sigma0_perdnu1': sigma_const * 0.320, 'JL': 5.18e14
    }

    try:
        orbit_line = orbit_lines[TARGET_TAA_LINE_NUMBER]
    except IndexError:
        print(f"エラー: 行番号 {TARGET_TAA_LINE_NUMBER} は orbit360.txt の範囲外です。")
        sys.exit()

    TAA, AU, lon, lat, Vms_ms = map(float, orbit_line.split())

    print(f"原子密度シミュレーションを開始します (N={N_PARTICLES})")
    print(f"TAA = {TAA:.1f}度 (orbit360.txt の {TARGET_TAA_LINE_NUMBER + 1}行目)")

    task_args = {
        'consts': constants, 'settings': settings, 'spec': spec_data_dict,
        'orbit': (TAA, AU, lon, lat, Vms_ms), 'grid_params': grid_params
    }
    tasks = [task_args] * N_PARTICLES

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(simulate_single_particle_for_density, tasks), total=N_PARTICLES,
                            desc=f"TAA={TAA:.1f}"))

    print("全粒子の結果を集計しています...")
    master_density_grid = np.sum(results, axis=0)
    cell_area_cm2 = (2 * grid_params['max_r'] * 1e5 / GRID_SIZE) ** 2
    column_density_grid = (TOTAL_SOURCE_FLUX / N_PARTICLES) * (master_density_grid / cell_area_cm2)

    base_filename = f"density_map_taa{TAA:.0f}_temp_dependent_sticking"
    full_path_npy = os.path.join(OUTPUT_DIRECTORY, f"{base_filename}.npy")
    np.save(full_path_npy, column_density_grid)
    print(f"結果を {full_path_npy} に保存しました。")

    # (プロット処理は変更なし)
    print("結果をプロットしています...")
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_data = np.log10(np.where(column_density_grid > 0, column_density_grid, np.nan))
    img = ax.imshow(plot_data, cmap='inferno',
                    extent=[-GRID_RADIUS_RM, GRID_RADIUS_RM, -GRID_RADIUS_RM, GRID_RADIUS_RM],
                    origin='lower')
    mercury_circle = plt.Circle((0, 0), 1, color='lightgray', zorder=10)
    ax.add_artist(mercury_circle)
    ax.arrow(GRID_RADIUS_RM * 0.9, 0, -GRID_RADIUS_RM * 0.3, 0,
             head_width=0.2, head_length=0.2, fc='yellow', ec='black', zorder=11, label='Sunlight')
    ax.text(GRID_RADIUS_RM * 0.6, 0.3, "Sun", color="yellow", zorder=11)
    ax.set_title(f"Na Column Density (PSD, Temp-Dependent Sticking, TAA={TAA:.1f}, N={N_PARTICLES})", fontsize=16)
    ax.set_xlabel("X [$R_M$] (anti-sunward)")
    ax.set_ylabel("Y [$R_M$]")
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    ax.grid(True, linestyle='--', alpha=0.3)
    cbar = fig.colorbar(img)
    cbar.set_label('Log10(Column Density [atoms/cm$^2$])')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()