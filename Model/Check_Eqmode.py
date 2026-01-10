import numpy as np
import matplotlib.pyplot as plt
import os

# === 設定 ===
ORBIT_FILE_OLD = 'orbit2025_v6.txt'
ORBIT_FILE_NEW = 'orbit2025_spice_unwrapped.txt'

YEARS_TO_SIMULATE = 3.0
DT_STEP = 3600.0
INIT_SURF_DENS = 7.5e14 * (100 ** 2) * 0.0053
RM = 2.440e6
EV_J = 1.602e-19
KB = 1.380649e-23
ORBITAL_PERIOD = 87.969 * 86400
ROTATION_PERIOD = 58.6462 * 86400


# === Old Code から抽出したヘルパー関数 ===

def calculate_temp_exact(lon, lat, r_au, sub_lon):
    # Old Codeの関数そのまま (np.where等でエラー回避のみ追加)
    T0, T1 = 100.0, 600.0
    scaling = np.sqrt(0.306 / r_au)  # 最初のコードはsqrtでした
    cos_theta = np.cos(lat) * np.cos(lon - sub_lon)

    # 計算エラー回避のためのクリップ
    cos_clamped = np.maximum(cos_theta, 0.0)
    T = T0 + T1 * (cos_clamped ** 0.25) * scaling
    T[cos_theta <= 0] = T0
    return T


def get_orbital_params_cyclic_logic(time_sec, orbit_data):
    # Old Codeの get_orbital_params_cyclic を再現
    # ファイルの時間を基準にループさせる
    cycle_sec = ORBITAL_PERIOD

    # データの開始時間を基準(t_peri)とする
    time_col = orbit_data[:, 2]
    t_start_file = time_col[0]

    # 経過時間を周期で割った余り
    dt = time_sec
    time_in_cycle = dt % cycle_sec

    t_lookup = t_start_file + time_in_cycle

    # 補間 (データ範囲外エラーを防ぐためクリップは入れるが、理論上範囲内になる)
    t_lookup = np.clip(t_lookup, time_col[0], time_col[-1])

    au = np.interp(t_lookup, time_col, orbit_data[:, 1])
    taa_deg = np.interp(t_lookup, time_col, orbit_data[:, 0])

    # SubLon計算 (Old Code方式)
    taa_rad = np.deg2rad(taa_deg)
    omega_rot = 2 * np.pi / ROTATION_PERIOD
    rotation_angle = omega_rot * time_sec
    sub_lon = taa_rad - rotation_angle
    sub_lon = (sub_lon + np.pi) % (2 * np.pi) - np.pi

    return au, sub_lon


def load_orbit_raw(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}. Using Mock.")
        # Mock
        return np.zeros((100, 6))
    return np.loadtxt(filepath)


# === シミュレーション実行 ===
def run_simulation(mode_name, orbit_data_raw):
    print(f"Running simulation: {mode_name}")

    n_lon, n_lat = 36, 18
    lon = np.linspace(-np.pi, np.pi, n_lon)
    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    LON, LAT = np.meshgrid(lon, lat)
    cell_area = (4 * np.pi * RM ** 2) / (n_lon * n_lat)

    density = np.full_like(LON, INIT_SURF_DENS)

    history_time = []
    history_total_inventory = []
    history_total_released = []
    cumulative_released = 0.0
    supply_flux = 1e4

    # 時間設定
    t_start = 0.0
    t_max = YEARS_TO_SIMULATE * 365.25 * 86400
    t_curr = t_start
    dt = DT_STEP

    # Old Codeの条件再現用変数
    # Old Codeでは t_start_spinup から開始していた
    # ここでは t=0 を t_start_spinup とみなす
    t_start_spinup = 0.0

    while t_curr < t_max:

        # 軌道パラメータ取得 (モード別)
        if "Old" in mode_name:
            # Cyclic読み込み (これでデータが短くても無限に続く)
            r_au, s_lon = get_orbital_params_cyclic_logic(t_curr, orbit_data_raw)
        else:
            # New読み込み (線形補間、足りなければクリップされるがNewファイルは長いはず)
            # 簡易的にCyclicを使う（比較のため条件を揃える）
            r_au, s_lon = get_orbital_params_cyclic_logic(t_curr, orbit_data_raw)

        T = calculate_temp_exact(LON, LAT, r_au, s_lon)

        # レート計算 (簡易)
        rate = np.zeros_like(T)
        mask = T > 50
        # U=1.85eV
        rate[mask] = 1e13 * np.exp(- (1.85 * EV_J) / (KB * T[mask]))

        step_loss = np.zeros_like(density)

        # ==========================================
        # ロジック完全再現
        # ==========================================
        timescale = 1.0 / (rate + 1e-30)

        if mode_name == "Old_Code_Exact":
            # 【Old Code そのまま】
            # if timescale <= dt_accumulated and t_curr > t_start_spinup:

            is_eq_condition = (timescale <= dt) & (t_curr > t_start_spinup)

            # --- 分岐処理 (ベクトル化) ---

            # 1. 平衡モード (在庫投げ捨て発生)
            # dens_eq = flux_in / rate
            # density = dens_eq
            dens_eq = supply_flux / (rate + 1e-30)

            # 2. 通常モード
            # loss = dens * rate * dt
            # density -= loss
            loss_normal = density * rate * dt

            # 適用
            # Is Eq?
            # Yes -> Density becomes dens_eq. Lost amount is implicit (Inventory - dens_eq).
            #        But in the "released particles" graph, we must count what was released.
            #        Old Code: temp_loss_per_sec = surface_density * area * rate
            #        If overwritten, surface_density is low. So released is low.

            # Bugged logic reproduction:
            # If Eq: Density is overwritten. Release is calculated based on NEW density.
            # If Normal: Density is updated. Release is calculated based on OLD density.

            # 放出量 (Old Codeの temp_loss_per_sec の計算位置に基づく)
            # Old Code: 更新処理の後で loss を計算していた
            # Normal: dens -= loss. Final dens is lower. Loss calc uses final dens?
            # No, Old Code:
            #   loss_dens = dens * rate * dt
            #   surface_density += gain - loss_dens
            #   temp_loss = surface_density * rate ... (Wait, this is post-update!)

            # Let's simplify to "What actually left the surface"

            # Case Normal:
            loss_actual_normal = density * rate * dt

            # Case Eq:
            # Old Code: dens = dens_eq.
            # The "loss" variable wasn't used to update particles in Eq mode in Old Code logic!
            # Instead, temp_loss_per_sec was calculated at the end.
            # temp_loss = dens_eq * rate * area = (flux_in / rate) * rate = flux_in.
            # So in Eq mode, Old Code released "flux_in".
            # It DID throw away the inventory difference.
            loss_actual_eq = supply_flux * dt

            actual_loss = np.where(is_eq_condition, loss_actual_eq, loss_normal)

            # 密度更新
            # Eq: dens = dens_eq
            # Normal: dens -= loss_normal + supply
            dens_next_eq = dens_eq
            dens_next_normal = density - loss_normal + supply_flux * dt

            density = np.where(is_eq_condition, dens_next_eq, dens_next_normal)

            step_loss = actual_loss

        elif mode_name == "New_Bugged":
            # Step 1 の保護 (t_curr > start) がない
            is_eq_condition = (timescale <= dt)  # & True

            dens_eq = supply_flux / (rate + 1e-30)
            loss_actual_eq = supply_flux * dt
            loss_normal = density * rate * dt

            actual_loss = np.where(is_eq_condition, loss_actual_eq, loss_normal)

            dens_next_eq = dens_eq
            dens_next_normal = density - loss_normal + supply_flux * dt
            density = np.where(is_eq_condition, dens_next_eq, dens_next_normal)
            step_loss = actual_loss

        elif mode_name == "New_Fixed":
            # Step 1 保護あり (Old Codeと同じ条件) + 在庫保護ロジック
            is_eq_condition = (timescale <= dt) & (t_curr > t_start_spinup)

            total_in = density + supply_flux * dt
            dens_eq = supply_flux / (rate + 1e-30)

            # Fixed: 在庫差分を放出する
            loss_eq_fixed = np.maximum(0.0, total_in - dens_eq)
            loss_normal = density * rate * dt

            actual_loss = np.where(is_eq_condition, loss_eq_fixed, loss_normal)

            density = total_in - actual_loss
            step_loss = actual_loss

        density[density < 0] = 0

        # 記録
        if int(t_curr) % 86400 == 0:  # 1日1回
            total_inv = np.sum(density) * cell_area
            cumulative_released += np.sum(step_loss) * cell_area

            history_time.append(t_curr / 86400)
            history_total_inventory.append(total_inv)
            history_total_released.append(cumulative_released)

        t_curr += dt

    return history_time, history_total_inventory, history_total_released


# === メイン実行 ===
def main():
    # 生データを読み込む
    orbit_old = load_orbit_raw(ORBIT_FILE_OLD)
    orbit_new = load_orbit_raw(ORBIT_FILE_NEW)

    print(f"Simulating for {YEARS_TO_SIMULATE} years...")

    # 1. Old Code Exact (青)
    # 軌道: Old, ロジック: Old (t>start判定あり)
    t1, inv1, rel1 = run_simulation("Old_Code_Exact", orbit_old)

    # 2. New Bugged (赤)
    # 軌道: New, ロジック: Bugged (t>start判定なし = いきなりEq)
    t2, inv2, rel2 = run_simulation("New_Bugged", orbit_new)

    # 3. New Fixed (緑)
    # 軌道: New, ロジック: Fixed (t>start判定あり + 在庫保護)
    t3, inv3, rel3 = run_simulation("New_Fixed", orbit_new)

    # === グラフ描画 ===
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # 上段: 表面在庫
    ax1.plot(t1, inv1, label='1. Old Code (Exact Logic)', color='blue')
    ax1.plot(t2, inv2, label='2. New Bugged (No Init Protect)', color='red', linestyle='--')
    ax1.plot(t3, inv3, label='3. New Fixed (Init Protect + Save Inv)', color='green', linestyle='-.')
    ax1.set_ylabel("Total Surface Atoms")
    ax1.set_yscale('log')
    ax1.set_title("Surface Inventory Comparison")
    ax1.grid(True)
    ax1.legend()

    # 下段: 累積放出数
    ax2.plot(t1, rel1, label='1. Old Code', color='blue')
    ax2.plot(t2, rel2, label='2. New Bugged', color='red', linestyle='--')
    ax2.plot(t3, rel3, label='3. New Fixed', color='green', linestyle='-.')
    ax2.set_ylabel("Cumulative Released Particles")
    ax2.set_yscale('log')
    ax2.set_xlabel("Time (Earth Days)")
    ax2.set_title("Cumulative Released Particles")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()