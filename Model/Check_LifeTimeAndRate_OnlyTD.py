# -*- coding: utf-8 -*-
"""
==============================================================================
事後解析ツール (Global Loss Estimation Version):
【修正版】マルチバウンド(Bounce)対応
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import glob
import re

# ==============================================================================
# 1. 設定・定数
# ==============================================================================
# 解析対象ディレクトリ
TARGET_DIR = r"./SimulationResult_202512/DynamicGrid72x36_EqMode_Hard_DT500_T0100_4.0"
ORBIT_FILE = 'orbit2025_spice_unwrapped.txt'

# --- ★解析設定 ---
DT_SIMULATION = 500.0  # 【重要】検証したいタイムステップ [s]
TARGET_TAA_CENTER = 180.0
TAA_WIDTH = 10.0

# グリッド設定
N_LON_FIXED = 72
N_LAT = 36

# 物理定数
PHYSICAL_CONSTANTS = {
    'PI': np.pi,
    'MASS_NA': 3.8175e-26,  # [kg]
    'K_BOLTZMANN': 1.380649e-23,  # [J/K]
    'GM_MERCURY': 2.2032e13,  # [m^3/s^2]
    'RM': 2.440e6,  # [m]
    'EV_TO_JOULE': 1.602e-19
}


# ==============================================================================
# 2. 物理計算関数群 (コアロジック)
# ==============================================================================

def calculate_surface_temperature_leblanc(lon_rad, lat_rad, AU, subsolar_lon_rad):
    T_BASE = 100.0
    T_ANGLE = 600.0
    scaling = np.sqrt(0.306 / AU)
    cos_theta = np.cos(lat_rad) * np.cos(lon_rad - subsolar_lon_rad)
    if cos_theta <= 0:
        return T_BASE
    return T_BASE + T_ANGLE * (cos_theta ** 0.25) * scaling


def calculate_thermal_desorption_rate(temp_k):
    """放出率係数 k [s^-1] の計算"""
    if temp_k < 10.0: return 0.0
    KB = PHYSICAL_CONSTANTS['K_BOLTZMANN']
    VIB_FREQ = 1e13
    U_MEAN, U_MIN, U_MAX, SIGMA = 1.85, 1.40, 2.70, 0.20

    u_ev_grid = np.linspace(U_MIN, U_MAX, 50)
    u_joule_grid = u_ev_grid * PHYSICAL_CONSTANTS['EV_TO_JOULE']

    pdf = np.exp(- (u_ev_grid - U_MEAN) ** 2 / (2 * SIGMA ** 2))
    norm_pdf = pdf / np.sum(pdf)

    exponent = -u_joule_grid / (KB * temp_k)
    rates = np.zeros_like(u_ev_grid)
    mask = exponent > -700
    rates[mask] = VIB_FREQ * np.exp(exponent[mask])

    return np.sum(rates * norm_pdf)


def calculate_sticking_probability(surface_temp_K):
    """付着確率の計算 (Yakshinskiy et al. based)"""
    A = 0.0804
    B = 458.0
    porosity = 0.8

    if surface_temp_K <= 0: return 1.0

    p_stick = A * np.exp(B / surface_temp_K)
    p_stick_eff = p_stick / (1.0 - (1.0 - p_stick) * porosity)

    return min(p_stick_eff, 1.0)


def get_orbital_info_from_time(rel_hours, orbit_data):
    ROTATION_PERIOD = 58.6462 * 86400
    idx_peri = np.argmin(np.abs(orbit_data[:, 0]))
    t_peri = orbit_data[idx_peri, 2]
    current_time = t_peri + rel_hours * 3600.0
    time_col = orbit_data[:, 2]

    idx = np.searchsorted(time_col, current_time)
    if idx >= len(time_col): idx = len(time_col) - 1
    if idx > 0 and abs(current_time - time_col[idx - 1]) < abs(current_time - time_col[idx]):
        idx -= 1

    au = orbit_data[idx, 1]
    taa_deg_raw = orbit_data[idx, 0]
    omega_rot = 2 * np.pi / ROTATION_PERIOD
    rotation_angle = omega_rot * (current_time - t_peri)
    taa_rad = np.deg2rad(taa_deg_raw)
    sub_lon = taa_rad - rotation_angle
    sub_lon = (sub_lon + np.pi) % (2 * np.pi) - np.pi
    return taa_deg_raw % 360.0, au, sub_lon


# ==============================================================================
# 3. 消失率計算エンジン (マルチバウンド対応版)
# ==============================================================================

class LossRatioCalculator:
    def __init__(self, dt_threshold, num_samples=50000):
        self.dt = dt_threshold
        self.num_samples = num_samples
        self.interp_func = self._build_lookup_table()

    def _run_monte_carlo(self, temp_k):
        """
        指定温度でのLost RatioをMC計算
        【変更点】バウンド処理を追加
        """
        mass = PHYSICAL_CONSTANTS['MASS_NA']
        g_surf = PHYSICAL_CONSTANTS['GM_MERCURY'] / (PHYSICAL_CONSTANTS['RM'] ** 2)
        kb = PHYSICAL_CONSTANTS['K_BOLTZMANN']
        kt = kb * temp_k

        # sticking probability at this temperature
        p_stick = calculate_sticking_probability(temp_k)

        # 粒子の状態追跡用: 残り時間 (最初は全員 DT 持っている)
        remaining_time = np.full(self.num_samples, self.dt)
        alive_mask = np.ones(self.num_samples, dtype=bool)  # まだ判定中かどうか
        final_lost = np.zeros(self.num_samples, dtype=bool)  # 最終的に死んだか(Stuck)

        # エネルギー適応係数 (シミュレーション設定に合わせる)
        BETA = 1.0  # 完全順応と仮定

        # 最大ループ回数 (無限ループ防止)
        MAX_BOUNCES = 50

        current_temp = temp_k  # 最初の温度

        for _ in range(MAX_BOUNCES):
            # まだ判定中の粒子だけ計算
            if not np.any(alive_mask):
                break

            n_active = np.sum(alive_mask)

            # 1. 速度サンプリング (Gamma(2, kT))
            # 再放出時はその場の温度(current_temp)で熱化すると仮定
            kt_curr = kb * current_temp
            E = np.random.gamma(2.0, kt_curr, n_active)
            speeds = np.sqrt(2.0 * E / mass)

            # 2. 角度サンプリング
            u2 = np.random.random(n_active)
            cos_thetas = np.sqrt(1.0 - u2)
            v_z = speeds * cos_thetas

            # 3. 飛行時間
            t_flight = 2.0 * v_z / g_surf

            # --- 判定ロジック ---
            # A. 飛行時間が残り時間より長い -> 生存確定 (Alive)
            survived_step = t_flight >= remaining_time[alive_mask]

            # B. 飛行時間が残り時間より短い -> 着地 (Bounce check)
            landed = ~survived_step

            # インデックス操作用の一時配列
            current_indices = np.where(alive_mask)[0]

            # 生存確定したものをマスクから外す (これ以上追跡不要)
            alive_mask[current_indices[survived_step]] = False

            # 着地したものについて Stick 判定
            if np.any(landed):
                n_landed = np.sum(landed)

                # Stick判定: random < p_stick なら吸着(Lost確定)
                # 注: 厳密には着地地点の温度でp_stickが変わりますが、
                # ここでは「局所的な温度変化はない(Short hop)」と仮定して同じ温度を使います
                r_vals = np.random.random(n_landed)
                stuck = r_vals < p_stick

                landed_indices = current_indices[landed]

                # 吸着した粒子 -> Lost確定, マスクから外す
                final_lost[landed_indices[stuck]] = True
                alive_mask[landed_indices[stuck]] = False

                # 跳ね返る粒子 -> 残り時間を減らしてループ継続
                bounced_mask = ~stuck
                bounced_indices = landed_indices[bounced_mask]

                # 着地した粒子の飛行時間を引く
                # (配列操作がややこしいので、landed全体から抽出して減算)
                subset_time = remaining_time[landed_indices]
                subset_flight = t_flight[landed]

                # 跳ね返る分だけ更新
                remaining_time[bounced_indices] -= subset_flight[bounced_mask]

                # エネルギー順応 (次の温度)
                # E_out = Beta * E_wall + (1-Beta)*E_in だが、
                # ここでは簡単のため E_wall (current_temp) で完全熱化(Beta=1)として次へ
                pass

                # ループを抜けてもまだ alive_mask が True のものは
        # (MAX_BOUNCES回跳ねても時間が余っている -> 稀だが一応生存扱い)
        # final_lost は False のままなのでOK

        loss_count = np.sum(final_lost)
        return loss_count / self.num_samples

    def _build_lookup_table(self):
        print(f"Building Loss Ratio Lookup Table (DT={self.dt}s) with Multi-Bounce...")
        # 50K から 1000K までをカバー
        temps = np.concatenate([
            np.linspace(50, 200, 10),
            np.linspace(200, 800, 20),
            np.linspace(800, 1500, 5)
        ])
        ratios = []
        for t in temps:
            ratios.append(self._run_monte_carlo(t))

        # 補間関数作成
        return interp1d(temps, ratios, kind='linear', fill_value="extrapolate")

    def get_loss_ratio(self, temp_k):
        """温度(配列可)から消失率を返す"""
        # 10K以下は計算不能/ゼロとする
        t_safe = np.maximum(temp_k, 10.0)
        return np.clip(self.interp_func(t_safe), 0.0, 1.0)


# ==============================================================================
# 4. 全球消失率解析ルーチン
# ==============================================================================

def analyze_weighted_global_loss(target_taa, width, files, orbit_data, loss_calculator):
    print(f"\n--- Analyzing Global Loss Ratio (Weighted by Production) ---")

    # --- 対象ファイル抽出 ---
    target_files = []
    taa_min = target_taa - width / 2.0
    taa_max = target_taa + width / 2.0
    cross_zero = (taa_min < 0) or (taa_max > 360)

    if taa_min < 0: taa_min += 360
    if taa_max > 360: taa_max -= 360

    for fpath in files:
        match = re.search(r"t(\d+)", fpath)
        if not match: continue
        rel_hours = int(match.group(1))
        taa, au, sub_lon = get_orbital_info_from_time(rel_hours, orbit_data)

        hit = False
        if not cross_zero:
            if taa_min <= taa <= taa_max: hit = True
        else:
            if taa >= taa_min or taa <= taa_max: hit = True

        if hit: target_files.append((fpath, au, sub_lon))

    if not target_files:
        print("No files found.")
        return

    # --- 集計用変数 ---
    total_produced_atoms_per_sec = 0.0  # 全球生成量 (weightの分母)
    total_lost_atoms_per_sec = 0.0  # 失われた量 (分子)

    # グリッド計算用
    lon_edges = np.linspace(-np.pi, np.pi, N_LON_FIXED + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    dlon = lon_edges[1] - lon_edges[0]
    # セル面積
    cell_areas = (PHYSICAL_CONSTANTS['RM'] ** 2) * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))
    area_grid = np.tile(cell_areas, (N_LON_FIXED, 1))

    # 中心座標
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    LON_GRID, LAT_GRID = np.meshgrid(lon_centers, lat_centers, indexing='ij')

    # ファイルループ (平均化のためすべて積算)
    count_files = 0

    for fpath, au, sub_lon in target_files:
        surf_dens = np.load(fpath)  # [lon, lat]

        # 1. 温度計算 (全グリッド一括)
        # cos_theta計算
        cos_z = np.cos(LAT_GRID) * np.cos(LON_GRID - sub_lon)
        # 温度マップ
        scaling = np.sqrt(0.306 / au)
        temp_map = np.full_like(cos_z, 100.0)  # Night default
        mask_day = cos_z > 0
        temp_map[mask_day] = 100.0 + 600.0 * (cos_z[mask_day] ** 0.25) * scaling

        # 2. 生成フラックス計算 (Flux = k * density)
        production_map = np.zeros_like(surf_dens)

        # 温度マップから放出率係数kを計算するのも重いので、ここだけループ
        for i in range(N_LON_FIXED):
            for j in range(N_LAT):
                t_val = temp_map[i, j]
                dens_val = surf_dens[i, j]
                if t_val < 100.1 and dens_val < 1e5: continue  # 夜側＆低密度はスキップ

                k_val = calculate_thermal_desorption_rate(t_val)
                flux_val = k_val * dens_val
                production_map[i, j] = flux_val * area_grid[i, j]

        # 3. 消失率 (Lost Ratio) の取得 (Lookup Table)
        loss_ratio_map = loss_calculator.get_loss_ratio(temp_map)

        # 4. 消失量マップ
        lost_amount_map = production_map * loss_ratio_map

        # 5. 積算
        total_produced_atoms_per_sec += np.sum(production_map)
        total_lost_atoms_per_sec += np.sum(lost_amount_map)
        count_files += 1

    # --- 結果算出 ---
    if total_produced_atoms_per_sec > 0:
        global_loss_ratio = (total_lost_atoms_per_sec / total_produced_atoms_per_sec) * 100.0

        print(f"\n=== RESULT: Global Weighted Loss Ratio (DT={DT_SIMULATION}s) ===")
        print(f" Analyzed Files : {count_files}")
        print(f" Total Production : {total_produced_atoms_per_sec:.4e} [atoms/s]")
        print(f" Total Lost       : {total_lost_atoms_per_sec:.4e} [atoms/s]")
        print(f" --------------------------------------------------")
        print(f" GLOBAL LOSS RATIO: {global_loss_ratio:.2f} %")
        print(f" --------------------------------------------------")

        if global_loss_ratio > 50.0:
            print(" [WARNING] More than half of generated particles are lost!")
            print(" Consider reducing DT or using residence-time weighting.")
    else:
        print("Total production is zero. Check simulation data.")


# ==============================================================================
# 5. メイン実行
# ==============================================================================
def main():
    # 1. ファイル検索
    pattern = os.path.join(TARGET_DIR, "surface_density_*.npy")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found in {TARGET_DIR}")
        return

    # 2. 軌道データ読み込み
    orbit_data = np.loadtxt(ORBIT_FILE)
    orbit_data[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(orbit_data[:, 0])))

    # 3. 消失率テーブルの作成 (一度だけ実行)
    loss_calc = LossRatioCalculator(DT_SIMULATION)

    # 4. 解析実行
    analyze_weighted_global_loss(TARGET_TAA_CENTER, TAA_WIDTH, files, orbit_data, loss_calc)


if __name__ == "__main__":
    main()