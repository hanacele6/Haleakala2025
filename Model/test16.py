# -*- coding: utf-8 -*-
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401
from matplotlib.ticker import LogLocator, NullFormatter

# ==========================================
# 設定（ご自身の環境に合わせて調整してください）
# ==========================================
BASE_DIR = r"./SimulationResult_202607"
MODELS = {
    "Q2.0 (Standard)": "ParabolicHop_72x36_NoEq_DT100_0716_BD0.8_UG1.85_Q2.0_A2.0e+07_LT190k_15yr_test2",
    "Q0.3 (Weak PSD)": "ParabolicHop_72x36_NoEq_DT100_0716_BD0.8_UG1.85_Q0.3_A2.0e+07_LT190k_15yr_test2",
}
PROFILE_TAAS = [120, 140, 160, 180]
ORBIT_FILE_PATH = 'orbit2025_spice_unwrapped.txt'

N_LON, N_LAT = 72, 36
R_BODY_KM = 2439.7
ZENITH_N_BINS = 20

# ==========================================
# 依存関数群
# ==========================================
def load_orbit_data(orbit_file_path):
    try:
        orbit_data = np.loadtxt(orbit_file_path)
        orbit_data[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(orbit_data[:, 0])))
        t_file_start = orbit_data[0, 2]
        return orbit_data, t_file_start
    except Exception as e:
        print(f"軌道ファイル読み込みエラー: {e}")
        return None, None

def get_subsolar_longitude_linear(time_h, t_file_start, orbit_data):
    time_col = orbit_data[:, 2]
    current_t = t_file_start + (float(time_h) * 3600.0)
    t_lookup = np.clip(current_t, time_col[0], time_col[-1])
    return np.interp(t_lookup, time_col, orbit_data[:, 5])

def calculate_cell_areas(n_lon, n_lat, r_body_km):
    dlon_rad = 2.0 * np.pi / n_lon
    lat_edges = np.linspace(-np.pi/2, np.pi/2, n_lat + 1)
    areas_cm2 = np.zeros((n_lat, n_lon))
    r_body_cm = r_body_km * 1e5
    for i in range(n_lat):
        factor = np.sin(lat_edges[i+1]) - np.sin(lat_edges[i])
        areas_cm2[i, :] = (r_body_cm ** 2) * dlon_rad * factor
    return areas_cm2

def compute_effcos_grid(n_lon, n_lat, subsolar_lon_deg):
    lon_centers = np.linspace(-180, 180, n_lon + 1)
    lon_centers = (lon_centers[:-1] + lon_centers[1:]) / 2.0
    lat_centers = np.linspace(-90, 90, n_lat + 1)
    lat_centers = (lat_centers[:-1] + lat_centers[1:]) / 2.0
    lon_sun = (np.deg2rad(lon_centers)[None, :] - np.deg2rad(subsolar_lon_deg) + np.pi) % (2 * np.pi) - np.pi
    lat2d = np.deg2rad(lat_centers)[:, None]
    eff_cos = np.cos(lat2d) * np.cos(lon_sun)
    is_dawn = np.broadcast_to((lon_sun < 0.0), eff_cos.shape)
    return eff_cos, is_dawn

def get_closest_surface_density_file(target_dir, target_taa):
    grid_files = glob.glob(os.path.join(target_dir, "density_grid_*.npy"))
    if not grid_files: return None, None, None
    min_diff = float('inf')
    best = (None, None, None)
    for f in grid_files:
        m = re.search(r'_t(\d+)_taa(\d+)\.npy$', os.path.basename(f))
        if m:
            time_id, taa = int(m.group(1)), int(m.group(2))
            diff = abs(taa - target_taa)
            if diff < min_diff:
                surf = os.path.join(target_dir, f"surface_density_t{time_id:05d}.npy")
                if os.path.exists(surf):
                    min_diff = diff
                    best = (surf, time_id, taa)
    return best

def load_and_align_density(filepath, time_h, orbit_data, t_start):
    data = np.load(filepath)
    if data.ndim == 3: data = np.sum(data, axis=2)
    subsolar_lon_deg = get_subsolar_longitude_linear(time_h, t_start, orbit_data)
    sun_index = int(np.round(((subsolar_lon_deg + 180.0) % 360.0) / (360.0 / N_LON))) % N_LON
    data = np.roll(data, shift=(N_LON // 2) - sun_index, axis=0)
    data_T = np.nan_to_num(data.T, nan=0.0) / 10000.0  # cm-2 への変換等
    eff_cos, is_dawn = compute_effcos_grid(N_LON, N_LAT, 0.0)
    return data_T, eff_cos, is_dawn

# ==========================================
# メインプロット＆数値比較関数
# ==========================================
def plot_zenith_density_profile_both_sides(models, taa_list, orbit_data, t_start, n_bins=20):
    if orbit_data is None:
        print("軌道データがないためプロットを中止します。")
        return

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    areas_cm2 = calculate_cell_areas(N_LON, N_LAT, R_BODY_KM)

    # 結果を保持する辞書: results[model_label][actual_taa] = {'dawn': count, 'dusk': count}
    results = {}

    for label, subdir in models.items():
        target_dir = os.path.join(BASE_DIR, subdir)
        results[label] = {}
        
        print("\n" + "="*60)
        print(f" モデル: {label}")
        print("="*60)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        cmap = plt.get_cmap('viridis')
        colors = [cmap(x) for x in np.linspace(0, 0.85, len(taa_list))]
        
        for ci, target_taa in enumerate(taa_list):
            filepath, time_h, actual_taa = get_closest_surface_density_file(target_dir, target_taa)
            if not filepath: 
                continue
            try:
                data_T, eff_cos, is_dawn = load_and_align_density(filepath, time_h, orbit_data, t_start)
            except Exception as e:
                print(f"  [エラー] ファイル読み込み失敗 {filepath}: {e}")
                continue
            
            day_mask = eff_cos > 0.0
            
            total_atoms_dawn = 0.0
            total_atoms_dusk = 0.0
            
            # Dawn側とDusk側のマスク処理とプロット
            for ax_idx, side_name in enumerate(["DAWN", "DUSK"]):
                ax = axes[ax_idx]
                if side_name == "DAWN":
                    side_mask = day_mask & is_dawn
                else:
                    side_mask = day_mask & (~is_dawn)
                
                ec_flat = eff_cos[side_mask]
                dens_flat = data_T[side_mask]
                area_flat = areas_cm2[side_mask]
                
                total_atoms_side = np.sum(dens_flat * area_flat)
                if side_name == "DAWN":
                    total_atoms_dawn = total_atoms_side
                else:
                    total_atoms_dusk = total_atoms_side
                
                # ビニング処理（プロット用）
                prof = np.full(n_bins, np.nan)
                for b in range(n_bins):
                    m = (ec_flat >= bin_edges[b]) & (ec_flat < bin_edges[b+1])
                    if np.any(m):
                        prof[b] = np.sum(dens_flat[m] * area_flat[m]) / np.sum(area_flat[m])
                
                ax.plot(bin_centers, prof, '-o', color=colors[ci], markersize=5, lw=1.8, 
                        label=f'TAA={actual_taa}°')

            # 結果を辞書に保存
            results[label][actual_taa] = {
                'dawn': total_atoms_dawn,
                'dusk': total_atoms_dusk
            }

            # --- [1] 同一モデル内での Dawn vs Dusk 比較 ---
            diff_atoms = total_atoms_dawn - total_atoms_dusk
            ratio = total_atoms_dawn / total_atoms_dusk if total_atoms_dusk > 0 else float('nan')
            
            print(f" 🌟 TAA = {actual_taa}° (ファイル指定TAA: {target_taa}°):")
            print(f"    - Dawn側 全原子数: {total_atoms_dawn:.3e} atoms")
            print(f"    - Dusk側 全原子数: {total_atoms_dusk:.3e} atoms")
            print(f"    - 差分 (Dawn - Dusk): {diff_atoms:+.3e} atoms")
            print(f"    - 比率 (Dawn / Dusk): {ratio:.3f}")
            print("-" * 45)

        # サブプロットごとのレイアウト調整
        for ax_idx, side_title in enumerate(["明け方側 (Dawn)", "夕方側 (Dusk)"]):
            ax = axes[ax_idx]
            ax.set_xlabel('eff_cos [0=ターミネーター → 1=SSP]', fontsize=11)
            ax.set_title(f'{side_title}', fontsize=12, fontweight='bold')
            ax.set_yscale('log')
            ax.grid(True, which='both', ls='--', alpha=0.5)
            
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=10))
            ax.yaxis.set_minor_formatter(NullFormatter())
            
            if ax_idx == 0:
                ax.set_ylabel('平均表面密度 [atoms/cm²]', fontsize=11)
                ax.legend(title='True Anomaly', loc='lower left')

        fig.suptitle(f'{label} — 天頂角に対する表面密度プロファイル (Dawn vs Dusk 比較)', fontsize=14, y=0.98)
        plt.tight_layout()
        plt.show()

    # ==========================================
    # --- [2] モデル間 (Qの違い) の比較プリント ---
    # ==========================================
    model_labels = list(models.keys())
    if len(model_labels) >= 2:
        m1, m2 = model_labels[0], model_labels[1]  # Q2.0 vs Q0.3
        
        print("\n" + "#"*60)
        print(f" 🔬 Qの違いによる比較: [{m1}]  vs  [{m2}]")
        print("#"*60)
        
        # 共通して存在するTAAで比較
        common_taas = sorted(list(set(results[m1].keys()) & set(results[m2].keys())))
        
        for taa in common_taas:
            d1, d2 = results[m1][taa]['dawn'], results[m2][taa]['dawn']
            k1, k2 = results[m1][taa]['dusk'], results[m2][taa]['dusk']
            
            print(f"\n 📍 TAA = {taa}°:")
            
            # --- Dawn同士の比較 ---
            dawn_diff = d1 - d2
            dawn_ratio = d1 / d2 if d2 > 0 else float('nan')
            print(f"   【Dawn同士】")
            print(f"      • {m1}: {d1:.3e} atoms")
            print(f"      • {m2}: {d2:.3e} atoms")
            print(f"      • 差分 ({m1} - {m2}): {dawn_diff:+.3e} atoms")
            print(f"      • 比率 ({m1} / {m2}): {dawn_ratio:.3f}")
            
            # --- Dusk同士の比較 ---
            dusk_diff = k1 - k2
            dusk_ratio = k1 / k2 if k2 > 0 else float('nan')
            print(f"   【Dusk同士】")
            print(f"      • {m1}: {k1:.3e} atoms")
            print(f"      • {m2}: {k2:.3e} atoms")
            print(f"      • 差分 ({m1} - {m2}): {dusk_diff:+.3e} atoms")
            print(f"      • 比率 ({m1} / {m2}): {dusk_ratio:.3f}")
            print("-" * 50)

# ==========================================
# 実行部
# ==========================================
if __name__ == "__main__":
    if os.path.exists(ORBIT_FILE_PATH):
        orbit_data, t_start = load_orbit_data(ORBIT_FILE_PATH)
        plot_zenith_density_profile_both_sides(MODELS, PROFILE_TAAS, orbit_data, t_start, n_bins=ZENITH_N_BINS)
    else:
        print(f"[エラー] 軌道ファイル ({ORBIT_FILE_PATH}) が見つかりません。")