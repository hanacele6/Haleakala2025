# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import glob
import os
import re
import sys

# ==============================================================================
# 設定
# ==============================================================================
BASE_OUTPUT_DIRECTORY = r"./SimulationResult_202606"

# 基準(Reference)と、比較したいターゲットを指定
REF_LABEL = 'dt = 1s (Ref)'
REF_DIR = os.path.join(BASE_OUTPUT_DIRECTORY, "ParabolicHop_72x36_NoEq_DT1_0616_Multi_BD0.5_UG_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)")

TGT_LABEL = 'dt = 100s'
TGT_DIR = os.path.join(BASE_OUTPUT_DIRECTORY, "ParabolicHop_72x36_NoEq_DT100_0609_Multi_BD0.5_UG_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)_1e24")

TARGET_TAAS = [100, 180, 300]

N_LON, N_LAT = 72, 36
ALIGN_SUN_TO_CENTER = True
ORBIT_FILE_PATH = 'orbit2025_spice_unwrapped.txt'

COLOR_VMIN = 10 ** 9.5
COLOR_VMAX = 10 ** 14.5

# 対象天体の半径 (km) (例: 月=1737.4, 水星=2439.7)
R_BODY_KM = 1737.4

# ==============================================================================
# データ読み込み・計算関数
# ==============================================================================
def load_orbit_data(orbit_file_path):
    try:
        orbit_data = np.loadtxt(orbit_file_path)
        orbit_data[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(orbit_data[:, 0])))
        return orbit_data, orbit_data[0, 2]
    except Exception as e:
        print(f"軌道ファイル読み込みエラー: {e}")
        sys.exit(1)

def get_subsolar_longitude(time_h, t_start, orbit_data):
    current_t_sec = t_start + (float(time_h) * 3600.0)
    t_lookup = np.clip(current_t_sec, orbit_data[0, 2], orbit_data[-1, 2])
    return np.interp(t_lookup, orbit_data[:, 2], orbit_data[:, 5])

def find_closest_surface_file(target_dir, target_taa):
    search_path = os.path.join(target_dir, "density_grid_*_taa*.npy")
    grid_files = glob.glob(search_path)
    if not grid_files: return None, None
        
    closest_diff = 999
    best_time_h, best_surf_path = None, None
    for f in grid_files:
        match = re.search(r'_t(\d+)_taa(\d+)\.npy$', os.path.basename(f))
        if match:
            time_h, taa = int(match.group(1)), int(match.group(2))
            diff = abs(taa - target_taa)
            if diff < closest_diff or (diff == closest_diff and best_time_h is not None and time_h > best_time_h):
                surf_path = os.path.join(target_dir, f"surface_density_t{time_h:05d}.npy")
                if os.path.exists(surf_path):
                    closest_diff, best_time_h, best_surf_path = diff, time_h, surf_path
    return best_surf_path, best_time_h

def load_and_format_data(filepath, time_h, t_start, orbit_data):
    try:
        data = np.load(filepath)
        if data.ndim == 3: data = np.sum(data, axis=2)
    except Exception as e:
        return np.zeros((N_LAT, N_LON))
        
    subsolar_lon = get_subsolar_longitude(time_h, t_start, orbit_data)
    
    if ALIGN_SUN_TO_CENTER:
        dlon = 360.0 / N_LON
        sun_pos_norm = (subsolar_lon + 180.0) % 360.0
        sun_index = int(np.round(sun_pos_norm / dlon)) % N_LON
        shift = (N_LON // 2) - sun_index
        data = np.roll(data, shift=shift, axis=0)
        
    data_T = np.nan_to_num(data.T, nan=0.0)
    data_T /= 10000.0  # cm^2 に変換
    return data_T

def calculate_cell_areas(n_lon, n_lat, r_body_km):
    """
    球体モデルに基づき、各グリッドの物理面積(cm^2)を計算して返す
    """
    r_body_cm = r_body_km * 1e5  # km -> cm
    dlon_rad = 2.0 * np.pi / n_lon
    
    # 緯度のエッジ（-90度から90度まで）をラジアンで等分割
    lat_edges = np.linspace(-np.pi/2, np.pi/2, n_lat + 1)
    
    areas = np.zeros((n_lat, n_lon))
    for i in range(n_lat):
        # 該当緯度帯の面積: R^2 * dlon * (sin(lat_upper) - sin(lat_lower))
        area_lat = (r_body_cm ** 2) * dlon_rad * (np.sin(lat_edges[i+1]) - np.sin(lat_edges[i]))
        areas[i, :] = area_lat
        
    return areas

# ==============================================================================
# インタラクティブ比較ビューワー クラス
# ==============================================================================
class InteractiveDiffViewer:
    def __init__(self, ref_data, tgt_data, diff_data, target_taa):
        self.ref_data = ref_data
        self.tgt_data = tgt_data
        self.diff_data = diff_data
        
        self.fig, self.axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        self.fig.canvas.manager.set_window_title(f"Comparison TAA = {target_taa}")
        
        lon_edges = np.linspace(-180, 180, N_LON + 1)
        lat_edges = np.linspace(-90, 90, N_LAT + 1)
        
        # カラースケール設定
        norm_abs = LogNorm(vmin=COLOR_VMIN, vmax=COLOR_VMAX)
        diff_max = np.max(np.abs(self.diff_data)) * 0.7
        if diff_max == 0: diff_max = 1.0
        norm_diff = Normalize(vmin=-diff_max, vmax=diff_max)
        
        # 1. Reference
        im_ref = self.axes[0].pcolormesh(lon_edges, lat_edges, self.ref_data, cmap='inferno', norm=norm_abs, shading='flat')
        self.axes[0].set_title(REF_LABEL)
        self.fig.colorbar(im_ref, ax=self.axes[0], fraction=0.046, pad=0.04)
        
        # 2. Target
        im_tgt = self.axes[1].pcolormesh(lon_edges, lat_edges, self.tgt_data, cmap='inferno', norm=norm_abs, shading='flat')
        self.axes[1].set_title(TGT_LABEL)
        self.fig.colorbar(im_tgt, ax=self.axes[1], fraction=0.046, pad=0.04)
        
        # 3. Difference
        im_diff = self.axes[2].pcolormesh(lon_edges, lat_edges, self.diff_data, cmap='RdBu_r', norm=norm_diff, shading='flat')
        self.axes[2].set_title(f"Difference ({TGT_LABEL} - Ref)")
        self.fig.colorbar(im_diff, ax=self.axes[2], fraction=0.046, pad=0.04)
        
        x_label = "Longitude (Sun Centered)" if ALIGN_SUN_TO_CENTER else "Longitude"
        
        for ax in self.axes:
            ax.set_aspect('equal')
            ax.set_xlabel(x_label)
            ax.set_ylabel("Latitude")
            ax.axvline(-90, color='white' if ax != self.axes[2] else 'black', linestyle='--', alpha=0.5)
            ax.axvline(90, color='white' if ax != self.axes[2] else 'black', linestyle='--', alpha=0.5)
            
        # ホバー情報のテキストボックス (図の下部に配置)
        self.info_text = self.fig.text(
            0.5, 0.02, '', transform=self.fig.transFigure,
            ha='center', va='bottom', fontsize=11, color='black',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9)
        )
        
        self.fig.suptitle(f"Surface Density at TAA = {target_taa}$^\circ$", fontsize=14, fontweight='bold')
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def on_mouse_move(self, event):
        if event.inaxes not in self.axes:
            self.info_text.set_text('')
            self.fig.canvas.draw_idle()
            return
            
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
            
        # 座標から配列インデックスを計算
        lon_idx = int((x + 180) / 360 * N_LON)
        lat_idx = int((y + 90) / 180 * N_LAT)
        lon_idx = np.clip(lon_idx, 0, N_LON - 1)
        lat_idx = np.clip(lat_idx, 0, N_LAT - 1)
        
        val_ref = self.ref_data[lat_idx, lon_idx]
        val_tgt = self.tgt_data[lat_idx, lon_idx]
        val_diff = self.diff_data[lat_idx, lon_idx]
        
        # テキストの更新
        self.info_text.set_text(
            f"Lon: {x:+.1f}°, Lat: {y:+.1f}° | "
            f"Ref: {val_ref:.2e} | Tgt: {val_tgt:.2e} | Diff: {val_diff:+.2e}"
        )
        self.fig.canvas.draw_idle()

# ==============================================================================
# メイン処理
# ==============================================================================
def main():
    orbit_data, t_start = load_orbit_data(ORBIT_FILE_PATH)
    
    # 緯度による面積の違いを考慮した各セルの物理面積(cm^2)配列を生成
    cell_areas = calculate_cell_areas(N_LON, N_LAT, R_BODY_KM)
    
    for target_taa in TARGET_TAAS:
        print(f"\n========================================")
        print(f" TAA = {target_taa} の解析")
        print(f"========================================")
        
        ref_path, ref_time = find_closest_surface_file(REF_DIR, target_taa)
        tgt_path, tgt_time = find_closest_surface_file(TGT_DIR, target_taa)
        
        if not ref_path or not tgt_path:
            print("データが揃わないためスキップします。")
            continue
            
        ref_data = load_and_format_data(ref_path, ref_time, t_start, orbit_data)
        tgt_data = load_and_format_data(tgt_path, tgt_time, t_start, orbit_data)
        diff_data = tgt_data - ref_data
        
        # --- 総原子数の計算（密度 cm^-2 × 面積 cm^2） ---
        sum_ref_atoms = np.sum(ref_data * cell_areas)
        sum_tgt_atoms = np.sum(tgt_data * cell_areas)
        sum_diff_atoms = sum_tgt_atoms - sum_ref_atoms
        diff_percent = (sum_diff_atoms / sum_ref_atoms) * 100 if sum_ref_atoms > 0 else 0
        
        print(f"[総原子数の比較]")
        print(f"  {REF_LABEL} : {sum_ref_atoms:.4e} atoms")
        print(f"  {TGT_LABEL}   : {sum_tgt_atoms:.4e} atoms")
        print(f"  総原子数の差分 : {sum_diff_atoms:+.4e} atoms ({diff_percent:+.2f}%)")
        print(f"----------------------------------------")
        print(f"※ ウィンドウを閉じると次のTAAへ進みます。")
        
        # ウィンドウを立ち上げて描画・ホバー待機
        viewer = InteractiveDiffViewer(ref_data, tgt_data, diff_data, target_taa)
        plt.show()

if __name__ == "__main__":
    main()