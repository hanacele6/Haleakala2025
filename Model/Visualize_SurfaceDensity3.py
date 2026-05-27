# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.widgets import Button, Slider, CheckButtons, RadioButtons
import glob
import os
import re
import sys

# ==============================================================================
# 設定
# ==============================================================================

USE_PAPER_SCALE = True  
COLOR_VMIN = 1.0e10
COLOR_VMAX = 1.0e18

N_LON, N_LAT = 72, 36
BASE_OUTPUT_DIRECTORY = r"./SimulationResult_202605"
#BASE_OUTPUT_DIRECTORY = r"./SimulationResult_202604"
#RUN_NAME = "ParabolicHop_72x36_NoEq_DT100_0501_Multi_BD0.4SD0.6,4e3_U1.85_Q0.27_Bouncetau30s_A0.5_LongLT_CHECK"
#RUN_NAME = "ParabolicHop_72x36_NoEq_DT100_0505_Fixed_BD0.4_U1.85_Q0.27_Bouncetau30s_A0.5_LongLT"
RUN_NAME = "ParabolicHop_72x36_NoEq_DT100_0508_Fixed_BD0.4_U1.85_Q2.0_Bouncetau30s_A1.0_LongLT(Fulle)"

#output_dir = r"./SimulationResult_202604/ParabolicHop_72x36_NoEq_DT100_0427_Multi_0.4Denabled_U1.85_Q0.27_Bouncetau30s_A0.5_LongLT"
#
#RUN_NAME = "ParabolicHop_72x36_NoEq_DT100_0504_Multi_BD0.4SD0.6,4e3_U1.85_Q0.27_Bouncetau30s_A0.5_LongLT_CHECK_定常状態どんなもん"
#RUN_NAME = "ParabolicHop_72x36_NoEq_DT100_0427_Multi_0.4Denabled_U1.85_Q0.27_Bouncetau30s_A0.5_LongLT"



INITIAL_TARGET_TAA = 100
ORBIT_FILE_PATH = 'orbit2025_spice_unwrapped.txt'
ALIGN_SUN_TO_CENTER = True
USE_LOG_SCALE = True
MERCURY_YEAR_SEC = 87.969 * 86400
SPIN_UP_YEARS = 2.0


# ==============================================================================
# 関数群 (計算・IOロジック)
# ==============================================================================

def load_orbit_data(orbit_file_path):
    try:
        orbit_data = np.loadtxt(orbit_file_path)
        orbit_data[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(orbit_data[:, 0])))
        t_file_start = orbit_data[0, 2]
        print(f"軌道ファイル読み込み: {orbit_file_path}")
        return orbit_data, t_file_start
    except Exception as e:
        print(f"軌道ファイル読み込みエラー: {e}")
        sys.exit(1)


def get_subsolar_longitude_linear(time_h, t_file_start, orbit_data):
    time_col_original = orbit_data[:, 2]
    # 注意: ここでのtime_hは「シミュレーション開始(1年目)」からの経過時間である前提
    current_t_sec = t_file_start + (float(time_h) * 3600.0)
    t_lookup = np.clip(current_t_sec, time_col_original[0], time_col_original[-1])
    sub_lon_deg = np.interp(t_lookup, time_col_original, orbit_data[:, 5])
    return sub_lon_deg


def get_all_files_grouped_by_year(target_dir):
    """ディレクトリ内の全ファイルをTAA順にソートし、年(公転)ごとにグループ化して返す"""
    search_path_grid = os.path.join(target_dir, "density_grid_*.npy")
    grid_files = glob.glob(search_path_grid)

    if not grid_files:
        print(f"エラー: {target_dir} に density_grid ファイルがありません。")
        return {}

    all_files = []
    for f in grid_files:
        match = re.search(r'_t(\d+)_taa(\d+)\.npy$', os.path.basename(f))
        if match:
            time_id = int(match.group(1))
            taa = int(match.group(2))
            surf_filename = f"surface_density_t{time_id:05d}.npy"
            surf_filepath = os.path.join(target_dir, surf_filename)
            if os.path.exists(surf_filepath):
                all_files.append({
                    'taa': taa,
                    'time_h': time_id,
                    'path': surf_filepath
                })

    all_files.sort(key=lambda x: x['time_h']) # 時間順に並べる

    # 年ごとに分割 (1年 = MERCURY_YEAR_SEC)
    grouped_files = {"Year 1": [], "Year 2": [], "Year 3 (Final)": []}
    
    for f in all_files:
        time_sec = f['time_h'] * 3600.0
        
        # どの年に属するかを判定
        if time_sec < MERCURY_YEAR_SEC:
            grouped_files["Year 1"].append(f)
        elif time_sec < 2 * MERCURY_YEAR_SEC:
            grouped_files["Year 2"].append(f)
        else:
            grouped_files["Year 3 (Final)"].append(f)

    # 各グループ内でTAA順にソート（念のため）
    for key in grouped_files:
        grouped_files[key].sort(key=lambda x: x['taa'])

    # 空のグループを削除
    return {k: v for k, v in grouped_files.items() if len(v) > 0}


# ==============================================================================
# ビューワークラス
# ==============================================================================

class SimulationViewer:
    def __init__(self, grouped_files, orbit_data, t_start):
        self.grouped_files = grouped_files
        self.orbit_data = orbit_data
        self.t_start = t_start
        
        # 利用可能な年のリスト
        self.years_available = list(self.grouped_files.keys())
        # デフォルトは最新の年（最後の要素）
        self.current_year_key = self.years_available[-1]
        self.file_list = self.grouped_files[self.current_year_key]

        # 表示設定
        self.n_lon = N_LON
        self.n_lat = N_LAT
        self.use_log = USE_LOG_SCALE
        self.align_sun = ALIGN_SUN_TO_CENTER
        self.current_display_data = None 
        self.show_contours = False
        self.contour_set = None

        if USE_PAPER_SCALE:
            self.vmin = 10 ** 9.5  
            self.vmax = 10 ** 14.5  
            self.unit_label = '[atoms/cm²]'
            print(f"Paper Scale ON: Vmin={self.vmin:.2e}, Vmax={self.vmax:.2e} {self.unit_label}")
        else:
            self.vmin = COLOR_VMIN
            self.vmax = COLOR_VMAX
            self.unit_label = '[atoms/m²]'
            print(f"Paper Scale OFF: Vmin={self.vmin:.2e}, Vmax={self.vmax:.2e} {self.unit_label}")

        self.current_idx = 0
        self._find_initial_index()

        # 図の準備
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        plt.subplots_adjust(bottom=0.25, left=0.2) # ラジオボタン用に左の余白を空ける

        import copy
        self.cmap = copy.copy(plt.get_cmap('inferno'))
        self.cmap.set_bad('black')

        if self.use_log:
            self.norm = LogNorm(vmin=self.vmin, vmax=self.vmax)
        else:
            self.norm = Normalize(vmin=self.vmin, vmax=self.vmax)

        dummy_data = np.zeros((self.n_lat, self.n_lon))
        self.mesh = self.ax.pcolormesh(dummy_data, cmap=self.cmap, norm=self.norm)
        cbar_title = f'Surface Density {self.unit_label}'
        self.cbar = plt.colorbar(self.mesh, ax=self.ax, label=cbar_title)

        self.info_text = self.ax.text(
            0.98, 0.95, '',
            transform=self.ax.transAxes,
            ha='right', va='top',
            fontsize=10, color='white', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="white", alpha=0.7)
        )

        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        self.update_plot()

        # === ウィジェット ===
        # Slider
        self.ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.slider = Slider(self.ax_slider, 'TAA', 0, len(self.file_list) - 1, valinit=self.current_idx, valfmt='%d')
        self.slider.on_changed(self.on_slider_change)

        # Prev Button
        ax_prev = plt.axes([0.25, 0.05, 0.1, 0.04])
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_prev.on_clicked(self.prev_frame)

        # Next Button
        ax_next = plt.axes([0.36, 0.05, 0.1, 0.04])
        self.btn_next = Button(ax_next, 'Next')
        self.btn_next.on_clicked(self.next_frame)

        # Plate Button
        ax_plate = plt.axes([0.55, 0.05, 0.15, 0.04])
        self.btn_plate = Button(ax_plate, 'Show Plate')
        self.btn_plate.on_clicked(self.generate_plate_view)

        # Checkbox (Contours)
        ax_check = plt.axes([0.82, 0.05, 0.12, 0.1])
        self.check = CheckButtons(ax_check, ['Contours'], [self.show_contours])
        self.check.on_clicked(self.toggle_contours)
        
        # ★ 新規追加: Year切り替え用ラジオボタン
        ax_radio = plt.axes([0.02, 0.1, 0.15, 0.15], facecolor='lightgrey')
        self.radio = RadioButtons(ax_radio, self.years_available, active=len(self.years_available)-1)
        self.radio.on_clicked(self.change_year)

    def _find_initial_index(self):
        """現在のファイルリストの中で INITIAL_TARGET_TAA に最も近いインデックスを探す"""
        min_diff = float('inf')
        self.current_idx = 0
        for i, f in enumerate(self.file_list):
            diff = abs(f['taa'] - INITIAL_TARGET_TAA)
            if diff < min_diff:
                min_diff = diff
                self.current_idx = i

    def change_year(self, label):
        """ラジオボタンで年が切り替えられた時の処理"""
        self.current_year_key = label
        self.file_list = self.grouped_files[label]
        self._find_initial_index()
        
        # スライダーの最大値と現在値を更新
        self.slider.valmax = len(self.file_list) - 1
        self.slider.ax.set_xlim(self.slider.valmin, self.slider.valmax)
        self.slider.set_val(self.current_idx)
        
        self.update_plot()

    def toggle_contours(self, label):
        if label == 'Contours':
            self.show_contours = not self.show_contours
            self.update_plot()

    def update_plot(self):
        data_info = self.file_list[self.current_idx]
        filepath = data_info['path']
        time_h = data_info['time_h']
        taa = data_info['taa']

        try:
            data = np.load(filepath)
            if data.ndim == 3:
                #data = data[:, :, 0]
                data = np.sum(data, axis=2)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return

        subsolar_lon_deg = get_subsolar_longitude_linear(time_h, self.t_start, self.orbit_data)

        xlabel = "Longitude (Planet)"
        title_mode = "(Planet)"

        if self.align_sun:
            dlon = 360.0 / self.n_lon
            sun_pos_norm = (subsolar_lon_deg + 180.0) % 360.0
            sun_index = int(np.round(sun_pos_norm / dlon)) % self.n_lon
            shift = (self.n_lon // 2) - sun_index
            data = np.roll(data, shift=shift, axis=0)
            title_mode = "(Sun Centered)"
            xlabel = "Longitude"

        data_T = data.T
        data_T = np.nan_to_num(data_T, nan=0.0)

        if USE_PAPER_SCALE:
            data_T = data_T / 10000.0

        self.current_display_data = data_T

        if self.contour_set is not None:
            try:
                for coll in self.contour_set.collections:
                    coll.remove()
            except Exception:
                pass
            self.contour_set = None

        if self.mesh:
            self.mesh.remove()

        lon_edges = np.linspace(-180, 180, self.n_lon + 1)
        lat_edges = np.linspace(-90, 90, self.n_lat + 1)

        self.mesh = self.ax.pcolormesh(lon_edges, lat_edges, data_T, cmap=self.cmap, norm=self.norm, shading='flat')

        if self.show_contours:
            try:
                lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
                lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
                X, Y = np.meshgrid(lon_centers, lat_centers)
                data_contour = data_T.copy()
                data_contour[data_contour <= 0] = np.nan

                if np.nanmax(data_contour) >= self.vmin:
                    exp_min = np.floor(np.log10(self.vmin))
                    exp_max = np.ceil(np.log10(self.vmax))
                    levels = np.logspace(exp_min, exp_max, num=int(exp_max - exp_min) + 1)
                    self.contour_set = self.ax.contour(
                        X, Y, data_contour, levels=levels, colors='cyan', linewidths=0.8
                    )
                    self.ax.clabel(self.contour_set, inline=True, fontsize=8, fmt='%.0e', colors='white')
            except Exception as e:
                print(f"Contour plot warning: {e}")

        # タイトルに現在表示中の年を追加
        self.ax.set_title(
            f"[{self.current_year_key}] Surface Density {title_mode}\nTAA: {taa} deg (Time: {time_h}h, SunLon: {subsolar_lon_deg:.1f})")
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel("Latitude")

        [line.remove() for line in self.ax.lines]
        if self.align_sun:
            self.ax.axvline(0, color='white', linestyle='--', alpha=0.5)
            self.ax.axvline(-90, color='white', linestyle=':', alpha=0.3)
            self.ax.axvline(90, color='white', linestyle=':', alpha=0.3)
        else:
            self.ax.axvline(subsolar_lon_deg, color='white', linestyle='--')

        self.ax.axhline(0, color='white', linestyle=':', alpha=0.3)
        self.info_text.set_zorder(100)
        self.fig.canvas.draw_idle()

    def on_mouse_move(self, event):
        if event.inaxes != self.ax:
            self.info_text.set_text('')
            self.fig.canvas.draw_idle()
            return
        if self.current_display_data is None:
            return

        x, y = event.xdata, event.ydata
        lon_idx = int((x + 180) / 360 * self.n_lon)
        lat_idx = int((y + 90) / 180 * self.n_lat)

        lon_idx = np.clip(lon_idx, 0, self.n_lon - 1)
        lat_idx = np.clip(lat_idx, 0, self.n_lat - 1)

        val = self.current_display_data[lat_idx, lon_idx]
        self.info_text.set_text(f"Lon: {x:.1f}\nLat: {y:.1f}\nVal: {val:.2e}")
        self.fig.canvas.draw_idle()

    def generate_plate_view(self, event):
        target_taas = [0, 60, 180, 240, 300, 359]
        fig_plate, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
        axes = axes.flatten()

        print(f"\n--- Generating Plate View for {self.current_year_key} ---")

        lon_edges = np.linspace(-180, 180, self.n_lon + 1)
        lat_edges = np.linspace(-90, 90, self.n_lat + 1)
        im = None

        for i, target_taa in enumerate(target_taas):
            ax = axes[i]
            closest_file = min(self.file_list, key=lambda x: abs(x['taa'] - target_taa))
            actual_taa = closest_file['taa']

            try:
                data = np.load(closest_file['path'])
                if data.ndim == 3:
                    #data = data[:, :, 0]
                    data = np.sum(data, axis=2)

                time_h = closest_file['time_h']
                subsolar_lon_deg = get_subsolar_longitude_linear(time_h, self.t_start, self.orbit_data)
                
                if self.align_sun:
                    dlon = 360.0 / self.n_lon
                    sun_pos_norm = (subsolar_lon_deg + 180.0) % 360.0
                    sun_index = int(np.round(sun_pos_norm / dlon)) % self.n_lon
                    shift = (self.n_lon // 2) - sun_index
                    data = np.roll(data, shift=shift, axis=0)

                data_T = data.T
                data_T = np.nan_to_num(data_T, nan=0.0)

                if USE_PAPER_SCALE:
                    data_T = data_T / 10000.0

                im = ax.pcolormesh(lon_edges, lat_edges, data_T, cmap=self.cmap, norm=self.norm, shading='flat')

                ax.set_title(f"TAA = {actual_taa}$^\circ$", fontsize=12, fontweight='bold')
                if i in [4, 5]:  
                    ax.set_xlabel("Longitude")
                else:
                    ax.set_xticklabels([])

                if i in [0, 2, 4]: 
                    ax.set_ylabel("Latitude")
                else:
                    ax.set_yticklabels([])

                ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5) 
                ax.axvline(-90, color='black', linestyle='--', linewidth=0.5, alpha=0.5) 
                ax.axvline(90, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
                ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5) 

                ax.set_aspect('equal')
                ax.set_xlim(-180, 180)
                ax.set_ylim(-90, 90)

            except Exception as e:
                print(f"Error plotting TAA {target_taa}: {e}")
                ax.text(0, 0, "Data Error", ha='center')

        cbar_label = f"Surface Na Density {self.unit_label}"
        if im:
            cbar = fig_plate.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05, shrink=0.8)
            cbar.set_label(cbar_label, fontsize=12)

        fig_plate.suptitle(f"Mercury Surface Sodium Density ({self.current_year_key})\nRun: {RUN_NAME}", fontsize=14)
        plt.show()

    def next_frame(self, event):
        if self.current_idx < len(self.file_list) - 1:
            self.slider.set_val(self.current_idx + 1)

    def prev_frame(self, event):
        if self.current_idx > 0:
            self.slider.set_val(self.current_idx - 1)

    def on_slider_change(self, val):
        idx = int(val)
        if idx != self.current_idx:
            self.current_idx = idx
            self.update_plot()


# ==============================================================================
# メイン
# ==============================================================================
if __name__ == "__main__":
    if not os.path.exists(ORBIT_FILE_PATH):
        print(f"エラー: 軌道ファイルなし {ORBIT_FILE_PATH}")
        sys.exit(1)

    full_dir = os.path.join(BASE_OUTPUT_DIRECTORY, RUN_NAME)
    if not os.path.exists(full_dir):
        print(f"エラー: 結果フォルダなし {full_dir}")
        sys.exit(1)

    orbit_data, t_start = load_orbit_data(ORBIT_FILE_PATH)

    print("ファイルリストを作成中...")
    grouped_files = get_all_files_grouped_by_year(full_dir)
    
    total_files = sum(len(v) for v in grouped_files.values())
    print(f"合計 {total_files} 個のタイムステップが見つかりました。")

    if not grouped_files:
        sys.exit(0)

    viewer = SimulationViewer(grouped_files, orbit_data, t_start)
    print("表示中... ウィンドウを閉じると終了します。")
    plt.show()