# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.widgets import Button, Slider, CheckButtons, RadioButtons
import matplotlib.animation as animation
import glob
import os
import japanize_matplotlib
import re
import sys

# ==============================================================================
# 設定
# ==============================================================================

SAVE_GIF = False       # 起動時に自動でGIFアニメーションを保存するかどうか (True / False)
GIF_FPS = 10          # GIFのフレームレート (1秒あたりのコマ数)

# コマ送り時の時間ステップ（Mode: Time のとき、何時間ごとに進めるか）
TIME_STEP_HOURS = 24.0 

# 追加解析プロットのオンオフ
PLOT_TOTAL_ATOMS_BY_TAA = False
PLOT_DAWN_DUSK_SPLIT    = False
PLOT_ZENITH_PROFILE     = False

ZENITH_PROFILE_TAAS = [0, 30, 60, 90, 120]
ZENITH_PROFILE_SIDE = "DAWN"
ZENITH_N_BINS = 20

USE_PAPER_SCALE = True
COLOR_VMIN = 1.0e10
COLOR_VMAX = 1.0e18

# 経度プロファイルのY軸固定最大値
LON_PROFILE_YMAX = 1.8e30

N_LON, N_LAT = 72, 36
N_LON, N_LAT = 144, 72
BASE_OUTPUT_DIRECTORY = r"./SimulationResult_202607"
#RUN_NAME = "ParabolicHop_72x36_NoEq_DT100_0710_BD0.8_UG1.85_Q0.3_A2.0e+07_LT190k_15yr"
#RUN_NAME = "ParabolicHop_72x36_NoEq_DT100_0713_BD0.8_UG1.85_Q2.0_A2.0e+07_LT190k_15yr"
RUN_NAME = "ParabolicHop_144x72_NoEq_DT100_0728_BD0.8_UG1.85_Q0.3_A2.0e+07_LT190k_15yr"


INITIAL_TARGET_TAA = 100
ORBIT_FILE_PATH = 'orbit2025_spice_unwrapped.txt'
ALIGN_SUN_TO_CENTER = True
USE_LOG_SCALE = True
MERCURY_YEAR_SEC = 87.969 * 86400
SPIN_UP_YEARS = 14.0

R_BODY_KM = 2439.7

MAX_NA_SURF_DENS_M2 = 7.5e14 * (100 ** 2)
INIT_SURF_DENS_M2 = MAX_NA_SURF_DENS_M2 * 0.0053

MAX_NA_SURF_DENS_CM2 = 7.5e14
INIT_SURF_DENS_CM2 = MAX_NA_SURF_DENS_CM2 * 0.0053

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
    current_t_sec = t_file_start + (float(time_h) * 3600.0)
    t_lookup = np.clip(current_t_sec, time_col_original[0], time_col_original[-1])
    sub_lon_deg = np.interp(t_lookup, time_col_original, orbit_data[:, 5])
    return sub_lon_deg


def get_all_files_grouped_by_year(target_dir):
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

    all_files.sort(key=lambda x: x['time_h'])

    grouped_files = {}
    for f in all_files:
        time_sec = f['time_h'] * 3600.0
        year_num = int(time_sec // MERCURY_YEAR_SEC) + 1
        year_key = f"Year {year_num}"

        if year_key not in grouped_files:
            grouped_files[year_key] = []
        grouped_files[year_key].append(f)

    sorted_grouped_files = {
        k: grouped_files[k] for k in sorted(grouped_files.keys(), key=lambda x: int(x.split()[1]))
    }

    for key in sorted_grouped_files:
        sorted_grouped_files[key].sort(key=lambda x: x['time_h'])

    return sorted_grouped_files


def calculate_cell_areas(n_lon, n_lat, r_body_km):
    r_body_m = r_body_km * 1e3
    r_body_cm = r_body_km * 1e5
    dlon_rad = 2.0 * np.pi / n_lon
    lat_edges = np.linspace(-np.pi/2, np.pi/2, n_lat + 1)
    areas_m2 = np.zeros((n_lat, n_lon))
    areas_cm2 = np.zeros((n_lat, n_lon))

    for i in range(n_lat):
        factor = np.sin(lat_edges[i+1]) - np.sin(lat_edges[i])
        areas_m2[i, :] = (r_body_m ** 2) * dlon_rad * factor
        areas_cm2[i, :] = (r_body_cm ** 2) * dlon_rad * factor

    return areas_m2, areas_cm2


def compute_effcos_grid(n_lon, n_lat, subsolar_lon_deg):
    lon_centers = np.linspace(-180, 180, n_lon + 1)
    lon_centers = (lon_centers[:-1] + lon_centers[1:]) / 2.0
    lat_centers = np.linspace(-90, 90, n_lat + 1)
    lat_centers = (lat_centers[:-1] + lat_centers[1:]) / 2.0

    lon_rad = np.deg2rad(lon_centers)
    lat_rad = np.deg2rad(lat_centers)
    sub_rad = np.deg2rad(subsolar_lon_deg)

    lon_sun = (lon_rad[None, :] - sub_rad + np.pi) % (2 * np.pi) - np.pi
    lat2d = lat_rad[:, None]

    eff_cos = np.cos(lat2d) * np.cos(lon_sun)
    is_dawn = (lon_sun < 0.0)
    is_dawn = np.broadcast_to(is_dawn, eff_cos.shape)
    return eff_cos, is_dawn


# ==============================================================================
# ビューワークラス
# ==============================================================================

class SimulationViewer:
    def __init__(self, grouped_files, orbit_data, t_start):
        self.grouped_files = grouped_files
        self.orbit_data = orbit_data
        self.t_start = t_start
        self.is_saving_gif = False

        self.years_available = list(self.grouped_files.keys())
        self.current_year_key = self.years_available[-1]
        self.file_list = self.grouped_files[self.current_year_key]

        self.n_lon = N_LON
        self.n_lat = N_LAT
        self.use_log = USE_LOG_SCALE
        self.align_sun = ALIGN_SUN_TO_CENTER
        self.current_display_data = None
        
        # 表示コントロール用フラグ
        self.show_contours = False
        self.show_ssp = True
        self.show_terminator = True
        self.show_planet_zero = True
        self.show_lon_profile = False    
        self.step_mode = "TAA"
        
        self.contour_set = None
        self.lon_profile_line = None    
        self.drawn_vlines = []          

        areas_m2, areas_cm2 = calculate_cell_areas(self.n_lon, self.n_lat, R_BODY_KM)

        if USE_PAPER_SCALE:
            self.vmin = 10 ** 9.5
            self.vmax = 10 ** 14.5
            self.unit_label = '[atoms/cm²]'
            self.cell_areas = areas_cm2
        else:
            self.vmin = COLOR_VMIN
            self.vmax = COLOR_VMAX
            self.unit_label = '[atoms/m²]'
            self.cell_areas = areas_m2

        self.current_idx = 0
        self._find_initial_index()

        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        plt.subplots_adjust(bottom=0.25, left=0.12, right=0.78)

        self.ax_twin = self.ax.twinx()
        self.ax_twin.yaxis.set_visible(False)

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
        self.ax_cbar = plt.axes([0.87, 0.25, 0.02, 0.65])
        self.cbar = plt.colorbar(self.mesh, cax=self.ax_cbar, label=cbar_title)

        self.info_text = self.ax.text(
            0.98, 0.95, '',
            transform=self.ax.transAxes,
            ha='right', va='top',
            fontsize=10, color='white', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="white", alpha=0.7)
        )

        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        # === UIレイアウト ===
        self.ax_slider = plt.axes([0.20, 0.14, 0.55, 0.03], facecolor='lightgoldenrodyellow')
        self.slider = Slider(self.ax_slider, 'Index/TAA', 0, len(self.file_list) - 1, valinit=self.current_idx, valfmt='%d')
        self.slider.on_changed(self.on_slider_change)

        self.ax_prev = plt.axes([0.13, 0.04, 0.07, 0.05])
        self.btn_prev = Button(self.ax_prev, 'Previous')
        self.btn_prev.on_clicked(self.prev_frame)

        self.ax_next = plt.axes([0.21, 0.04, 0.07, 0.05])
        self.btn_next = Button(self.ax_next, 'Next')
        self.btn_next.on_clicked(self.next_frame)

        self.ax_mode = plt.axes([0.29, 0.04, 0.10, 0.05])
        self.btn_mode = Button(self.ax_mode, f'Mode: {self.step_mode}', color='lightblue')
        self.btn_mode.on_clicked(self.toggle_step_mode)

        self.ax_plate = plt.axes([0.40, 0.04, 0.09, 0.05])
        self.btn_plate = Button(self.ax_plate, 'Show Plate')
        self.btn_plate.on_clicked(self.generate_plate_view)

        self.ax_total = plt.axes([0.50, 0.04, 0.09, 0.05])
        self.btn_total = Button(self.ax_total, 'Plot Total')
        self.btn_total.on_clicked(self.plot_total_atoms)

        self.ax_gif = plt.axes([0.01, 0.21, 0.10, 0.04])
        self.btn_gif = Button(self.ax_gif, 'Save GIF', color='salmon')
        self.btn_gif.on_clicked(self.on_gif_btn_clicked)

        self.ax_check = plt.axes([0.61, 0.01, 0.16, 0.12])
        self.check = CheckButtons(
            self.ax_check, 
            ['Contours', 'SSP', 'Terminator', 'Planet 0°', 'Lon Profile'], 
            [self.show_contours, self.show_ssp, self.show_terminator, self.show_planet_zero, self.show_lon_profile]
        )
        self.check.on_clicked(self.toggle_checks)

        self.ax_radio = plt.axes([0.01, 0.04, 0.10, 0.15], facecolor='lightgrey')
        self.radio = RadioButtons(self.ax_radio, self.years_available, active=len(self.years_available)-1)
        self.radio.on_clicked(self.change_year)

        self.ui_axes = [
            self.ax_slider, self.ax_prev, self.ax_next, self.ax_mode, 
            self.ax_plate, self.ax_total, self.ax_check, self.ax_radio, self.ax_gif
        ]

        self.update_plot()

    def _find_initial_index(self):
        min_diff = float('inf')
        self.current_idx = 0
        for i, f in enumerate(self.file_list):
            diff = abs(f['taa'] - INITIAL_TARGET_TAA)
            if diff < min_diff:
                min_diff = diff
                self.current_idx = i

    def change_year(self, label):
        self.current_year_key = label
        self.file_list = self.grouped_files[label]
        self._find_initial_index()

        self.slider.valmax = len(self.file_list) - 1
        self.slider.ax.set_xlim(self.slider.valmin, self.slider.valmax)
        self.slider.set_val(self.current_idx)
        self.update_plot()

    def toggle_checks(self, label):
        if label == 'Contours':
            self.show_contours = not self.show_contours
        elif label == 'SSP':
            self.show_ssp = not self.show_ssp
        elif label == 'Terminator':
            self.show_terminator = not self.show_terminator
        elif label == 'Planet 0°':
            self.show_planet_zero = not self.show_planet_zero
        elif label == 'Lon Profile':
            self.show_lon_profile = not self.show_lon_profile
        self.update_plot()

    def toggle_step_mode(self, event):
        if self.step_mode == "TAA":
            self.step_mode = "Time"
            self.btn_mode.label.set_text("Mode: Time")
            self.btn_mode.ax.set_facecolor('lightgreen')
        else:
            self.step_mode = "TAA"
            self.btn_mode.label.set_text("Mode: TAA")
            self.btn_mode.ax.set_facecolor('lightblue')
        self.fig.canvas.draw_idle()

    def _load_and_align(self, data_info):
        filepath = data_info['path']
        time_h = data_info['time_h']

        data = np.load(filepath)
        if data.ndim == 3:
            data = np.sum(data, axis=2)

        subsolar_lon_deg = get_subsolar_longitude_linear(time_h, self.t_start, self.orbit_data)

        if self.align_sun:
            dlon = 360.0 / self.n_lon
            sun_pos_norm = (subsolar_lon_deg + 180.0) % 360.0
            sun_index = int(np.round(sun_pos_norm / dlon)) % self.n_lon
            shift = (self.n_lon // 2) - sun_index
            data = np.roll(data, shift=shift, axis=0)
            effcos_sublon = 0.0
        else:
            effcos_sublon = subsolar_lon_deg

        data_T = np.nan_to_num(data.T, nan=0.0)
        if USE_PAPER_SCALE:
            data_T = data_T / 10000.0

        eff_cos, is_dawn = compute_effcos_grid(self.n_lon, self.n_lat, effcos_sublon)
        return data_T, eff_cos, is_dawn

    def update_plot(self):
        data_info = self.file_list[self.current_idx]
        filepath = data_info['path']
        time_h = data_info['time_h']
        taa = data_info['taa']

        try:
            data = np.load(filepath)
            if data.ndim == 3:
                data = np.sum(data, axis=2)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return

        subsolar_lon_deg = get_subsolar_longitude_linear(time_h, self.t_start, self.orbit_data)

        xlabel = "Longitude (Planet)"
        title_mode = "(Planet Fixed)"

        if self.align_sun:
            dlon = 360.0 / self.n_lon
            sun_pos_norm = (subsolar_lon_deg + 180.0) % 360.0
            sun_index = int(np.round(sun_pos_norm / dlon)) % self.n_lon
            shift = (self.n_lon // 2) - sun_index
            data = np.roll(data, shift=shift, axis=0)
            title_mode = "(Sun Centered)"
            xlabel = "Longitude (Sun-Relative)"

        data_T = data.T
        data_T = np.nan_to_num(data_T, nan=0.0)

        if USE_PAPER_SCALE:
            data_T = data_T / 10000.0

        self.current_display_data = data_T

        if self.contour_set is not None:
            try:
                self.contour_set.remove()
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
                print(f"Contour drawing error: {e}")

        # === 経度プロファイルの描画 ===
        if self.show_lon_profile:
            atoms_per_lon = np.sum(data_T * self.cell_areas, axis=0) 
            lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2.0
            
            self.ax_twin.yaxis.set_visible(True)
            self.ax_twin.set_ylabel('Total Atoms [atoms]', color='magenta', fontsize=11, fontweight='bold')
            
            if self.lon_profile_line is not None:
                self.lon_profile_line.set_data(lon_centers, atoms_per_lon)
            else:
                self.lon_profile_line, = self.ax_twin.plot(
                    lon_centers, atoms_per_lon, color='magenta', linestyle='-', linewidth=2.2, label='Lon Profile', zorder=10
                )
            self.ax_twin.tick_params(axis='y', labelcolor='magenta')
            self.ax_twin.set_ylim(0, LON_PROFILE_YMAX)
        else:
            self.ax_twin.yaxis.set_visible(False)
            if self.lon_profile_line is not None:
                self.lon_profile_line.remove()
                self.lon_profile_line = None

        self.ax.set_title(
            f"[{self.current_year_key}] Surface Density {title_mode}\nTAA: {taa} deg (Time: {time_h}h, SunLon: {subsolar_lon_deg:.1f})")
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel("Latitude")

        # === 縦線（動的・静的ライン）の安全なリセットと再描画 ===
        for line in self.drawn_vlines:
            try:
                line.remove()
            except Exception:
                pass
        self.drawn_vlines.clear()

        # 赤道線
        eq_line = self.ax.axhline(0, color='white', linestyle=':', alpha=0.3)
        self.drawn_vlines.append(eq_line)
        
        ssp_planet_lon = (subsolar_lon_deg + 180) % 360 - 180

        if self.align_sun:
            if self.show_ssp:
                l1 = self.ax.axvline(0.0, color='#FFBF00', linestyle='-', linewidth=2.0, alpha=0.8, label='SSP (Fixed)')
                self.drawn_vlines.append(l1)
            if self.show_terminator:
                l2 = self.ax.axvline(-90.0, color='white', linestyle='--', linewidth=1.5, alpha=0.7, label='Terminator (Dawn, Fixed)')
                l3 = self.ax.axvline(90.0, color='white', linestyle='--', linewidth=1.5, alpha=0.7, label='Terminator (Dusk, Fixed)')
                self.drawn_vlines.append(l2)
                self.drawn_vlines.append(l3)
            
            planet_zero_disp = (-ssp_planet_lon + 180) % 360 - 180
            if self.show_planet_zero:
                l4 = self.ax.axvline(planet_zero_disp, color='lime', linestyle='-.', linewidth=2.0, alpha=0.8, label='Planet 0°')
                self.drawn_vlines.append(l4)
            
            if self.show_terminator:
                moving_dawn = (-ssp_planet_lon - 90 + 180) % 360 - 180
                moving_dusk = (-ssp_planet_lon + 90 + 180) % 360 - 180
                l5 = self.ax.axvline(moving_dawn, color='cyan', linestyle=':', linewidth=1.2, alpha=0.6, label='Moving Term (Dawn)')
                l6 = self.ax.axvline(moving_dusk, color='cyan', linestyle=':', linewidth=1.2, alpha=0.6, label='Moving Term (Dusk)')
                self.drawn_vlines.append(l5)
                self.drawn_vlines.append(l6)
        else:
            if self.show_planet_zero:
                l1 = self.ax.axvline(0.0, color='lime', linestyle='-.', linewidth=2.0, alpha=0.8, label='Planet 0° (Fixed)')
                self.drawn_vlines.append(l1)
            if self.show_ssp:
                l2 = self.ax.axvline(ssp_planet_lon, color='#FFBF00', linestyle='-', linewidth=2.0, alpha=0.8, label='SSP (Moving)')
                self.drawn_vlines.append(l2)
            if self.show_terminator:
                term_dawn_disp = (ssp_planet_lon - 90 + 180) % 360 - 180
                term_dusk_disp = (ssp_planet_lon + 90 + 180) % 360 - 180
                l3 = self.ax.axvline(term_dawn_disp, color='white', linestyle='--', linewidth=1.5, alpha=0.7, label='Terminator (Dawn, Moving)')
                l4 = self.ax.axvline(term_dusk_disp, color='white', linestyle='--', linewidth=1.5, alpha=0.7, label='Terminator (Dusk, Moving)')
                self.drawn_vlines.append(l3)
                self.drawn_vlines.append(l4)

        if not self.is_saving_gif:
            self.fig.canvas.draw_idle()

    # ★修正：マウス座標を元のマップ(ax)の座標系に逆変換してから緯度・経度を判定する
    def on_mouse_move(self, event):
        # 判定に ax_twin も含めることで、右軸表示時でもカーソル情報を取得可能にする
        if event.inaxes not in [self.ax, self.ax_twin]:
            self.info_text.set_text('')
            self.fig.canvas.draw_idle()
            return
            
        if self.current_display_data is None:
            return

        # event.x, event.y（画面上のピクセル位置）から、self.axのデータ座標（経度・緯度）を逆算する
        x, y = self.ax.transData.inverted().transform((event.x, event.y))

        lon_idx = int((x + 180) / 360 * self.n_lon)
        lat_idx = int((y + 90) / 180 * self.n_lat)

        lon_idx = np.clip(lon_idx, 0, self.n_lon - 1)
        lat_idx = np.clip(lat_idx, 0, self.n_lat - 1)

        val = self.current_display_data[lat_idx, lon_idx]

        if USE_PAPER_SCALE:
            pct_of_max = (val / MAX_NA_SURF_DENS_CM2) * 100.0
            ratio_to_init = val / INIT_SURF_DENS_CM2 if INIT_SURF_DENS_CM2 > 0 else 0
        else:
            pct_of_max = (val / MAX_NA_SURF_DENS_M2) * 100.0
            ratio_to_init = val / INIT_SURF_DENS_M2 if INIT_SURF_DENS_M2 > 0 else 0

        self.info_text.set_text(
            f"Lon: {x:.1f}\nLat: {y:.1f}\n"
            f"Val: {val:.2e}\n"
            f"Cap: {pct_of_max:.2f}%\n"
            f"vs Init: {ratio_to_init:.2f}x"
        )
        self.fig.canvas.draw_idle()

    def _get_next_index_by_mode(self, start_idx):
        if self.step_mode == "TAA":
            return min(start_idx + 1, len(self.file_list) - 1)
        else:
            current_t = self.file_list[start_idx]['time_h']
            for i in range(start_idx + 1, len(self.file_list)):
                if self.file_list[i]['time_h'] - current_t >= TIME_STEP_HOURS:
                    return i
            return len(self.file_list) - 1

    def _get_prev_index_by_mode(self, start_idx):
        if self.step_mode == "TAA":
            return max(start_idx - 1, 0)
        else:
            current_t = self.file_list[start_idx]['time_h']
            for i in range(start_idx - 1, -1, -1):
                if current_t - self.file_list[i]['time_h'] >= TIME_STEP_HOURS:
                    return i
            return 0

    def next_frame(self, event):
        next_idx = self._get_next_index_by_mode(self.current_idx)
        if next_idx != self.current_idx:
            self.slider.set_val(next_idx)

    def prev_frame(self, event):
        prev_idx = self._get_prev_index_by_mode(self.current_idx)
        if prev_idx != self.current_idx:
            self.slider.set_val(prev_idx)

    def on_slider_change(self, val):
        idx = int(val)
        if idx != self.current_idx:
            self.current_idx = idx
            self.update_plot()

    def on_gif_btn_clicked(self, event):
        self.generate_gif()

    def generate_gif(self):
        gif_name = f"{RUN_NAME}_{self.current_year_key.replace(' ', '_')}_{self.step_mode}Mode.gif"
        print(f"\n--- GIFアニメーションの作成を開始します ({self.step_mode}モード送り) ---")
        print(f"出力ファイル: {gif_name}")

        frames_to_record = []
        idx = 0
        frames_to_record.append(idx)
        while idx < len(self.file_list) - 1:
            next_idx = self._get_next_index_by_mode(idx)
            if next_idx <= idx:
                break
            idx = next_idx
            frames_to_record.append(idx)

        print(f"フレーム数: {len(frames_to_record)}")
        original_idx = self.current_idx
        self.is_saving_gif = True
        
        for ax_ui in self.ui_axes:
            ax_ui.set_visible(False)
        self.info_text.set_text('')

        try:
            def update(frame_idx):
                self.slider.set_val(frame_idx)
                return [self.mesh]

            anim = animation.FuncAnimation(
                self.fig, update, frames=frames_to_record, blit=False, repeat=False
            )
            anim.save(gif_name, writer=animation.PillowWriter(fps=GIF_FPS))
            print(f"GIFの保存が完了しました！: {gif_name}\n")
        except Exception as e:
            print(f"GIF保存エラー: {e}")
        finally:
            self.is_saving_gif = False
            for ax_ui in self.ui_axes:
                ax_ui.set_visible(True)
            self.slider.set_val(original_idx)

    def generate_plate_view(self, event):
        target_taas = [0, 60, 180, 240, 300, 359]
        fig_plate, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
        axes = axes.flatten()

        lon_edges = np.linspace(-180, 180, self.n_lon + 1)
        lat_edges = np.linspace(-90, 90, self.n_lat + 1)
        im = None

        for i, target_taa in enumerate(target_taas):
            ax = axes[i]
            closest_file = min(self.file_list, key=lambda x: abs(x['taa'] - target_taa))
            actual_taa = closest_file['taa']

            try:
                data = np.load(closest_file['path'])
                if data.ndim == 3: data = np.sum(data, axis=2)
                time_h = closest_file['time_h']
                subsolar_lon_deg = get_subsolar_longitude_linear(time_h, self.t_start, self.orbit_data)

                if self.align_sun:
                    dlon = 360.0 / self.n_lon
                    sun_pos_norm = (subsolar_lon_deg + 180.0) % 360.0
                    sun_index = int(np.round(sun_pos_norm / dlon)) % self.n_lon
                    shift = (self.n_lon // 2) - sun_index
                    data = np.roll(data, shift=shift, axis=0)

                data_T = np.nan_to_num(data.T, nan=0.0)
                if USE_PAPER_SCALE: data_T = data_T / 10000.0

                im = ax.pcolormesh(lon_edges, lat_edges, data_T, cmap=self.cmap, norm=self.norm, shading='flat')
                ax.set_title(f"TAA = {actual_taa}$^\\circ$", fontsize=12, fontweight='bold')
                
                if i in [4, 5]: ax.set_xlabel("Longitude")
                else: ax.set_xticklabels([])
                if i in [0, 2, 4]: ax.set_ylabel("Latitude")
                else: ax.set_yticklabels([])

                ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
                ax.axvline(-90, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
                ax.axvline(90, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
                ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
                ax.set_aspect('equal')
            except Exception as e:
                ax.text(0, 0, "Data Error", ha='center')

        if im:
            cbar = fig_plate.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05, shrink=0.8)
            cbar.set_label(f"Surface Na Density {self.unit_label}", fontsize=12)
        fig_plate.suptitle(f"Mercury Surface Sodium Density ({self.current_year_key})\nRun: {RUN_NAME}", fontsize=14)
        plt.show()

    def plot_total_atoms(self, event):
        fig_total, ax_total = plt.subplots(figsize=(8, 6))
        for year_key, file_list in self.grouped_files.items():
            taas, totals = [], []
            for f in file_list:
                try:
                    data = np.load(f['path'])
                    if data.ndim == 3: data = np.sum(data, axis=2)
                    data_T = np.nan_to_num(data.T, nan=0.0)
                    if USE_PAPER_SCALE: data_T = data_T / 10000.0
                    total_atoms = np.sum(data_T * self.cell_areas)
                    taas.append(f['taa'])
                    totals.append(total_atoms)
                except Exception: pass

            if taas and totals:
                sorted_indices = np.argsort(taas)
                ax_total.plot(np.array(taas)[sorted_indices], np.array(totals)[sorted_indices], marker='o', linestyle='-', markersize=4, label=year_key)

        ax_total.set_xlabel('True Anomaly (TAA) [deg]')
        ax_total.set_ylabel('Total Number of Atoms')
        ax_total.set_title(f'Total Surface Atoms vs TAA\nRun: {RUN_NAME}')
        ax_total.grid(True, linestyle='--', alpha=0.7)
        ax_total.legend()
        plt.tight_layout()
        plt.show()


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
    
    if SAVE_GIF:
        viewer.step_mode = "TAA"
        viewer.generate_gif()

    print("表示中... ウィンドウを閉じると終了します。")
    plt.show()