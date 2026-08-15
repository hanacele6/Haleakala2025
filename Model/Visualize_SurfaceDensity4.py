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

SAVE_GIF = False       # GIFアニメーションを保存するかどうか (True / False)
GIF_FPS = 10          # GIFのフレームレート (1秒あたりのコマ数)

# ★★★ 新規: 追加解析プロットのオンオフ (GIF同様 True/False で制御) ★★★
PLOT_TOTAL_ATOMS_BY_TAA = True   # [1] TAAに対する表面総原子量 (Dawn/Dusk/全体)
PLOT_DAWN_DUSK_SPLIT    = True   # [2] Dawn側・Dusk側に絞った総原子量 vs TAA
PLOT_ZENITH_PROFILE     = True   # [3] 天頂角(eff_cos) vs 平均表面密度 (複数TAA重ね)

# [3] 天頂角プロファイルで重ねるTAA(度)。TAAによる移動を1枚で見る。
# ZENITH_PROFILE_TAAS = [60, 120, 150, 180]
ZENITH_PROFILE_TAAS = [0, 30, 60, 90, 120,]
# [3] 天頂角プロファイルをどの半球で見るか: "DAWN" / "DUSK" / "BOTH"(全昼側)
ZENITH_PROFILE_SIDE = "DAWN"
# [3] eff_cos のビン数 (ターミネーター eff_cos=0 → 太陽直下点 eff_cos=1)
ZENITH_N_BINS = 20

USE_PAPER_SCALE = True
COLOR_VMIN = 1.0e10
COLOR_VMAX = 1.0e18

N_LON, N_LAT = 72, 36
BASE_OUTPUT_DIRECTORY = r"./SimulationResult_202607"
#RUN_NAME = "ParabolicHop_72x36_NoEq_DT100_0713_BD0.8_UG1.85_Q2.0_A2.0e+07_LT190k_15yr"
RUN_NAME = "ParabolicHop_72x36_NoEq_DT100_0710_BD0.8_UG1.85_Q0.3_A2.0e+07_LT190k_15yr"

INITIAL_TARGET_TAA = 100
ORBIT_FILE_PATH = 'orbit2025_spice_unwrapped.txt'
ALIGN_SUN_TO_CENTER = True
USE_LOG_SCALE = True
MERCURY_YEAR_SEC = 87.969 * 86400
SPIN_UP_YEARS = 14.0

# --- 対象天体の半径 (水星: 2439.7 km) ---
R_BODY_KM = 2439.7

MAX_NA_SURF_DENS_M2 = 7.5e14 * (100 ** 2)  # 7.5e18 [atoms/m²]
INIT_SURF_DENS_M2 = MAX_NA_SURF_DENS_M2 * 0.0053  # 初期値 (0.53%)

MAX_NA_SURF_DENS_CM2 = 7.5e14  # 7.5e16 [atoms/cm²]
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
    """ディレクトリ内の全ファイルをTAA順にソートし、年(公転)ごとに動的にグループ化して返す"""
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
        sorted_grouped_files[key].sort(key=lambda x: x['taa'])

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
    """各セル(lat, lon)の eff_cos = cos(太陽天頂角) を計算する。
    戻り値 shape (n_lat, n_lon) で、data_T と同じ並び。
    eff_cos>0 が昼側。lon_sun<0(太陽直下点より西)を Dawn とする。"""
    lon_centers = np.linspace(-180, 180, n_lon + 1)
    lon_centers = (lon_centers[:-1] + lon_centers[1:]) / 2.0
    lat_centers = np.linspace(-90, 90, n_lat + 1)
    lat_centers = (lat_centers[:-1] + lat_centers[1:]) / 2.0

    lon_rad = np.deg2rad(lon_centers)
    lat_rad = np.deg2rad(lat_centers)
    sub_rad = np.deg2rad(subsolar_lon_deg)

    # lon_sun: 太陽直下点からの経度差 (-pi, pi]
    lon_sun = (lon_rad[None, :] - sub_rad + np.pi) % (2 * np.pi) - np.pi  # (1, n_lon) -> broadcast
    lat2d = lat_rad[:, None]  # (n_lat, 1)

    eff_cos = np.cos(lat2d) * np.cos(lon_sun)  # (n_lat, n_lon)
    is_dawn = (lon_sun < 0.0)                  # (1, n_lon) broadcast -> (n_lat, n_lon)
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
        self.show_contours = False
        self.contour_set = None

        areas_m2, areas_cm2 = calculate_cell_areas(self.n_lon, self.n_lat, R_BODY_KM)

        if USE_PAPER_SCALE:
            self.vmin = 10 ** 9.5
            self.vmax = 10 ** 14.5
            self.unit_label = '[atoms/cm²]'
            self.cell_areas = areas_cm2
            print(f"Paper Scale ON: Vmin={self.vmin:.2e}, Vmax={self.vmax:.2e} {self.unit_label}")
        else:
            self.vmin = COLOR_VMIN
            self.vmax = COLOR_VMAX
            self.unit_label = '[atoms/m²]'
            self.cell_areas = areas_m2
            print(f"Paper Scale OFF: Vmin={self.vmin:.2e}, Vmax={self.vmax:.2e} {self.unit_label}")

        self.current_idx = 0
        self._find_initial_index()

        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        plt.subplots_adjust(bottom=0.25, left=0.2)

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

        # === ウィジェット ===
        self.ax_slider = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.slider = Slider(self.ax_slider, 'TAA', 0, len(self.file_list) - 1, valinit=self.current_idx, valfmt='%d')
        self.slider.on_changed(self.on_slider_change)

        self.ax_prev = plt.axes([0.25, 0.05, 0.1, 0.05])
        self.btn_prev = Button(self.ax_prev, 'Previous')
        self.btn_prev.on_clicked(self.prev_frame)

        self.ax_next = plt.axes([0.36, 0.05, 0.1, 0.05])
        self.btn_next = Button(self.ax_next, 'Next')
        self.btn_next.on_clicked(self.next_frame)

        self.ax_plate = plt.axes([0.47, 0.05, 0.12, 0.05])
        self.btn_plate = Button(self.ax_plate, 'Show Plate')
        self.btn_plate.on_clicked(self.generate_plate_view)

        self.ax_total = plt.axes([0.60, 0.05, 0.12, 0.05])
        self.btn_total = Button(self.ax_total, 'Plot Total')
        self.btn_total.on_clicked(self.plot_total_atoms)

        self.ax_check = plt.axes([0.74, 0.05, 0.15, 0.05])
        self.check = CheckButtons(self.ax_check, ['Contours'], [self.show_contours])
        self.check.on_clicked(self.toggle_contours)

        self.ax_radio = plt.axes([0.02, 0.05, 0.15, 0.15], facecolor='lightgrey')
        self.radio = RadioButtons(self.ax_radio, self.years_available, active=len(self.years_available)-1)
        self.radio.on_clicked(self.change_year)

        self.ui_axes = [
            self.ax_slider, self.ax_prev, self.ax_next, self.ax_plate,
            self.ax_total, self.ax_check, self.ax_radio
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

    def toggle_contours(self, label):
        if label == 'Contours':
            self.show_contours = not self.show_contours
            self.update_plot()

    # -------------------------------------------------------------------------
    # 共通: あるファイルから (Sun-centered化した data_T, eff_cos, is_dawn) を返す
    # -------------------------------------------------------------------------
    def _load_and_align(self, data_info):
        """1ファイル読み込み、太陽中心化した表面密度[cm^2 or m^2]と、
        それに整合する eff_cos / is_dawn グリッドを返す。
        data_T: (n_lat, n_lon)。align_sun=True の場合、中心が太陽直下点。"""
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
            # 太陽中心化したので、eff_cos計算上の太陽直下点は経度0にある
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
            except Exception:
                pass

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

        if not self.is_saving_gif:
            self.fig.canvas.draw_idle()

        if not self.is_saving_gif:
            ref_max = MAX_NA_SURF_DENS_CM2 if USE_PAPER_SCALE else MAX_NA_SURF_DENS_M2
            ref_init = INIT_SURF_DENS_CM2 if USE_PAPER_SCALE else INIT_SURF_DENS_M2
            max_val = np.max(data_T)
            min_val = np.min(data_T)
            mean_val = np.mean(data_T)
            total_atoms = np.sum(data_T * self.cell_areas)
            init_total_atoms = ref_init * np.sum(self.cell_areas)
            ratio_to_init_total = total_atoms / init_total_atoms if init_total_atoms > 0 else 0

            print(f"\n--- Statistical Summary for TAA: {taa}° ({self.current_year_key}) ---")
            print(f"  [Max]   {max_val:.2e} ({self.unit_label}) -> Cap: {(max_val/ref_max)*100:.2f}%, vs Init: {max_val/ref_init:.2f}x")
            print(f"  [Mean]  {mean_val:.2e} ({self.unit_label}) -> Cap: {(mean_val/ref_max)*100:.2f}%, vs Init: {mean_val/ref_init:.2f}x")
            print(f"  [Min]   {min_val:.2e} ({self.unit_label}) -> Cap: {(min_val/ref_max)*100:.2f}%, vs Init: {min_val/ref_init:.2f}x")
            print(f"  [Total] {total_atoms:.4e} atoms -> vs Init Total: {ratio_to_init_total:.2f}x")

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

                ax.set_title(f"TAA = {actual_taa}$^\\circ$", fontsize=12, fontweight='bold')
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

    def plot_total_atoms(self, event):
        print("\n--- Generating Total Atoms vs TAA Plot ---")
        fig_total, ax_total = plt.subplots(figsize=(8, 6))

        for year_key, file_list in self.grouped_files.items():
            taas = []
            totals = []
            for f in file_list:
                try:
                    data = np.load(f['path'])
                    if data.ndim == 3:
                        data = np.sum(data, axis=2)

                    data_T = data.T
                    data_T = np.nan_to_num(data_T, nan=0.0)

                    if USE_PAPER_SCALE:
                        data_T = data_T / 10000.0

                    total_atoms = np.sum(data_T * self.cell_areas)

                    taas.append(f['taa'])
                    totals.append(total_atoms)
                except Exception:
                    pass

            if taas and totals:
                sorted_indices = np.argsort(taas)
                taas_sorted = np.array(taas)[sorted_indices]
                totals_sorted = np.array(totals)[sorted_indices]

                ax_total.plot(taas_sorted, totals_sorted, marker='o', linestyle='-', markersize=4, label=year_key)

        ax_total.set_xlabel('True Anomaly (TAA) [deg]')
        ax_total.set_ylabel('Total Number of Atoms')
        ax_total.set_title(f'Total Surface Atoms vs TAA\nRun: {RUN_NAME}')
        ax_total.grid(True, linestyle='--', alpha=0.7)
        ax_total.legend()
        plt.tight_layout()
        plt.show()

    # =========================================================================
    # ★★★ 新規機能 [1][2]: TAAに対する総原子量 (全体 / Dawn / Dusk) ★★★
    # =========================================================================
    def plot_total_atoms_dawn_dusk(self, split=True):
        """現在の年について、TAAに対する表面総原子量を
        split=False: 全体のみ
        split=True : 全体 + Dawn側 + Dusk側 に分けてプロット。
        Dawn = 太陽直下点より西(lon_sun<0)側。"""
        which = "Dawn/Dusk分離" if split else "全体"
        print(f"\n--- [新規] TAA vs 総原子量 ({which}) : {self.current_year_key} ---")

        fig, ax = plt.subplots(figsize=(9, 6))
        file_list = self.file_list  # 現在選択中の年

        taas, tot_all, tot_dawn, tot_dusk = [], [], [], []
        for f in file_list:
            try:
                data_T, eff_cos, is_dawn = self._load_and_align(f)
                # 昼側のみ(eff_cos>0)を対象にすると"表面に存在する原子"の議論に沿う。
                # ただし総原子量としては夜側在庫も含めたいので、ここでは全セルを対象にする。
                atoms = data_T * self.cell_areas  # (n_lat, n_lon)

                taas.append(f['taa'])
                tot_all.append(np.sum(atoms))
                if split:
                    is_dusk = ~is_dawn
                    tot_dawn.append(np.sum(atoms[is_dawn]))
                    tot_dusk.append(np.sum(atoms[is_dusk]))
            except Exception as e:
                print(f"  skip taa={f['taa']}: {e}")

        order = np.argsort(taas)
        taas = np.array(taas)[order]
        tot_all = np.array(tot_all)[order]

        ax.plot(taas, tot_all, '-', color='black', lw=2.2, marker='o', markersize=3, label='Total (全体)')
        if split:
            tot_dawn = np.array(tot_dawn)[order]
            tot_dusk = np.array(tot_dusk)[order]
            ax.plot(taas, tot_dawn, '-', color='royalblue', lw=2, marker='^', markersize=3, label='Dawn (明け方側)')
            ax.plot(taas, tot_dusk, '-', color='crimson', lw=2, marker='v', markersize=3, label='Dusk (夕方側)')

        ax.axvline(180, color='gray', ls=':', alpha=0.6, label='遠日点(TAA=180)')
        ax.set_xlabel('True Anomaly Angle (TAA) [deg]')
        ax.set_ylabel('表面に存在する総原子量 [atoms]')
        ax.set_title(f'TAAに対する表面総原子量 ({which})\n{self.current_year_key}')
        ax.set_xlim(0, 360)
        ax.grid(True, ls='--', alpha=0.6)
        ax.legend()
        plt.tight_layout()
        plt.show()

    # =========================================================================
    # ★★★ 新規機能 [3]: 天頂角(eff_cos) vs 平均表面密度 (複数TAA重ね) ★★★
    # =========================================================================
    def plot_zenith_profile(self, taa_list, side="DAWN", n_bins=20):
        """複数のTAAについて、eff_cos(局所天頂角)ビンごとの平均表面密度をプロット。
        Image4の模式図(ターミネーター→太陽直下点で在庫がどう変わるか)の実データ版。
        side: "DAWN"(西側) / "DUSK"(東側) / "BOTH"(全昼側)"""
        print(f"\n--- [新規] 天頂角プロファイル (side={side}) : {self.current_year_key} ---")

        fig, ax = plt.subplots(figsize=(9, 6))
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        cmap = plt.get_cmap('viridis')
        colors = [cmap(x) for x in np.linspace(0, 0.85, len(taa_list))]

        for ci, target_taa in enumerate(taa_list):
            closest = min(self.file_list, key=lambda x: abs(x['taa'] - target_taa))
            actual_taa = closest['taa']
            try:
                data_T, eff_cos, is_dawn = self._load_and_align(closest)
            except Exception as e:
                print(f"  skip taa={target_taa}: {e}")
                continue

            # 対象半球のマスク(昼側 eff_cos>0 のみ)
            day_mask = eff_cos > 0.0
            if side == "DAWN":
                side_mask = day_mask & is_dawn
            elif side == "DUSK":
                side_mask = day_mask & (~is_dawn)
            else:
                side_mask = day_mask

            ec_flat = eff_cos[side_mask]
            dens_flat = data_T[side_mask]

            # eff_cosビンごとに平均表面密度(面積重み付き平均)
            area_flat = self.cell_areas[side_mask]
            prof = np.full(n_bins, np.nan)
            for b in range(n_bins):
                m = (ec_flat >= bin_edges[b]) & (ec_flat < bin_edges[b+1])
                if np.any(m):
                    # 面積重み付き平均密度
                    prof[b] = np.sum(dens_flat[m] * area_flat[m]) / np.sum(area_flat[m])

            ax.plot(bin_centers, prof, '-o', color=colors[ci], markersize=4,
                    lw=1.8, label=f'TAA={actual_taa}°')

        ax.set_xlabel('eff_cos = cos(太陽天頂角)  [0=ターミネーター → 1=太陽直下点]')
        ax.set_ylabel(f'平均表面密度 {self.unit_label}')
        side_jp = {"DAWN": "明け方側", "DUSK": "夕方側", "BOTH": "全昼側"}.get(side, side)
        ax.set_title(f'天頂角に対する表面密度プロファイル ({side_jp})\n{self.current_year_key} — 各TAAで重ね描き')
        if self.use_log:
            ax.set_yscale('log')
        ax.grid(True, which='both', ls='--', alpha=0.5)
        ax.legend(title='True Anomaly')
        # 補助: 模式図の向き(左=ターミネーター, 右=太陽直下点)を明示
        ax.annotate('ターミネーター', xy=(0.02, 0.02), xycoords='axes fraction',
                    fontsize=9, color='gray')
        ax.annotate('太陽直下点', xy=(0.80, 0.02), xycoords='axes fraction',
                    fontsize=9, color='gray')
        plt.tight_layout()
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

    def generate_gif(self):
        gif_name = f"{RUN_NAME}_{self.current_year_key.replace(' ', '_')}.gif"
        print(f"\n--- GIFアニメーションの作成を開始します ---")
        print(f"出力ファイル: {gif_name}")
        print(f"フレーム数: {len(self.file_list)}")

        original_idx = self.current_idx

        self.is_saving_gif = True
        for ax in self.ui_axes:
            ax.set_visible(False)
        self.info_text.set_text('')

        try:
            def update(frame):
                self.slider.set_val(frame)
                if frame % 10 == 0:
                    print(f"  Frame {frame}/{len(self.file_list)} processed...")
                return [self.mesh]

            anim = animation.FuncAnimation(
                self.fig, update, frames=len(self.file_list), blit=False, repeat=False
            )

            anim.save(gif_name, writer=animation.PillowWriter(fps=GIF_FPS))
            print(f"GIFの保存が完了しました！: {gif_name}\n")

        except Exception as e:
            print(f"GIF保存エラー: {e}")
        finally:
            self.is_saving_gif = False
            for ax in self.ui_axes:
                ax.set_visible(True)
            self.slider.set_val(original_idx)


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

    # --- GIF(既存) ---
    if SAVE_GIF:
        viewer.generate_gif()

    # --- ★新規: 追加解析プロット(それぞれ True/False で制御) ---
    if PLOT_TOTAL_ATOMS_BY_TAA:
        viewer.plot_total_atoms_dawn_dusk(split=False)

    if PLOT_DAWN_DUSK_SPLIT:
        viewer.plot_total_atoms_dawn_dusk(split=True)

    if PLOT_ZENITH_PROFILE:
        viewer.plot_zenith_profile(ZENITH_PROFILE_TAAS, side=ZENITH_PROFILE_SIDE, n_bins=ZENITH_N_BINS)

    print("表示中... ウィンドウを閉じると終了します。")
    plt.show()