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

# プロファイルプロット用のY軸固定最大値
LON_PROFILE_YMAX = 2.4e30
TEMP_PROFILE_YMAX = 800.0   # 温度プロファイルの最大値 [K]
UV_PROFILE_YMAX = 2.0e15    # 紫外線フラックスプロファイルの最大値 [photons/cm²/s]
FLUX_PROFILE_YMAX = 1.5e26  # 総放出量の最大値 [atoms/s]
FLUX_PROFILE_AUTOSCALE = False  # True にすると毎フレーム自動スケール

# ==============================================================================
# ★ 本体コード (mkNaColumnDensity9_9_1.py) と一致させる物理パラメータ
#    ここがズレていると Total Flux が桁違いになる
# ==============================================================================
EV_TO_J   = 1.602176634e-19
K_B       = 1.380649e-23
NU_0      = 1.0e13           # TD の頻度因子 [1/s]

DT_SIM               = 100.0     # SIMULATION_SETTINGS['DT_RATE_UPDATE'] と同じ値にする
TEMP_BASE            = 100.0     # SIMULATION_SETTINGS['TEMP_BASE']
TEMP_AMP             = 600.0     # SIMULATION_SETTINGS['TEMP_AMP']
TEMP_NIGHT           = 100.0     # SIMULATION_SETTINGS['TEMP_NIGHT']
USE_AREA_WEIGHTED_FLUX = False   # SIMULATION_SETTINGS['USE_AREA_WEIGHTED_FLUX']

U_MIN_EV  = 1.4              # SIMULATION_SETTINGS['U_MIN']
U_MAX_EV  = 2.7              # SIMULATION_SETTINGS['U_MAX']

F_UV_1AU_M2 = 1.5e14 * (100 ** 2)     # [photons/m^2/s]
# ★本体: base_q = (q_psd_base * 1.0e-20) / (100 ** 2)  → 単位は m^2
Q_PSD_UNIT_CONV = 1.0 / (100.0 ** 2)

# --- 太陽風スパッタリング (SWS_PARAMS) ---
INCLUDE_SWS   = True
SWS_FLUX_1AU  = 10.0 * 100 ** 3 * 400e3
SWS_YIELD     = 0.06
SWS_REF_DENS  = 7.5e14 * 100 ** 2
SWS_LON_RANGE = (-40.0, 40.0)     # sun-relative [deg]
SWS_LAT_N     = (20.0, 80.0)
SWS_LAT_S     = (-80.0, -20.0)

# --- 保存された表面密度は「1ステップ放出した後」の値なので、
#     放出前の在庫に戻してから放出量を計算する補正 (昼側の過小評価を防ぐ) ---
PRE_STEP_DENSITY_CORRECTION = True
RATE_DT_CAP = 5.0   # exp() 発散防止のため rate*dt の上限

# --- 光電離寿命 (本体: tau_ion = T1AU * AU**2) ---
T1AU_DEFAULT = 190000.0   # RUN_NAME の LT###k から自動取得し、失敗時はこの値
ATMOS_PLOT_ALL_YEARS = False  # True にすると全年を重ね描き (重い)
TAA_PLOT_MODE = 'cumulative'  # 'cumulative' = 累計放出量 ∫P dt / 'ionization' = P·τ の釣り合い
TAA_SPLIT_DAWN_DUSK = True    # 正午-真夜子午線で dawn / dusk 半球に分けて描く
DAWN_DUSK_OVERRIDE = 0        # 0=軌道データから自動判定, +1 or -1 で手動固定 (左右が逆なら符号を反転)

# --- Fastest Tau の設定 ---
TAU_INCLUDE_PSD = True      # PSD (と SWS) も含めた実効タイムスケールにする
TAU_SHOW_COMPONENTS = True  # TD単体 / PSD単体 の細線も重ねて切り替わりを見る
TAU_OCCUPANCY_THRESHOLD = 1.0e5   # 「在庫あり」とみなす最小密度 [atoms/m^2]

# --- データ検査 ---
VALIDATE_ON_LOAD = True     # 読み込みのたびに NaN/負値/形状をチェックする
SANITIZE_ON_LOAD = True     # NaN/Inf/負値を 0 に丸めて描画を続行する (警告は出す)
CAPACITY_TOLERANCE = 1.05   # 1MLキャパシティの何倍までを許容するか

# --- スピンアップを考慮した年ラベル ---
LABEL_SPINUP_YEARS = True   # 最初の SPIN_UP_YEARS 年を 'Spin-up' として区別する

N_LON, N_LAT = 72, 36
BASE_OUTPUT_DIRECTORY = r"./SimulationResult_202607"
RUN_NAME = "Def2_ParabolicHop_72x36_NoEq_DT100_0729_BD0.8_UG1.85_Q0.3_A2.0e+07_LT190k_15yr"
#RUN_NAME = "ParabolicHop_72x36_NoEq_DT100_0710_BD0.8_UG1.85_Q0.3_A2.0e+07_LT190k_15yr"
#RUN_NAME = "ParabolicHop_72x36_NoEq_DT100_0713_BD0.8_UG1.85_Q2.0_A2.0e+07_LT190k_15yr"
#RUN_NAME = "Def2_ParabolicHop_72x36_NoEq_DT100_0729_BD0.8_UG1.85_Q2.0_A2.0e+07_LT190k_15yr"

INITIAL_TARGET_TAA = 100
ORBIT_FILE_PATH = 'orbit2025_spice_unwrapped.txt'
ALIGN_SUN_TO_CENTER = True
USE_LOG_SCALE = True
MERCURY_YEAR_SEC = 87.969 * 86400
SPIN_UP_YEARS = 14.0

R_BODY_KM = 2440.0   # ★本体の PHYSICAL_CONSTANTS['RM'] = 2.440e6 m に合わせる

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
        # ★本体も col5 (subsolar lon) を unwrap している。
        #   これがないと ±180° をまたぐところで np.interp が崩れ、太陽中心揃えがずれる
        orbit_data[:, 5] = np.rad2deg(np.unwrap(np.deg2rad(orbit_data[:, 5])))
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
        f['abs_year'] = year_num
        # ★スピンアップ期間を区別する
        if LABEL_SPINUP_YEARS and year_num <= int(SPIN_UP_YEARS):
            year_key = f"SpinUp {year_num}"
        elif LABEL_SPINUP_YEARS:
            year_key = f"Year {year_num - int(SPIN_UP_YEARS)}"
        else:
            year_key = f"Year {year_num}"
        f['year_key'] = year_key

        if year_key not in grouped_files:
            grouped_files[year_key] = []
        grouped_files[year_key].append(f)

    def _year_sort_key(k):
        # SpinUp を先に、その後本番を並べる
        n = int(k.split()[1])
        return (0, n) if k.startswith("SpinUp") else (1, n)

    sorted_grouped_files = {
        k: grouped_files[k] for k in sorted(grouped_files.keys(), key=_year_sort_key)
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

def validate_surface_array(arr, n_lon, n_lat, path=""):
    """表面密度配列の壊れを検出して問題リストを返す。"""
    problems = []
    if arr is None:
        return ["読み込み失敗"]
    if arr.ndim not in (2, 3):
        problems.append(f"ndim={arr.ndim} (2 or 3 を期待)")
        return problems
    if arr.shape[0] != n_lon or arr.shape[1] != n_lat:
        problems.append(f"shape={arr.shape} (期待: ({n_lon}, {n_lat}, nbins))")
    if arr.size == 0:
        problems.append("空配列")
        return problems

    n_bad = int(np.count_nonzero(~np.isfinite(arr)))
    if n_bad:
        problems.append(f"NaN/Inf が {n_bad} 個 ({100.0*n_bad/arr.size:.3f}%)")

    finite = arr[np.isfinite(arr)]
    if finite.size:
        n_neg = int(np.count_nonzero(finite < 0.0))
        if n_neg:
            problems.append(f"負の密度が {n_neg} 個 (min={finite.min():.3e})")
        vmax = float(finite.max())
        cap = MAX_NA_SURF_DENS_M2 * CAPACITY_TOLERANCE
        if vmax > cap:
            problems.append(f"1MLキャパシティ超過: max={vmax:.3e} > {cap:.3e} [atoms/m^2]")
        if vmax == 0.0:
            problems.append("全要素が 0 (未初期化 or 上書き失敗の痕跡)")
    return problems


def parse_t1au_from_run_name(run_name):
    """RUN_NAME の 'LT190k' から T1AU [s] を取り出す。"""
    m = re.search(r'LT(\d+)k', run_name)
    if m:
        return float(m.group(1)) * 1000.0
    return T1AU_DEFAULT


def parse_simulation_parameters_from_run_name(run_name):
    u_model = 'gaussian_random' if 'UG' in run_name else ('fixed' if 'UF' in run_name else 'fixed')
    u_mu = 1.85
    u_match = re.search(r'U[GF](\d+\.\d+)', run_name)
    if u_match:
        u_mu = float(u_match.group(1))
    q_base = 2.0
    q_match = re.search(r'Q(\d+\.\d+)', run_name)
    if q_match:
        q_base = float(q_match.group(1))
    print(f"【パラメータ自動回収結果】Model Type: {u_model}, U_mu: {u_mu} eV, Q_base_coeff: {q_base}")
    return u_model, u_mu, q_base

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
        self.show_temp_profile = False  
        self.show_uv_profile = False    
        self.show_flux_profile = False  
        self.show_tau_profile = False   
        self.step_mode = "TAA"
        
        self.contour_set = None
        self.lon_profile_line = None    
        self.temp_profile_line = None   
        self.uv_profile_line = None     
        self.flux_profile_line = None   
        self.tau_profile_line = None    
        self.tau_td_line = None
        self.tau_psd_line = None
        self.drawn_vlines = []          

        areas_m2, areas_cm2 = calculate_cell_areas(self.n_lon, self.n_lat, R_BODY_KM)
        self.cell_areas_m2 = areas_m2 

        # ★ RUN_NAME の解析は1回だけ (毎フレーム呼ぶと print が大量に出る)
        self.u_model_type, self.u_mu, self.q_psd_base_coeff = \
            parse_simulation_parameters_from_run_name(RUN_NAME)
        self.reported_bad_files = set()
        self.dawn_sign = self._detect_dawn_sign()
        _side = "lon_sun < 0" if self.dawn_sign > 0 else "lon_sun > 0"
        print(f"【dawn/dusk 判定】dawn 半球 = {_side}  "
              f"(sign={self.dawn_sign:+d}, DAWN_DUSK_OVERRIDE で上書き可)")
        self.t1au = parse_t1au_from_run_name(RUN_NAME)
        print(f"【光電離寿命】T1AU = {self.t1au:.4e} s (tau_ion = T1AU * AU^2)")
        self.total_release_all = 0.0

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

        self.fig, self.ax = plt.subplots(figsize=(14, 7))
        plt.subplots_adjust(bottom=0.25, left=0.08, right=0.52)

        self.ax_twin = self.ax.twinx()
        self.ax_twin.yaxis.set_visible(False)
        
        self.ax_twin_temp = self.ax.twinx()
        self.ax_twin_temp.yaxis.set_visible(False)
        self.ax_twin_temp.spines['right'].set_position(('outward', 50))

        self.ax_twin_uv = self.ax.twinx()
        self.ax_twin_uv.yaxis.set_visible(False)
        self.ax_twin_uv.spines['right'].set_position(('outward', 105)) 

        self.ax_twin_flux = self.ax.twinx()
        self.ax_twin_flux.yaxis.set_visible(False)
        self.ax_twin_flux.spines['right'].set_position(('outward', 160))

        self.ax_twin_tau = self.ax.twinx()
        self.ax_twin_tau.yaxis.set_visible(False)
        self.ax_twin_tau.spines['right'].set_position(('outward', 225))

        import copy
        self.cmap = copy.copy(plt.get_cmap('inferno'))
        self.cmap.set_bad('black')

        if self.use_log:
            self.norm = LogNorm(vmin=self.vmin, vmax=self.vmax)
        else:
            self.norm = Normalize(vmin=self.vmin, vmax=self.vmax)

        dummy_data = np.zeros((self.n_lat, self.n_lon))
        self.mesh = self.ax.pcolormesh(dummy_data, cmap=self.cmap, norm=self.norm)
        
        self.ax_cbar = plt.axes([0.94, 0.25, 0.015, 0.65])
        self.cbar = plt.colorbar(self.mesh, cax=self.ax_cbar, label=f'Surface Density {self.unit_label}')

        self.info_text = self.ax.text(
            0.98, 0.95, '',
            transform=self.ax.transAxes,
            ha='right', va='top',
            fontsize=10, color='white', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="white", alpha=0.7)
        )

        self.warn_text = self.ax.text(
            0.02, 0.03, '', transform=self.ax.transAxes,
            ha='left', va='bottom', fontsize=10, color='yellow', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="darkred", ec="yellow", alpha=0.85),
            zorder=200, visible=False
        )

        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        # === UIレイアウト ===
        self.ax_slider = plt.axes([0.15, 0.165, 0.42, 0.03], facecolor='lightgoldenrodyellow')
        self.slider = Slider(self.ax_slider, 'Index/TAA', 0, len(self.file_list) - 1, valinit=self.current_idx, valfmt='%d')
        self.slider.on_changed(self.on_slider_change)

        self.ax_prev = plt.axes([0.075, 0.06, 0.048, 0.05])
        self.btn_prev = Button(self.ax_prev, 'Previous')
        self.btn_prev.on_clicked(self.prev_frame)

        self.ax_next = plt.axes([0.126, 0.06, 0.048, 0.05])
        self.btn_next = Button(self.ax_next, 'Next')
        self.btn_next.on_clicked(self.next_frame)

        self.ax_mode = plt.axes([0.177, 0.06, 0.075, 0.05])
        self.btn_mode = Button(self.ax_mode, f'Mode: {self.step_mode}', color='lightblue')
        self.btn_mode.on_clicked(self.toggle_step_mode)

        self.ax_plate = plt.axes([0.255, 0.06, 0.075, 0.05])
        self.btn_plate = Button(self.ax_plate, 'Show Plate')
        self.btn_plate.on_clicked(self.generate_plate_view)

        self.ax_total = plt.axes([0.333, 0.06, 0.075, 0.05])
        self.btn_total = Button(self.ax_total, 'Plot Total')
        self.btn_total.on_clicked(self.plot_total_atoms)

        self.ax_gif = plt.axes([0.01, 0.21, 0.06, 0.04])
        self.btn_gif = Button(self.ax_gif, 'Save GIF', color='salmon')
        self.btn_gif.on_clicked(self.on_gif_btn_clicked)

        # ★大気量 (電離寿命考慮) プロットボタン
        self.ax_atmos = plt.axes([0.411, 0.06, 0.075, 0.05])
        self.btn_atmos = Button(self.ax_atmos, 'Plot Cumul.', color='lightgreen')
        self.btn_atmos.on_clicked(self.plot_atmosphere_vs_taa)

        # ★データ検査ボタン
        self.ax_verify = plt.axes([0.489, 0.06, 0.075, 0.05])
        self.btn_verify = Button(self.ax_verify, 'Verify Data', color='khaki')
        self.btn_verify.on_clicked(self.verify_all_data)

        # ★チェックボックスは 2 パネルに分けて大きくする
        self.ax_check = plt.axes([0.60, 0.015, 0.155, 0.205])
        self.check = CheckButtons(
            self.ax_check,
            ['Contours', 'SSP', 'Terminator', 'Planet 0°', 'Lon Profile'],
            [self.show_contours, self.show_ssp, self.show_terminator,
             self.show_planet_zero, self.show_lon_profile]
        )
        self.check.on_clicked(self.toggle_checks)

        self.ax_check2 = plt.axes([0.775, 0.015, 0.155, 0.205])
        self.check2 = CheckButtons(
            self.ax_check2,
            ['Temp Profile', 'UV Flux', 'Total Flux', 'Fastest Tau'],
            [self.show_temp_profile, self.show_uv_profile,
             self.show_flux_profile, self.show_tau_profile]
        )
        self.check2.on_clicked(self.toggle_checks)

        for chk in (self.check, self.check2):
            for lab in chk.labels:
                lab.set_fontsize(10)

        self.ax_radio = plt.axes([0.01, 0.04, 0.06, 0.15], facecolor='lightgrey')
        self.radio = RadioButtons(self.ax_radio, self.years_available, active=len(self.years_available)-1)
        self.radio.on_clicked(self.change_year)

        self.ui_axes = [
            self.ax_slider, self.ax_prev, self.ax_next, self.ax_mode,
            self.ax_plate, self.ax_total, self.ax_atmos, self.ax_verify,
            self.ax_check, self.ax_check2, self.ax_radio, self.ax_gif
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
        self._set_warning("")
        self.file_list = self.grouped_files[label]
        self._find_initial_index()

        self.slider.valmax = len(self.file_list) - 1
        self.slider.ax.set_xlim(self.slider.valmin, self.slider.valmax)
        self.slider.set_val(self.current_idx)
        self.update_plot()

    def toggle_checks(self, label):
        if label == 'Contours': self.show_contours = not self.show_contours
        elif label == 'SSP': self.show_ssp = not self.show_ssp
        elif label == 'Terminator': self.show_terminator = not self.show_terminator
        elif label == 'Planet 0°': self.show_planet_zero = not self.show_planet_zero
        elif label == 'Lon Profile': self.show_lon_profile = not self.show_lon_profile
        elif label == 'Temp Profile': self.show_temp_profile = not self.show_temp_profile
        elif label == 'UV Flux': self.show_uv_profile = not self.show_uv_profile
        elif label == 'Total Flux': self.show_flux_profile = not self.show_flux_profile
        elif label == 'Fastest Tau': self.show_tau_profile = not self.show_tau_profile
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

    def _get_bins(self, n_bins_file):
        """本体の setup_binding_energy_bins と同じ U ビン・Q_PSD ビンを再現する。"""
        if self.u_model_type == 'fixed':
            u_bins = np.full(n_bins_file, self.u_mu)
        else:
            u_bins = np.linspace(U_MIN_EV, U_MAX_EV, n_bins_file)
        q_psd_bins = np.full(n_bins_file,
                             self.q_psd_base_coeff * 1.0e-20 * Q_PSD_UNIT_CONV)
        return u_bins, q_psd_bins

    def _sws_rate_map(self, lon_centers, lat_centers, subsolar_lon_deg, AU):
        """太陽風スパッタリング率マップ [1/s], shape=(LAT, LON)"""
        if not INCLUDE_SWS:
            return 0.0
        if self.align_sun:
            lon_sun = lon_centers
        else:
            lon_sun = (lon_centers - subsolar_lon_deg + 180.0) % 360.0 - 180.0
        mask_lon = (lon_sun >= SWS_LON_RANGE[0]) & (lon_sun <= SWS_LON_RANGE[1])
        mask_lat = (((lat_centers >= SWS_LAT_N[0]) & (lat_centers <= SWS_LAT_N[1])) |
                    ((lat_centers >= SWS_LAT_S[0]) & (lat_centers <= SWS_LAT_S[1])))
        mask = mask_lat[:, np.newaxis] & mask_lon[np.newaxis, :]
        sw_flux = SWS_FLUX_1AU / (AU ** 2)
        return np.where(mask, sw_flux * SWS_YIELD / SWS_REF_DENS, 0.0)

    def _safe_load(self, path, tag=""):
        """npy を読み、壊れていたら (None, 問題リスト) を返す。"""
        try:
            arr = np.load(path, allow_pickle=False)
        except Exception as e:
            msg = [f"np.load 失敗: {type(e).__name__}: {e}"]
            self._report_problems(path, msg, tag)
            return None, msg
        problems = validate_surface_array(arr, self.n_lon, self.n_lat, path) if VALIDATE_ON_LOAD else []
        if problems:
            self._report_problems(path, problems, tag)

        # ★形状が違うものは致命的 -> None を返して呼び出し側にスキップさせる
        fatal = (arr.ndim not in (2, 3) or
                 arr.shape[0] != self.n_lon or arr.shape[1] != self.n_lat or arr.size == 0)
        if fatal:
            return None, problems

        # ★NaN/Inf/負値は丸めて続行 (警告は出した上で)
        if SANITIZE_ON_LOAD and problems:
            if not np.all(np.isfinite(arr)):
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if np.any(arr < 0.0):
                arr = np.clip(arr, 0.0, None)
        return arr, problems

    def _report_problems(self, path, problems, tag=""):
        key = os.path.basename(path)
        if key in self.reported_bad_files:
            return
        self.reported_bad_files.add(key)
        head = f"\u26a0 データ異常 [{tag}] {key}" if tag else f"\u26a0 データ異常 {key}"
        print(head)
        for p_ in problems:
            print(f"    - {p_}")

    def _set_warning(self, text):
        if not hasattr(self, 'warn_text') or self.warn_text is None:
            return
        self.warn_text.set_text(text)
        self.warn_text.set_visible(bool(text))

    def verify_all_data(self, event=None, all_years=False):
        """現在の年 (または全年) のファイルを一括検査する。"""
        keys = list(self.grouped_files.keys()) if all_years else [self.current_year_key]
        n_ok = n_bad = 0
        bad_list = []
        print("=" * 70)
        print(f"データ検査開始: {keys}")
        for k in keys:
            for f in self.grouped_files[k]:
                try:
                    arr = np.load(f['path'], allow_pickle=False)
                    problems = validate_surface_array(arr, self.n_lon, self.n_lat, f['path'])
                except Exception as e:
                    problems = [f"np.load 失敗: {type(e).__name__}: {e}"]
                if problems:
                    n_bad += 1
                    bad_list.append((k, f['time_h'], f['taa'], problems))
                    print(f"  NG  {k} t={f['time_h']}h taa={f['taa']}  {os.path.basename(f['path'])}")
                    for p_ in problems:
                        print(f"        - {p_}")
                else:
                    n_ok += 1
        print(f"検査完了: OK={n_ok}  NG={n_bad}")
        print("=" * 70)
        if n_bad:
            self._set_warning(f"[!] {n_bad} files broken (see console)")
        else:
            self._set_warning("")
        self.fig.canvas.draw_idle()
        return bad_list

    def _release_total_from_surfdata(self, surf_data, subsolar_lon_deg, AU):
        """惑星固定フレームのまま放出率 [atoms/s] を返す。
           戻り値: (全球, dawn 半球, dusk 半球)
           (update_plot 内の計算と同じ式。回転は結果に影響しないので省略)"""
        if surf_data.ndim != 3:
            return 0.0, 0.0, 0.0
        n_bins_file = surf_data.shape[2]
        u_bins, q_psd_bins = self._get_bins(n_bins_file)

        lon_edges = np.linspace(-180, 180, self.n_lon + 1)
        lat_edges = np.linspace(-90, 90, self.n_lat + 1)
        lon_c = (lon_edges[:-1] + lon_edges[1:]) / 2.0
        lat_c = (lat_edges[:-1] + lat_edges[1:]) / 2.0

        lon_sun = (lon_c - subsolar_lon_deg + 180.0) % 360.0 - 180.0
        cos2d = (np.cos(np.deg2rad(lat_c))[:, np.newaxis] *
                 np.cos(np.deg2rad(lon_sun))[np.newaxis, :])
        cos_safe = np.maximum(cos2d, 0.0)
        temp = TEMP_BASE + TEMP_AMP * (cos_safe ** 0.25) * np.sqrt(0.306 / AU)

        if USE_AREA_WEIGHTED_FLUX:
            dlon_val = 360.0 / self.n_lon
            shw = np.sin(np.deg2rad(dlon_val / 2.0))
            illum = np.clip((cos2d + shw) / (2.0 * shw), 0.0, 1.0)
        else:
            illum = np.where(cos2d > 0.0, 1.0, 0.0)

        # SWS
        if INCLUDE_SWS:
            mask_lon = (lon_sun >= SWS_LON_RANGE[0]) & (lon_sun <= SWS_LON_RANGE[1])
            mask_lat = (((lat_c >= SWS_LAT_N[0]) & (lat_c <= SWS_LAT_N[1])) |
                        ((lat_c >= SWS_LAT_S[0]) & (lat_c <= SWS_LAT_S[1])))
            mask = mask_lat[:, np.newaxis] & mask_lon[np.newaxis, :]
            rate_sws = np.where(mask, (SWS_FLUX_1AU / (AU ** 2)) * SWS_YIELD / SWS_REF_DENS, 0.0)
        else:
            rate_sws = 0.0

        psd_geom = (F_UV_1AU_M2 / (AU ** 2)) * cos_safe * illum

        dawn_col = self._dawn_mask(lon_sun)          # (LON,)
        dawn_2d = np.broadcast_to(dawn_col[np.newaxis, :], (self.n_lat, self.n_lon))

        total = 0.0
        total_dawn = 0.0
        for b in range(n_bins_file):
            u_j = u_bins[b] * EV_TO_J
            rate_psd = psd_geom * q_psd_bins[b]
            exp_day = np.maximum(-u_j / (K_B * temp), -700.0)
            rate_day = np.where(temp >= 10.0, NU_0 * np.exp(exp_day), 0.0)
            rate_night = NU_0 * np.exp(max(-u_j / (K_B * TEMP_NIGHT), -700.0))
            if USE_AREA_WEIGHTED_FLUX:
                rate_td = rate_day * illum + rate_night * (1.0 - illum)
            else:
                rate_td = np.where(illum > 0.5, rate_day, rate_night)
            rate_tot = rate_psd + rate_td + rate_sws
            if PRE_STEP_DENSITY_CORRECTION:
                x = np.minimum(rate_tot * DT_SIM, RATE_DT_CAP)
                avg = np.where(rate_tot > 1e-30, (np.exp(x) - 1.0) / DT_SIM, 0.0)
            else:
                avg = np.where(rate_tot > 1e-30, (1.0 - np.exp(-rate_tot * DT_SIM)) / DT_SIM, 0.0)
            flux_map = surf_data[:, :, b].T * avg * self.cell_areas_m2
            total += float(np.sum(flux_map))
            total_dawn += float(np.sum(np.where(dawn_2d, flux_map, 0.0)))
        return total, total_dawn, total - total_dawn

    def _detect_dawn_sign(self):
        """表面の点が sun-relative 経度上をどちら向きに進むかを軌道データから判定する。

        lon_sun = lon - sub_lon なので d(lon_sun)/dt = -d(sub_lon)/dt。
        戻り値 +1: lon_sun は時間とともに増加 → 点は lon_sun<0 側から昇ってくる
                        → dawn (朝側) = lon_sun < 0
        戻り値 -1: その逆。
        水星は近日点付近で太陽が逆行するので、瞬時の傾きではなく全期間の平均傾きを使う。
        """
        if DAWN_DUSK_OVERRIDE in (+1, -1):
            return int(DAWN_DUSK_OVERRIDE)
        t = self.orbit_data[:, 2]
        sl = self.orbit_data[:, 5]   # unwrap 済み
        if t[-1] == t[0]:
            return 1
        slope = (sl[-1] - sl[0]) / (t[-1] - t[0])
        sign = -1 if slope > 0 else 1
        return sign

    def _dawn_mask(self, lon_sun_deg):
        """dawn 半球 True / dusk 半球 False の 1D マスク (経度方向)。"""
        if self.dawn_sign > 0:
            return lon_sun_deg < 0.0
        return lon_sun_deg > 0.0

    def _orbit_at(self, time_h):
        tcol = self.orbit_data[:, 2]
        t_lookup = np.clip(self.t_start + float(time_h) * 3600.0, tcol[0], tcol[-1])
        sub_lon = np.interp(t_lookup, tcol, self.orbit_data[:, 5])
        AU = np.interp(t_lookup, tcol, self.orbit_data[:, 1])
        return sub_lon, AU

    def _load_and_align(self, data_info):
        filepath = data_info['path']
        time_h = data_info['time_h']
        data = np.load(filepath)
        if data.ndim == 3: data = np.sum(data, axis=2)
        subsolar_lon_deg = get_subsolar_longitude_linear(time_h, self.t_start, self.orbit_data)
        if self.align_sun:
            dlon = 360.0 / self.n_lon
            sun_pos_norm = (subsolar_lon_deg + 180.0) % 360.0
            sun_index = int(np.floor(sun_pos_norm / dlon)) % self.n_lon
            shift = (self.n_lon // 2) - sun_index
            data = np.roll(data, shift=shift, axis=0)
        data_T = np.nan_to_num(data.T, nan=0.0)
        if USE_PAPER_SCALE: data_T = data_T / 10000.0
        eff_cos, is_dawn = compute_effcos_grid(self.n_lon, self.n_lat, 0.0 if self.align_sun else subsolar_lon_deg)
        return data_T, eff_cos, is_dawn

    def update_plot(self):
        data_info = self.file_list[self.current_idx]
        filepath = data_info['path']
        time_h = data_info['time_h']
        taa = data_info['taa']

        # ★検査付き読み込み
        surf_data, problems = self._safe_load(filepath, tag=self.current_year_key)
        if surf_data is None:
            reason = problems[0] if problems else "unknown"
            print(f"[update_plot] スキップ: {os.path.basename(filepath)} ({reason})")
            self._set_warning(f"[!] SKIPPED: {os.path.basename(filepath)}\n{reason[:70]}")
            self.fig.canvas.draw_idle()
            return
        if problems:
            self._set_warning("[!] " + " / ".join(problems)[:90])
        else:
            self._set_warning("")
        n_bins_file = surf_data.shape[2] if surf_data.ndim == 3 else 1
        data = np.sum(surf_data, axis=2) if surf_data.ndim == 3 else surf_data

        time_col_original = self.orbit_data[:, 2]
        current_t_sec = self.t_start + (float(time_h) * 3600.0)
        t_lookup = np.clip(current_t_sec, time_col_original[0], time_col_original[-1])
        subsolar_lon_deg = np.interp(t_lookup, time_col_original, self.orbit_data[:, 5])
        AU = np.interp(t_lookup, time_col_original, self.orbit_data[:, 1])
        ssp_planet_lon = (subsolar_lon_deg + 180) % 360 - 180

        xlabel = "Longitude (Planet)"
        title_mode = "(Planet Fixed)"

        shift = 0
        if self.align_sun:
            dlon = 360.0 / self.n_lon
            sun_pos_norm = (subsolar_lon_deg + 180.0) % 360.0
            sun_index = int(np.floor(sun_pos_norm / dlon)) % self.n_lon
            shift = (self.n_lon // 2) - sun_index
            data = np.roll(data, shift=shift, axis=0)
            title_mode = "(Sun Centered)"
            xlabel = "Longitude (Sun-Relative)"

        data_T = data.T
        data_T = np.nan_to_num(data_T, nan=0.0)
        
        display_data_T = data_T / 10000.0 if USE_PAPER_SCALE else data_T
        self.current_display_data = display_data_T

        if self.contour_set is not None:
            try: self.contour_set.remove()
            except: pass
            self.contour_set = None
        if self.mesh: self.mesh.remove()

        lon_edges = np.linspace(-180, 180, self.n_lon + 1)
        lat_edges = np.linspace(-90, 90, self.n_lat + 1)
        lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2.0
        lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2.0

        self.mesh = self.ax.pcolormesh(lon_edges, lat_edges, display_data_T, cmap=self.cmap, norm=self.norm, shading='flat')

        if self.show_contours:
            try:
                X, Y = np.meshgrid(lon_centers, lat_centers)
                data_contour = display_data_T.copy()
                data_contour[data_contour <= 0] = np.nan
                if np.nanmax(data_contour) >= self.vmin:
                    exp_min = np.floor(np.log10(self.vmin))
                    exp_max = np.ceil(np.log10(self.vmax))
                    levels = np.logspace(exp_min, exp_max, num=int(exp_max - exp_min) + 1)
                    self.contour_set = self.ax.contour(X, Y, data_contour, levels=levels, colors='cyan', linewidths=0.8)
                    self.ax.clabel(self.contour_set, inline=True, fontsize=8, fmt='%.0e', colors='white')
            except Exception as e: print(f"Contour error: {e}")

        # === [1] 経度原子数プロファイル ===
        if self.show_lon_profile:
            atoms_per_lon = np.sum(display_data_T * self.cell_areas, axis=0) 
            self.ax_twin.yaxis.set_visible(True)
            self.ax_twin.set_ylabel('Total Atoms [atoms]', color='magenta', fontsize=11, fontweight='bold')
            if self.lon_profile_line is not None: self.lon_profile_line.set_data(lon_centers, atoms_per_lon)
            else: self.lon_profile_line, = self.ax_twin.plot(lon_centers, atoms_per_lon, color='magenta', linestyle='-', linewidth=2.2, label='Atoms', zorder=10)
            self.ax_twin.tick_params(axis='y', labelcolor='magenta')
            self.ax_twin.set_ylim(0, LON_PROFILE_YMAX)
        else:
            self.ax_twin.yaxis.set_visible(False)
            if self.lon_profile_line is not None: self.lon_profile_line.remove(); self.lon_profile_line = None

        # 幾何学・日照条件 (shape: LAT, LON)
        lon_rad = np.deg2rad(lon_centers)
        lat_rad = np.deg2rad(lat_centers)
        if self.align_sun: cos_theta_2d = np.cos(lat_rad[:, np.newaxis]) * np.cos(lon_rad[np.newaxis, :])
        else:              cos_theta_2d = np.cos(lat_rad[:, np.newaxis]) * np.cos(lon_rad[np.newaxis, :] - np.deg2rad(subsolar_lon_deg))
        cos_theta_safe = np.maximum(cos_theta_2d, 0.0)

        dlon_val = 360.0 / self.n_lon
        sin_half_width = np.sin(np.deg2rad(dlon_val / 2.0))
        illum_map = np.where(cos_theta_2d > sin_half_width, 1.0, np.where(cos_theta_2d < -sin_half_width, 0.0, (cos_theta_2d + sin_half_width) / (2.0 * sin_half_width)))
        
        scaling = np.sqrt(0.306 / AU)
        # ★本体と同一: t_day = TEMP_BASE + TEMP_AMP * (max(0,cos)^0.25) * scaling
        #   (夜側は cos_theta_safe = 0 なので自動的に TEMP_BASE になる)
        temp_map = TEMP_BASE + TEMP_AMP * (cos_theta_safe ** 0.25) * scaling

        # === [2] 表面温度プロファイル（赤道） ===
        if self.show_temp_profile:
            self.ax_twin_temp.yaxis.set_visible(True)
            self.ax_twin_temp.set_ylabel('Equatorial Temp [K]', color='red', fontsize=11, fontweight='bold')
            if self.temp_profile_line is not None: self.temp_profile_line.set_data(lon_centers, temp_map[self.n_lat // 2, :])
            else: self.temp_profile_line, = self.ax_twin_temp.plot(lon_centers, temp_map[self.n_lat // 2, :], color='red', linestyle='--', linewidth=2.0, label='Temp', zorder=9)
            self.ax_twin_temp.tick_params(axis='y', labelcolor='red')
            self.ax_twin_temp.set_ylim(0, TEMP_PROFILE_YMAX)
        else:
            self.ax_twin_temp.yaxis.set_visible(False)
            if self.temp_profile_line is not None: self.temp_profile_line.remove(); self.temp_profile_line = None

        # === [3] 紫外線フラックスプロファイル ===
        f_uv_m2 = F_UV_1AU_M2 / (AU ** 2)
        if self.show_uv_profile:
            f_uv_mercury = 1.5e14 / (AU ** 2)
            uv_profile = np.where(cos_theta_2d[self.n_lat // 2, :] > 0, f_uv_mercury * cos_theta_2d[self.n_lat // 2, :], 0.0)
            self.ax_twin_uv.yaxis.set_visible(True)
            self.ax_twin_uv.set_ylabel('UV Flux [photons/cm²/s]', color='teal', fontsize=11, fontweight='bold')
            if self.uv_profile_line is not None: self.uv_profile_line.set_data(lon_centers, uv_profile)
            else: self.uv_profile_line, = self.ax_twin_uv.plot(lon_centers, uv_profile, color='teal', linestyle='-.', linewidth=2.0, label='UV Flux', zorder=8)
            self.ax_twin_uv.tick_params(axis='y', labelcolor='teal')
            self.ax_twin_uv.set_ylim(0, UV_PROFILE_YMAX)
        else:
            self.ax_twin_uv.yaxis.set_visible(False)
            if self.uv_profile_line is not None: self.uv_profile_line.remove(); self.uv_profile_line = None

        # ======================================================================
        # ★放出量の逆算 (本体の update_surface_maps_numba と式を一致させる)
        # ======================================================================
        u_bins, q_psd_bins = self._get_bins(n_bins_file)

        total_release_flux_per_lon = np.zeros(self.n_lon)
        fastest_tau_profile = np.full(self.n_lon, np.nan)
        self.total_release_all = 0.0

        # ★使わないときは重いループを回さない
        need_flux = self.show_flux_profile or self.show_tau_profile

        tau_td_profile = np.full(self.n_lon, np.nan)
        tau_psd_profile = np.full(self.n_lon, np.nan)

        if need_flux and surf_data.ndim == 3:
            # 太陽中心の場合は3D配列もシフトさせる
            surf_data_aligned = np.roll(surf_data, shift=shift, axis=0) if self.align_sun else surf_data

            # --- 照度: 本体は USE_AREA_WEIGHTED_FLUX=False のとき二値 ---
            if USE_AREA_WEIGHTED_FLUX:
                illum_eff = illum_map
            else:
                illum_eff = np.where(cos_theta_2d > 0.0, 1.0, 0.0)

            # --- 太陽風スパッタリング率 [1/s] ---
            rate_sws = self._sws_rate_map(lon_centers, lat_centers, subsolar_lon_deg, AU)
            if np.isscalar(rate_sws):
                rate_sws = np.full_like(cos_theta_2d, float(rate_sws))

            # --- PSD率 [1/s] (全ビンで q が同じなのでループ外で作る) ---
            psd_geom = f_uv_m2 * np.maximum(cos_theta_2d, 0.0) * illum_eff

            # ------------------------------------------------------------------
            # ★最速タイムスケール (赤道上) : PSD + TD + SWS を含める
            #    在庫のある最も浅いビン (= 最大 rate) で評価
            # ------------------------------------------------------------------
            eq_j = self.n_lat // 2
            t_eq = temp_map[eq_j, :]
            psd_eq = psd_geom[eq_j, :]
            sws_eq = rate_sws[eq_j, :]
            illum_eq = illum_eff[eq_j, :]
            occupied = surf_data_aligned[:, eq_j, :] > TAU_OCCUPANCY_THRESHOLD  # (LON, BIN)

            for i in range(self.n_lon):
                idx = np.flatnonzero(occupied[i])
                if idx.size == 0:
                    fastest_tau_profile[i] = 1e10
                    continue
                b = int(idx[0])                      # 在庫のある最も浅いビン
                u_j = u_bins[b] * EV_TO_J
                temp_b = t_eq[i] if illum_eq[i] > 0.5 else TEMP_NIGHT
                exp_val = max(-u_j / (K_B * temp_b), -700.0)
                r_td = NU_0 * np.exp(exp_val) if temp_b >= 10.0 else 0.0
                r_psd = psd_eq[i] * q_psd_bins[b]
                r_sws = sws_eq[i]

                r_use = (r_td + r_psd + r_sws) if TAU_INCLUDE_PSD else r_td
                fastest_tau_profile[i] = 1.0 / r_use if r_use > 1e-30 else 1e10
                tau_td_profile[i] = 1.0 / r_td if r_td > 1e-30 else 1e10
                r_p = r_psd + r_sws
                tau_psd_profile[i] = 1.0 / r_p if r_p > 1e-30 else 1e10

            # --- 総放出フラックスの計算 (ビンごと) ---
            for b in range(n_bins_file):
                u_j = u_bins[b] * EV_TO_J
                rate_psd = psd_geom * q_psd_bins[b]

                # TD率 [1/s] : 昼側は temp_map、夜側は TEMP_NIGHT
                exp_day = np.maximum(-u_j / (K_B * temp_map), -700.0)
                rate_day = np.where(temp_map >= 10.0, NU_0 * np.exp(exp_day), 0.0)

                exp_night = max(-u_j / (K_B * TEMP_NIGHT), -700.0)
                rate_night = NU_0 * np.exp(exp_night)

                if USE_AREA_WEIGHTED_FLUX:
                    rate_td = rate_day * illum_map + rate_night * (1.0 - illum_map)
                else:
                    rate_td = np.where(illum_eff > 0.5, rate_day, rate_night)

                rate_tot = rate_psd + rate_td + rate_sws

                # dt 間の減衰を考慮した平均放出率 (1 - exp(-nu*dt))/dt
                if PRE_STEP_DENSITY_CORRECTION:
                    # d_after * (exp(r*dt)-1)/dt == d_before * (1-exp(-r*dt))/dt
                    x = np.minimum(rate_tot * DT_SIM, RATE_DT_CAP)
                    avg_loss_rate = np.where(rate_tot > 1e-30,
                                             (np.exp(x) - 1.0) / DT_SIM, 0.0)
                else:
                    avg_loss_rate = np.where(rate_tot > 1e-30,
                                             (1.0 - np.exp(-rate_tot * DT_SIM)) / DT_SIM,
                                             0.0)

                # 密度 [atoms/m^2] * 平均放出率 [1/s] * 面積 [m^2]
                dens_T = surf_data_aligned[:, :, b].T # shape: [LAT, LON]
                flux_map_b = dens_T * avg_loss_rate * self.cell_areas_m2

                total_release_flux_per_lon += np.sum(flux_map_b, axis=0)

            self.total_release_all = float(np.sum(total_release_flux_per_lon))

        # === [4] 総放出量プロファイルの描画 ===
        if self.show_flux_profile:
            self.ax_twin_flux.yaxis.set_visible(True)
            self.ax_twin_flux.set_ylabel('Total Release [atoms/s]', color='blue', fontsize=11, fontweight='bold')
            if self.flux_profile_line is not None: self.flux_profile_line.set_data(lon_centers, total_release_flux_per_lon)
            else: self.flux_profile_line, = self.ax_twin_flux.plot(lon_centers, total_release_flux_per_lon, color='blue', linestyle='-', linewidth=2.2, label='Release Flux', zorder=11)
            self.ax_twin_flux.tick_params(axis='y', labelcolor='blue')
            if FLUX_PROFILE_AUTOSCALE:
                fmax = float(np.nanmax(total_release_flux_per_lon)) if total_release_flux_per_lon.size else 0.0
                self.ax_twin_flux.set_ylim(0, fmax * 1.15 if fmax > 0 else 1.0)
            else:
                self.ax_twin_flux.set_ylim(0, FLUX_PROFILE_YMAX)
            self.ax_twin_flux.set_ylabel(
                f'Total Release [atoms/s]  (\u03a3={self.total_release_all:.3e})',
                color='blue', fontsize=11, fontweight='bold')
        else:
            self.ax_twin_flux.yaxis.set_visible(False)
            if self.flux_profile_line is not None: self.flux_profile_line.remove(); self.flux_profile_line = None

        # === [5] 最速放出タイムスケールプロファイルの描画 ===
        if self.show_tau_profile:
            self.ax_twin_tau.yaxis.set_visible(True)
            self.ax_twin_tau.set_ylabel(
                'Fastest Tau [s] (PSD+TD+SWS)' if TAU_INCLUDE_PSD else 'Fastest TD Tau [s]',
                color='darkorange', fontsize=11, fontweight='bold')
            self.ax_twin_tau.set_yscale('log') 
            if self.tau_profile_line is not None: self.tau_profile_line.set_data(lon_centers, fastest_tau_profile)
            else: self.tau_profile_line, = self.ax_twin_tau.plot(lon_centers, fastest_tau_profile, color='darkorange', linestyle=':', marker='x', markersize=3, linewidth=1.8, label='Fastest Tau', zorder=12)
            # ★ TD単体 / PSD+SWS単体 を重ねて、どこで支配過程が入れ替わるかを見る
            if TAU_SHOW_COMPONENTS:
                if self.tau_td_line is not None: self.tau_td_line.set_data(lon_centers, tau_td_profile)
                else: self.tau_td_line, = self.ax_twin_tau.plot(lon_centers, tau_td_profile, color='red', linestyle='--', linewidth=1.0, alpha=0.65, label='TD only', zorder=12)
                if self.tau_psd_line is not None: self.tau_psd_line.set_data(lon_centers, tau_psd_profile)
                else: self.tau_psd_line, = self.ax_twin_tau.plot(lon_centers, tau_psd_profile, color='deepskyblue', linestyle='--', linewidth=1.0, alpha=0.65, label='PSD+SWS only', zorder=12)
            self.ax_twin_tau.tick_params(axis='y', labelcolor='darkorange')
            self.ax_twin_tau.set_ylim(1e-1, 1e8) 
        else:
            self.ax_twin_tau.yaxis.set_visible(False)
            if self.tau_profile_line is not None: self.tau_profile_line.remove(); self.tau_profile_line = None
            if self.tau_td_line is not None: self.tau_td_line.remove(); self.tau_td_line = None
            if self.tau_psd_line is not None: self.tau_psd_line.remove(); self.tau_psd_line = None

        # === 縦線（SSP/境界線）の更新 ===
        for line in self.drawn_vlines:
            try: line.remove()
            except: pass
        self.drawn_vlines.clear()

        eq_line = self.ax.axhline(0, color='white', linestyle=':', alpha=0.3)
        self.drawn_vlines.append(eq_line)

        if self.align_sun:
            if self.show_ssp:
                l1 = self.ax.axvline(0.0, color='#FFBF00', linestyle='-', linewidth=2.0, alpha=0.8)
                self.drawn_vlines.append(l1)
            if self.show_terminator:
                l2 = self.ax.axvline(-90.0, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
                l3 = self.ax.axvline(90.0, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
                self.drawn_vlines.append(l2); self.drawn_vlines.append(l3)
                
                moving_dawn = (-ssp_planet_lon - 90 + 180) % 360 - 180
                moving_dusk = (-ssp_planet_lon + 90 + 180) % 360 - 180
                l5 = self.ax.axvline(moving_dawn, color='cyan', linestyle=':', linewidth=1.2, alpha=0.6)
                l6 = self.ax.axvline(moving_dusk, color='cyan', linestyle=':', linewidth=1.2, alpha=0.6)
                self.drawn_vlines.append(l5); self.drawn_vlines.append(l6)
        else:
            if self.show_planet_zero:
                l1 = self.ax.axvline(0.0, color='lime', linestyle='-.', linewidth=2.0, alpha=0.8)
                self.drawn_vlines.append(l1)
            if self.show_ssp:
                l2 = self.ax.axvline(ssp_planet_lon, color='#FFBF00', linestyle='-', linewidth=2.0, alpha=0.8)
                self.drawn_vlines.append(l2)
            if self.show_terminator:
                term_dawn_disp = (ssp_planet_lon - 90 + 180) % 360 - 180
                term_dusk_disp = (ssp_planet_lon + 90 + 180) % 360 - 180
                l3 = self.ax.axvline(term_dawn_disp, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
                l4 = self.ax.axvline(term_dusk_disp, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
                self.drawn_vlines.append(l3); self.drawn_vlines.append(l4)

        self.info_text.set_text(f"Index: {self.current_idx}")
        self.info_text.set_zorder(100)
        if not self.is_saving_gif: self.fig.canvas.draw_idle()

    def on_mouse_move(self, event):
        if event.inaxes not in [self.ax, self.ax_twin, self.ax_twin_temp, self.ax_twin_uv, self.ax_twin_flux, self.ax_twin_tau]: return
        if self.current_display_data is None: return
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
        self.info_text.set_text(f"Lon: {x:.1f}\nLat: {y:.1f}\nVal: {val:.2e}\nCap: {pct_of_max:.2f}%\nvs Init: {ratio_to_init:.2f}x")
        self.fig.canvas.draw_idle()

    def _get_next_index_by_mode(self, start_idx):
        if self.step_mode == "TAA": return min(start_idx + 1, len(self.file_list) - 1)
        else:
            current_t = self.file_list[start_idx]['time_h']
            for i in range(start_idx + 1, len(self.file_list)):
                if self.file_list[i]['time_h'] - current_t >= TIME_STEP_HOURS: return i
            return len(self.file_list) - 1

    def _get_prev_index_by_mode(self, start_idx):
        if self.step_mode == "TAA": return max(start_idx - 1, 0)
        else:
            current_t = self.file_list[start_idx]['time_h']
            for i in range(start_idx - 1, -1, -1):
                if current_t - self.file_list[i]['time_h'] >= TIME_STEP_HOURS: return i
            return 0

    def next_frame(self, event):
        next_idx = self._get_next_index_by_mode(self.current_idx)
        if next_idx != self.current_idx: self.slider.set_val(next_idx)

    def prev_frame(self, event):
        prev_idx = self._get_prev_index_by_mode(self.current_idx)
        if prev_idx != self.current_idx: self.slider.set_val(prev_idx)

    def on_slider_change(self, val):
        idx = int(val)
        if idx != self.current_idx: self.current_idx = idx; self.update_plot()

    def on_gif_btn_clicked(self, event): self.generate_gif()

    def generate_gif(self):
        gif_name = f"{RUN_NAME}_{self.current_year_key.replace(' ', '_')}_{self.step_mode}Mode.gif"
        frames_to_record = [0]
        idx = 0
        while idx < len(self.file_list) - 1:
            next_idx = self._get_next_index_by_mode(idx)
            if next_idx <= idx: break
            idx = next_idx; frames_to_record.append(idx)
        self.is_saving_gif = True
        for ax_ui in self.ui_axes: ax_ui.set_visible(False)
        self.info_text.set_text('')
        try:
            anim = animation.FuncAnimation(self.fig, lambda f: self.slider.set_val(f), frames=frames_to_record, blit=False, repeat=False)
            anim.save(gif_name, writer=animation.PillowWriter(fps=GIF_FPS))
        except Exception as e: print(f"GIF Error: {e}")
        finally:
            self.is_saving_gif = False
            for ax_ui in self.ui_axes: ax_ui.set_visible(True)
            self.slider.set_val(self.current_idx)

    def generate_plate_view(self, event):
        target_taas = [0, 60, 180, 240, 300, 359]
        fig_plate, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
        axes = axes.flatten()
        lon_edges = np.linspace(-180, 180, self.n_lon + 1)
        lat_edges = np.linspace(-90, 90, self.n_lat + 1)
        im = None
        for i, target_taa in enumerate(target_taas):
            ax = axes[i]; closest_file = min(self.file_list, key=lambda x: abs(x['taa'] - target_taa))
            data, problems = self._safe_load(closest_file['path'], tag='Plate')
            if data is None:
                ax.text(0.5, 0.5, "LOAD FAILED", ha='center', va='center', transform=ax.transAxes, color='red')
                continue
            if problems:
                ax.set_title("[!] " + problems[0][:40], fontsize=7, color='red')
            try:
                if data.ndim == 3: data = np.sum(data, axis=2)
                if self.align_sun:
                    subsolar_lon_deg = get_subsolar_longitude_linear(closest_file['time_h'], self.t_start, self.orbit_data)
                    shift = (self.n_lon // 2) - (int(np.round(((subsolar_lon_deg + 180.0) % 360.0) / (360.0 / self.n_lon))) % self.n_lon)
                    data = np.roll(data, shift=shift, axis=0)
                data_T = np.nan_to_num(data.T, nan=0.0)
                if USE_PAPER_SCALE: data_T = data_T / 10000.0
                im = ax.pcolormesh(lon_edges, lat_edges, data_T, cmap=self.cmap, norm=self.norm, shading='flat')
                ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
                ax.set_aspect('equal')
            except Exception as e:
                print(f"[Plate] {os.path.basename(closest_file['path'])}: {type(e).__name__}: {e}")
                ax.text(0.5, 0.5, f"Data Error\n{type(e).__name__}", ha='center', va='center',
                        transform=ax.transAxes, color='red')
        if im: fig_plate.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05, shrink=0.8)
        plt.show()

    def plot_total_atoms(self, event):
        fig_total, ax_total = plt.subplots(figsize=(8, 6))
        for year_key, file_list in self.grouped_files.items():
            taas, totals = [], []
            n_skip = 0
            for f in file_list:
                data, problems = self._safe_load(f['path'], tag=year_key)
                if data is None:
                    n_skip += 1
                    continue
                if problems:
                    n_skip += 1   # 壊れていても描くが、件数は数えて報告する
                if data.ndim == 3: data = np.sum(data, axis=2)
                data_T = np.nan_to_num(data.T, nan=0.0)
                if USE_PAPER_SCALE: data_T = data_T / 10000.0
                taas.append(f['taa']); totals.append(np.sum(data_T * self.cell_areas))
            if n_skip:
                print(f"[Plot Total] {year_key}: 異常/読込失敗 {n_skip} 件")
            if taas and totals:
                ax_total.plot(np.array(taas)[np.argsort(taas)], np.array(totals)[np.argsort(taas)],
                              marker='o', linestyle='-', markersize=4, label=year_key)
        ax_total.set_xlabel('True Anomaly Angle [deg]', fontsize=11, fontweight='bold')
        ax_total.set_ylabel('Total surface Na [atoms]', fontsize=11, fontweight='bold')
        ax_total.set_xlim(0, 360)
        ax_total.grid(True, linestyle='--', alpha=0.7); ax_total.legend(fontsize=8); plt.show()

    # ------------------------------------------------------------------
    # ★ 電離寿命を考慮した大気量 (TAA依存) のプロット
    #    dN/dt = P(t) - N / tau_ion,   tau_ion = T1AU * AU^2
    # ------------------------------------------------------------------
    def _atmos_series_for_year(self, file_list):
        times, taas, P_list, tau_list = [], [], [], []
        Pd_list, Pk_list = [], []   # dawn, dusk
        n_skip = 0
        for f in file_list:
            surf, problems = self._safe_load(f['path'], tag='Atmos')
            if surf is None or surf.ndim != 3:
                n_skip += 1
                continue
            if problems:
                n_skip += 1
            sub_lon, AU = self._orbit_at(f['time_h'])
            P, P_dawn, P_dusk = self._release_total_from_surfdata(surf, sub_lon, AU)
            times.append(f['time_h'] * 3600.0)
            taas.append(f['taa'])
            P_list.append(P)
            Pd_list.append(P_dawn)
            Pk_list.append(P_dusk)
            tau_list.append(self.t1au * AU ** 2)

        if n_skip:
            print(f"[Plot Atmos] 異常/スキップ {n_skip} 件")
        if not times:
            return None

        times = np.array(times); taas = np.array(taas)
        P_arr = np.array(P_list); tau_arr = np.array(tau_list)
        Pd_arr = np.array(Pd_list); Pk_arr = np.array(Pk_list)

        # ★時間順に並べ直す (累計積分は必ず時系列順で行う)
        tsort = np.argsort(times)
        times, taas = times[tsort], taas[tsort]
        P_arr, tau_arr = P_arr[tsort], tau_arr[tsort]
        Pd_arr, Pk_arr = Pd_arr[tsort], Pk_arr[tsort]

        # --- 累計放出量 C(t) = ∫ P dt  [atoms] (台形則、年の先頭を 0 とする) ---
        def _cum(arr):
            out = np.zeros_like(arr)
            if len(times) > 1:
                out[1:] = np.cumsum(0.5 * (arr[:-1] + arr[1:]) * np.diff(times))
            return out
        C = _cum(P_arr)
        Cd = _cum(Pd_arr)
        Ck = _cum(Pk_arr)

        # --- 参考: 電離寿命との釣り合い N(t) ---
        N = np.zeros_like(P_arr)
        N[0] = P_arr[0] * tau_arr[0]
        for k in range(len(times) - 1):
            dt = times[k + 1] - times[k]
            tau = 0.5 * (tau_arr[k] + tau_arr[k + 1])
            Pm = 0.5 * (P_arr[k] + P_arr[k + 1])
            e = np.exp(-dt / tau) if dt / tau < 700 else 0.0
            N[k + 1] = N[k] * e + Pm * tau * (1.0 - e)
        return dict(taa=taas, P=P_arr, tau=tau_arr, N=N,
                    C=C, P_dawn=Pd_arr, P_dusk=Pk_arr, C_dawn=Cd, C_dusk=Ck)

    def plot_atmosphere_vs_taa(self, event):
        fig_a, (ax_n, ax_p) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                           constrained_layout=True)
        year_keys = (list(self.grouped_files.keys())
                     if ATMOS_PLOT_ALL_YEARS else [self.current_year_key])

        cumulative = (TAA_PLOT_MODE == 'cumulative')

        prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0'])

        for iy, year_key in enumerate(year_keys):
            res = self._atmos_series_for_year(self.grouped_files[year_key])
            if res is None:
                continue
            col = prop_cycle[iy % len(prop_cycle)]
            t_s = res['taa']; P_s = res['P']; tau_s = res['tau']
            N_s = res['N']; C_s = res['C']

            if cumulative:
                ax_n.plot(t_s, C_s, color=col, marker='o', markersize=3, linewidth=1.8,
                          label=f"{year_key} total ({C_s[-1]:.3e})")
                ax_p.plot(t_s, P_s, color=col, marker='.', markersize=3, linewidth=1.5,
                          label=f"{year_key} total")
                if TAA_SPLIT_DAWN_DUSK:
                    ax_n.plot(t_s, res['C_dawn'], color=col, linestyle='--', linewidth=1.4,
                              alpha=0.85, label=f"{year_key} dawn ({res['C_dawn'][-1]:.3e})")
                    ax_n.plot(t_s, res['C_dusk'], color=col, linestyle=':', linewidth=1.8,
                              alpha=0.85, label=f"{year_key} dusk ({res['C_dusk'][-1]:.3e})")
                    ax_p.plot(t_s, res['P_dawn'], color=col, linestyle='--', linewidth=1.1,
                              alpha=0.85, label=f"{year_key} dawn")
                    ax_p.plot(t_s, res['P_dusk'], color=col, linestyle=':', linewidth=1.4,
                              alpha=0.85, label=f"{year_key} dusk")
            else:
                ax_n.plot(t_s, N_s, color=col, marker='o', markersize=3, linewidth=1.8,
                          label=f"{year_key}  (max {np.max(N_s):.3e})")
                ax_n.plot(t_s, P_s * tau_s, color=col, linestyle='--', linewidth=1.0, alpha=0.6,
                          label=f"{year_key} steady (P·τ)")
                ax_p.plot(t_s, P_s, color=col, marker='.', markersize=3, linewidth=1.5,
                          label=year_key)

            span_h = (self.grouped_files[year_key][-1]['time_h'] -
                      self.grouped_files[year_key][0]['time_h'])
            tot = C_s[-1]
            d_end, k_end = res['C_dawn'][-1], res['C_dusk'][-1]
            ratio = (d_end / k_end) if k_end > 0 else float('nan')
            print(f"[{year_key}] 累計放出量 = {tot:.4e} atoms "
                  f"(TAA {t_s[0]:.0f}°→{t_s[-1]:.0f}°, {len(t_s)} 点, 約 {span_h} h)")
            print(f"          dawn = {d_end:.4e} ({100.0*d_end/tot if tot else 0:.1f}%),  "
                  f"dusk = {k_end:.4e} ({100.0*k_end/tot if tot else 0:.1f}%),  dawn/dusk = {ratio:.3f}")
            print(f"          max P = {np.max(P_s):.4e} atoms/s (TAA={t_s[np.argmax(P_s)]:.0f}°),  "
                  f"[参考] max N = {np.max(N_s):.4e} atoms")

        if cumulative:
            ax_n.set_ylabel('Cumulative release ∫P dt [atoms]', fontsize=11, fontweight='bold')
            _side = 'lon_sun<0' if self.dawn_sign > 0 else 'lon_sun>0'
            ax_n.set_title('Cumulative Na release over the orbit '
                           '(each year reset to 0 at its first snapshot)\n'
                           + (f'solid = total,  dashed = dawn ({_side}),  dotted = dusk'
                              if TAA_SPLIT_DAWN_DUSK else ''), fontsize=10)
        else:
            ax_n.set_ylabel('Exospheric Na content N [atoms]', fontsize=11, fontweight='bold')
            ax_n.set_title('Ionization-limited exospheric content  '
                           f'(T1AU = {self.t1au:.3e} s, τ = T1AU·AU²)', fontsize=11)
        ax_n.grid(True, linestyle='--', alpha=0.6)
        ax_n.legend(fontsize=8)

        ax_p.set_ylabel('Total release rate P [atoms/s]', fontsize=11, fontweight='bold')
        ax_p.set_xlabel('True Anomaly Angle [deg]', fontsize=11, fontweight='bold')
        ax_p.set_xlim(0, 360)
        ax_p.set_xticks(np.arange(0, 361, 30))
        ax_p.grid(True, linestyle='--', alpha=0.6)
        ax_p.legend(fontsize=8)
        plt.show()


if __name__ == "__main__":
    orbit_data, t_start = load_orbit_data(ORBIT_FILE_PATH)
    grouped_files = get_all_files_grouped_by_year(os.path.join(BASE_OUTPUT_DIRECTORY, RUN_NAME))
    viewer = SimulationViewer(grouped_files, orbit_data, t_start)
    plt.show()