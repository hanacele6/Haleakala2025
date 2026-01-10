# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.widgets import Button, Slider, CheckButtons
import glob
import os
import re
import sys

# ==============================================================================
# 設定
# ==============================================================================

# ★ 論文 (Leblanc & Johnson 2003, Plate 1) と同じ単位(Na/cm^2)とスケールを使用する
USE_PAPER_SCALE = True  # True: Na/cm^2 に換算して表示, False: atoms/m^2 (生データ)で表示

# デフォルト設定 (USE_PAPER_SCALE = False の場合に使用)
COLOR_VMIN = 1.0e10
COLOR_VMAX = 1.0e18

# その他設定
N_LON, N_LAT = 72, 36
BASE_OUTPUT_DIRECTORY = r"./SimulationResult_202512"
# RUN_NAME = "ParabolicHop_72x36_EqMode_DT500_PLeblanc_DSuzuki"
# RUN_NAME = "Parabolichop_72x36_NoEq_DT100_1224_s1"
RUN_NAME = "ParabolicHop_72x36_NoEq_DT100_0109_0.4Denabled_2.7_HalfQ"
# RUN_NAME = "ParabolicHop_72x36_NoEq_DT100_0102_Denabled"
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
    t_start_run = t_file_start + SPIN_UP_YEARS * MERCURY_YEAR_SEC
    current_t_sec = t_start_run + (float(time_h) * 3600.0)
    t_lookup = np.clip(current_t_sec, time_col_original[0], time_col_original[-1])
    sub_lon_deg = np.interp(t_lookup, time_col_original, orbit_data[:, 5])
    return sub_lon_deg


def get_all_files_sorted(target_dir):
    """ディレクトリ内の全ファイルをTAA順にソートしてリスト化する"""
    search_path_grid = os.path.join(target_dir, "density_grid_*.npy")
    grid_files = glob.glob(search_path_grid)

    if not grid_files:
        print(f"エラー: {target_dir} に density_grid ファイルがありません。")
        return []

    file_list = []
    for f in grid_files:
        match = re.search(r'_t(\d+)_taa(\d+)\.npy$', os.path.basename(f))
        if match:
            time_id = int(match.group(1))
            taa = int(match.group(2))
            surf_filename = f"surface_density_t{time_id:05d}.npy"
            surf_filepath = os.path.join(target_dir, surf_filename)
            if os.path.exists(surf_filepath):
                file_list.append({
                    'taa': taa,
                    'time_h': time_id,
                    'path': surf_filepath
                })

    file_list.sort(key=lambda x: x['taa'])
    return file_list


# ==============================================================================
# ビューワークラス
# ==============================================================================

class SimulationViewer:
    def __init__(self, file_list, orbit_data, t_start):
        self.file_list = file_list
        self.orbit_data = orbit_data
        self.t_start = t_start

        # 表示設定
        self.n_lon = N_LON
        self.n_lat = N_LAT
        self.use_log = USE_LOG_SCALE
        self.align_sun = ALIGN_SUN_TO_CENTER
        self.current_display_data = None  # ★ 現在表示中のデータを保持する変数

        # ★ 等高線表示フラグ (初期値: False)
        self.show_contours = False
        self.contour_set = None

        # ★カラースケールの設定ロジック (Na/cm^2 対応)
        if USE_PAPER_SCALE:
            # 論文 Plate 1 のスケール (Na/cm^2)
            self.vmin = 10 ** 9.5  # 約 3.16e9
            # self.vmax = 10 ** 13.5  # 約 3.16e13
            self.vmax = 10 ** 14.5  # 約 3.16e13
            self.unit_label = '[atoms/cm^2]'
            print(f"Paper Scale ON: Vmin={self.vmin:.2e}, Vmax={self.vmax:.2e} {self.unit_label}")
        else:
            # デフォルト設定 (atoms/m^2)
            self.vmin = COLOR_VMIN
            self.vmax = COLOR_VMAX
            self.unit_label = '[atoms/m^2]'
            print(f"Paper Scale OFF: Vmin={self.vmin:.2e}, Vmax={self.vmax:.2e} {self.unit_label}")

        # 初期インデックス
        self.current_idx = 0
        min_diff = float('inf')
        for i, f in enumerate(self.file_list):
            diff = abs(f['taa'] - INITIAL_TARGET_TAA)
            if diff < min_diff:
                min_diff = diff
                self.current_idx = i

        # 図の準備
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        plt.subplots_adjust(bottom=0.25)

        # カラーマップ設定
        import copy
        self.cmap = copy.copy(plt.get_cmap('inferno'))
        self.cmap.set_bad('black')

        if self.use_log:
            self.norm = LogNorm(vmin=self.vmin, vmax=self.vmax)
        else:
            self.norm = Normalize(vmin=self.vmin, vmax=self.vmax)

        # カラーバー作成
        dummy_data = np.zeros((self.n_lat, self.n_lon))
        self.mesh = self.ax.pcolormesh(dummy_data, cmap=self.cmap, norm=self.norm)

        cbar_title = f'Surface Density {self.unit_label}'
        if USE_PAPER_SCALE:
            #cbar_title += ' (Plate 1 Scale)'
            cbar_title += ''
        self.cbar = plt.colorbar(self.mesh, ax=self.ax, label=cbar_title)

        # ★ カーソル情報のテキスト表示用 (右上に配置)
        # transform=self.ax.transAxes を使うことで、軸に対する相対座標(0.0~1.0)で指定
        self.info_text = self.ax.text(
            0.98, 0.95, '',
            transform=self.ax.transAxes,
            ha='right', va='top',
            fontsize=10, color='white', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="white", alpha=0.7)
        )

        # ★ マウスイベントの接続
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        # 最初の描画
        self.update_plot()

        # === ウィジェット ===
        # Slider
        ax_slider = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.slider = Slider(ax_slider, 'TAA', 0, len(self.file_list) - 1, valinit=self.current_idx, valfmt='%d')
        self.slider.on_changed(self.on_slider_change)

        # Prev Button
        ax_prev = plt.axes([0.15, 0.05, 0.1, 0.04])
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_prev.on_clicked(self.prev_frame)

        # Next Button
        ax_next = plt.axes([0.26, 0.05, 0.1, 0.04])
        self.btn_next = Button(ax_next, 'Next')
        self.btn_next.on_clicked(self.next_frame)

        # ★ Plate Button (New!)
        ax_plate = plt.axes([0.45, 0.05, 0.15, 0.04])
        self.btn_plate = Button(ax_plate, 'Show Plate')
        self.btn_plate.on_clicked(self.generate_plate_view)

        # Checkbox (Contours)
        ax_check = plt.axes([0.82, 0.05, 0.12, 0.1])
        self.check = CheckButtons(ax_check, ['Contours'], [self.show_contours])
        self.check.on_clicked(self.toggle_contours)

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

        # ★ 表示中のデータを保存 (マウスホバーイベント用)
        self.current_display_data = data_T

        # === 既存の等高線を消去 ===
        if self.contour_set is not None:
            try:
                for coll in self.contour_set.collections:
                    coll.remove()
            except Exception:
                pass
            self.contour_set = None
            # テキスト消去時にinfo_textまで消さないように注意 (ax.textsをフィルタリングするか再描画)
            # ここではシンプルに contoursのラベルだけ消すのは難しいので、info_textは__init__で保持し再利用する

        # pcolormeshの更新 (removeして再作成が最も確実)
        if self.mesh:
            self.mesh.remove()

        lon_edges = np.linspace(-180, 180, self.n_lon + 1)
        lat_edges = np.linspace(-90, 90, self.n_lat + 1)

        self.mesh = self.ax.pcolormesh(lon_edges, lat_edges, data_T, cmap=self.cmap, norm=self.norm, shading='flat')

        # === 等高線 ===
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

        self.ax.set_title(
            f"Surface Density {title_mode}\nTAA: {taa} deg (Time: {time_h}, SunLon: {subsolar_lon_deg:.1f})")
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel("Latitude")

        # グリッド線の再描画（mesh.remove()で消える可能性があるため、毎回ではなく必要な場合に追加）
        # ここでは描画順序を保つためlinesをクリアしてから書き直し
        [line.remove() for line in self.ax.lines]
        if self.align_sun:
            self.ax.axvline(0, color='white', linestyle='--', alpha=0.5)
            self.ax.axvline(-90, color='white', linestyle=':', alpha=0.3)
            self.ax.axvline(90, color='white', linestyle=':', alpha=0.3)
        else:
            self.ax.axvline(subsolar_lon_deg, color='white', linestyle='--')

        self.ax.axhline(0, color='white', linestyle=':', alpha=0.3)

        # テキストボックスを最前面へ
        self.info_text.set_zorder(100)

        self.fig.canvas.draw_idle()

    # ==========================================================================
    # ★ 新規追加: マウス移動時のハンドラ
    # ==========================================================================
    def on_mouse_move(self, event):
        """マウスカーソル下の値を表示する"""
        if event.inaxes != self.ax:
            # プロットエリア外ならテキストを隠す
            self.info_text.set_text('')
            self.fig.canvas.draw_idle()
            return

        if self.current_display_data is None:
            return

        # マウス座標 (経度, 緯度)
        x, y = event.xdata, event.ydata

        # グリッドインデックスの計算
        # 経度: -180 ~ 180 -> 0 ~ N_LON
        # 緯度: -90 ~ 90 -> 0 ~ N_LAT
        lon_idx = int((x + 180) / 360 * self.n_lon)
        lat_idx = int((y + 90) / 180 * self.n_lat)

        # インデックス範囲チェック
        lon_idx = np.clip(lon_idx, 0, self.n_lon - 1)
        lat_idx = np.clip(lat_idx, 0, self.n_lat - 1)

        # 値の取得 (current_display_dataは [Lat, Lon] の形状になっている)
        val = self.current_display_data[lat_idx, lon_idx]

        # 表示テキスト更新
        self.info_text.set_text(f"Lon: {x:.1f}\nLat: {y:.1f}\nVal: {val:.2e}")
        self.fig.canvas.draw_idle()

    # ==========================================================================
    # ★ 新規追加: Plate作成機能
    # ==========================================================================
    def generate_plate_view(self, event):
        """TAA 0, 60, 180, 240, 300, 360 の6枚をPlateとして表示する"""

        target_taas = [0, 60, 180, 240, 300, 360]

        # 新しいウィンドウを作成 (3行2列)
        fig_plate, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
        axes = axes.flatten()

        print("\n--- Generating Plate View ---")

        # 共通の座標軸定義
        lon_edges = np.linspace(-180, 180, self.n_lon + 1)
        lat_edges = np.linspace(-90, 90, self.n_lat + 1)

        # 描画用のメッシュオブジェクト（カラーバー用）
        im = None

        for i, target_taa in enumerate(target_taas):
            ax = axes[i]

            # 最も近いTAAを持つファイルを探す
            closest_file = min(self.file_list, key=lambda x: abs(x['taa'] - target_taa))
            actual_taa = closest_file['taa']

            print(f"Target: {target_taa} -> Actual: {actual_taa} (File: {os.path.basename(closest_file['path'])})")

            # データ読み込みと処理（update_plotと同じロジック）
            try:
                data = np.load(closest_file['path'])
                time_h = closest_file['time_h']

                # 太陽方向へのアライメント
                subsolar_lon_deg = get_subsolar_longitude_linear(time_h, self.t_start, self.orbit_data)
                if self.align_sun:
                    dlon = 360.0 / self.n_lon
                    sun_pos_norm = (subsolar_lon_deg + 180.0) % 360.0
                    sun_index = int(np.round(sun_pos_norm / dlon)) % self.n_lon
                    shift = (self.n_lon // 2) - sun_index
                    data = np.roll(data, shift=shift, axis=0)

                data_T = data.T
                data_T = np.nan_to_num(data_T, nan=0.0)

                # 単位変換
                if USE_PAPER_SCALE:
                    data_T = data_T / 10000.0

                # プロット
                im = ax.pcolormesh(lon_edges, lat_edges, data_T, cmap=self.cmap, norm=self.norm, shading='flat')

                # 装飾
                ax.set_title(f"TAA = {actual_taa}$^\circ$", fontsize=12, fontweight='bold')

                # 軸ラベルは外側のみにするなどの調整
                if i in [4, 5]:  # 下段のみXラベル
                    ax.set_xlabel("Longitude")
                else:
                    ax.set_xticklabels([])

                if i in [0, 2, 4]:  # 左側のみYラベル
                    ax.set_ylabel("Latitude")
                else:
                    ax.set_yticklabels([])

                # グリッド線（Plate 1に似せる）
                ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)  # Subsolar
                ax.axvline(-90, color='black', linestyle='--', linewidth=0.5, alpha=0.5)  # Dawn/Dusk roughly
                ax.axvline(90, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
                ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)  # Equator

                # アスペクト比を固定
                ax.set_aspect('equal')

                # 範囲設定
                ax.set_xlim(-180, 180)
                ax.set_ylim(-90, 90)

            except Exception as e:
                print(f"Error plotting TAA {target_taa}: {e}")
                ax.text(0, 0, "Data Error", ha='center')

        # 共通カラーバーを追加
        cbar_label = f"Surface Na Density {self.unit_label}"
        if USE_PAPER_SCALE:
            cbar_label += " (Log Scale)"

        # Figure全体に対してカラーバーを配置
        if im:
            cbar = fig_plate.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05, shrink=0.8)
            cbar.set_label(cbar_label, fontsize=12)

        fig_plate.suptitle(f"Mercury Surface Sodium Density (Plate View)\nRun: {RUN_NAME}", fontsize=14)
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
    file_list = get_all_files_sorted(full_dir)
    print(f"合計 {len(file_list)} 個のタイムステップが見つかりました。")

    if not file_list:
        sys.exit(0)

    viewer = SimulationViewer(file_list, orbit_data, t_start)
    print("表示中... ウィンドウを閉じると終了します。")
    plt.show()