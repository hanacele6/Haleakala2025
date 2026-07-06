# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Circle
import os
import glob
import re
import sys
import copy

# ==============================================================================
# ★★★ ユーザー設定 ★★★
# ==============================================================================

DIR_A = r"./SimulationResult_202606/ParabolicHop_72x36_NoEq_DT100_0617_Multi_BD0.5_UG_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)_1res"
NAME_A = "Resolution 101"

DIR_B = r"./SimulationResult_202606/ParabolicHop_72x36_NoEq_DT100_0616_Multi_BD0.5_UG_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)_5res"
NAME_B = "Resolution 501"

INITIAL_TARGET_TAA = 100

GRID_RESOLUTION_A = 101
GRID_RESOLUTION_B = 501
GRID_MAX_RM = 5.0
RM_METERS = 2.440e6

PLOT_UNIT = 'cm2'
VIEW_FROM = 'Z'

# ズーム設定
ZOOM_X = [0.8, 1.2]
ZOOM_Y = [-0.2, 0.2]

# カラーバーのレンジ (奥行き1グリッド分の柱密度に合わせて調整してください)
VMIN_MANUAL = 1e8
VMAX_MANUAL = 1e12

# ==============================================================================
# 関数群
# ==============================================================================

def get_all_grid_files_sorted(target_dir):
    search_path = os.path.join(target_dir, "density_grid_*.npy")
    files = glob.glob(search_path)
    if not files: return []
    file_list = []
    for f in files:
        match = re.search(r'_t(\d+)_taa(\d+)\.npy$', os.path.basename(f))
        if match:
            file_list.append({'taa': int(match.group(2)), 'time_h': int(match.group(1)), 'path': f})
    file_list.sort(key=lambda x: (x['taa'], x['time_h']))
    return file_list

# ==============================================================================
# ★比較ビューワークラス
# ==============================================================================

class CompareGrid3DViewer:
    def __init__(self, list_A, list_B, view_from='Z'):
        self.list_A = list_A
        self.list_B = list_B
        self.view_from = view_from
        self.show_overlay = True

        grid_min_m = -GRID_MAX_RM * RM_METERS
        grid_max_m =  GRID_MAX_RM * RM_METERS
        self.cell_size_A_m = (grid_max_m - grid_min_m) / GRID_RESOLUTION_A
        self.cell_size_B_m = (grid_max_m - grid_min_m) / GRID_RESOLUTION_B
        self.plot_extent = [-GRID_MAX_RM, GRID_MAX_RM, -GRID_MAX_RM, GRID_MAX_RM]

        self.current_idx = min(range(len(self.list_A)), key=lambda i: abs(self.list_A[i]['taa'] - INITIAL_TARGET_TAA))

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 7))
        plt.subplots_adjust(bottom=0.25, wspace=0.15)
        self.ax1.set_facecolor('black')
        self.ax2.set_facecolor('black')

        if self.view_from == 'Z':
            self.integration_axis = 2
            self.xlabel, self.ylabel = "X [$R_M$]", "Y [$R_M$]"
            self.view_name = "View: +Z (X-Y Plane)"
        elif self.view_from == 'Y':
            self.integration_axis = 1
            self.xlabel, self.ylabel = "X [$R_M$]", "Z [$R_M$]"
            self.view_name = "View: -Y (X-Z Plane)"

        self.unit_factor = 1e-4 if PLOT_UNIT == 'cm2' else 1.0
        self.cbar_label = f"Slab Column Density [atoms/{PLOT_UNIT}]"
        self.view_name += " (Slab: Depth of 1 Coarse Cell)"

        self.cmap = copy.copy(plt.get_cmap('inferno'))
        self.cmap.set_bad('black')
        self.norm = mcolors.LogNorm(vmin=VMIN_MANUAL, vmax=VMAX_MANUAL)

        data_A, data_B, taa_A, taa_B = self.get_sync_data(self.current_idx)
        self.im1 = self.ax1.imshow(data_A.T, origin='lower', extent=self.plot_extent, cmap=self.cmap, norm=self.norm)
        self.im2 = self.ax2.imshow(data_B.T, origin='lower', extent=self.plot_extent, cmap=self.cmap, norm=self.norm)

        self.cbar = self.fig.colorbar(self.im2, ax=[self.ax1, self.ax2], pad=0.02, shrink=0.8)
        self.cbar.set_label(self.cbar_label, fontsize=12)

        self.add_mercury_circle(self.ax1)
        self.add_mercury_circle(self.ax2)
        
        self.overlay_lines = []
        self.apply_grid_and_zoom(self.ax1, GRID_RESOLUTION_A, GRID_RESOLUTION_B)
        self.apply_grid_and_zoom(self.ax2, GRID_RESOLUTION_B, GRID_RESOLUTION_A)
        
        self.update_titles(taa_A, taa_B)

        # UI Widgets
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='lightgray')
        self.slider = Slider(ax_slider, 'Time/TAA', 0, len(self.list_A) - 1, valinit=self.current_idx, valfmt='%d')
        self.slider.on_changed(self.on_slider_change)

        self.btn_prev = Button(plt.axes([0.7, 0.025, 0.1, 0.04]), 'Previous')
        self.btn_next = Button(plt.axes([0.81, 0.025, 0.1, 0.04]), 'Next')
        self.btn_prev.on_clicked(self.prev_frame)
        self.btn_next.on_clicked(self.next_frame)

        self.btn_plate = Button(plt.axes([0.45, 0.025, 0.15, 0.04]), 'Show Compare Plate')
        self.btn_plate.on_clicked(self.generate_plate_view)

        self.btn_toggle = Button(plt.axes([0.35, 0.025, 0.08, 0.04]), 'Grid ON/OFF')
        self.btn_toggle.on_clicked(self.toggle_grid)

    def add_mercury_circle(self, ax):
        ax.add_patch(Circle((0, 0), 1.0, color='white', fill=False, linestyle='--', linewidth=1.5, zorder=10))

    def apply_grid_and_zoom(self, ax, main_res, overlay_res=None):
        ax.set_xlim(ZOOM_X)
        ax.set_ylim(ZOOM_Y)
        ax.set_aspect('equal')

        ticks_main = np.linspace(-GRID_MAX_RM, GRID_MAX_RM, main_res + 1)
        ax.vlines(ticks_main, ymin=-GRID_MAX_RM, ymax=GRID_MAX_RM, color='cyan', linestyle='-', linewidth=0.5, alpha=0.6)
        ax.hlines(ticks_main, xmin=-GRID_MAX_RM, xmax=GRID_MAX_RM, color='cyan', linestyle='-', linewidth=0.5, alpha=0.6)

        if overlay_res:
            ticks_over = np.linspace(-GRID_MAX_RM, GRID_MAX_RM, overlay_res + 1)
            vl = ax.vlines(ticks_over, ymin=-GRID_MAX_RM, ymax=GRID_MAX_RM, color='red', linestyle='-', linewidth=1.5, alpha=0.8)
            hl = ax.hlines(ticks_over, xmin=-GRID_MAX_RM, xmax=GRID_MAX_RM, color='red', linestyle='-', linewidth=1.5, alpha=0.8)
            self.overlay_lines.extend([vl, hl])

    def toggle_grid(self, event):
        self.show_overlay = not self.show_overlay
        for line in self.overlay_lines:
            line.set_visible(self.show_overlay)
        self.fig.canvas.draw_idle()

    def load_process_file(self, filepath, resolution, cell_size_m):
        """
        指定されたファイルからデータを読み込み、「スラブ柱密度（特定の厚さでの積分）」を計算する
        """
        try:
            data_3d = np.load(filepath)
            
            # ★ 101分割の1セル分の厚さ(基準)を計算し、それに対応する積分セル数を割り出す
            base_resolution = GRID_RESOLUTION_A
            num_cells_to_integrate = max(1, int(resolution / base_resolution))
            
            # 中央から必要なセル数分だけスライスして取り出す
            mid = resolution // 2
            start_idx = mid - (num_cells_to_integrate // 2)
            end_idx = start_idx + num_cells_to_integrate
            
            if self.view_from == 'Z': slab_data = data_3d[:, :, start_idx:end_idx]
            elif self.view_from == 'Y': slab_data = data_3d[:, start_idx:end_idx, :]
            else: slab_data = data_3d[start_idx:end_idx, :, :]
            
            # スラブの奥行き方向で積分し、物理サイズを掛けて柱密度化
            column_density_slab = np.sum(slab_data, axis=self.integration_axis) * cell_size_m
            
            return column_density_slab * self.unit_factor
            
        except Exception as e:
            print(f"Read Error [{filepath}]: {e}")
            return np.zeros((resolution, resolution))

    def get_sync_data(self, idx_A):
        info_A = self.list_A[idx_A]
        _, info_B = min(enumerate(self.list_B), key=lambda x: abs(x[1]['taa'] - info_A['taa']))
        data_A = self.load_process_file(info_A['path'], GRID_RESOLUTION_A, self.cell_size_A_m)
        data_B = self.load_process_file(info_B['path'], GRID_RESOLUTION_B, self.cell_size_B_m)
        return data_A, data_B, info_A['taa'], info_B['taa']

    def update_titles(self, taa_A, taa_B):
        self.ax1.set_title(f"[{NAME_A}]\nTAA: {taa_A:03d}°", fontsize=12)
        self.ax2.set_title(f"[{NAME_B}]\nTAA: {taa_B:03d}°", fontsize=12)
        self.ax1.set_xlabel(self.xlabel); self.ax1.set_ylabel(self.ylabel)
        self.ax2.set_xlabel(self.xlabel)
        self.fig.suptitle(f"Mercury Na Exosphere Grid Artifact Comparison\n{self.view_name}", fontsize=14)

    def update_plot(self):
        data_A, data_B, taa_A, taa_B = self.get_sync_data(self.current_idx)
        self.im1.set_data(data_A.T)
        self.im2.set_data(data_B.T)
        self.update_titles(taa_A, taa_B)
        self.fig.canvas.draw_idle()

    def next_frame(self, event):
        if self.current_idx < len(self.list_A) - 1: self.slider.set_val(self.current_idx + 1)
    def prev_frame(self, event):
        if self.current_idx > 0: self.slider.set_val(self.current_idx - 1)
    def on_slider_change(self, val):
        if int(val) != self.current_idx:
            self.current_idx = int(val)
            self.update_plot()

    def generate_plate_view(self, event):
        target_taas = [0, 60, 180, 300]
        fig_plate, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
        im_object = None

        for col, target_taa in enumerate(target_taas):
            _, file_A = min(enumerate(self.list_A), key=lambda x: abs(x[1]['taa'] - target_taa))
            data_A = self.load_process_file(file_A['path'], GRID_RESOLUTION_A, self.cell_size_A_m)
            ax_A = axes[0, col]
            ax_A.set_facecolor('black')
            ax_A.imshow(data_A.T, origin='lower', extent=self.plot_extent, cmap=self.cmap, norm=self.norm)
            self.add_mercury_circle(ax_A)
            self.apply_grid_and_zoom(ax_A, GRID_RESOLUTION_A)
            ax_A.set_title(f"TAA = {file_A['taa']}$^\circ$", fontsize=12)

            _, file_B = min(enumerate(self.list_B), key=lambda x: abs(x[1]['taa'] - target_taa))
            data_B = self.load_process_file(file_B['path'], GRID_RESOLUTION_B, self.cell_size_B_m)
            ax_B = axes[1, col]
            ax_B.set_facecolor('black')
            im_object = ax_B.imshow(data_B.T, origin='lower', extent=self.plot_extent, cmap=self.cmap, norm=self.norm)
            self.add_mercury_circle(ax_B)
            self.apply_grid_and_zoom(ax_B, GRID_RESOLUTION_B)

            if col == 0:
                ax_A.set_ylabel(f"{NAME_A}\n\n{self.ylabel}", fontsize=12, fontweight='bold')
                ax_B.set_ylabel(f"{NAME_B}\n\n{self.ylabel}", fontsize=12, fontweight='bold')
            else:
                ax_A.set_yticklabels([]); ax_B.set_yticklabels([])
            ax_A.set_xticklabels([]); ax_B.set_xlabel(self.xlabel, fontsize=10)

        if im_object:
            cbar = fig_plate.colorbar(im_object, ax=axes, orientation='horizontal', fraction=0.04, pad=0.06, shrink=0.5)
            cbar.set_label(self.cbar_label, fontsize=12)
        fig_plate.suptitle(f"Grid Artifact Comparison Plate | {self.view_name}", fontsize=16)
        plt.show()

if __name__ == "__main__":
    list_A = get_all_grid_files_sorted(DIR_A)
    list_B = get_all_grid_files_sorted(DIR_B)
    if not list_A or not list_B: sys.exit(1)
    viewer = CompareGrid3DViewer(list_A, list_B, view_from=VIEW_FROM)
    plt.show()