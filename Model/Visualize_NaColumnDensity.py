# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

# 1. シミュレーション結果ディレクトリ
#SIMULATION_RUN_DIRECTORY = r"./SimulationResult_202512/ParabolicHop_72x36_EqMode_DT500_PLeblanc_DLeblanc"
#SIMULATION_RUN_DIRECTORY = r"./SimulationResult_202512/ParabolicHop_72x36_NoEq_DT100_0105_LmtDenabled_2.05"
#SIMULATION_RUN_DIRECTORY = r"./SimulationResult_202512/ParabolicHop_72x36_NoEq_DT100_0105_0.4Denabled_2.7"
SIMULATION_RUN_DIRECTORY = r"./SimulationResult_202512/ParabolicHop_72x36_NoEq_DT100_0102_Denabled"

# 2. 最初に表示したいTAA
INITIAL_TARGET_TAA = 100

# 3. グリッド設定
GRID_RESOLUTION = 101  # グリッド解像度
GRID_MAX_RM = 5.0  # グリッド範囲 [RM]

# 4. 物理定数
RM_METERS = 2.440e6  # 水星半径 [m]

# 5. プロット設定
PLOT_IN_CM2 = True  # True: atoms/cm^2, False: atoms/m^2
VIEW_FROM = 'Z'  # 'Z': X-Y平面 (Face-on), 'Y': X-Z平面 (Side-on)

# 6. カラーバーのレンジ (対数スケール)
# ※ PLOT_IN_CM2=True なら cm2 単位の値を入れてください
VMIN_MANUAL = 1e8
VMAX_MANUAL = 1e13


# ==============================================================================
# 関数群
# ==============================================================================

def get_all_grid_files_sorted(target_dir):
    """ディレクトリ内の density_grid_*.npy を全て探し、TAA順にソートして返す"""
    search_path = os.path.join(target_dir, "density_grid_*.npy")
    files = glob.glob(search_path)

    if not files:
        print(f"エラー: {target_dir} に density_grid ファイルがありません。")
        return []

    file_list = []
    for f in files:
        match = re.search(r'_t(\d+)_taa(\d+)\.npy$', os.path.basename(f))
        if match:
            time_id = int(match.group(1))
            taa = int(match.group(2))
            file_list.append({
                'taa': taa,
                'time_h': time_id,
                'path': f
            })

    file_list.sort(key=lambda x: (x['taa'], x['time_h']))
    return file_list


# ==============================================================================
# ★ビューワークラス
# ==============================================================================

class Grid3DViewer:
    def __init__(self, file_list, view_from='Z'):
        self.file_list = file_list
        self.view_from = view_from

        # 物理パラメータ計算
        grid_min_m = -GRID_MAX_RM * RM_METERS
        grid_max_m = GRID_MAX_RM * RM_METERS
        self.cell_size_m = (grid_max_m - grid_min_m) / GRID_RESOLUTION
        self.plot_extent = [-GRID_MAX_RM, GRID_MAX_RM, -GRID_MAX_RM, GRID_MAX_RM]

        # 初期インデックス
        self.current_idx = 0
        min_diff = float('inf')
        for i, f in enumerate(self.file_list):
            diff = abs(f['taa'] - INITIAL_TARGET_TAA)
            if diff < min_diff:
                min_diff = diff
                self.current_idx = i

        # --- 図の準備 ---
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.25)
        self.ax.set_facecolor('black')

        # 軸ラベルと積分軸の設定
        if self.view_from == 'Z':
            self.integration_axis = 2
            self.xlabel = "X [$R_M$]"
            self.ylabel = "Y [$R_M$]"
            self.view_name = "View: +Z (X-Y Plane)"
        elif self.view_from == 'Y':
            self.integration_axis = 1
            self.xlabel = "X [$R_M$]"
            self.ylabel = "Z [$R_M$]"
            self.view_name = "View: -Y (X-Z Plane)"
        else:
            raise ValueError("VIEW_FROM must be 'Z' or 'Y'")

        # 単位設定
        if PLOT_IN_CM2:
            self.unit_factor = 1e-4
            self.cbar_label = "Column Density [atoms/cm²]"
        else:
            self.unit_factor = 1.0
            self.cbar_label = "Column Density [atoms/m²]"

        # --- カラーマップ設定 ---
        self.cmap = copy.copy(plt.get_cmap('inferno'))
        self.cmap.set_bad('black')
        self.norm = mcolors.LogNorm(vmin=VMIN_MANUAL, vmax=VMAX_MANUAL)

        # 初回データロード
        initial_data = self.load_and_process(self.current_idx)

        # 画像描画
        self.im = self.ax.imshow(
            initial_data.T,
            origin='lower',
            extent=self.plot_extent,
            cmap=self.cmap,
            norm=self.norm
        )

        # カラーバー
        self.cbar = self.fig.colorbar(self.im, ax=self.ax, pad=0.02)
        self.cbar.set_label(self.cbar_label, fontsize=12)

        # 水星の円
        self.add_mercury_circle(self.ax)

        self.ax.legend(loc='upper right', facecolor='black', labelcolor='white')
        self.ax.set_aspect('equal')
        self.update_title()

        # === ウィジェット設定 ===
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='lightgray')
        self.slider = Slider(ax_slider, 'Time/TAA', 0, len(self.file_list) - 1,
                             valinit=self.current_idx, valfmt='%d')
        self.slider.on_changed(self.on_slider_change)

        ax_prev = plt.axes([0.7, 0.025, 0.1, 0.04])
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_prev.on_clicked(self.prev_frame)

        ax_next = plt.axes([0.81, 0.025, 0.1, 0.04])
        self.btn_next = Button(ax_next, 'Next')
        self.btn_next.on_clicked(self.next_frame)

        # ★ Plate Button
        ax_plate = plt.axes([0.45, 0.025, 0.15, 0.04])
        self.btn_plate = Button(ax_plate, 'Show Plate')
        self.btn_plate.on_clicked(self.generate_plate_view)

    def add_mercury_circle(self, ax):
        """指定されたaxに水星の円を描画する"""
        mercury_circle = Circle((0, 0), 1.0, color='white', fill=False,
                                linestyle='--', linewidth=1, label='Mercury')
        ax.add_patch(mercury_circle)

    def load_and_process(self, idx):
        """指定インデックスのファイルを読み込み、柱密度(2D)を計算して返す"""
        filepath = self.file_list[idx]['path']
        try:
            # 3Dグリッド [x, y, z]
            data_3d = np.load(filepath)
            # 積分
            column_density = np.sum(data_3d, axis=self.integration_axis) * self.cell_size_m
            # 単位変換
            data = column_density * self.unit_factor
            return data
        except Exception as e:
            print(f"Read Error: {e}")
            return np.zeros((GRID_RESOLUTION, GRID_RESOLUTION))

    def update_title(self):
        """タイトルと軸ラベルの更新"""
        info = self.file_list[self.current_idx]
        taa = info['taa']
        time_h = info['time_h']
        filename = os.path.basename(info['path'])

        title_str = (f"Mercury Na Exosphere Column Density\n"
                     f"{self.view_name} | TAA: {taa:03d}° (TimeID: {time_h})\n"
                     f"File: {filename}")

        self.ax.set_title(title_str, fontsize=12)
        self.ax.set_xlabel(self.xlabel, fontsize=12)
        self.ax.set_ylabel(self.ylabel, fontsize=12)

    def update_plot(self):
        """プロット内容の更新"""
        new_data = self.load_and_process(self.current_idx)
        self.im.set_data(new_data.T)
        self.update_title()
        self.fig.canvas.draw_idle()

    # --- イベントハンドラ ---
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

    # ==========================================================================
    # ★ 新規追加: Plate作成機能 (2x2)
    # ==========================================================================
    def generate_plate_view(self, event):
        """TAA 0, 60, 180, 300 の4枚を 2x2 Plate として表示する"""

        target_taas = [0, 60, 180, 300]

        # 新しいウィンドウを作成 (2行2列)
        fig_plate, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
        # 背景を少し暗く設定(任意)
        # fig_plate.patch.set_facecolor('#202020')
        axes = axes.flatten()

        print("\n--- Generating Plate View (2x2) ---")

        im_object = None  # カラーバー用に保持

        for i, target_taa in enumerate(target_taas):
            ax = axes[i]
            ax.set_facecolor('black')

            # 最も近いTAAを持つファイルのインデックスを探す
            # file_list は辞書のリストなので、enumerateでインデックスを取得
            closest_idx, closest_file = min(enumerate(self.file_list),
                                            key=lambda x: abs(x[1]['taa'] - target_taa))

            actual_taa = closest_file['taa']
            print(f"Target: {target_taa}° -> Actual: {actual_taa}° (File: {os.path.basename(closest_file['path'])})")

            # データの読み込み (既存メソッド再利用)
            data = self.load_and_process(closest_idx)

            # プロット
            im_object = ax.imshow(
                data.T,
                origin='lower',
                extent=self.plot_extent,
                cmap=self.cmap,
                norm=self.norm
            )

            # 水星の円を追加
            self.add_mercury_circle(ax)

            # タイトルとラベル
            ax.set_title(f"TAA = {actual_taa}$^\circ$", fontsize=14, fontweight='bold')  # color='white' if dark bg
            ax.set_aspect('equal')

            # 軸ラベル (外側だけ残す等の調整はお好みで。ここでは全てにつける)
            if i >= 2:  # 下段
                ax.set_xlabel(self.xlabel, fontsize=10)
            else:
                ax.set_xticklabels([])  # 上段はラベルなし

            if i % 2 == 0:  # 左列
                ax.set_ylabel(self.ylabel, fontsize=10)
            else:
                ax.set_yticklabels([])  # 右列はラベルなし

            # 目盛りの色調整 (必要なら)
            # ax.tick_params(colors='white')

        # 共通カラーバーを追加
        if im_object:
            cbar = fig_plate.colorbar(im_object, ax=axes, orientation='horizontal',
                                      fraction=0.05, pad=0.05, shrink=0.8)
            cbar.set_label(self.cbar_label, fontsize=12)
            # cbar.ax.xaxis.set_tick_params(color='white')
            # plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='white')

        fig_plate.suptitle(f"Mercury Exosphere: {self.view_name}", fontsize=16)
        plt.show()


# ==============================================================================
# メイン実行部
# ==============================================================================
if __name__ == "__main__":

    if not os.path.exists(SIMULATION_RUN_DIRECTORY):
        print(f"エラー: フォルダが見つかりません {SIMULATION_RUN_DIRECTORY}")
        sys.exit(1)

    print("ファイルリストを作成中...")
    file_list = get_all_grid_files_sorted(SIMULATION_RUN_DIRECTORY)
    print(f"合計 {len(file_list)} 個のグリッドファイルが見つかりました。")

    if not file_list:
        sys.exit(0)

    print("ビューワーを起動します...")
    viewer = Grid3DViewer(file_list, view_from=VIEW_FROM)

    print("表示中... ウィンドウを閉じると終了します。")
    plt.show()