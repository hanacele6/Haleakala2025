# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button, Slider
import os
import glob
import re
import sys

# ==============================================================================
# ★★★ ユーザー設定 ★★★
# ==============================================================================

# 1. シミュレーション結果ディレクトリ
SIMULATION_RUN_DIRECTORY = r"./SimulationResult_202605/ParabolicHop_72x36_NoEq_DT100_0518_Multi_BD0.9_UG_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)"

# 2. 最初に表示したいTAA
INITIAL_TARGET_TAA = 100

# 3. グリッド設定
GRID_RESOLUTION = 101  # グリッド解像度
GRID_MAX_RM = 5.0  # グリッド範囲 [RM]

# 4. 表示する密度のしきい値
VMIN_MANUAL = 1e6
VMAX_MANUAL = 1e13

# 5. 描画パフォーマンス設定 (最大プロット点数)
MAX_PLOT_POINTS = 10000

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

    # TAA順、同TAAなら時間順でソート
    file_list.sort(key=lambda x: (x['taa'], x['time_h']))
    return file_list

# ==============================================================================
# ★ 改良版インタラクティブ3Dビューワークラス
# ==============================================================================

class Grid3DVolumeViewer:
    def __init__(self, file_list):
        self.file_list = file_list
        
        # 3D座標グリッドの事前計算 (1回だけ計算)
        x = np.linspace(-GRID_MAX_RM, GRID_MAX_RM, GRID_RESOLUTION)
        y = np.linspace(-GRID_MAX_RM, GRID_MAX_RM, GRID_RESOLUTION)
        z = np.linspace(-GRID_MAX_RM, GRID_MAX_RM, GRID_RESOLUTION)
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing='ij')

        # 水星本体のポリゴン (1回だけ計算)
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 60)
        self.merc_x = 1.0 * np.outer(np.cos(u), np.sin(v))
        self.merc_y = 1.0 * np.outer(np.sin(u), np.sin(v))
        self.merc_z = 1.0 * np.outer(np.ones(np.size(u)), np.cos(v))

        # 初期インデックスの検索
        self.current_idx = 0
        min_diff = float('inf')
        for i, f in enumerate(self.file_list):
            diff = abs(f['taa'] - INITIAL_TARGET_TAA)
            if diff < min_diff:
                min_diff = diff
                self.current_idx = i

        # カラーマップの準備
        self.cmap = plt.get_cmap('viridis')

        # --- 図の準備 ---
        self.fig = plt.figure(figsize=(10, 9))
        
        self.fig.patch.set_facecolor('black') 
        plt.subplots_adjust(bottom=0.25)
        
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 初期カメラアングル
        self.ax.view_init(elev=10, azim=180)

        # 初回プロット
        self.update_plot(initial=True)

        # === ウィジェット設定 ===
        ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor='#333333')
        self.slider = Slider(ax_slider, 'Time/TAA', 0, len(self.file_list) - 1,
                             valinit=self.current_idx, valfmt='%d')
        self.slider.label.set_color('white')
        self.slider.valtext.set_color('white')
        self.slider.on_changed(self.on_slider_change)

        ax_prev = plt.axes([0.3, 0.025, 0.1, 0.05])
        self.btn_prev = Button(ax_prev, 'Prev')
        self.btn_prev.on_clicked(self.prev_frame)

        ax_next = plt.axes([0.45, 0.025, 0.1, 0.05])
        self.btn_next = Button(ax_next, 'Next')
        self.btn_next.on_clicked(self.next_frame)

        ax_save = plt.axes([0.65, 0.025, 0.15, 0.05])
        self.btn_save = Button(ax_save, 'Save PNG')
        self.btn_save.on_clicked(self.save_transparent_png)

    def load_data(self, idx):
        """指定インデックスのファイルを読み込む"""
        filepath = self.file_list[idx]['path']
        try:
            return np.load(filepath)
        except Exception as e:
            print(f"Read Error: {e}")
            return np.zeros((GRID_RESOLUTION, GRID_RESOLUTION, GRID_RESOLUTION))

    def update_plot(self, initial=False):
        """3Dプロットの更新"""
        info = self.file_list[self.current_idx]
        data = self.load_data(self.current_idx)

        # カメラアングルを保持
        elev, azim = self.ax.elev, self.ax.azim
        
        self.ax.clear()

        # 背景を透過に設定
        self.ax.set_facecolor('none')

        # --- 黒背景用のグリッド線・目盛り・ラベルのカスタマイズ ---
        # 3Dの壁面（ペイン）を透明にする
        self.ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        self.ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        self.ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        
        # グリッド線をグレーにする
        self.ax.xaxis._axinfo["grid"].update({"color": (0.5, 0.5, 0.5, 0.5)})
        self.ax.yaxis._axinfo["grid"].update({"color": (0.5, 0.5, 0.5, 0.5)})
        self.ax.zaxis._axinfo["grid"].update({"color": (0.5, 0.5, 0.5, 0.5)})

        # 目盛りラベルと軸ラベルを白にする
        self.ax.tick_params(axis='x', colors='white', labelcolor='white')
        self.ax.tick_params(axis='y', colors='white', labelcolor='white')
        self.ax.tick_params(axis='z', colors='white', labelcolor='white')
        self.ax.set_xlabel('X [RM]', color='white')
        self.ax.set_ylabel('Y [RM]', color='white')
        self.ax.set_zlabel('Z [RM]', color='white')
        # --------------------------------------------------------

        # 1. データのフィルタリング
        mask = data > VMIN_MANUAL
        X_f = self.X[mask]
        Y_f = self.Y[mask]
        Z_f = self.Z[mask]
        vals = data[mask]

        # 2. ランダム・ダウンサンプリング
        total_points = len(vals)
        if total_points > MAX_PLOT_POINTS:
            indices = np.random.choice(total_points, MAX_PLOT_POINTS, replace=False)
            X_f = X_f[indices]
            Y_f = Y_f[indices]
            Z_f = Z_f[indices]
            vals = vals[indices]
            displayed_points = MAX_PLOT_POINTS
        else:
            displayed_points = total_points

        # 3. 対数スケールによる正規化
        if len(vals) > 0:
            log_vals = np.log10(vals)
            log_vmin = np.log10(VMIN_MANUAL)
            log_vmax = np.log10(VMAX_MANUAL)
            
            norm_vals = (log_vals - log_vmin) / (log_vmax - log_vmin)
            norm_vals = np.clip(norm_vals, 0.0, 1.0)

            # 4. 色と透明度(Alpha)の計算
            colors = self.cmap(norm_vals) 
            alpha_mapped = np.clip(norm_vals * 1.5 + 0.1, 0.1, 1.0)
            colors[:, 3] = alpha_mapped

            # 描画: 1. 外気圏パーティクル
            self.ax.scatter(X_f, Y_f, Z_f, c=colors, s=15, marker='o', edgecolors='none', depthshade=False, zorder=1)

        # 描画: 2. 水星本体 (不透明な球体)
        self.ax.plot_surface(self.merc_x, self.merc_y, self.merc_z, 
                             color='silver', edgecolor='dimgray', antialiased=False, alpha=1.0, zorder=10)

        # カメラと表示範囲の再設定
        self.ax.set_xlim([-GRID_MAX_RM, GRID_MAX_RM])
        self.ax.set_ylim([-GRID_MAX_RM, GRID_MAX_RM])
        self.ax.set_zlim([-GRID_MAX_RM, GRID_MAX_RM])
        self.ax.set_box_aspect([1, 1, 1])
        
        if not initial:
            self.ax.view_init(elev=elev, azim=azim)

        # タイトル表示
        title_str = (f"TAA: {info['taa']:03d}° | TimeID: {info['time_h']}\n"
                     f"Points: {displayed_points} / {total_points} (VMIN: {VMIN_MANUAL:.1e})")
        self.ax.set_title(title_str, color='white', fontsize=12)

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

    def save_transparent_png(self, event):
        """現在の表示を背景透過PNGとして保存する"""
        info = self.file_list[self.current_idx]
        filename = f"exosphere_3d_taa{info['taa']:03d}_t{info['time_h']}.png"
        
        # 1. 状態の退避
        original_facecolor = self.fig.get_facecolor()
        original_title = self.ax.get_title()
        
        # 2. 保存用の設定
        self.fig.patch.set_alpha(0.0) # 背景を完全に透過
        self.ax.set_title("") # タイトルを非表示
        
        # 文字を黒色に変更
        self.ax.tick_params(axis='x', colors='black', labelcolor='black')
        self.ax.tick_params(axis='y', colors='black', labelcolor='black')
        self.ax.tick_params(axis='z', colors='black', labelcolor='black')
        self.ax.xaxis.label.set_color('black')
        self.ax.yaxis.label.set_color('black')
        self.ax.zaxis.label.set_color('black')
        
        # ★ Matplotlibの3Dプロット仕様対策：強制再描画して黒文字設定を適用
        self.fig.canvas.draw()
        
        # 3. 画像の保存
        plt.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0.1, dpi=300)
        
        # 4. 元の表示状態(画面用: 黒背景、白文字)に復元
        self.fig.patch.set_facecolor(original_facecolor)
        self.fig.patch.set_alpha(1.0)
        self.ax.set_title(original_title, color='white', fontsize=12)
        
        self.ax.tick_params(axis='x', colors='white', labelcolor='white')
        self.ax.tick_params(axis='y', colors='white', labelcolor='white')
        self.ax.tick_params(axis='z', colors='white', labelcolor='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.zaxis.label.set_color('white')
        
        # 復元後も再描画
        self.fig.canvas.draw_idle()
        
        print(f"★画像を保存しました: {filename}")


# ==============================================================================
# メイン実行部
# ==============================================================================
if __name__ == "__main__":

    # ディレクトリが存在するか確認
    if not os.path.exists(SIMULATION_RUN_DIRECTORY):
        print(f"エラー: フォルダが見つかりません\n{SIMULATION_RUN_DIRECTORY}")
        sys.exit(1)

    print("ファイルリストを作成中...")
    file_list = get_all_grid_files_sorted(SIMULATION_RUN_DIRECTORY)
    print(f"合計 {len(file_list)} 個のグリッドファイルが見つかりました。")

    if not file_list:
        sys.exit(0)

    print("3Dビューワーを起動します...")
    print("※ 描画に少し時間がかかる場合があります。画面をドラッグすると視点を回転できます。")
    
    viewer = Grid3DVolumeViewer(file_list)
    plt.show()