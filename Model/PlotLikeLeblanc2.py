import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import os
import glob
import sys

# ==============================================================================
# 設定・定数
# ==============================================================================
# 1. シミュレーション結果が保存されているディレクトリのパス (末尾の / は不要)
#    ここに "DynamicGrid..." などのフォルダパスを指定してください
SIM_RESULT_DIR = r"./SimulationResult_202512/DynamicGrid72x36_NoEq_Hard_DT100_T0100_1.0"

# 2. 読み込みたい TAA (True Anomaly Angle) を指定
#    シミュレーションが出力したTAA（例: 125）を指定します
TARGET_TAA = 125

# 3. 観測位相角 (Figure 12の再現には120度を使用)
PHASE_ANGLE_DEG = 120.0

# 物理定数・グリッド設定
RM_M = 2.440e6  # 水星半径 [m]
GRID_RESOLUTION = 101  # グリッド分割数
GRID_MAX_RM = 5.0  # 計算領域半径 [RM]

# プロット設定
FIG_SIZE = (10, 8)
X_RANGE_KM = (-6000, 50000)  # プロットするX軸の範囲 (km)

# ==============================================================================
# 比較用データ (Potter et al., 2002 / Leblanc Fig 12b 近似値)
# ==============================================================================
# X: Heliocentric distance (km), Y: Column Density (10^9 Na/cm^2)
POTTER_DATA_X = np.array([-3, -1.5, 0, 1.5, 3, 4.5, 6, 10, 15, 20, 30, 40]) * 1000
POTTER_DATA_Y = np.array([3.5, 12, 18, 11, 4.5, 2.0, 1.0, 0.4, 0.2, 0.1, 0.05, 0.03])


# ==============================================================================
# ファイル検索・解析クラス
# ==============================================================================
def find_file_by_taa(directory, taa):
    """
    指定されたディレクトリ内から density_grid_t*_taa{taa}.npy を検索して返す。
    """
    # 検索パターン: taaは3桁ゼロ埋めされていると想定 (シミュレーションコードの仕様)
    pattern = f"density_grid_t*_taa{int(taa):03d}.npy"
    search_path = os.path.join(directory, pattern)

    files = glob.glob(search_path)

    if not files:
        # 見つからない場合はエラー
        return None

    # 複数見つかった場合は、タイムスタンプ(tの値)が一番大きいもの（最新）を使う
    # ファイル名例: density_grid_t00500_taa125.npy -> 文字列ソートで末尾が最新になるはず
    files.sort()
    return files[-1]


class MercuryExosphereAnalyzer:
    def __init__(self, npy_path, grid_res, max_rm):
        self.grid_res = grid_res
        self.max_rm = max_rm
        self.rm_m = RM_M

        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"File not found: {npy_path}")

        print(f"Loading grid data from: {os.path.basename(npy_path)}")
        self.density_grid = np.load(npy_path)

        # 補間関数の作成
        grid_coords = np.linspace(-max_rm, max_rm, grid_res)
        self.interpolator = RegularGridInterpolator(
            (grid_coords, grid_coords, grid_coords),
            self.density_grid,
            bounds_error=False,
            fill_value=0.0
        )

    def calculate_los_column_density(self, target_dist_km, phase_angle_deg):
        # 座標変換: Grid_X (Sun=+X) <-> Plot_X (Sun=-X)
        target_x_grid_m = -1.0 * target_dist_km * 1000.0
        target_x_rm = target_x_grid_m / self.rm_m

        # 視線積分セットアップ
        p_center = np.array([target_x_rm, 0.0, 0.0])
        theta_rad = np.deg2rad(phase_angle_deg)
        vec_obs = np.array([np.cos(theta_rad), np.sin(theta_rad), 0.0])

        # 積分
        ds_rm = 0.05
        ds_m = ds_rm * self.rm_m
        s_values = np.arange(-15.0, 15.0, ds_rm)
        sample_points = p_center + np.outer(s_values, vec_obs)

        densities = self.interpolator(sample_points)
        return np.sum(densities) * ds_m

    def get_profile(self, x_range_km, step_km=500):
        x_values = np.arange(x_range_km[0], x_range_km[1], step_km)
        y_values = []
        print("Calculating profiles...")
        for x in x_values:
            col_dens = self.calculate_los_column_density(x, PHASE_ANGLE_DEG)
            y_values.append(col_dens)
        return x_values, np.array(y_values)


# ==============================================================================
# メイン実行部
# ==============================================================================
def main():
    # 1. ファイルの自動検索
    print(f"Searching for TAA={TARGET_TAA} in {SIM_RESULT_DIR} ...")
    data_file_path = find_file_by_taa(SIM_RESULT_DIR, TARGET_TAA)

    if data_file_path is None:
        print(f"Error: No file found for TAA={TARGET_TAA} in the directory.")
        print(f"Check the directory path: {SIM_RESULT_DIR}")
        return

    print(f"File found: {data_file_path}")

    # 2. 解析とプロット
    try:
        analyzer = MercuryExosphereAnalyzer(data_file_path, GRID_RESOLUTION, GRID_MAX_RM)
    except Exception as e:
        print(f"Error during initialization: {e}")
        return

    x_km, col_dens_m2 = analyzer.get_profile(X_RANGE_KM, step_km=500)

    # 単位変換: [atoms/m^2] -> [10^9 atoms/cm^2] (factor 1e-13)
    col_dens_plot = col_dens_m2 * 1e-13

    # プロット
    plt.figure(figsize=FIG_SIZE)
    plt.plot(x_km / 1000.0, col_dens_plot, 'o', color='gray', mfc='none', label='Simulation')
    plt.plot(POTTER_DATA_X / 1000.0, POTTER_DATA_Y, 'k+', markersize=12, markeredgewidth=2,
             label='Potter et al. (2002)')

    plt.axvline(x=-2.44, color='k', linestyle='--', alpha=0.5)
    plt.axvline(x=2.44, color='k', linestyle='--', alpha=0.5)
    plt.text(0, 50, 'Mercury', ha='center', va='bottom', fontsize=10)

    plt.yscale('log')
    plt.xlim(X_RANGE_KM[0] / 1000, X_RANGE_KM[1] / 1000)
    plt.ylim(0.01, 100)

    plt.xlabel(r'Heliocentric distance to Mercury ($10^3$ km)', fontsize=14)
    plt.ylabel(r'Column density ($10^9$ Na/cm$^2$)', fontsize=14)
    plt.title(
        f'Mercury Sodium Exosphere Profile\nTAA ~ {TARGET_TAA}$^\circ$, Phase Angle = {int(PHASE_ANGLE_DEG)}$^\circ$',
        fontsize=16)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(fontsize=12)

    save_fname = f"fig12_taa{TARGET_TAA}.png"
    plt.savefig(save_fname, dpi=300)
    print(f"Figure saved to {save_fname}")
    plt.show()


if __name__ == "__main__":
    main()