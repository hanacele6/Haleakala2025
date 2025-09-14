import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re

# --- ★★★ 設定項目 (3D対応版) ★★★ ---
# 3Dシミュレーション結果が保存されている親フォルダを指定
RESULTS_DIR = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult3D"

# 読み込むファイルの命名規則（ワイルドカード * を使用）
FILE_PATTERN = "density3d_map_taa*_beta0.50_Q1.0_MW_pl24x48.npy"

# ★ 柱密度を計算したい水星表面の「地点」を角度で指定
# 極角 (緯度のようなもの): 0°=北極, 90°=赤道, 180°=南極
TARGET_THETA_DEG = 90.0
# 方位角 (経度のようなもの): 0°=太陽方向(昼の12時), 90°=朝側, 180°=反太陽方向(夜の0時), -90°=夕側
TARGET_PHI_DEG = 60.0

# ★ 柱密度を計算する「高さの範囲」を水星半径単位で指定 (地表からの高さ)
# 例: 1.0-1.7 は地表(1.0RM)から高度0.7RMまでの範囲
RADIAL_RANGE_RM_MIN = 1.0
RADIAL_RANGE_RM_MAX = 2.0

# --- シミュレーションの物理・グリッド定数 ---
# これらはシミュレーション本体のコードと一致させる必要があります
GRID_RADIUS_RM = 5.0  # シミュレーションで設定したグリッドの最大半径
RM_m = 2439.7e3  # 水星の半径 [m]


# --- ここまで設定項目 ---


def plot_and_save_column_density_vs_taa(results_dir, file_pattern, target_theta_deg, target_phi_deg, radial_range_rm,
                                        grid_radius_rm, mercury_radius_m):
    """
    3D球座標のシミュレーション結果(NPY)を読み込み、
    指定された表面地点からの半径方向の柱密度とTAAの関係をプロットし、
    結果をCSVファイルとして保存する。
    """
    search_path = os.path.join(results_dir, '**', file_pattern)
    file_list = glob.glob(search_path, recursive=True)

    if not file_list:
        print(f"エラー: '{search_path}' に一致するファイルが見つかりません。")
        return

    print(f"{len(file_list)}個のファイルを検出しました。")

    taas = []
    column_densities = []  # 柱密度を格納するリスト

    for filepath in file_list:
        try:
            filename = os.path.basename(filepath)
            match = re.search(r'taa(\d+)', filename)
            if not match:
                print(f"警告: ファイル名 '{filename}' からTAAを抽出できませんでした。スキップします。")
                continue
            taa = int(match.group(1))

            if not filename.endswith('.npy'):
                continue

            # 3Dの数密度グリッドを読み込む [atoms/cm^3]
            number_density_grid_cm3 = np.load(filepath)
            N_R, N_THETA, N_PHI = number_density_grid_cm3.shape

            # --- ★ ここから計算ロジックを3D用に変更 ---

            # 1. 半径方向の1セルの厚みを計算 [cm]
            DR_m = (mercury_radius_m * grid_radius_rm) / N_R
            DR_cm = DR_m * 100

            # 2. 指定された角度から、対応するグリッドのインデックス(itheta, iphi)を計算
            # 極角 (theta)
            target_theta_rad = np.deg2rad(target_theta_deg)
            itheta = int((target_theta_rad / np.pi) * N_THETA)
            itheta = np.clip(itheta, 0, N_THETA - 1)  # 範囲内に収める

            # 方位角 (phi)
            target_phi_rad = np.deg2rad(target_phi_deg)  # [-pi, pi] の範囲
            # インデックス計算のため [0, 2pi] に変換
            iphi = int(((target_phi_rad + np.pi) / (2 * np.pi)) * N_PHI)
            iphi = np.clip(iphi, 0, N_PHI - 1)

            # 3. 半径方向のインデックス範囲(ir_start, ir_end)を計算
            r_min, r_max = radial_range_rm
            ir_start = int((r_min / grid_radius_rm) * N_R)
            ir_end = int((r_max / grid_radius_rm) * N_R)

            if not (0 <= ir_start < ir_end <= N_R):
                print(f"警告: 半径範囲 {radial_range_rm} RM はグリッド範囲外です (TAA={taa})。スキップします。")
                continue

            # 4. 指定した地点の真上に伸びる「柱」のデータをスライス
            radial_column = number_density_grid_cm3[ir_start:ir_end, itheta, iphi]

            # 5. 柱に沿って数密度を積分（合計 * 厚み）して柱密度を計算
            if radial_column.size > 0:
                # 柱密度 [atoms/cm^2] = Σ (数密度 [atoms/cm^3] * セルの厚み [cm])
                column_density_value = np.sum(radial_column) * DR_cm
            else:
                column_density_value = 0

            # --- ★ 変更ここまで ---

            taas.append(taa)
            column_densities.append(column_density_value)

        except Exception as e:
            print(f"ファイル '{filepath}' の処理中にエラーが発生しました: {e}")

    if not taas:
        print("処理できるデータがありませんでした。")
        return

    # TAAでソート
    sorted_pairs = sorted(zip(taas, column_densities))
    taas_sorted, densities_sorted = zip(*sorted_pairs)

    # --- 出力ファイル名とグラフタイトルを更新 ---
    r_min, r_max = radial_range_rm
    output_filename_base = f"column_density_at_T{target_theta_deg}_P{target_phi_deg}_from_{r_min:.1f}-{r_max:.1f}RM"
    output_csv_filename = f"{output_filename_base}.csv"

    output_data = np.column_stack((taas_sorted, densities_sorted))
    np.savetxt(output_csv_filename, output_data, fmt='%.6e', delimiter=',', header='TAA,Column_Density_atoms_cm2',
               comments='')
    print(f"\n結果を '{output_csv_filename}' に保存しました。")

    # --- グラフ描画 ---
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(taas_sorted, densities_sorted, marker='o', linestyle='-', color='darkblue')

    # タイトルとラベルを更新
    title = (
        f"Column Density at $\\theta$={target_theta_deg}°, $\\phi$={target_phi_deg}° (from {r_min}$R_M$ to {r_max}$R_M$) vs. TAA")
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("True Anomaly Angle (TAA) [degrees]", fontsize=12)
    ax.set_ylabel("Column Density [atoms/cm$^2$]", fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, which="both", linestyle='--', alpha=0.7)

    plt.tight_layout()

    output_png_filename = f"{output_filename_base}.png"
    plt.savefig(output_png_filename)
    print(f"グラフを '{output_png_filename}' に保存しました。")
    plt.show()


if __name__ == '__main__':
    radial_range = (RADIAL_RANGE_RM_MIN, RADIAL_RANGE_RM_MAX)

    plot_and_save_column_density_vs_taa(
        RESULTS_DIR,
        FILE_PATTERN,
        TARGET_THETA_DEG,
        TARGET_PHI_DEG,
        radial_range,
        GRID_RADIUS_RM,
        RM_m
    )