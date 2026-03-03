import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
import pandas as pd

# --- ★★★ 設定項目 (リージョン分析対応版) ★★★ ---
# 3Dシミュレーション結果が保存されている親フォルダを指定
RESULTS_DIR = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult3D"

# 読み込むファイルの命名規則（ワイルドカード * を使用）
FILE_PATTERN = "density3d_taa*_beta0.50_Q3.0_MW_ISO_PD_pl24x24.npy"
#FILE_PATTERN = "density3d_taa*_test.npy"
#density3d_beta0.50_Q3.0_MW_ISO_PD_pl24x24 density3d_PSD_ISO_TD_COS_beta0.50_pl24x24


# 解析結果のグラフとCSVを保存するフォルダの名前
OUTPUT_DIR = "Analysis_Results"

# 解析する面の定義
REGIONS_TO_ANALYZE = [
    {'label': 'Dayside', 'theta_range_deg': (10, 170), 'phi_range_deg': (-80, 80)},
    {'label': 'Dawnside', 'theta_range_deg': (10, 170), 'phi_range_deg': (10, 80)},
    {'label': 'VDuskside', 'theta_range_deg': (10, 170), 'phi_range_deg': (-10, 10)},
    {'label': 'Duskside', 'theta_range_deg': (10, 170), 'phi_range_deg': (-80, -10)},
    {'label': 'Nightside', 'theta_range_deg': (30, 150), 'phi_range_deg': (135, -135)},
]

PLOT_TARGET_LABEL = 'Duskside'

# 高さの範囲
RADIAL_RANGE_RM_MIN = 1.1
RADIAL_RANGE_RM_MAX = 3.0

# シミュレーションの物理・グリッド定数
GRID_RADIUS_RM = 5.0
RM_m = 2439.7e3


# --- ここまで設定項目 ---


def plot_and_save_regional_column_density(results_dir, file_pattern, regions_to_analyze, radial_range_rm,
                                          grid_radius_rm, mercury_radius_m, output_dir):
    """
    3D球座標のシミュレーション結果から、指定された複数の「面」(リージョン)の
    平均柱密度を計算し、TAAとの関係を一つのグラフにまとめてプロット・保存する。
    """
    search_path = os.path.join(results_dir, '**', file_pattern)
    file_list = glob.glob(search_path, recursive=True)

    if not file_list:
        print(f"エラー: '{search_path}' に一致するファイルが見つかりません。")
        return

    print(f"{len(file_list)}個のファイルを検出しました。")

    results = {region['label']: [] for region in regions_to_analyze}


    for filepath in sorted(file_list):
        try:
            filename = os.path.basename(filepath)
            match = re.search(r'taa(\d+)', filename)
            if not match:
                print(f"警告: ファイル名 '{filename}' からTAAを抽出できませんでした。スキップします。")
                continue
            taa = int(match.group(1))

            if not filename.endswith('.npy'):
                continue

            number_density_grid_cm3 = np.load(filepath)
            N_R, N_THETA, N_PHI = number_density_grid_cm3.shape

            DR_m = (mercury_radius_m * grid_radius_rm) / N_R
            DR_cm = DR_m * 100
            r_min, r_max = radial_range_rm
            ir_start = int((r_min / grid_radius_rm) * N_R)
            ir_end = int((r_max / grid_radius_rm) * N_R)

            if not (0 <= ir_start < ir_end <= N_R):
                print(f"警告: 半径範囲 {radial_range_rm} RM はグリッド範囲外です (TAA={taa})。スキップします。")
                continue

            for region in regions_to_analyze:
                label = region['label']
                theta_min_deg, theta_max_deg = region['theta_range_deg']
                phi_min_deg, phi_max_deg = region['phi_range_deg']

                itheta_start = int((np.deg2rad(theta_min_deg) / np.pi) * N_THETA)
                itheta_end = int((np.deg2rad(theta_max_deg) / np.pi) * N_THETA)
                itheta_start = np.clip(itheta_start, 0, N_THETA - 1)
                itheta_end = np.clip(itheta_end, 0, N_THETA)

                iphi_start = int(((np.deg2rad(phi_min_deg) + np.pi) / (2 * np.pi)) * N_PHI)
                iphi_end = int(((np.deg2rad(phi_max_deg) + np.pi) / (2 * np.pi)) * N_PHI)

                if iphi_start > iphi_end:
                    part1 = number_density_grid_cm3[ir_start:ir_end, itheta_start:itheta_end, iphi_start:]
                    part2 = number_density_grid_cm3[ir_start:ir_end, itheta_start:itheta_end, :iphi_end]
                    regional_subgrid = np.concatenate((part1, part2), axis=2)
                else:
                    regional_subgrid = number_density_grid_cm3[ir_start:ir_end, itheta_start:itheta_end,
                                       iphi_start:iphi_end]

                if regional_subgrid.size > 0:
                    column_densities_map = np.sum(regional_subgrid, axis=0) * DR_cm
                    average_column_density = np.mean(column_densities_map)
                else:
                    average_column_density = 0

                results[label].append({'taa': taa, 'density': average_column_density})

        except Exception as e:
            print(f"ファイル '{filepath}' の処理中にエラーが発生しました: {e}")

    print("\n全ファイルの計算が完了しました。結果を統合して保存・プロットします。")

    fig, ax = plt.subplots(figsize=(12, 8))
    df = pd.DataFrame()

    for label, data_points in results.items():
        if not data_points:
            print(f"リージョン '{label}' のデータが見つかりませんでした。スキップします。")
            continue

        sorted_data = sorted(data_points, key=lambda x: x['taa'])
        taas = [d['taa'] for d in sorted_data]
        densities = [d['density'] for d in sorted_data]

        if df.empty:
            df['TAA'] = taas
        df[label] = densities

        ax.plot(taas, densities, marker='o', linestyle='none', label=label)

    r_min, r_max = radial_range_rm
    title = f"Na Column Density on Mercury's Surface Regions\n(Altitude: {r_min:.1f} - {r_max:.1f} $R_M$) vs. TAA"
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("True Anomaly Angle (TAA) [degrees]", fontsize=12)
    ax.set_ylabel("Na Column Density [atoms/cm$^2$]", fontsize=12)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    ax.grid(True, which="both", linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()

    # --- 保存 ---

    # 1. 出力フォルダを作成 (既に存在する場合は何もしない)
    os.makedirs(output_dir, exist_ok=True)

    # 2. ファイル名を決定
    if len(regions_to_analyze) == 1:
        region_label_for_filename = regions_to_analyze[0]['label']
    else:
        region_label_for_filename = "All_Regions"
    output_filename_base = f"column_density_{region_label_for_filename}_from_{r_min:.1f}-{r_max:.1f}RM"

    # 3. フォルダパスとファイル名を結合して保存
    # グラフを保存
    output_png_filename = os.path.join(output_dir, f"{output_filename_base}.png")
    plt.savefig(output_png_filename)
    print(f"\n統合グラフを '{output_png_filename}' に保存しました。")
    plt.show()

    # CSVファイルを保存
    if not df.empty:
        output_csv_filename = os.path.join(output_dir, f"{output_filename_base}.csv")
        df.to_csv(output_csv_filename, index=False, float_format='%.6e')
        print(f"統合データを '{output_csv_filename}' に保存しました。")


# --- メインの実行部分 ---
if __name__ == '__main__':
    radial_range = (RADIAL_RANGE_RM_MIN, RADIAL_RANGE_RM_MAX)

    if PLOT_TARGET_LABEL:
        regions_for_analysis = [
            region for region in REGIONS_TO_ANALYZE if region['label'] == PLOT_TARGET_LABEL
        ]
        if not regions_for_analysis:
            print(f"エラー: 指定されたラベル '{PLOT_TARGET_LABEL}' がREGIONS_TO_ANALYZE内に見つかりません。")
            exit()
    else:
        regions_for_analysis = REGIONS_TO_ANALYZE


    plot_and_save_regional_column_density(
        RESULTS_DIR,
        FILE_PATTERN,
        regions_for_analysis,
        radial_range,
        GRID_RADIUS_RM,
        RM_m,
        OUTPUT_DIR  # ここで出力フォルダ名を渡す
    )