import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
import pandas as pd
import sys

# --- ★★★ 設定項目 ★★★ ---
# mercury_na_simulation_taa_sync.py の出力フォルダを指定
#RESULTS_DIR = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult3D/density3d_test2"
RESULTS_DIR = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult3D/density3d_beta0.50_Q3.0_MW_ISO_PD_pl24x24_nostick"

# 読み込むファイルのパターン
FILE_PATTERN = "density3d_taa*.npy"
#FILE_PATTERN = "atmospheric_density_t*_taa*.npy"

# 解析結果のグラフとCSVを保存するフォルダの名前
OUTPUT_DIR_NAME = "Analysis_Results_ColumnDensity"

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★ プロットしたい領域を一つだけ指定（'Dayside', 'Nightside'など）。
# ★ すべてプロットする場合は None にする
#PLOT_TARGET_LABEL = 'Subsolar_Region'
#PLOT_TARGET_LABEL = 'Duskside'
#PLOT_TARGET_LABEL = 'Dusk_Terminator'
PLOT_TARGET_LABEL = 'Dayside'
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

# 解析する領域の定義 [theta: 0-180 (北極-南極), phi: -180-180 (反太陽方向が0)]
REGIONS_TO_ANALYZE = [
    {'label': 'Dayside', 'theta_range_deg': (10, 170), 'phi_range_deg': (-80, 80)},
    {'label': 'Nightside', 'theta_range_deg': (10, 170), 'phi_range_deg': (100, -100)},  # 180度をまたぐ例
    {'label': 'Dawnside', 'theta_range_deg': (10, 170), 'phi_range_deg': (10, 80)},  # Y>0
    {'label': 'Duskside', 'theta_range_deg': (10, 170), 'phi_range_deg': (-80, -10)},  # Y<0
    {'label': 'Subsolar_Region', 'theta_range_deg': (10, 170), 'phi_range_deg': (165, -165)},
    {'label': 'Dusk_Terminator', 'theta_range_deg': (10, 170), 'phi_range_deg': (84, 96)},
    {'label': 'Dawn_Terminator', 'theta_range_deg': (10, 170), 'phi_range_deg': (-96, -84)},
    {'label': 'North_Polar', 'theta_range_deg': (0, 30), 'phi_range_deg': (-180, 180)},
    {'label': 'South_Polar', 'theta_range_deg': (150, 180), 'phi_range_deg': (-180, 180)},
]

# 柱密度を計算する高さの範囲 (水星半径 RM 単位)
RADIAL_RANGE_RM_MIN = 1.0
RADIAL_RANGE_RM_MAX = 4.0

# --- シミュレーション実行時のパラメータ (mercury_na_simulation_taa_sync.py と一致させる) ---
N_R = 100
N_THETA = 24
N_PHI = 24
GRID_RADIUS_RM = 5.0
RM_m = 2439.7e3


# --- ここまで設定項目 ---


def analyze_and_plot_column_density(results_dir, file_pattern, output_dir_name, regions, radial_range_rm):
    """
    3Dシミュレーション結果から柱密度を計算し、グラフとCSVで出力する。
    """
    # --- 1. ファイル検索 ---
    search_path = os.path.join(results_dir, file_pattern)
    file_list = glob.glob(search_path)

    if not file_list:
        print(f"エラー: '{search_path}' に一致するファイルが見つかりません。")
        print("RESULTS_DIRとFILE_PATTERNの設定を確認してください。")
        return

    print(f"{len(file_list)}個のデータファイルを検出しました。解析を開始します...")

    # 結果を格納する辞書を初期化
    results_data = {region['label']: [] for region in regions}
    processed_taas = set()

    # --- 2. 各ファイルをループ処理 ---
    for filepath in sorted(file_list):
        try:
            filename = os.path.basename(filepath)
            # ファイル名からTAAを抽出
            match = re.search(r'taa(\d+)', filename)
            if not match:
                print(f"警告: ファイル名 '{filename}' からTAAを抽出できませんでした。スキップします。")
                continue
            taa = int(match.group(1))

            # TAAが重複している場合はスキップ (複数年のシミュレーションなどで発生しうる)
            if taa in processed_taas:
                continue
            processed_taas.add(taa)

            # --- 3. データの読み込みとパラメータ計算 ---
            number_density_grid_cm3 = np.load(filepath)

            # 半径方向のステップ幅を計算 [cm]
            dr_m = (RM_m * GRID_RADIUS_RM) / N_R
            dr_cm = dr_m * 100

            # 柱密度を計算する半径方向のインデックス範囲を決定
            ir_start = int(N_R * (radial_range_rm[0] / GRID_RADIUS_RM))
            ir_end = int(N_R * (radial_range_rm[1] / GRID_RADIUS_RM))

            # インデックスがグリッド範囲内に収まっているか確認
            ir_start = np.clip(ir_start, 0, N_R)
            ir_end = np.clip(ir_end, 0, N_R)
            if ir_start >= ir_end:
                print(f"警告: TAA={taa}で半径範囲のインデックスが無効です。スキップします。")
                continue

            # --- 4. 各領域の柱密度を計算 ---
            for region in regions:
                label = region['label']
                theta_min_deg, theta_max_deg = region['theta_range_deg']
                phi_min_deg, phi_max_deg = region['phi_range_deg']

                # 角度をグリッドのインデックスに変換
                itheta_start = int((np.deg2rad(theta_min_deg) / np.pi) * N_THETA)
                itheta_end = int((np.deg2rad(theta_max_deg) / np.pi) * N_THETA)
                itheta_start = np.clip(itheta_start, 0, N_THETA)
                itheta_end = np.clip(itheta_end, 0, N_THETA)

                iphi_start = int(((np.deg2rad(phi_min_deg) + np.pi) / (2 * np.pi)) * N_PHI) % N_PHI
                iphi_end = int(((np.deg2rad(phi_max_deg) + np.pi) / (2 * np.pi)) * N_PHI) % N_PHI

                # 指定された領域のサブグリッドを切り出す
                target_grid = number_density_grid_cm3[ir_start:ir_end, itheta_start:itheta_end, :]

                # 方位角(phi)が180度をまたぐ場合の処理
                if iphi_start > iphi_end:
                    part1 = target_grid[:, :, iphi_start:]
                    part2 = target_grid[:, :, :iphi_end]
                    regional_subgrid = np.concatenate((part1, part2), axis=2)
                else:
                    regional_subgrid = target_grid[:, :, iphi_start:iphi_end]

                # 柱密度を計算して平均化
                if regional_subgrid.size > 0:
                    # 1. 半径方向に積分して、各角度点での柱密度マップを作成 [atoms/cm^2]
                    column_densities_map = np.sum(regional_subgrid, axis=0) * dr_cm
                    # 2. 領域内で柱密度を平均
                    average_column_density = np.mean(column_densities_map)
                else:
                    average_column_density = 0

                results_data[label].append({'taa': taa, 'density': average_column_density})

        except Exception as e:
            print(f"ファイル '{filepath}' の処理中にエラーが発生しました: {e}")

    print("\n全ファイルの計算が完了しました。結果を統合して保存・プロットします。")

    # --- 5. グラフのプロット ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    df_data = {}  # CSV保存用のデータを準備

    for label, data_points in results_data.items():
        if not data_points:
            print(f"リージョン '{label}' のデータが見つかりませんでした。")
            continue

        sorted_data = sorted(data_points, key=lambda x: x['taa'])
        taas = [d['taa'] for d in sorted_data]
        densities = [d['density'] for d in sorted_data]

        # データを辞書に追加
        if 'TAA' not in df_data:
            df_data['TAA'] = taas
        df_data[label] = densities

        ax.plot(taas, densities, marker='o', linestyle='none', label=label, markersize=5)

    # グラフの装飾
    title = f"Average Na Column Density vs. TAA\n(Altitude: {radial_range_rm[0]:.1f} - {radial_range_rm[1]:.1f} $R_M$)"
    ax.set_title(title, fontsize=18, pad=15)
    ax.set_xlabel("True Anomaly Angle (TAA) [degrees]", fontsize=14)
    ax.set_ylabel("Na Column Density [atoms/cm$^2$]", fontsize=14)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.legend(fontsize=12)
    plt.tight_layout()

    # --- 6. 結果の保存 ---
    output_path = os.path.join(results_dir, output_dir_name)
    os.makedirs(output_path, exist_ok=True)

    # 保存するファイル名のベース部分を作成
    if len(regions) == 1:
        region_label_for_filename = regions[0]['label']
    else:
        region_label_for_filename = "All_Regions"
    filename_base = f"column_density_{region_label_for_filename}_vs_taa_{radial_range_rm[0]:.1f}-{radial_range_rm[1]:.1f}RM"

    # グラフを保存
    output_png_path = os.path.join(output_path, f"{filename_base}.png")
    plt.savefig(output_png_path, dpi=150)
    print(f"\nグラフを '{output_png_path}' に保存しました。")
    plt.show()

    # CSVファイルを保存
    if df_data:
        df = pd.DataFrame(df_data)
        output_csv_path = os.path.join(output_path, f"{filename_base}.csv")
        df.to_csv(output_csv_path, index=False, float_format='%.5e')
        print(f"データを '{output_csv_path}' に保存しました。")


# --- メインの実行部分 ---
if __name__ == '__main__':
    # ★★★ ここでプロットする領域をフィルタリング ★★★
    if PLOT_TARGET_LABEL:
        # PLOT_TARGET_LABELで指定されたリージョンのみを抽出
        regions_for_analysis = [
            region for region in REGIONS_TO_ANALYZE if region['label'] == PLOT_TARGET_LABEL
        ]
        if not regions_for_analysis:
            print(f"エラー: 指定されたラベル '{PLOT_TARGET_LABEL}' がREGIONS_TO_ANALYZE内に見つかりません。")
            sys.exit()
    else:
        # 指定がない場合は、定義されたすべてのリージョンを解析
        regions_for_analysis = REGIONS_TO_ANALYZE

    analyze_and_plot_column_density(
        RESULTS_DIR,
        FILE_PATTERN,
        OUTPUT_DIR_NAME,
        regions_for_analysis,  # フィルタリングされたリストを渡す
        (RADIAL_RANGE_RM_MIN, RADIAL_RANGE_RM_MAX)
    )

