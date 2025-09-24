import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
import pandas as pd
import sys

# --- ★★★ 設定項目 ★★★ ---
# 実行したシミュレーションの出力フォルダを指定
# 例: "./SimulationResult3D_snapshot_corrected/snapshot_SPperCell10_Q2.0_corrected"
RESULTS_DIR = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult3D_test/snapshot_SPperCell1000_Q2.0"

# 読み込むファイルのパターン
FILE_PATTERN = "snapshot_t*_taa*.npy"

# 解析結果のグラフとCSVを保存するフォルダの名前
OUTPUT_DIR_NAME = "Analysis_Results_ColumnDensity"

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★ プロットしたい領域を一つだけ指定（'Dayside', 'Nightside'など）。
# ★ すべてプロottoする場合は None にする
PLOT_TARGET_LABEL = 'Duskside'
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

# 解析する領域の定義 [theta: 0-180 (北極-南極), phi: -180-180 (反太陽方向が0)]
# 座標系: x軸が太陽方向、z軸が北極方向
REGIONS_TO_ANALYZE = [
    {'label': 'Dayside', 'theta_range_deg': (5, 175), 'phi_range_deg': (-85, 85)},
    {'label': 'Nightside', 'theta_range_deg': (5, 175), 'phi_range_deg': (95, -95)},  # 180度をまたぐ例
    {'label': 'Dawnside', 'theta_range_deg': (5, 175), 'phi_range_deg': (5, 85)},  # Y>0
    {'label': 'Duskside', 'theta_range_deg': (5, 175), 'phi_range_deg': (-85, -5)},  # Y<0
    {'label': 'North_Polar', 'theta_range_deg': (0, 30), 'phi_range_deg': (-180, 180)},
    {'label': 'South_Polar', 'theta_range_deg': (150, 180), 'phi_range_deg': (-180, 180)},
]

# 柱密度を計算する高さの範囲 (水星半径 RM 単位)
RADIAL_RANGE_RM_MIN = 1.0
RADIAL_RANGE_RM_MAX = 5.0

# --- シミュレーション定数 (スナップショットの座標解釈に必要) ---
RM_m = 2439.7e3


# --- ここまで設定項目 ---

def calculate_column_density_from_snapshot(snapshot_data, region, radial_range_rm):
    """
    単一のスナップショットデータから、指定された領域の平均柱密度を計算する。
    """
    if snapshot_data.shape[0] == 0:
        return 0

    # --- 1. データの準備 ---
    pos = snapshot_data[:, 0:3]  # 位置(x,y,z) [m]
    weights = snapshot_data[:, 6]  # 重み [atoms/superparticle]

    # 直交座標(x,y,z)から球座標(r, theta, phi)へ変換
    r_m = np.sqrt(np.sum(pos ** 2, axis=1))
    theta_rad = np.arccos(np.clip(pos[:, 2] / r_m, -1.0, 1.0))  # 天頂角 [0, pi]
    phi_rad = np.arctan2(pos[:, 1], pos[:, 0])  # 方位角 [-pi, pi]

    # --- 2. 領域内に存在する粒子をフィルタリング ---
    # 領域の範囲をメートルとラジアンに変換
    r_min_m, r_max_m = radial_range_rm[0] * RM_m, radial_range_rm[1] * RM_m
    theta_min_rad, theta_max_rad = np.deg2rad(region['theta_range_deg'])
    phi_min_rad, phi_max_rad = np.deg2rad(region['phi_range_deg'])

    # 半径と天頂角でフィルタリング
    mask = (r_m >= r_min_m) & (r_m < r_max_m) & \
           (theta_rad >= theta_min_rad) & (theta_rad < theta_max_rad)

    # 方位角(phi)でフィルタリング (180度をまたぐ場合も考慮)
    if phi_min_rad > phi_max_rad:  # 例: (100, -100) -> 100~180度 と -180~-100度
        mask &= (phi_rad >= phi_min_rad) | (phi_rad < phi_max_rad)
    else:
        mask &= (phi_rad >= phi_min_rad) & (phi_rad < phi_max_rad)

    # 領域内の粒子の重みを合計して、総原子数を計算
    total_atoms_in_region = np.sum(weights[mask])

    # --- 3. 領域の底面積を計算 ---
    # 単位: [m^2]
    if phi_min_rad > phi_max_rad:
        # 180度をまたぐ場合の角度範囲
        delta_phi_rad = (np.pi - phi_min_rad) + (phi_max_rad - (-np.pi))
    else:
        delta_phi_rad = phi_max_rad - phi_min_rad

    delta_cos_theta = np.cos(theta_min_rad) - np.cos(theta_max_rad)
    surface_area_m2 = RM_m ** 2 * delta_phi_rad * delta_cos_theta
    surface_area_cm2 = surface_area_m2 * (100 ** 2)

    if surface_area_cm2 == 0:
        return 0

    # --- 4. 平均柱密度を計算 ---
    # (領域内の総原子数) / (領域の底面積) [atoms/cm^2]
    average_column_density = total_atoms_in_region / surface_area_cm2

    return average_column_density


def main_analysis(results_dir, file_pattern, output_dir_name, regions, radial_range_rm):
    """
    メインの解析処理を実行する関数
    """
    search_path = os.path.join(results_dir, file_pattern)
    file_list = sorted(glob.glob(search_path))

    if not file_list:
        print(f"エラー: '{search_path}' に一致するファイルが見つかりません。")
        return

    print(f"{len(file_list)}個のスナップショットファイルを検出しました。解析を開始します...")

    results_data = {region['label']: [] for region in regions}
    processed_taas = set()

    for filepath in file_list:
        try:
            filename = os.path.basename(filepath)
            match = re.search(r'taa(\d+)', filename)
            if not match:
                continue
            taa = int(match.group(1))

            if taa in processed_taas:
                continue
            processed_taas.add(taa)

            snapshot_data = np.load(filepath)

            for region in regions:
                label = region['label']
                avg_col_density = calculate_column_density_from_snapshot(snapshot_data, region, radial_range_rm)
                results_data[label].append({'taa': taa, 'density': avg_col_density})

        except Exception as e:
            print(f"ファイル '{filepath}' の処理中にエラーが発生しました: {e}")

    print("\n全ファイルの計算が完了しました。")

    # --- グラフ描画とCSV保存 (ご提示のコードとほぼ同じ) ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    df_data = {}

    for label, data_points in results_data.items():
        if not data_points: continue
        sorted_data = sorted(data_points, key=lambda x: x['taa'])
        taas = [d['taa'] for d in sorted_data]
        densities = [d['density'] for d in sorted_data]
        if 'TAA' not in df_data: df_data['TAA'] = taas
        df_data[label] = densities
        ax.plot(taas, densities, marker='o', linestyle='-', label=label, markersize=5)

    title = f"Average Na Column Density vs. TAA\n(Altitude: {radial_range_rm[0]:.1f} - {radial_range_rm[1]:.1f} $R_M$)"
    ax.set_title(title, fontsize=18, pad=15)
    ax.set_xlabel("True Anomaly Angle (TAA) [degrees]", fontsize=14)
    ax.set_ylabel("Na Column Density [atoms/cm$^2$]", fontsize=14)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.legend(fontsize=12)
    plt.tight_layout()

    output_path = os.path.join(results_dir, output_dir_name)
    os.makedirs(output_path, exist_ok=True)

    region_label_for_filename = regions[0]['label'] if len(regions) == 1 else "All_Regions"
    filename_base = f"column_density_{region_label_for_filename}_vs_taa_{radial_range_rm[0]:.1f}-{radial_range_rm[1]:.1f}RM"

    output_png_path = os.path.join(output_path, f"{filename_base}.png")
    plt.savefig(output_png_path, dpi=150)
    print(f"\nグラフを '{output_png_path}' に保存しました。")
    plt.show()

    if df_data:
        df = pd.DataFrame(df_data)
        output_csv_path = os.path.join(output_path, f"{filename_base}.csv")
        df.to_csv(output_csv_path, index=False, float_format='%.5e')
        print(f"データを '{output_csv_path}' に保存しました。")


if __name__ == '__main__':
    if PLOT_TARGET_LABEL:
        regions_for_analysis = [r for r in REGIONS_TO_ANALYZE if r['label'] == PLOT_TARGET_LABEL]
        if not regions_for_analysis:
            print(f"エラー: 指定されたラベル '{PLOT_TARGET_LABEL}' が見つかりません。")
            sys.exit()
    else:
        regions_for_analysis = REGIONS_TO_ANALYZE

    main_analysis(
        RESULTS_DIR,
        FILE_PATTERN,
        OUTPUT_DIR_NAME,
        regions_for_analysis,
        (RADIAL_RANGE_RM_MIN, RADIAL_RANGE_RM_MAX)
    )