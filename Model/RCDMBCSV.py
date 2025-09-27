import numpy as np
import sys
import os
# (Pool, cpu_count, tqdm は不要になるのでコメントアウトまたは削除してもOK)
# from multiprocessing import Pool, cpu_count
# from tqdm import tqdm
import matplotlib.pyplot as plt
import csv


# --- 物理定数と各種関数 ---
# (元のコードと同じなので、この部分は省略)
# PHYSICAL_CONSTANTS = {...}
# calculate_surface_temperature, calculate_sticking_probability, etc.

# --- ★★★ CSVから読み取って再計算する新しい関数 ★★★ ---
def recalculate_density_from_csv(input_csv_path, new_phi_0, new_t1au, source_exponent=2.0):
    """
    事前に計算されたt_Mを含むCSVファイルを読み込み、
    新しいパラメータで柱密度を再計算する。
    """
    # --- 入力ファイルの存在確認 ---
    if not os.path.exists(input_csv_path):
        print(f"エラー: 入力CSVファイルが見つかりません: {input_csv_path}", file=sys.stderr)
        return
    try:
        orbit_lines = open('orbit360.txt', 'r').readlines()
    except FileNotFoundError:
        print("エラー: orbit360.txt が見つかりません。", file=sys.stderr)
        return

    # --- 軌道データを読み込み、TAAをキーにした辞書を作成 ---
    orbit_data = {}
    for line in orbit_lines:
        TAA, AU, _, _, _ = map(float, line.split())
        orbit_data[TAA] = AU

    # --- CSVから計算済みのt_Mを読み込む ---
    tm_data = []
    with open(input_csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # ヘッダーをスキップ
        for row in reader:
            # CSVの列: 'TAA_deg', 't_M_hr', 'tau_hr', 'N_bar_cm-2'
            taa, tm_hr, _, _ = row
            tm_data.append((float(taa), float(tm_hr)))

    print("--- CSVデータからの柱密度再計算 ---")
    print(f"入力ファイル: {input_csv_path}")
    print(f"新しいパラメータ: phi_0 = {new_phi_0:.2e}, T1AU = {new_t1au:.2e}, source_exp = {source_exponent}")
    print("-" * 60)
    print(f"{'TAA':>6s} {'t_M (hr)':>10s} {'new_tau (hr)':>12s} {'new_N_bar (cm-2)':>18s}")
    print("-" * 60)

    # --- 再計算ループ ---
    results_data = []
    for taa, tm_hr in tm_data:
        if taa not in orbit_data:
            print(f"警告: TAA={taa} の軌道データが見つかりません。スキップします。")
            continue

        au = orbit_data[taa]
        tm_sec = tm_hr * 3600

        # --- 新しいパラメータで τ と φ を計算 ---
        new_tau_sec = new_t1au * (au ** 2)

        R_p = 0.307  # 近日点距離 (AU)
        phi_cm2 = new_phi_0 * (R_p / au) ** source_exponent
        #phi_cm2 = 2.0e7
        phi_m2 = phi_cm2 * 1e4

        # --- 新しい柱密度を計算 ---
        exponent = -tm_sec / new_tau_sec if new_tau_sec > 0 else -float('inf')
        N_bar_m2 = phi_m2 * new_tau_sec * (1.0 - np.exp(exponent))
        new_N_bar_cm2 = N_bar_m2 / 1e4

        # 結果を画面に出力
        new_tau_hr = new_tau_sec / 3600
        print(f"{taa:6.1f} {tm_hr:10.2f} {new_tau_hr:12.2f} {new_N_bar_cm2:18.2e}")

        results_data.append([taa, new_N_bar_cm2])

    # --- グラフのプロット ---
    taa_values = [row[0] for row in results_data]
    n_bar_values = [row[1] for row in results_data]

    plt.figure(figsize=(12, 7))
    plt.plot(taa_values, n_bar_values, marker='.', linestyle='-', color='red',
             label=f'phi_0={new_phi_0:.1e}, T1AU={new_t1au:.1e}')

    plt.xlabel("True Anomaly Angle (deg)", fontsize=14)
    plt.ylabel("Recalculated Column Density (atoms cm⁻²)", fontsize=14)
    plt.title(f"Recalculated Density from {os.path.basename(input_csv_path)}", fontsize=16)
    plt.yscale('log')
    plt.xticks(np.arange(0, 361, 30))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 元の main_smyth_method() の呼び出しはコメントアウト
    # main_smyth_method()

    # --- ★★★ 再計算の実行設定 ★★★ ---
    # 1. 事前に計算したt_Mを含むCSVファイルを指定
    input_csv = "smyth_method_results/smyth_results_beta_0.5_start_60deg.csv"

    # 2. 変更したいパラメータをここで設定
    # (例: 前回の質問で修正した、より妥当な値)
    new_phi_0_val = 4.6e7  # 新しい供給率 (at 近日点) [atoms/cm^2/s]
    #new_phi_0_val = 2.6e7
    new_t1au_val = 5.4e4  # 新しい光電離寿命 (at 1AU) [s]
    #new_t1au_val = 1.9e5

    # --- 再計算を実行 ---
    recalculate_density_from_csv(
        input_csv_path=input_csv,
        new_phi_0=new_phi_0_val,
        new_t1au=new_t1au_val
    )