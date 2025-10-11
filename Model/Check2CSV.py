import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- ★★★ 設定項目 ★★★ ---

# 比較したいCSVファイルを2つ指定します
CSV_FILE_1 = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult3D\density3d_beta0.50_Q1.0_MW_ISO_PD_pl24x48_nostick\Analysis_Results_ColumnDensity\column_density_Duskside_vs_taa_1.0-4.0RM.csv"
CSV_FILE_2 = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult3D\density3d_beta0.50_Q3.0_MW_ISO_PD_pl24x24_nostick\Analysis_Results_ColumnDensity\column_density_Dayside_vs_taa_1.0-4.0RM.csv"

# --- グラフの凡例や軸ラベルに使うテキスト ---
# 1つ目のデータセットのラベル (凡例で表示されます)
LABEL_1 = 'Q=1.0 (Duskside)'
# 2つ目のデータセットのラベル (凡例で表示されます)
LABEL_2 = 'Q=3.0 (Duskside)'

# --- 読み込む列の名前 ---
# 横軸(X軸)として使用する列の名前 (通常は 'TAA' のまま)
X_COLUMN_NAME = 'TAA'
# 1つ目のCSVファイルから読み込む縦軸(Y軸)の列名
# (元のコードで指定した'label'が列名になっています)
Y1_COLUMN_NAME = 'Duskside'
# 2つ目のCSVファイルから読み込む縦軸(Y軸)の列名
Y2_COLUMN_NAME = 'Dayside'

# --- グラフの装飾 ---
# グラフのタイトル
PLOT_TITLE = 'Comparison of Na Column Density vs. TAA (Duskside)'
# 出力する画像ファイル名
OUTPUT_FILENAME = 'comparison_plot_Q1_vs_Q3_Duskside.png'
# 画像を保存するフォルダ (CSV_FILE_1 と同じ場所になります)
OUTPUT_DIR = os.path.dirname(CSV_FILE_1)

# --- ★★★ 設定はここまで ★★★ ---


def plot_dual_axis_comparison(csv1, csv2, x_col, y1_col, y2_col, label1, label2, title, output_path):
    """
    2つのCSVファイルを読み込み、2つのY軸を持つ比較グラフを作成して保存する。
    """
    # --- 1. データの読み込み ---
    try:
        df1 = pd.read_csv(csv1)
        df2 = pd.read_csv(csv2)
        print(f"1つ目のCSVファイルを読み込みました: {os.path.basename(csv1)}")
        print(f"2つ目のCSVファイルを読み込みました: {os.path.basename(csv2)}")
    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません。パスを確認してください。 -> {e}")
        return

    # --- 2. グラフの作成準備 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # X軸を共有する2つ目のY軸 (ax2) を作成
    ax2 = ax1.twinx()

    # --- 3. 1つ目のデータをプロット (左側のY軸) ---
    color1 = 'tab:blue'
    # 元のスクリプトに合わせて、マーカーのみのプロットにする場合は scatter を使うと便利です
    ax1.scatter(df1[x_col], df1[y1_col], label=label1, color=color1, marker='o')
    ax1.set_xlabel("True Anomaly Angle (TAA) [degrees]", fontsize=14)
    ax1.set_ylabel(f"{label1} Density [atoms/cm$^2$]", fontsize=14, color=color1)
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor=color1, which='both') # 目盛りの色も変更

    # --- 4. 2つ目のデータをプロット (右側のY軸) ---
    color2 = 'tab:red'
    ax2.scatter(df2[x_col], df2[y2_col], label=label2, color=color2, marker='^') # マーカーを変えると見やすい
    ax2.set_ylabel(f"{label2} Density [atoms/cm$^2$]", fontsize=14, color=color2)
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor=color2, which='both')

    # --- 5. グラフ全体の装飾 ---
    ax1.set_title(title, fontsize=18, pad=15)
    ax1.set_xticks(np.arange(0, 361, 30))
    ax1.tick_params(axis='x', which='major', labelsize=12)

    # 凡例を結合して表示
    # ax1とax2から凡例の情報をそれぞれ取得し、リストを結合して1つの凡例を作成します
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=12)

    plt.tight_layout()

    # --- 6. グラフの保存と表示 ---
    full_output_path = os.path.join(output_path, OUTPUT_FILENAME)
    plt.savefig(full_output_path, dpi=150)
    print(f"\nグラフを '{full_output_path}' に保存しました。")
    plt.show()


# --- メインの実行部分 ---
if __name__ == '__main__':
    plot_dual_axis_comparison(
        csv1=CSV_FILE_1,
        csv2=CSV_FILE_2,
        x_col=X_COLUMN_NAME,
        y1_col=Y1_COLUMN_NAME,
        y2_col=Y2_COLUMN_NAME,
        label1=LABEL_1,
        label2=LABEL_2,
        title=PLOT_TITLE,
        output_path=OUTPUT_DIR
    )