import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 設定項目 ---
# シミュレーション結果が保存されている親ディレクトリ
# このディレクトリの中に 'density3d_...' といったサブフォルダがあると想定
OUTPUT_DIRECTORY = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult3D"

# シミュレーション実行時に設定したのと同じパラメータを使って、
# 正しいサブフォルダ名を特定します。
# もしフォルダ名が異なる場合は、下の 'sub_folder_name' を直接編集してください。
settings = {
    'BETA': 0.50,
    'speed_distribution': 'maxwellian',
    'ejection_direction_model': 'isotropic',
    'ionization_model': 'particle_death',
}
N_THETA = 24
N_PHI = 24

# --- フォルダ名の生成 (シミュレーションコードから流用) ---
dist_tag = "CO" if settings['ejection_direction_model'] == 'cosine' else "ISO"
speed_tag = "MW" if settings['speed_distribution'] == 'maxwellian' else "WB"
ion_tag = "WD" if settings['ionization_model'] == 'weight_decay' else "PD"
base_name_template = f"density3d_beta{settings['BETA']:.2f}_Q1.0_{speed_tag}_{dist_tag}_{ion_tag}_pl{N_THETA}x{N_PHI}"
sub_folder_name = base_name_template

# --- メイン処理 ---
def plot_death_statistics():
    """
    death_statistics.csvを読み込み、死因の割合をTAAに対してプロットする。
    """
    # CSVファイルのフルパスを構築
    csv_file_path = os.path.join(OUTPUT_DIRECTORY, sub_folder_name, "death_statistics.csv")

    # CSVファイルの存在チェック
    if not os.path.exists(csv_file_path):
        print(f"エラー: ファイルが見つかりません。パスを確認してください: {csv_file_path}")
        return

    print(f"'{csv_file_path}' からデータを読み込んでいます...")
    # pandasを使ってCSVファイルを読み込む
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"エラー: CSVファイルの読み込み中に問題が発生しました。 {e}")
        return

    # グラフのスタイルを設定
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # 各死因の割合をプロット
    ax.plot(df['TAA'], df['Ionized_Percent'], marker='o', linestyle='-', label='Ionized', color='red')
    ax.plot(df['TAA'], df['Stuck_Percent'], marker='o', linestyle='--', label='Stuck', color='blue')
    ax.plot(df['TAA'], df['Escaped_Percent'], marker='o', linestyle=':', label='Escaped', color='green')

    # グラフの装飾
    ax.set_title('Particle Fate vs. True Anomaly Angle (TAA)', fontsize=16)
    ax.set_xlabel('True Anomaly Angle (TAA) [degrees]', fontsize=12)
    ax.set_ylabel('Percentage of Particles [%]', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xticks(range(0, 361, 30))  # X軸の目盛りを30度ごとに設定
    ax.set_ylim(0, 100) # Y軸を0%から100%に固定
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # グラフを画像ファイルとして保存
    output_image_path = os.path.join(OUTPUT_DIRECTORY, sub_folder_name, "death_statistics_vs_TAA.png")
    plt.savefig(output_image_path, dpi=300)
    print(f"グラフを '{output_image_path}' に保存しました。")

    # グラフを表示
    plt.show()


if __name__ == '__main__':
    plot_death_statistics()