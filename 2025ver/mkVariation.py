import numpy as np
import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# ==============================================================================
# スクリプトの実行部
# ==============================================================================
if __name__ == '__main__':
    # --- 基本設定 ---
    # 処理したい日付のリスト。ここに日付を追加すれば、自動で集計対象になります。
    #days_to_process = ["20250701", "20250702", "20250703", "20250705", "20250706", "20250707", "20250709", "20250710","20250711", "20250712", "20250713", "20250716", "20250717", "20250720"]
    days_to_process = ["20250613", "20250614", "20250615", "20250616", "20250617", "20250630"]

    base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")

    # --- 入力ファイルと出力ファイル ---
    # このスクリプトは、複数の日の結果を一つのファイルにまとめます
    output_filename = base_dir / 'output' / 'DailyVariation_Summary_202506.dat' #ここ変えるでい
    # 位相角補正用のファイル
    correction_file = base_dir / 'PA_correction2.txt'

    print("--- 日々の変動の集計を開始します ---")

    # --- 1. 位相角(PA)補正の準備 ---
    try:
        # 補正ファイルを読み込み、2次の多項式でフィットする
        pa_data, cf_data = np.loadtxt(correction_file, usecols=(0, 1), unpack=True)
        pa_correction_coeffs = np.polyfit(pa_data, cf_data, 2)
        print("位相角補正ファイルを読み込み、多項式フィットを実行しました。")
    except FileNotFoundError:
        print(f"エラー: 位相角補正ファイルが見つかりません: {correction_file}")
        sys.exit()
    except Exception as e:
        print(f"エラー: 位相角補正ファイルの処理中にエラーが発生しました: {e}")
        sys.exit()

    # --- 2. 各日付のデータをループ処理 ---
    final_results_list = []
    taa_for_plot = []
    correction_factors_for_plot = []

    for day in days_to_process:
        print(f"\n-> {day} のデータを処理中...")

        # --- 2a. 前の工程で作成したサマリーファイルを読み込む ---
        summary_file = base_dir / "output" / day / f'Final_Summary_{day}.txt'
        try:
            # ファイルから PA, TAA, 平均原子数, 統合誤差 を読み込む
            PA, TAA, avg_atoms, integrated_err = np.loadtxt(summary_file)
        except FileNotFoundError:
            print(f"    -> 警告: サマリーファイルが見つかりません: {summary_file.name}。この日付をスキップします。")
            continue
        except Exception as e:
            print(f"    -> 警告: サマリーファイルの読み込み中にエラーが発生しました ({e})。この日付をスキップします。")
            continue

        # --- 2b. その日の詳細なパラメータをCSVから読み込む ---
        csv_file_path = base_dir / "2025ver" / f"mcparams{day}.csv"
        try:
            df = pd.read_csv(csv_file_path)
            
            type_match_df = df[df['Type'].str.strip() == 'MERCURY']
            if type_match_df.empty:
                print(f"    -> 警告: CSVにTypeが'MERCURY'の行が見つかりません。スキップします。")
                continue

            # 条件を満たす最初の行を取得
            obs_row = type_match_df.iloc[0]

            # ↓↓↓ CSVのヘッダー名に合わせて、以下のキーを修正してください ↓↓↓
            Rms = obs_row['mercury_sun_distance_au']  # 太陽心距離
            beta = obs_row['ecliptic_latitude_deg']  # 黄緯
            elon = obs_row['ecliptic_longitude_deg']  # 黄経

        except (FileNotFoundError, KeyError, IndexError) as e:
            print(
                f"    -> 警告: {csv_file_path.name} から {day} のパラメータを読み込めませんでした ({e})。スキップします。")
            continue

        # --- 2c. 物理量を計算 ---
        d2r = np.pi / 180.0

        # 位相角補正係数を計算
        pa_correction_factor = np.polyval(pa_correction_coeffs, PA)

        # 太陽光散乱の幾何学的要因
        sf = np.pi / (np.pi - PA * d2r)

        # 黄道面からの高さ
        h = Rms * np.sin(beta * d2r)

        # 補正後の原子数密度と誤差
        corrected_atoms = avg_atoms / pa_correction_factor
        corrected_err = integrated_err / pa_correction_factor

        # --- 2d. 結果をリストに追加 ---
        # IDL版の出力フォーマットに合わせて7つの値を格納
        result_line = [TAA, h, Rms, elon, beta, corrected_atoms, corrected_err]
        final_results_list.append(result_line)
        print(f"    -> 処理完了: TAA={TAA:.2f}, 補正後原子数={corrected_atoms:.3e}")

        taa_for_plot.append(TAA)
        correction_factors_for_plot.append(pa_correction_factor)

    # --- 3. 最終結果をファイルに書き出し ---
    if final_results_list:
        # NumPy配列に変換して保存
        final_data = np.array(final_results_list)
        # IDLのフォーマット '(2f15.5,5e15.6)' に合わせてフォーマット文字列を作成
        fmt_string = ['%15.5f'] * 2 + ['%15.6e'] * 5

        np.savetxt(output_filename, final_data, fmt=fmt_string,
                   header="      TAA              h            Rms           elon           elat      corrected_atoms    corrected_err")

        print(f"\n全ての処理が完了しました。最終結果を {output_filename.name} に保存しました。")
    else:
        print("\n処理対象のデータが見つからなかったため、出力ファイルは作成されませんでした。")

    print("\nTAAごとの補正係数をプロットしています...")

    try:
        plt.figure(figsize=(10, 6))
        # 散布図を作成
        plt.scatter(taa_for_plot, correction_factors_for_plot, c='blue', marker='o',
                    label='Calculated Correction Factor')

        # グラフのタイトルとラベルを設定
        plt.title('Correction Factor vs. TAA')
        plt.xlabel('TAA (deg)')
        plt.ylabel('PA Correction Factor')
        plt.grid(True)  # グリッドを表示
        plt.legend()

        # プロットをファイルに保存
        plot_filename = base_dir / 'output' / 'Correction_Factor_vs_TAA.png'
        plt.savefig(plot_filename)
        print(f"プロットを {plot_filename.name} として保存しました。")

    except Exception as e:
        print(f"プロットの作成中にエラーが発生しました: {e}")

    print(f"\n全ての処理が完了しました。")

else:
    print("\n処理対象のデータが見つからなかったため、出力ファイルは作成されませんでした。")