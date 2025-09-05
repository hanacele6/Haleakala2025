import numpy as np
import pandas as pd
from pathlib import Path
import sys

# ==============================================================================
# スクリプトの実行部
# ==============================================================================
if __name__ == '__main__':
    # --- 基本設定 ---
    day = "20250824"
    base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
    data_dir = base_dir / "output" / day
    csv_file_path = base_dir / "2025ver" / f"mcparams{day}.csv"

    # --- 入力ファイルと出力ファイル ---
    # 前のスクリプトが生成したファイルを入力とする
    input_filename = data_dir / 'Na_atoms_final.dat'
    output_filename = data_dir / f'Final_Summary_{day}.txt'

    print(f"--- 最終結果の集計を開始します ---")
    print(f"入力ファイル: {input_filename.name}")

    # --- 1. 天文学的パラメータをCSVから読み込む ---
    try:
        df = pd.read_csv(csv_file_path)
        # その日の最初の観測データ行を取得
        first_obs_row = df[df['Type'] == 'MERCURY'].iloc[0]

        # ↓↓↓ CSVのヘッダー名に合わせて、以下のキーを修正してください ↓↓↓
        PA = first_obs_row['phase_angle_deg']  # 位相角 (Phase Angle)
        AU = first_obs_row['mercury_sun_distance_au']  # 太陽心距離 (Heliocentric Distance)
        Rp = 0.307502  # 近日点距離 (perihelion)
        Ra = 0.466697  # 遠日点距離 (aphelion)

        print(f"CSVからパラメータを読み込みました: PA={PA:.4f}, AU={AU:.4f}")

    except FileNotFoundError:
        print(f"エラー: CSVファイルが見つかりません: {csv_file_path}")
        sys.exit()
    except (KeyError, IndexError) as e:
        print(f"エラー: CSVファイルからパラメータを読み込めませんでした。列名を確認してください: {e}")
        sys.exit()

    # --- 2. 真近点角 (True Anomaly Angle) を計算 ---
    try:
        numerator = -(AU * (Rp + Ra) - 2 * Rp * Ra)
        denominator = AU * (Ra - Rp)
        # np.arccosはラジアンを返すため、度に変換
        TAA = np.arccos(numerator / denominator) * 180 / np.pi
        print(f"真近点角を計算しました: TAA={TAA:.4f}")
    except ZeroDivisionError:
        print("エラー: 真近点角の計算中にゼロ除算が発生しました。RpとRaの値を確認してください。")
        sys.exit()

    # --- 3. 原子数密度の結果ファイルを読み込み、集計 ---
    try:
        # ファイルを読み込む (列: Index, Na_Atoms, Error)
        data = np.loadtxt(input_filename)

        # データが1行しかない場合も正しく扱えるようにする
        if data.ndim == 1:
            data = data.reshape(1, -1)

        N = data.shape[0]  # 観測数を動的に取得
        print(f"{N} 個の観測データを読み込みました。")

        # 必要な列を抽出
        na_atoms_col = data[:, 1]
        error_col = data[:, 2]

        # na_atoms_col の値が0より大きいデータのみを対象とするためのマスクを作成
        non_zero_mask = na_atoms_col > 0

        # 0より大きいデータの数を取得
        N_non_zero = np.count_nonzero(non_zero_mask)

        # 0より大きいデータが1つ以上ある場合のみ計算を実行
        if N_non_zero > 0:
            print(f"有効なデータ数（0より大きい）: {N_non_zero}")

            # 0より大きいデータとその誤差のみを抽出
            na_atoms_non_zero = na_atoms_col[non_zero_mask]
            error_non_zero = error_col[non_zero_mask]

            # 合計値を計算
            b_sum = np.sum(na_atoms_non_zero)
            err2_sum = np.sum(error_non_zero ** 2)

            # 平均原子数密度と統合誤差を、0より大きいデータの数で割って計算
            b = b_sum / N_non_zero
            err = np.sqrt(err2_sum) / N_non_zero

        else:
            # 0より大きいデータがなかった場合、平均と誤差を0にする
            print("警告: 有効な（0より大きい）データがありませんでした。")
            b = 0.0
            err = 0.0

        print(f"平均原子数密度: {b:.4e}")
        print(f"統合誤差: {err:.4e}")

    except FileNotFoundError:
        print(f"エラー: 入力ファイルが見つかりません: {input_filename}")
        sys.exit()
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        sys.exit()

    # --- 4. 最終結果をファイルに書き込み ---
    try:
        with open(output_filename, 'w') as f_out:
            # 各変数をスペース区切りで書き込む
            f_out.write(f"{PA} {TAA} {b} {err}\n")
        print(f"\n処理が完了しました。最終結果を {output_filename.name} に保存しました。")

    except Exception as e:
        print(f"エラー: ファイルの書き込みに失敗しました: {e}")

