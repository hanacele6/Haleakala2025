import pandas as pd
import matplotlib.pyplot as plt
import os


# 1. 比較したいCSVファイルのファイル名を設定
RESULTS_DIR = 'Analysis_Results'

# お手元のファイル名に書き換えてください
file1 = 'dusk.csv'  # 6列あるCSVファイル
#model_filename = 'dusk_model_output.csv'  # 2列あるCSVファイル
model_filename = 'column_density_Duskside_from_1.0-3.0RM.csv'
#model_filename = 'column_density_Dayside_from_1.0-3.0RM.csv'

file2 = os.path.join(RESULTS_DIR, model_filename)

try:
    # ヘッダーがないCSVとしてファイルを読み込む
    df1 = pd.read_csv(file1, encoding='shift_jis')
    df2 = pd.read_csv(file2, encoding='shift_jis')

    # 2. グラフ用のデータを抽出
    # df1から4行目(インデックス3)をx軸、6行目(インデックス5)をy軸として抽出
    x1 = df1.iloc[:, 3]
    y1 = df1.iloc[:, 4]

    # df2から1列目(インデックス0)をx軸、2列目(インデックス1)をy軸として抽出
    x2 = df2.iloc[:, 0]
    y2 = df2.iloc[:, 1]

    # 3. グラフの描画
    plt.figure(figsize=(10, 6))  # グラフのサイズを設定

    # 1つ目のCSVのデータをプロット (点と線で表示)
    plt.plot(x1, y1, marker='o', linestyle='none', label='observation')

    # 2つ目のCSVのデータをプロット (点と線で表示)
    plt.plot(x2, y2, marker='o', linestyle='none', label='model')

    # グラフのタイトルやラベルを設定
    plt.title('obs_vs_model', fontsize=16)
    plt.xlabel('TAA[°]', fontsize=12)
    plt.ylabel('Na_Column_Density [atoms/$cm^2$]', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.legend()  # 凡例を表示
    plt.grid(True)  # グリッド線を表示

    # 4. グラフを画像ファイルとして保存
    plt.savefig('obs_vs_model.png')

    print("グラフを保存しました。")

    plt.show()

except FileNotFoundError as e:
    print(f"エラー: ファイルが見つかりません。'{e.filename}' という名前のファイルがあるか確認してください。")
except IndexError:
    print("エラー: 指定された行または列が存在しません。CSVファイルの形式（6列、2列）と内容を確認してください。")
except Exception as e:
    print(f"予期せぬエラーが発生しました: {e}")