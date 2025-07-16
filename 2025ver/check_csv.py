import pandas as pd

# ★★★ ご自身のCSVファイル名に合わせてください ★★★
csv_file_path = "mcparams202505.csv"

try:
    # まずは普通に読み込んでみる
    df_test = pd.read_csv(csv_file_path)

    print("--- ファイル読み込み成功 ---")
    print("■ 検出された列名:")
    print(list(df_test.columns))

    print("\n■ 読み込まれたデータの先頭5行:")
    print(df_test.head())

except Exception as e:
    print(f"エラーが発生しました: {e}")