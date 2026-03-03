import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# CSVファイルのパスを指定 (日付は適宜変更してください)
csv_file = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/2026ver/csvs/mcparams20250819.csv")

if csv_file.exists():
    df = pd.read_csv(csv_file)
    target_df = df[df['Type'] == 'MERCURY']

    if 'g_factor' in target_df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(target_df.index, target_df['g_factor'], 'o-', label='g-factor')
        plt.xlabel('Image Index')
        plt.ylabel('g-factor')
        plt.title(f'g-factor Variation: {csv_file.name}')
        plt.grid(True)
        plt.legend()
        plt.show()

        print("g-factor 統計:")
        print(target_df['g_factor'].describe())
    else:
        print("エラー: g_factor列が見つかりません。Step 11 を実行してください。")
else:
    print("エラー: CSVファイルが見つかりません。")