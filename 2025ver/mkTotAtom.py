import numpy as np
import os

# 基本的な変数の設定
day = 'test'
# Pythonでは、パスの区切り文字にバックスラッシュを使う場合、
# エスケープシーケンスと解釈されるのを防ぐために 'r' を先頭に付けます。
file_path = r'C:\Users\hanac\University\Senior\Mercury\Haleakala2025'
file_dir = os.path.join(file_path, 'output', day)

# IDLコードの N=4-1+1 に相当
N = 4

# 定数の設定
AU = 0.365763  # Rms
PA = 90.624233  # 位相角 (phase angle)
Rp = 0.307502  # 近日点距離 (perihelion)
Ra = 0.466697  # 遠日点距離 (aphelion)

# --- 計算 ---

# 真近点角 (True Anomaly Angle) の計算
# IDLの計算式: TAA = acos(-(AU*(Rp+Ra)-2*Rp*Ra)/(AU*(Ra-Rp)))*180/pi
numerator = -(AU * (Rp + Ra) - 2 * Rp * Ra)
denominator = AU * (Ra - Rp)
# np.arccosはラジアンを返すため、度に変換
TAA = np.arccos(numerator / denominator) * 180 / np.pi

# --- ファイル処理 ---

# 計算用の変数を初期化
b_sum = 0.0
err0 = []  # IDLのerr0配列に相当
err2_sum = 0.0

# 入出力ファイル名の定義
input_filename = os.path.join(file_dir, 'Na_atoms2_python.dat')
output_filename = os.path.join(file_dir, f'{day}num2_python.txt')

try:
    # ファイルを読み込み、データを処理
    # 'with' を使うことで、ファイルが自動的に閉じられます
    with open(input_filename, 'r') as f_in:
        # N行だけ読み込む
        for i in range(N):
            line = f_in.readline()
            if not line:
                print(f"警告: ファイル '{input_filename}' には {N} 行未満しかありません。")
                N = i  # 実際に読み込んだ行数にNを更新
                break

            # 行をスペースで分割し、浮動小数点数に変換
            # t0, a0, a1 に相当する値を取得
            parts = list(map(float, line.strip().split()))

            a0 = parts[1]
            a1 = parts[2]

            # IDLのループ内計算
            b_sum += a0
            err0.append(a1)
            err2_sum += a1 ** 2

    # --- 最終計算 ---

    # bの平均値を計算
    b = b_sum / N

    # エラーを計算
    # IDLの計算式: err = sqrt(err2)/N
    err = np.sqrt(err2_sum) / N

    # --- ファイル書き込み ---

    # 計算結果を指定されたファイルに書き込む
    with open(output_filename, 'w') as f_out:
        # 各変数をスペース区切りで書き込む
        f_out.write(f"{PA} {TAA} {b} {err}\n")

    print('end')

except FileNotFoundError:
    print(f"エラー: 入力ファイル '{input_filename}' が見つかりません。")
except Exception as e:
    print(f"予期せぬエラーが発生しました: {e}")