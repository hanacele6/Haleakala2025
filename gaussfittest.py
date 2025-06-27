import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- ファイルパスの設定 ---
fileF1 = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/output/test/")
idl_file = fileF1 / 'gaussfit_params_IDL.txt'
python_file = fileF1 / 'gaussfit_params_Python.txt'

# --- データの読み込み ---
try:
    # --- ここを修正 ---
    # IDLのファイルはスペース区切りなので、delimiterの指定を削除（デフォルトの空白区切りで読み込む）
    params_idl = np.loadtxt(idl_file, delimiter=',')

    # Pythonのファイルはカンマ区切りなので、こちらは delimiter=',' のままでOK
    params_py = np.loadtxt(python_file, delimiter=',')

except ValueError as e:
    print(f"エラー: ファイルの読み込みまたはデータ変換に失敗しました。")
    print(f"ファイルの内容と区切り文字が正しいか確認してください。")
    print(f"詳細: {e}")
    # プログラムを終了
    exit()
except FileNotFoundError as e:
    print(f"エラー: ファイルが見つかりません。パスを確認してください。\n{e}")
    exit()

# --- パラメータごとに差を計算 ---
# IDLの列: 0=ifib, 1=amplitude, 2=mean, 3=stddev, 4=offset
# Pythonの列も同じ順番で保存したので、対応は取れている
diff_amplitude = params_py[:, 1] - params_idl[:, 1]
diff_mean = params_py[:, 2] - params_idl[:, 2]  # 最も重要なパラメータ
diff_stddev = params_py[:, 3] - params_idl[:, 3]
diff_offset = params_py[:, 4] - params_idl[:, 4]
fiber_indices = params_py[:, 0]

# --- 差の統計情報を表示 ---
print("--- ガウス中心位置(mean)の差の統計 ---")
print(f"平均値: {np.mean(diff_mean):.6f}")
print(f"標準偏差: {np.std(diff_mean):.6f}")
print(f"最小値: {np.min(diff_mean):.6f}")
print(f"最大値: {np.max(diff_mean):.6f}")

# --- グラフで可視化 ---
plt.style.use('default')
plt.rcParams['font.family'] = 'Meiryo'
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(fiber_indices, diff_amplitude, 'o-', markersize=3)
plt.title('振幅 (Amplitude) の差')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(fiber_indices, diff_mean, 'o-', markersize=3, color='red')
plt.title('中心位置 (Mean) の差 (Python - IDL)')
plt.ylabel('差')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(fiber_indices, diff_stddev, 'o-', markersize=3)
plt.title('幅 (Stddev) の差')
plt.xlabel('ファイバー番号 (ifib)')
plt.ylabel('差')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(fiber_indices, diff_offset, 'o-', markersize=3)
plt.title('オフセット (Offset) の差')
plt.xlabel('ファイバー番号 (ifib)')
plt.grid(True)

plt.suptitle('ガウスフィット パラメータの差 (Python - IDL)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

max_abs_diff_idx = np.argmax(np.abs(diff_mean))
worst_fiber_ifib = int(fiber_indices[max_abs_diff_idx])
print(f"\n[調査] 最も中心位置の差が大きいファイバー: ifib = {worst_fiber_ifib}")