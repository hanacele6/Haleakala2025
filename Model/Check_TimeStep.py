import numpy as np
import os
import sys

# --- 定数 ---
# 水星の1年の秒数 (シミュレーションコードと同一)
MERCURY_YEAR_SEC = 87.97 * 24 * 3600
# 軌道データファイル名
#ORBIT_FILE = 'orbit360.txt'
ORBIT_FILE = 'orbit2025_v5.txt'


def preprocess_orbit_data(filename, mercury_year_sec):
    """
    軌道データファイルを読み込み、時間軸のラップアラウンドを処理する。

    ★重要: この関数は、'orbit360.txt' の
    1列目 (インデックス 0) が 時間 [sec]
    2列目 (インデックス 1) が TAA [degree]
    であることを前提としています。
    """
    try:
        data = np.loadtxt(filename)
    except FileNotFoundError:
        print(f"エラー: {filename} が見つかりません。")
        sys.exit()

    # 1列目（時間）が0から始まっていることを確認
    if data[0, 0] != 0:
        print("警告: 軌道データの1列目（時間）が0から始まっていません。")

    # 1年の最後と次の年の最初を繋げるための「ダミーデータ」を追加
    # 最後の行のデータをコピー
    last_row = data[-1, :].copy()
    # 時間に1年分を足す
    last_row[0] += mercury_year_sec
    # TAAは変更しない（359度 -> 360(0)度への補間用）

    # データを結合
    processed_data = np.vstack((data, last_row))

    # シミュレーションコードは orbit_data[:, 1] を TAA として使っているので...
    # [time, TAA, AU, ..., Vms] の並びにする必要があります。

    # シミュレーションコードのインデックスと合わせる
    # sim_code: orbit_data[:, 1] -> TAA
    # sim_code: orbit_data[:, 2] -> AU
    # sim_code: orbit_data[:, 5] -> Vms

    # この前提が正しいか確認が必要です。
    # もし orbit360.txt が [TAA, AU, Vms, ...] で「時間」を含まない場合、
    # 楕円軌道のシミュレーションは不可能です。

    # ★★★ 一旦、元のコードのロジックに戻します ★★★
    #
    # `orbit360.txt` が [TAA, AU, ..., Vms] という列を持っており、
    # 「時間」列を含まないと仮定します。
    #
    data = np.loadtxt(filename)

    # これが「円軌道」シミュレーションの原因です
    time_axis = np.linspace(0, mercury_year_sec, len(data), endpoint=False)

    # time_axis [0], data [1, 2, 3, 4, 5]
    # -> [time, data[0], data[1], data[2], data[3], data[4]]
    # orbit_data[:, 0] = time
    # orbit_data[:, 1] = data[0] (TAA)
    # orbit_data[:, 2] = data[1] (AU)
    # orbit_data[:, 5] = data[4] (Vms)
    return np.column_stack((time_axis, data))


def calculate_taa_speed():
    """
    TAA 1度ごとに、そこを通過するのにかかる時間（秒）を計算して表示する。
    """

    # --- 1. 必須ファイルの確認 ---
    if not os.path.exists(ORBIT_FILE):
        print(f"エラー: 必須ファイル '{ORBIT_FILE}' が見つかりません。")
        print("シミュレーションコードと同じディレクトリで実行してください。")
        sys.exit()

    # --- 2. 軌道データの読み込みと前処理 ---
    try:
        # orbit_data[:, 0] = time [sec]
        # orbit_data[:, 1] = TAA [degree]
        # (以降の列は AU, Vms など)
        orbit_data = preprocess_orbit_data(ORBIT_FILE, MERCURY_YEAR_SEC)
    except Exception as e:
        print(f"エラー: '{ORBIT_FILE}' の読み込みに失敗しました。{e}")
        sys.exit()

    print(f"'{ORBIT_FILE}' を読み込みました。{len(orbit_data)} 行のデータがあります。")

    # --- 3. TAAと時間の差分を計算 ---

    # np.diff() は (N)個の配列から (N-1)個の差分配列を計算する
    # 連続するデータポイント間の時間差 [sec]
    dt = np.diff(orbit_data[:, 0])

    # 連続するデータポイント間のTAA差 [degree]
    dTAA = np.diff(orbit_data[:, 1])

    # --- 4. TAAのラップアラウンド（359 -> 0）を処理 ---
    # (例: 0.1 - 359.9 = -359.8 となるのを +0.2 に補正)
    dTAA[dTAA < -180] += 360

    # 0除算や負の時間を避ける
    valid_steps = dTAA > 1e-6  # 非常に小さい変化や逆行を除外

    if not np.any(valid_steps):
        print("エラー: TAAが変化する有効なステップが見つかりません。")
        sys.exit()

    # --- 5. TAA 1度あたりの秒数を計算 ---
    # (sec / degree)
    # dt / dTAA で、1度進むのに何秒かかるかが求まる
    sec_per_degree = dt[valid_steps] / dTAA[valid_steps]

    # この速度データが、どのTAAに対応するかを設定
    # 各インターバルの開始点のTAAを使用する。
    taa_values = orbit_data[:-1, 1][valid_steps]

    # --- 6. TAA 0度から359度までを1度刻みで補間 ---

    # 補間するターゲットTAA (0, 1, 2, ..., 359)
    target_taas = np.arange(0, 360, 1)

    # np.interpで線形補間
    # 'orbit360.txt' のTAA列（2列目）が単調増加（0->360）している必要がある
    interp_sec_per_degree = np.interp(target_taas, taa_values, sec_per_degree, period=360)

    # --- 7. 結果の表示 ---

    # 秒を時間に変換するヘルパー関数
    def sec_to_hms(seconds):
        seconds = abs(seconds)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}時間 {minutes}分 {secs}秒"

    print("\n--- TAA 1度あたりに進むのにかかる時間（秒）の計算結果 ---")
    print("TAA [度] | 1度進む時間 [秒] | (参考: 時/分/秒)")
    print("-" * 60)

    total_sec = 0
    for taa, sec in zip(target_taas, interp_sec_per_degree):
        # 1時間 (3600秒) より速いステップに印をつける
        marker = "★" if sec < 3600 else ""
        print(f" {taa:3d} -> {taa + 1:3d} | {sec:16.2f} | ({sec_to_hms(sec)}) {marker}")
        total_sec += sec

    print("-" * 60)
    avg_sec = np.mean(interp_sec_per_degree)
    print(f"  平均時間: {avg_sec:16.2f} | ({sec_to_hms(avg_sec)})")
    print(f"  合計時間: {total_sec:16.2f} | ({sec_to_hms(total_sec)})")
    print(f"  (参考) 1年: {MERCURY_YEAR_SEC:16.2f} | ({sec_to_hms(MERCURY_YEAR_SEC)})")

    min_sec = np.min(interp_sec_per_degree)
    min_taa = target_taas[np.argmin(interp_sec_per_degree)]
    max_sec = np.max(interp_sec_per_degree)
    max_taa = target_taas[np.argmax(interp_sec_per_degree)]

    print("\n--- 最も速い/遅い ポイント ---")
    print(f"  ★ 最も速い: TAA {min_taa}度 付近 (1度あたり約 {min_sec:.2f} 秒)")
    print(f"  ・ 最も遅い: TAA {max_taa}度 付近 (1度あたり約 {max_sec:.2f} 秒)")
    print(f"\n  (1時間 = 3600秒。 ★マークは1時間以内に1度以上進む要注意区間)")


if __name__ == '__main__':
    calculate_taa_speed()