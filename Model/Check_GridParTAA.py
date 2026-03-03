# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 設定
# ==============================================================================
N_LON_FIXED = 500
N_LAT = 60
ORBIT_FILE = 'orbit2025_spice.txt'


def load_orbit_data_smart(filepath):
    """
    軌道データを読み込み、TAAが0度付近から始まるようにデータをスライスして整形する
    """
    try:
        # 生データを読み込む
        raw_data = np.loadtxt(filepath)

        # TAA列 (col 0)
        taa_raw = raw_data[:, 0]

        # 【修正ポイント】
        # ファイルの冒頭が "359..." のような年末データから始まっている場合、
        # unwrapすると全体が360度ズレてしまうため、
        # 最初の100行の中で「値が最小になる行（＝新年の始まり）」を探してそこからスタートする。
        start_idx = np.argmin(taa_raw[:100])
        print(f"データ開始位置を調整: 行番号 {start_idx} (TAA={taa_raw[start_idx]:.2f} deg) から使用します。")

        # データをスライス
        data = raw_data[start_idx:, :]

        # アンラップ処理 (0, 360, 720... と連続化)
        # col 0: TAA, col 5: SubSolarLon
        # ※ここで deg -> rad -> unwrap -> deg に戻す
        data[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(data[:, 0])))
        data[:, 5] = np.rad2deg(np.unwrap(np.deg2rad(data[:, 5])))

        return data
    except Exception as e:
        print(f"Error: {e}")
        return None


def get_subsolar_lon_at_taa(target_taa, orbit_data):
    """ 指定したTAAにおけるSubSolarLon(rad)を返す """
    taa_col = orbit_data[:, 0]  # 補正済みTAA (0.6 -> 360...)
    time_col = orbit_data[:, 2]  # Time
    sub_lon_col = orbit_data[:, 5]  # SubSolarLon

    # ターゲットTAAがデータの範囲内かチェック
    # データが 0.6度から始まっている場合、TAA=0 を要求すると範囲外になるため、
    # データの最小値より小さいTAAが来た場合は、最小値と同じとみなす（または+360して次年度を参照する手もあるが簡易的に）
    if target_taa < taa_col[0]:
        target_taa = taa_col[0]

    # TAA -> Time
    target_time = np.interp(target_taa, taa_col, time_col)
    # Time -> SubSolarLon
    target_sub_lon_deg = np.interp(target_time, time_col, sub_lon_col)

    return np.deg2rad(target_sub_lon_deg)


def main():
    # 1. グリッドの準備
    lon_edges = np.linspace(-np.pi, np.pi, N_LON_FIXED + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers, indexing='ij')

    # 2. 軌道データ読み込み (改良版関数を使用)
    orbit_data = load_orbit_data_smart(ORBIT_FILE)
    if orbit_data is None: return

    # TAA 0度から360度までループ
    taa_steps = np.arange(0, 361, 1)
    crossing_counts = []

    print("\n計算中... (TAA 0 -> 360)")

    # 初期状態
    sub_lon_prev = get_subsolar_lon_at_taa(taa_steps[0], orbit_data)
    cos_theta_prev = np.cos(lat_grid) * np.cos(lon_grid - sub_lon_prev)
    is_day_prev = cos_theta_prev > 0

    for i in range(1, len(taa_steps)):
        current_taa = taa_steps[i]
        sub_lon_curr = get_subsolar_lon_at_taa(current_taa, orbit_data)

        # 昼夜判定
        cos_theta_curr = np.cos(lat_grid) * np.cos(lon_grid - sub_lon_curr)
        is_day_curr = cos_theta_curr > 0

        # 変化カウント
        changed_mask = np.logical_xor(is_day_prev, is_day_curr)
        count = np.sum(changed_mask)
        crossing_counts.append(count)

        # 更新
        is_day_prev = is_day_curr
        sub_lon_prev = sub_lon_curr

    # ==============================================================================
    # 結果の可視化
    # ==============================================================================
    total_cells = N_LON_FIXED * N_LAT

    plt.figure(figsize=(10, 6))
    plt.plot(taa_steps[1:], crossing_counts, marker='o', markersize=3, label='Changed Cells')

    # 最大・最小
    max_idx = np.argmax(crossing_counts)
    max_val = crossing_counts[max_idx]
    max_taa = taa_steps[1:][max_idx]

    plt.title(f"Terminator Crossing (Grid: {N_LON_FIXED}x{N_LAT}={total_cells})")
    plt.xlabel("True Anomaly Angle (deg)")
    plt.ylabel("Number of Flipping Cells / 1 deg")
    plt.grid(True, alpha=0.5)
    plt.legend()

    msg = (f"Max Change: {max_val} cells @ TAA={max_taa} deg\n"
           f"Total Grid: {total_cells}")
    plt.text(0.02, 0.95, msg, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8), va='top')

    plt.tight_layout()
    plt.show()

    # ログ出力
    print(f"\n[Result] 最大変化数: {max_val} (TAA={max_taa}付近)")
    print("データの一部:")
    for k in range(0, 10):
        print(f"TAA {taa_steps[k]}->{taa_steps[k + 1]}: {crossing_counts[k]} cells")


if __name__ == "__main__":
    main()