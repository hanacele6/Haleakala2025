import numpy as np
from astropy.io import fits
from pathlib import Path


def create_2d_trace_map(input_filepath, output_filepath, config):
    """
    2D画像のまま、各ファイバーを波長方向に追跡し、
    その軌跡を記録したFITSファイルを作成する。
    """
    # --- 1. FITS読み込みと設定準備 ---
    try:
        with fits.open(input_filepath) as hdul:
            image_data = hdul[0].data.astype(np.float64)
            header = hdul[0].header
        print(f"--- 2Dトレースマップ作成開始: {input_filepath.name} ---")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {input_filepath}")
        return

    # ===================================================================
    # --- ステップ1: 基準となるファイバーのY座標を特定する ---
    # ===================================================================
    print("\n--- ステップ1: 基準ファイバー位置（Y座標）を特定します ---")

    # 信号の強い波長範囲で平均化し、安定した1D空間プロファイルを作成
    avg_x_start, avg_x_end = config['reference_avg_range_x']
    spatial_profile = np.mean(image_data[:, avg_x_start:avg_x_end], axis=1)

    # 1Dプロファイルからピーク(ファイバー)を探す
    window_cfg = config['peak_find_window']
    # ... (以前の1Dピーク探索コードと同じロジック) ...
    found_peaks_y = []
    last_peak_y = -config['fiber_separation_threshold'] - 1
    for iy in range(len(spatial_profile) - window_cfg['total_size'] + 1):
        window_slice = spatial_profile[iy: iy + window_cfg['total_size']]
        # ... (平坦なピークを探すロジック) ...
        # ここでは簡略化のため、単純なピーク検出にします
        # (以前のウィンドウロジックをここに戻しても良い)
        if (np.argmax(window_slice) == window_cfg['total_size'] // 2 and
                window_slice.max() > config['reference_peak_threshold'] and
                (iy - last_peak_y) > config['fiber_separation_threshold']):
            peak_y_abs = iy + window_cfg['total_size'] // 2
            found_peaks_y.append(peak_y_abs)
            last_peak_y = peak_y_abs

    found_peaks_y.sort()
    reference_fiber_positions = {fid + 1: y_coord for fid, y_coord in enumerate(found_peaks_y)}
    print(f"  > {len(reference_fiber_positions)} 本の基準ファイバー位置を特定しました。")

    # ===================================================================
    # --- ステップ2: 各ファイバーを波長方向に追跡（トレース） ---
    # ===================================================================
    print("\n--- ステップ2: 各ファイバーを個別に追跡します ---")

    pp1_data = np.zeros_like(image_data, dtype=np.int16)
    trace_x_start, trace_x_end = config['trace_range_x']
    y_search_radius = config['trace_y_search_radius']
    trace_intensity_threshold = config['trace_intensity_threshold']

    # 特定したファイバーを1本ずつループ
    for fiber_id, ref_y in reference_fiber_positions.items():
        if fiber_id % 20 == 0:
            print(f"  > ファイバー {fiber_id}/{len(reference_fiber_positions)} を追跡中...")

        # 波長方向に追跡ループ
        for x_wav in range(trace_x_start, trace_x_end):
            # 基準Y座標の周辺だけを切り出す
            y_min = max(0, ref_y - y_search_radius)
            y_max = min(image_data.shape[0], ref_y + y_search_radius + 1)
            search_strip = image_data[y_min:y_max, x_wav]

            # この狭い範囲で、閾値を超えるピークがあるか確認
            if search_strip.max() > trace_intensity_threshold:
                # ピークがあれば、その正確なY座標を見つけてファイバー番号を書き込む
                peak_offset = np.argmax(search_strip)
                actual_y = y_min + peak_offset
                pp1_data[actual_y, x_wav] = fiber_id

    # --- 結果の保存 ---
    fits.writeto(output_filepath, pp1_data, header, overwrite=True)
    print(f"\n2Dトレースマップを {output_filepath.name} に保存しました。")
    print("DS9などのビューアで開き、線が途中で消えているファイバーを探してください。")


if __name__ == "__main__":
    # 1. ファイルパス
    date = '20250501'
    base_dir = Path(f"C:/Users/hanac/University/Senior/Mercury/data/{date}")
    output_dir = Path(f"C:/Users/hanac/University/Senior/Mercury/Haleakala2025/output/{date}")
    input_file = output_dir / ' led01r_clf590n_ga7000fsp220_1_nhp_py.fits'
    output_file = output_dir / "pp1_trace_map.fits"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. 解析設定
    config = {
        # ステップ1で使用: 基準位置を決めるための平均化範囲 (信号が強く安定しているXの範囲)
        'reference_avg_range_x': (0, 1024),
        # ステップ1で使用: 基準位置を探すためのピーク検出パラメータ
        'peak_find_window': {'total_size': 4},  # ファイバーのおおよその幅
        'fiber_separation_threshold': 1,  # Y方向のファイバー分離距離
        'reference_peak_threshold': 10000,  # 1Dプロファイルでのピーク検出閾値 (要調整)

        # ステップ2で使用: ファイバーを追跡するパラメータ
        'trace_range_x': (0, 1024),  # 追跡する波長の全範囲
        'trace_y_search_radius': 3,  # 各ファイバーの基準Y座標から上下何ピクセルを探すか
        'trace_intensity_threshold': 5000,  # 追跡中に「ピーク」とみなすための輝度閾値 (背景値より十分高い値, 要調整)
    }

    # --- 実行 ---
    create_2d_trace_map(input_file, output_file, config)
