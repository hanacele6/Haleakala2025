import numpy as np
from astropy.io import fits
from pathlib import Path
from scipy.ndimage import median_filter  # 背景除去のためにインポート


def create_2d_trace_map_final(input_filepath, output_filepath, config):
    """
    背景除去機能を追加し、不均一な背景を持つデータから
    ファイバーの位置を特定し、2Dトレースマップを作成する。
    """
    # --- 1. FITS読み込み ---
    try:
        with fits.open(input_filepath) as hdul:
            image_data = hdul[0].data.astype(np.float64)
            header = hdul[0].header
        print(f"--- 2Dトレースマップ作成開始 (背景除去機能付き): {input_filepath.name} ---")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {input_filepath}")
        return

    # ===================================================================
    # --- ステップ1: 基準となるファイバーのY座標を特定する ---
    # ===================================================================
    print("\n--- ステップ1: 基準ファイバー位置（Y座標）を特定します ---")

    avg_x_start, avg_x_end = config['reference_avg_range_x']
    spatial_profile = np.mean(image_data[:, avg_x_start:avg_x_end], axis=1)

    # ★★★【今回の改善点】背景成分の推定と除去 ★★★
    # ファイバーの幅(4)や間隔(1)よりずっと大きいウィンドウでメディアンフィルタをかける
    background_profile = median_filter(spatial_profile, size=config['background_filter_size'])
    # 元のプロファイルから背景プロファイルを引く
    subtracted_profile = spatial_profile - background_profile
    print("  > 背景成分を推定し、除去しました。")

    # --- 平坦化されたプロファイルからピークを探す ---
    # パラメータ取得
    window_cfg = config['peak_find_window']
    sep_thresh = config['fiber_separation_threshold']
    # 新しい閾値: 背景からの「突出度」
    height_thresh = config['peak_height_above_background']

    found_peaks_y = []
    last_peak_y = -sep_thresh - 1
    # ピーク探索を `subtracted_profile` に対して行う
    for iy in range(len(subtracted_profile) - window_cfg['total_size'] + 1):
        window_slice = subtracted_profile[iy: iy + window_cfg['total_size']]

        # 単純なピーク検出ロジック (より洗練させることも可能)
        if (window_slice.max() > height_thresh and
                (iy - last_peak_y) > sep_thresh):

            # ピーク位置を特定
            peak_offset = np.argmax(window_slice)
            peak_y_abs = iy + peak_offset

            # 同じピークを複数回検出しないためのチェック
            if abs(peak_y_abs - last_peak_y) > sep_thresh:
                found_peaks_y.append(peak_y_abs)
                last_peak_y = peak_y_abs

    found_peaks_y.sort()
    reference_fiber_positions = {fid + 1: y_coord for fid, y_coord in enumerate(found_peaks_y)}
    print(f"  > {len(reference_fiber_positions)} 本の基準ファイバー位置を特定しました。")

    # (ステップ2のコードは前回と同じなので省略)
    # ===================================================================
    # --- ステップ2: 各ファイバーを波長方向に追跡（トレース） ---
    # ===================================================================
    print("\n--- ステップ2: 各ファイバーを個別に追跡します ---")
    pp1_data = np.zeros_like(image_data, dtype=np.int16)
    # ... (前回のコードと同じ)
    trace_x_start, trace_x_end = config['trace_range_x']
    y_search_radius = config['trace_y_search_radius']
    trace_intensity_threshold = config['trace_intensity_threshold']

    for fiber_id, ref_y in reference_fiber_positions.items():
        for x_wav in range(trace_x_start, trace_x_end):
            y_min = max(0, ref_y - y_search_radius)
            y_max = min(image_data.shape[0], ref_y + y_search_radius + 1)
            search_strip = image_data[y_min:y_max, x_wav]
            if search_strip.max() > trace_intensity_threshold:
                peak_offset = np.argmax(search_strip)
                actual_y = y_min + peak_offset
                pp1_data[actual_y, x_wav] = fiber_id

    fits.writeto(output_filepath, pp1_data, header, overwrite=True)
    print(f"\n2Dトレースマップを {output_filepath.name} に保存しました。")


if __name__ == "__main__":
    # ... ファイルパス設定は省略 ...
    date = '20250501'
    base_dir = Path(f"C:/Users/hanac/University/Senior/Mercury/data/{date}")
    output_dir = Path(f"C:/Users/hanac/University/Senior/Mercury/Haleakala2025/output/{date}")
    input_file = output_dir / ' led01r_clf590n_ga7000fsp220_1_nhp_py.fits'
    output_file = output_dir / "pp1_trace_map.fits"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'reference_avg_range_x': (0, 1024),

        # --- ステップ1のパラメータ ---
        'peak_find_window': {'total_size': 4},
        # あなたが見つけた最適な値
        'fiber_separation_threshold': 4,

        # ★★★ 新しいパラメータ ★★★
        # 背景推定に使うメディアンフィルタの窓幅。ファイバー間の距離(5-12)より十分大きい奇数。
        'background_filter_size': 51,
        # 背景除去後のグラフで、「背景からどれだけ突出していればピークとみなすか」という閾値。
        # 小さい値でOKなはず。
        'peak_height_above_background': 500,  # まずはこの値で試し、検出本数を見ながら微調整する

        # --- ステップ2のパラメータ ---
        'trace_range_x': (0, 1024),
        'trace_y_search_radius': 5,
        # この閾値も、背景レベルではなく「真の信号の強さ」を反映する値にできる
        'trace_intensity_threshold': 2500,
    }

    # --- 実行 ---
    create_2d_trace_map_final(input_file, output_file, config)