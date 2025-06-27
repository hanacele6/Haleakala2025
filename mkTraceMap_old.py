import numpy as np
from astropy.io import fits
from pathlib import Path


def create_peak_map_from_idl_logic(config):
    """
    IDLの `trace1` のロジックを、座標系の違いを考慮してPythonで忠実に再現する。
    """
    # --- 1. 設定とファイルの準備 ---
    input_file = config['input_file']
    output_dir = config['output_dir']
    output_file = config['output_file']
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. FITSファイルの読み込み ---
    try:
        with fits.open(input_file) as hdul:
            # IDLの double(readfits()) に相当。データ型をfloat64に変換。
            # astropyは自動的に (Y, X) のNumPy配列として読み込む。
            image_data = hdul[0].data.astype(np.float64)
            header = hdul[0].header
        print(f"--- 処理開始: {input_file.name} ---")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {input_file}")
        return

    # NumPy配列の形状を取得 (ny: Y方向のサイズ, nx: X方向のサイズ)
    # IDLの iym, ixm にそれぞれ対応
    ny, nx = image_data.shape
    print(f"  > 画像サイズ: Y={ny}, X={nx}")

    # --- 3. オリジナルデータのコピーを保存 (flat1.fits) ---
    flat1_path = output_dir / 'flat1.fits'
    fits.writeto(flat1_path, image_data, header, overwrite=True)
    print(f"  > オリジナルデータのコピーを {flat1_path.name} に保存しました。")

    # --- 4. 閾値の計算 ---
    # ★修正点: IDLの a[X,Y] に合わせ、Python/NumPyでは [Y,X] の順でアクセス
    y_idx, x_idx = config['threshold_ref_pixel_yx']
    threshold = image_data[y_idx, x_idx] + config['threshold_offset']
    print(f"  > 閾値ピクセル (Y={y_idx}, X={x_idx}) の値から閾値を計算: {threshold:.2f}")

    # --- 5. ピーク検出用のデータ準備 (マスク処理) ---
    proc_image = image_data.copy()

    # デッドファイバー領域をマスク (値を0にする)
    # IDLの a[X1:X2, *] = 0 は、特定の「列」を0にすることに相当
    if config['dead_fiber_regions_x']:
        print(f"  > {len(config['dead_fiber_regions_x'])}個のデッドファイバー領域(X方向)をマスクします。")
        for x_start, x_end in config['dead_fiber_regions_x']:
            # Pythonでは [:, X1:X2] で列(X方向)を選択
            proc_image[:, x_start:x_end + 1] = 0

    # GOTO文に相当する領域をマスクするためのブール配列を作成
    goto_mask = np.zeros_like(proc_image, dtype=bool)
    if config['goto_mask_regions_xyxy']:
        print(f"  > {len(config['goto_mask_regions_xyxy'])}個のGOTOマスク領域を適用します。")
        for x1, y1, x2, y2 in config['goto_mask_regions_xyxy']:
            # ★修正点: IDLの `ge` `le` (以上/以下) を正しくスライスで表現
            # sliceの第2引数は含まれないため、+1 する必要がある
            y_slice = slice(y1, y2 + 1)
            x_slice = slice(x1, x2 + 1)
            goto_mask[y_slice, x_slice] = True

    # --- 6. ピーク検出ループ (IDLのロジックを忠実に再現) ---
    print("  > ピーク検出処理を開始します...")
    # IDLの b = intarr(ixm, iym) に相当。NumPyでは(Y, X)の順。
    peak_map = np.zeros_like(proc_image, dtype=np.int16)

    y_start, y_end = config['analysis_range_y']
    min_x_detect = config['min_x_for_peak_detection']
    fiber_sep = config['fiber_separation_threshold']

    # IDL: for iy = 0, iym-1
    for iy in range(y_start, y_end + 1):
        fiber_id = 1
        last_peak_x = -fiber_sep - 1  # 初期化条件を確実に満たすため

        # IDL: for ix = 0, ixm-10
        for ix in range(nx - 10):
            # GOTOマスク領域ならスキップ
            if goto_mask[iy, ix]:
                continue

            # 7ピクセルのウィンドウを取得
            window_7px = proc_image[iy, ix:ix + 7]

            # 4つのサブウィンドウの最大値が全て同じかチェック
            # IDLのロジックをNumPyで効率的に記述
            max_vals = [np.max(window_7px[k:k + 4]) for k in range(4)]

            # 浮動小数点数なので、完全一致ではなく差が小さいことで判定
            if (np.max(max_vals) - np.min(max_vals)) < 1e-9:
                peak_value = max_vals[0]

                # IDLのピーク検出条件
                if (peak_value >= threshold and
                        ix >= min_x_detect and
                        (ix - last_peak_x) > fiber_sep):

                    # ウィンドウ内で最大値を持つ最初の位置を見つける
                    # IDLの `pos=where(...)` `pos[0]` に相当
                    pos_in_window = np.argmax(window_7px)
                    peak_x_abs = ix + pos_in_window

                    # 検出したピーク位置自体がマスクされていないか最終確認
                    if not goto_mask[iy, peak_x_abs]:
                        peak_map[iy, peak_x_abs] = fiber_id
                        fiber_id += 1
                        last_peak_x = peak_x_abs

    num_found = np.max(peak_map)
    print(f"  > ピーク検出完了。1行あたり最大 {num_found} 本のファイバーを検出しました。")

    # --- 7. 結果をFITSファイルに保存 ---
    fits.writeto(output_file, peak_map, header, overwrite=True)
    print(f"--- 処理完了: ピークマップを {output_file.name} に保存しました ---")


if __name__ == "__main__":
    # ===================================================================
    # --- ユーザー設定 ---
    # 環境に合わせてパスを修正してください
    date = 'test'
    base_dir = Path(f"C:/Users/hanac/University/Senior/Mercury/data/{date}")
    input_file_path = base_dir / 'wlflat01-001_nhp.fits'
    output_dir_path = Path(f"C:/Users/hanac/University/Senior/Mercury/Haleakala2025/output/{date}")

    # IDLのGOTO文マスク領域 [X1, Y1, X2, Y2] (0-based index)
    # IDLコードの数値は0ベースのインデックスとして解釈
    GOTO_MASK_REGIONS = [
        [1474, 951, 1486, 1329], [1476, 431, 1487, 951], [1477, 0, 1488, 431],
        [1538, 1046, 1548, 1294], [1541, 827, 1549, 905], [1541, 0, 1549, 66],
        [1808, 1044, 1820, 1329], [1810, 611, 1824, 1044], [1811, 177, 1824, 611],
        [1809, 0, 1824, 177], [1759, 989, 1771, 1329], [1760, 579, 1772, 989],
        [1761, 180, 1772, 579], [1761, 0, 1772, 180], [1660, 1056, 1671, 1329],
        [1662, 639, 1673, 1056], [1663, 287, 1673, 639], [1663, 0, 1674, 287],
        [1735, 979, 1746, 1328], [1736, 560, 1747, 979], [1737, 151, 1747, 560],
        [1736, 0, 1746, 151], [1635, 999, 1646, 1329], [1637, 565, 1649, 999],
        [1637, 0, 1648, 565], [1574, 886, 1584, 1329], [1576, 0, 1586, 886],
        [1388, 919, 1400, 1328], [1390, 490, 1401, 919], [1391, 0, 1403, 490],
        [1387, 1329, 1397, 1410], [1475, 1329, 1483, 1410], [1539, 1297, 1548, 1306],
        [1539, 1001, 1549, 1026], [1571, 1329, 1582, 1410], [1633, 1329, 1642, 1410],
        [1660, 1329, 1670, 1410], [1757, 1329, 1767, 1410], [1733, 1329, 1743, 1410],
        [1540, 821, 1550, 831], [1609, 1104, 1620, 1410], [1611, 809, 1622, 1104],
        [1612, 0, 1624, 809], [1607, 1395, 1613, 1401], [485, 999, 496, 1410],
        [485, 639, 498, 999], [487, 341, 500, 648], [490, 0, 503, 341],
        [384, 0, 392, 46]
    ]

    # --- 解析設定 ---
    config = {
        'input_file': input_file_path,
        'output_dir': output_dir_path,
        'output_file': output_dir_path / 'pp1_py_corrected.fits',

        # 解析領域 (Y座標, 0-based, inclusive)
        'analysis_range_y': (0, 1410),

        # 閾値決定ピクセル (Y, X) 順で指定 (0-based)
        # IDLの a[458-1, 462-1] -> X=457, Y=461
        'threshold_ref_pixel_yx': (461, 457),
        'threshold_offset': 450.0,

        'min_x_for_peak_detection': 387,
        'fiber_separation_threshold': 4,

        # デッドファイバー領域 (X座標, 0-based, inclusive)
        'dead_fiber_regions_x': [
            (1866, 2045), (1769, 1870), (1683, 1767), (1641, 1766)
        ],
        # GOTOマスク領域 (X1, Y1, X2, Y2), 0-based, inclusive
        'goto_mask_regions_xyxy': GOTO_MASK_REGIONS,
    }

    # --- 実行 ---
    create_peak_map_from_idl_logic(config)