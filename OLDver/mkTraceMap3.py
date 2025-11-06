import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter
from astropy.io import fits
from pathlib import Path
import os


def gaussian(x, amplitude, mean, stddev):
    """ガウス関数の定義"""
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))


def create_high_precision_pp1(data_filepath, dark_filepath, output_filepath, config):
    """
    高精度なガウスフィッティング追跡ロジックを使い、
    指定されたファイルからpp1.fitsを生成する。
    """
    print(f"--- 高精度トレース処理開始 ---")
    print(f"データファイル: {data_filepath.name}")
    print(f"ダークファイル: {dark_filepath.name}")

    # --- 1. パラメータとファイルの準備 ---
    # configからパラメータを読み込む
    nFibX = config['nFibX']
    nFibY = config['nFibY']
    iFibInact = config['iFibInact']
    yFib0 = config['yFib0']
    yFib1 = config['yFib1']
    ypixFibWid = config['ypixFibWid']
    finterval = config['trace_interval_x']

    # FITSファイルの読み込みとダーク減算
    try:
        with fits.open(data_filepath) as hd1:
            dat = hd1[0].data
            hd = hd1[0].header
        with fits.open(dark_filepath) as hd2:
            dk = hd2[0].data
    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません。 {e}")
        return

    data = dat - dk
    # データを滑らかにするメディアンフィルタ
    data = median_filter(data, size=(1, 5))

    nx = hd['NAXIS1']
    ny = hd['NAXIS2']

    # ファイバー番号のリストを作成
    iFib = np.arange(nFibX * nFibY)
    iFibAct = np.setdiff1d(iFib, iFibInact)
    # 各ファイバーの初期Y座標を計算
    yfibs = np.arange(len(iFib), dtype=float) * yFib1 + yFib0

    # 結果を格納する配列
    AARR = np.zeros((nx, len(iFib), 3), dtype=float)
    # pp1.fits用の空の配列
    pp1_data = np.zeros_like(data, dtype=np.int16)

    # --- 2. 各ファイバーをガウスフィッティングで追跡 ---
    xpix = np.arange(nx, dtype=int)
    xpixF = np.arange(0, nx, finterval)
    if xpixF.max() != (nx - 1):
        xpixF = np.append(xpixF, nx - 1)

    # 有効なファイバーを1本ずつループ
    for i in iFibAct:
        print(f"ファイバー {i}/{len(iFib) - 1} を追跡中...", end="")

        m2 = yfibs[i]  # 初期位置の推測値

        # 波長方向（X軸）にループして追跡
        for j in xpixF:
            # 現在の推測位置m2を中心に、フィッティングするY座標の範囲を決める
            ypix1 = (np.arange(ypixFibWid, dtype=float) - ypixFibWid / 2 + 0.5 + m2).astype(int)
            # 範囲外アクセスを防ぐ
            if ypix1.min() < 0 or ypix1.max() >= ny:
                continue

            ydat1 = data[ypix1, j]
            Aini = [np.max(ydat1), m2, ypixFibWid / 5.0]

            try:
                # ガウス関数でフィッティングして、より正確な中心位置を求める
                param_bounds = ([1, m2 - ypixFibWid, 0.1], [1e5, m2 + ypixFibWid, 3])
                par, cov = curve_fit(gaussian, ypix1, ydat1, p0=Aini, bounds=param_bounds)
                # フィッティングで得られた中心位置を、次のステップの推測値として更新
                m2 = par[1]
                # 結果を保存
                AARR[j, i, :] = par
            except RuntimeError:
                # フィッティングが失敗した場合は、前の位置を維持するなどの処理も可能
                AARR[j, i, :] = np.nan  # 失敗した箇所はNaNにする
                continue

        # --- 追跡した全点のデータから、滑らかなトレースカーブを再計算 ---
        # 追跡が成功した点だけを使って多項式フィッティング
        valid_points = np.isfinite(AARR[xpixF, i, 1])
        if np.count_nonzero(valid_points) > 10:  # 最低10点は成功していないとフィットしない
            coef = np.polyfit(xpixF[valid_points], AARR[xpixF, i, 1][valid_points], 6)
            # 全てのXピクセルに対して滑らかなY座標を計算
            smooth_y_trace = np.polyval(coef, xpix)
            AARR[:, i, 1] = smooth_y_trace

        print(" 完了")

    # --- 3. 最終的なトレース情報から pp1.fits を生成 ---
    print("\n--- 全ファイバーのトレース情報から pp1.fits を生成します ---")

    trace_width_radius = 1  # 線の太さ（半径）

    for i in iFibAct:
        y_trace = AARR[:, i, 1]
        for x_pos in range(nx):
            y_pos_float = y_trace[x_pos]
            if not np.isnan(y_pos_float):
                y_pos = int(round(y_pos_float))
                y_min = max(0, y_pos - trace_width_radius)
                y_max = min(ny, y_pos + trace_width_radius + 1)
                pp1_data[y_min:y_max, x_pos] = i + 1

    # --- 4. pp1.fits を保存 ---
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    hdu_pp1 = fits.PrimaryHDU(data=pp1_data, header=hd)
    hdul_pp1 = fits.HDUList([hdu_pp1])
    hdul_pp1.writeto(output_filepath, overwrite=True)
    print(f"高精度トレースマップを保存しました: {output_filepath}")


if __name__ == "__main__":
    # ===================================================================
    # --- ユーザー設定 ---
    # ===================================================================
    # 1. ファイルパスを直接指定
    date = '20250501'
    base_dir = Path(f"C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
    output_dir = Path(f"C:/Users/hanac/University/Senior/Mercury/Haleakala2025/output/{date}")
    dark_file = base_dir / "dk01h_20s.sp.fits"
    input_file = output_dir / ' led01r_clf590n_ga7000fsp220_1_nhp_py.fits'
    output_file = output_dir / "pp1_trace_map.fits"
    output_file_pp1 = output_dir / input_file.name.replace(".fits", ".pp1.fits")

    # 2. 解析パラメータを直接指定 (以前のCSVファイルの内容に相当)
    config = {
        # --- ファイバーの基本情報 ---
        'nFibX': 10,  # ファイバーバンドルの横の数
        'nFibY': 12,  # ファイバーバンドルの縦の数 (合計120本)
        'iFibInact': [6, 49, 69, 89, 94, 109, 117],  # 使っていないファイバーの番号

        # --- ファイバーの初期位置を定義するパラメータ ---
        'yFib0': 16.0,  # 0番目のファイバーのおおよそのY座標
        'yFib1': 8.3,  # ファイバー間のY方向のおおよその間隔
        'ypixFibWid': 4.0,  # ファイバーの幅（ピクセル単位）

        # --- トレースの挙動を制御するパラメータ ---
        'trace_interval_x': 16,  # 何ピクセルおきにフィッティングを行うか (小さいほど丁寧だが遅い)
    }

    # --- 実行 ---
    create_high_precision_pp1(input_file, dark_file, output_file_pp1, config)