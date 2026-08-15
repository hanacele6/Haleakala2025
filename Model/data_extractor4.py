import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import tkinter as tk
from tkinter import filedialog
import sys
import csv


def imread_japanese(filename, flags=cv2.IMREAD_COLOR):
    """日本語パス対応の画像読み込み関数"""
    try:
        n = np.fromfile(filename, np.uint8)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(f"画像読み込みエラー: {e}")
        return None


# ============================================================
# ★追加: ピクセル→実数値 変換関数（線形／対数スケール対応）
# ============================================================
def pixel_to_value(px, cal):
    """
    calの中に 'scale' キー（'linear' or 'log'）を持たせておくことで
    線形軸／対数軸のどちらにも対応した変換を行う。

    重要: cal['min_val'] / cal['max_val'] は「軸の最小値・最大値」である
    必要はない。目盛り上ではっきり数値がわかる、任意の2点（基準点）でよい。
    """
    scale = cal.get('scale', 'linear')

    if scale == 'log':
        # 対数軸の場合、ピクセル位置は log10(値) に対して線形になる
        log_min = np.log10(cal['min_val'])
        log_max = np.log10(cal['max_val'])
        log_val = (log_min
                   + (px - cal['min_px'])
                   * (log_max - log_min)
                   / (cal['max_px'] - cal['min_px']))
        return 10 ** log_val
    else:
        # 通常の線形軸
        return (cal['min_val']
                + (px - cal['min_px'])
                * (cal['max_val'] - cal['min_val'])
                / (cal['max_px'] - cal['min_px']))


# ============================================================
# モード1: 散布図マーカー抽出
# ============================================================
def extract_and_calibrate_points(img, hsv_ranges, calibration, legend_bounds=None,
                                 min_dist=15, param1=50, param2=12,
                                 min_radius=4, max_radius=15,
                                 fill_ratio_thresh=0.75):
    """
    Hough円検出による散布図マーカー抽出

    ★追加: fill_ratio_thresh
    破線の目盛り線などがたまたま円形に誤検出されるのを防ぐため、
    検出した円の内部がどれだけ隙間なく塗りつぶされているか(充填率)をチェックする。
    実際のデータ点マーカーはベタ塗りの円のため充填率が高いが、
    破線の交差やかすれた線は充填率が低くなるため、閾値未満のものは除外する。
    値を上げる(例: 0.85〜0.9)ほど判定は厳しくなる。
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lower, upper) in hsv_ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(
            hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)))

    x_cal, y_cal = calibration['x'], calibration['y']
    # ★変更: ROI（有効領域）は較正点ではなく、別途指定する画像全体の描画範囲を使う。
    #        calibration の min_px/max_px は「較正のための基準点」であり、
    #        軸の描画範囲（プロットエリアの端）とは限らないため。
    plot_area = calibration.get('plot_area_px', None)
    roi_mask = np.zeros_like(mask)
    if plot_area is not None:
        top, bottom = plot_area['y_top'], plot_area['y_bottom']
        left, right = plot_area['x_left'], plot_area['x_right']
    else:
        # plot_area が指定されない場合は、従来通り較正点の範囲を使う（後方互換）
        top    = int(min(y_cal['min_px'], y_cal['max_px']))
        bottom = int(max(y_cal['min_px'], y_cal['max_px']))
        left   = int(min(x_cal['min_px'], x_cal['max_px']))
        right  = int(max(x_cal['min_px'], x_cal['max_px']))
    margin = 5
    roi_mask[top+margin:bottom-margin, left+margin:right-margin] = 1

    if legend_bounds is not None:
        roi_mask[legend_bounds['y_min']:legend_bounds['y_max'],
                 legend_bounds['x_min']:legend_bounds['x_max']] = 0

    mask = mask * roi_mask
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist,
        param1=param1, param2=param2,
        minRadius=min_radius, maxRadius=max_radius
    )

    pixel_points, real_data_points = [], []
    display_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cX, cY, r = int(i[0]), int(i[1]), int(i[2])

            # ★追加: 円内の充填率チェック（破線・かすれ誤検出の除外）
            y1, y2 = max(cY - r, 0), min(cY + r + 1, mask.shape[0])
            x1, x2 = max(cX - r, 0), min(cX + r + 1, mask.shape[1])
            roi = mask[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
            cy_local, cx_local = cY - y1, cX - x1
            circle_area_mask = (xx - cx_local)**2 + (yy - cy_local)**2 <= r**2
            total_px = circle_area_mask.sum()
            if total_px == 0:
                continue
            filled_px = np.count_nonzero(roi[circle_area_mask])
            fill_ratio = filled_px / total_px

            if fill_ratio < fill_ratio_thresh:
                # 破線の交差点などベタ塗りでない箇所は誤検出として除外
                continue

            pixel_points.append((cX, cY))
            # ★変更: 線形固定の式をやめ、pixel_to_value() 経由に統一（log軸対応）
            data_x = pixel_to_value(cX, x_cal)
            data_y = pixel_to_value(cY, y_cal)
            real_data_points.append((data_x, data_y))
            cv2.circle(display_mask, (cX, cY), r, (0, 255, 0), 2)
            cv2.circle(display_mask, (cX, cY), 2, (255, 0, 0), 3)

    return real_data_points, pixel_points, display_mask


def extract_markers_distance_transform(img, hsv_ranges, calibration, legend_bounds=None,
                                        min_radius=4, max_radius=15,
                                        peak_min_dist=15, dist_thresh_ratio=0.6):
    """
    ★追加: 距離変換によるマーカー中心検出（誤差棒が付いた点への対応）

    Hough円検出は「きれいな円形」であることを前提にしているため、
    誤差棒(エラーバー)がくっついて輪郭が崩れた点は検出漏れしやすい。
    param2を下げれば拾えることもあるが、同時に無関係なノイズも
    大量に拾ってしまい、かえって不安定になりやすい。

    このため、円形状の仮定を使わない別アプローチとして距離変換を用いる。
    ・誤差棒の線は細い(数px幅)ため、線上では「背景までの距離」が小さい
    ・マーカー本体は太い円なので、中心付近は「背景までの距離」が大きい
    という性質を利用し、距離変換の値が局所的に最大となる点(ピーク)を
    マーカーの中心として検出する。線がくっついていても中心だけを拾える。
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lower, upper) in hsv_ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(
            hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)))

    x_cal, y_cal = calibration['x'], calibration['y']
    plot_area = calibration.get('plot_area_px', None)
    roi_mask = np.zeros_like(mask)
    if plot_area is not None:
        top, bottom = plot_area['y_top'], plot_area['y_bottom']
        left, right = plot_area['x_left'], plot_area['x_right']
    else:
        top    = int(min(y_cal['min_px'], y_cal['max_px']))
        bottom = int(max(y_cal['min_px'], y_cal['max_px']))
        left   = int(min(x_cal['min_px'], x_cal['max_px']))
        right  = int(max(x_cal['min_px'], x_cal['max_px']))
    margin = 5
    roi_mask[top+margin:bottom-margin, left+margin:right-margin] = 1

    if legend_bounds is not None:
        roi_mask[legend_bounds['y_min']:legend_bounds['y_max'],
                 legend_bounds['x_min']:legend_bounds['x_max']] = 0

    mask = mask * roi_mask

    # --- 距離変換: 各画素について「一番近い背景(mask=0)までの距離」を計算 ---
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # --- 局所最大(ピーク)の検出。scipyに依存せず cv2.dilate で代用 ---
    ksize = max(3, peak_min_dist | 1)  # 奇数サイズにする
    kernel = np.ones((ksize, ksize), np.uint8)
    dilated = cv2.dilate(dist, kernel)
    dist_thresh = min_radius * dist_thresh_ratio
    peak_mask = (dist >= dilated - 1e-6) & (dist > dist_thresh)

    ys, xs = np.where(peak_mask)
    candidates = [(int(xs[i]), int(ys[i]), dist[ys[i], xs[i]]) for i in range(len(xs))]
    # 距離値(≒推定半径)が大きい順に並べ、近すぎる候補は間引く(簡易NMS)
    candidates.sort(key=lambda c: -c[2])

    accepted = []
    for cX, cY, r in candidates:
        if r > max_radius * 1.5:
            continue
        too_close = False
        for (aX, aY, _) in accepted:
            if (cX - aX)**2 + (cY - aY)**2 < peak_min_dist**2:
                too_close = True
                break
        if not too_close:
            accepted.append((cX, cY, r))

    pixel_points, real_data_points = [], []
    display_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    for cX, cY, r in accepted:
        pixel_points.append((cX, cY))
        data_x = pixel_to_value(cX, x_cal)
        data_y = pixel_to_value(cY, y_cal)
        real_data_points.append((data_x, data_y))
        cv2.circle(display_mask, (cX, cY), int(r), (0, 255, 0), 2)
        cv2.circle(display_mask, (cX, cY), 2, (255, 0, 0), 3)

    return real_data_points, pixel_points, display_mask


# ============================================================
# モード2（スケルトン廃止・太線重心抽出・強力な隙間結合）
# ============================================================
def extract_curves_with_eraser_and_seeds(img, calibration, legend_bounds=None, dark_thresh=150):
    """
    【改良版】
    細線化（スケルトン化）によるささくれを防ぐため、太い線のまま抽出し、
    X座標ごとのY座標の中心を取るアプローチに変更。
    """
    x_cal, y_cal = calibration['x'], calibration['y']

    # --- Step 1: 消しゴム機能（手動での分離） ---
    print("\n" + "="*50)
    print("【消しゴムツール】")
    print("グラフの線が「矢印」や「軸」とくっついている部分を、マウスのドラッグで囲んで消去（白塗り）してください。")
    print("終わったら、ウィンドウの「×」ボタンを押して閉じてください。")
    print("="*50)

    working_img = img.copy()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Drag to erase (white out) connections. Close window when done.")
    img_display = ax.imshow(cv2.cvtColor(working_img, cv2.COLOR_BGR2RGB))

    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        cv2.rectangle(working_img, (min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2)), (255, 255, 255), -1)
        img_display.set_data(cv2.cvtColor(working_img, cv2.COLOR_BGR2RGB))
        fig.canvas.draw()

    rs = RectangleSelector(ax, onselect, useblit=True,
                           button=[1], minspanx=5, minspany=5,
                           spancoords='pixels', interactive=True)
    plt.show()

    # --- Step 2: 二値化と【強力な隙間埋め】 ---
    gray = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)

    if legend_bounds is not None:
        gray[legend_bounds['y_min']:legend_bounds['y_max'],
             legend_bounds['x_min']:legend_bounds['x_max']] = 255

    # 閾値で二値化（暗い画素が白）
    _, bin_img = cv2.threshold(gray, dark_thresh, 255, cv2.THRESH_BINARY_INV)

    # ★変更点: より大きい円形のカーネル(5x5)で強力に隙間を繋ぐ
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bin_img_closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ★変更点: スケルトン化を廃止し、太いままラベリングする
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img_closed, connectivity=8)

    # --- Step 3: シード（種）のクリック取得 ---
    print("\n" + "="*50)
    print("【抽出対象の選択】")
    print("抽出したい線の上をクリックしてください（複数可）。")
    print("すべて選び終わったら「Enter」キーを押すか、マウスの「中ボタン」をクリックしてください。")
    print("="*50)

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.set_title("Click on the target curves. Press Enter or Middle-Click when done.")
    # 表示するのもスケルトンではなく太い線
    ax2.imshow(bin_img_closed, cmap='gray')
    seeds = plt.ginput(n=-1, timeout=0)
    plt.close(fig2)

    valid_tracks = []
    extracted_mask = np.zeros_like(bin_img_closed)

    ys_mask, xs_mask = np.where(bin_img_closed > 0)
    if len(xs_mask) == 0 or len(seeds) == 0:
        return [], extracted_mask

    # --- Step 4: クリックした座標から対象ラベルを抽出し、重心を計算 ---
    extracted_labels = set()
    for (sx, sy) in seeds:
        distances = (xs_mask - sx)**2 + (ys_mask - sy)**2
        nearest_idx = np.argmin(distances)
        if distances[nearest_idx] > 1000:
            continue

        target_x = xs_mask[nearest_idx]
        target_y = ys_mask[nearest_idx]
        target_label = labels[target_y, target_x]

        if target_label != 0 and target_label not in extracted_labels:
            extracted_labels.add(target_label)
            extracted_mask[labels == target_label] = 255

            pts_y, pts_x = np.where(labels == target_label)
            unique_xs = np.unique(pts_x)
            track_pts = []

            # 同じX座標にある複数のYピクセル（太さ）の平均値を取る
            for ux in unique_xs:
                mean_y = np.mean(pts_y[pts_x == ux])

                # ★変更: pixel_to_value() 経由に統一（log軸対応）
                data_x = pixel_to_value(ux, x_cal)
                data_y = pixel_to_value(mean_y, y_cal)
                track_pts.append((data_x, data_y))

            valid_tracks.append(track_pts)

    debug_img = cv2.cvtColor(extracted_mask, cv2.COLOR_GRAY2RGB)
    return valid_tracks, debug_img


# ============================================================
# ★追加: 軸の較正情報を対話的に取得する関数
#   - 「軸の最小値・最大値」ではなく「数値がわかる任意の2点（基準点）」を
#     クリックしてもらう方式に変更。目盛りが軸の端にない場合でも対応可能。
#   - 対数スケールかどうかも聞く。
# ============================================================
def get_axis_calibration(img, axis_name):
    print("\n" + "="*50)
    print(f"【{axis_name}軸の較正】")
    print("目盛り上で数値がはっきりわかる点を「2点」クリックしてください。")
    print("※ 軸の最小値・最大値である必要はありません。離れた2点であるほど精度が上がります。")
    print("="*50)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Click 2 reference points on the {axis_name} axis")
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)
    if len(pts) != 2:
        sys.exit()

    if axis_name == "X":
        px1, px2 = pts[0][0], pts[1][0]
    else:
        px1, px2 = pts[0][1], pts[1][1]

    print(f"クリックした2点の実際の数値を入力してください。")
    try:
        val1 = float(input(f"1点目の{axis_name}軸の値: "))
        val2 = float(input(f"2点目の{axis_name}軸の値: "))
    except ValueError:
        sys.exit()

    scale_input = input(f"{axis_name}軸は対数(log)スケールですか？ (y/n): ").strip().lower()
    scale = 'log' if scale_input == 'y' else 'linear'

    if scale == 'log' and (val1 <= 0 or val2 <= 0):
        print("❌ 対数スケールでは0以下の値は使用できません。")
        sys.exit()

    return {
        'min_px': px1, 'min_val': val1,
        'max_px': px2, 'max_val': val2,
        'scale': scale
    }


# ============================================================
# 実行ブロック
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    image_path = filedialog.askopenfilename(
        title="解析するグラフ画像を選択してください",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp")]
    )
    if not image_path:
        sys.exit()

    # ── キャリブレーション ──────────────────────────────────────
    img = imread_japanese(image_path)
    if img is None:
        sys.exit()

    # ★変更: 従来の「4点まとめてクリック」方式をやめ、X軸・Y軸それぞれ
    #        個別に「基準点2点＋対数スケールかどうか」を聞く方式に変更。
    x_cal = get_axis_calibration(img, "X")
    y_cal = get_axis_calibration(img, "Y")

    calibration = {
        'x': x_cal,
        'y': y_cal
    }

    # ★追加: モード1（マーカー抽出）のROI計算にも使うプロットエリアのピクセル範囲。
    #        軸の較正点とは別に、プロットエリアの「左上」「右下」をクリックしてもらう。
    print("\n" + "="*50)
    print("プロットエリア（グラフの描画枠）の【左上】と【右下】をクリックしてください。")
    print("="*50)
    fig_area, ax_area = plt.subplots(figsize=(10, 8))
    ax_area.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax_area.set_title("Click Top-Left and Bottom-Right of the plot area")
    area_pts = plt.ginput(2, timeout=0)
    plt.close(fig_area)
    if len(area_pts) == 2:
        calibration['plot_area_px'] = {
            'x_left':  int(min(area_pts[0][0], area_pts[1][0])),
            'x_right': int(max(area_pts[0][0], area_pts[1][0])),
            'y_top':   int(min(area_pts[0][1], area_pts[1][1])),
            'y_bottom':int(max(area_pts[0][1], area_pts[1][1]))
        }

    # ★変更: 表示範囲(x_display_min/max)は手入力させず、プロットエリアの
    #        左端・右端ピクセル位置を較正式(pixel_to_value)で変換して自動算出する。
    #        目盛りがない位置の値でも、較正済みの対応関係さえあれば計算可能。
    if 'plot_area_px' in calibration:
        x_auto_min = pixel_to_value(calibration['plot_area_px']['x_left'], x_cal)
        x_auto_max = pixel_to_value(calibration['plot_area_px']['x_right'], x_cal)
        x_auto_min, x_auto_max = min(x_auto_min, x_auto_max), max(x_auto_min, x_auto_max)

        print("\n" + "="*50)
        print("【表示範囲（自動算出）】")
        print(f"プロットエリアの端から計算した X軸範囲: {x_auto_min:.5f} 〜 {x_auto_max:.5f}")
        print("この値でよければ何も入力せず Enter、修正したい場合のみ数値を入力してください。")
        print("="*50)
        x_min_input = input(f"X軸 最小値 [{x_auto_min:.5f}]: ").strip()
        x_max_input = input(f"X軸 最大値 [{x_auto_max:.5f}]: ").strip()
        x_display_min = float(x_min_input) if x_min_input else x_auto_min
        x_display_max = float(x_max_input) if x_max_input else x_auto_max
    else:
        # plot_area が取得できなかった場合のみ、やむを得ず手入力
        print("\n" + "="*50)
        print("【表示範囲の指定】プロットエリアが未指定のため、手入力してください。")
        print("="*50)
        try:
            x_display_min = float(input("X軸 表示範囲の最小値: "))
            x_display_max = float(input("X軸 表示範囲の最大値: "))
        except ValueError:
            sys.exit()

    # ── 凡例除外エリア ────────────────────────────────────────
    print("\n" + "="*50)
    print("凡例など誤検知しやすい領域を除外しますか？")
    exclude_legend = input("'y' / 'n': ").strip().lower()

    legend_bounds = None
    if exclude_legend == 'y':
        print("\n除外エリアの【左上】と【右下】をクリックしてください。")
        fig_leg, ax_leg = plt.subplots(figsize=(10, 8))
        ax_leg.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax_leg.set_title("Click Top-Left and Bottom-Right of Legend Area")
        leg_pts = plt.ginput(2, timeout=0)
        plt.close(fig_leg)
        if len(leg_pts) == 2:
            legend_bounds = {
                'x_min': int(min(leg_pts[0][0], leg_pts[1][0])),
                'x_max': int(max(leg_pts[0][0], leg_pts[1][0])),
                'y_min': int(min(leg_pts[0][1], leg_pts[1][1])),
                'y_max': int(max(leg_pts[0][1], leg_pts[1][1]))
            }
            print("✅ 凡例エリアを除外します。")

    # ── モード選択 ────────────────────────────────────────────
    print("\n" + "="*50)
    print("抽出モードを選択してください：")
    print(" [1] データ点抽出（散布図マーカー）")
    print(" [2] 黒系実線・破線トラッキング")
    mode_choice = input("番号を入力 (1 or 2): ").strip()

    color_ranges_hsv = [([0, 0, 0], [179, 255, 150])]  #これ黒
    # 抽出したい色変更後のコード赤色※赤color_ranges_hsv = [([0, 100, 100], [10, 255, 255]), ([170, 100, 100], [179, 255, 255])]
    # 青色color_ranges_hsv = [([100, 100, 50], [140, 255, 255])]
    # 緑色color_ranges_hsv = [([35, 100, 50], [85, 255, 255])]
    # 黄色color_ranges_hsv = [([20, 100, 100], [35, 255, 255])]
    # オレンジ色color_ranges_hsv = [([11, 100, 100], [25, 255, 255])]

    # ── 処理実行 ─────────────────────────────────────────────
    if mode_choice == '1':
        # ★変更: 較正・プロットエリア・凡例除外はやり直さず、
        #        パラメータ(検出方式・充填率など)だけを何度でも試せるループにする。
        #        気に入る結果になるまで 'r' で再試行、OKならそのまま次へ進む。
        while True:
            print("\n" + "-"*50)
            print("検出方式を選んでください:")
            print(" [1] Hough円検出 (通常のデータ点向け)")
            print(" [2] 距離変換 (誤差棒などが付いていて円が崩れている点向け)")
            method_input = input("番号 (未入力なら1): ").strip()
            method = method_input if method_input in ('1', '2') else '1'

            if method == '2':
                minr_input = input("マーカーの最小半径(px, 未入力なら4): ").strip()
                min_radius_val = int(minr_input) if minr_input else 4
                maxr_input = input("マーカーの最大半径(px, 未入力なら15): ").strip()
                max_radius_val = int(maxr_input) if maxr_input else 15
                pmd_input = input("マーカー同士の最小間隔(px, 未入力なら15): ").strip()
                peak_min_dist_val = int(pmd_input) if pmd_input else 15
                dtr_input = input("距離変換のしきい値比率(0〜1, 未入力なら0.6): ").strip()
                dist_thresh_ratio_val = float(dtr_input) if dtr_input else 0.6

                data_pts, _, mask_cleaned = extract_markers_distance_transform(
                    img, color_ranges_hsv, calibration,
                    legend_bounds=legend_bounds,
                    min_radius=min_radius_val, max_radius=max_radius_val,
                    peak_min_dist=peak_min_dist_val, dist_thresh_ratio=dist_thresh_ratio_val
                )
            else:
                fr_input = input("マーカー判定の厳しさ(円内充填率の閾値 0〜1, 未入力なら0.75): ").strip()
                fill_ratio_thresh = float(fr_input) if fr_input else 0.75
                minr_input = input("マーカーの最小半径(px, 未入力なら4): ").strip()
                min_radius_val = int(minr_input) if minr_input else 4
                param2_input = input("Hough検出の厳しさ(param2, 未入力なら12): ").strip()
                param2_val = int(param2_input) if param2_input else 12

                data_pts, _, mask_cleaned = extract_and_calibrate_points(
                    img, color_ranges_hsv, calibration,
                    legend_bounds=legend_bounds,
                    min_dist=15, param2=param2_val, min_radius=min_radius_val, max_radius=15,
                    fill_ratio_thresh=fill_ratio_thresh
                )

            # フィルタ範囲は較正基準点ではなく x_display_min/max を使う
            data_pts = [pt for pt in data_pts if x_display_min <= pt[0] <= x_display_max]

            # プレビュー表示
            fig_prev, ax_prev = plt.subplots(figsize=(8, 6))
            ax_prev.imshow(mask_cleaned)
            ax_prev.set_title(f"Preview (method={method}): {len(data_pts)} points detected")
            ax_prev.axis('off')
            plt.show()

            redo = input("この結果でよければ Enter、パラメータを変えて再試行するなら 'r': ").strip().lower()
            if redo != 'r':
                break

    elif mode_choice == '2':
        valid_tracks, mask_cleaned = extract_curves_with_eraser_and_seeds(
            img, calibration,
            legend_bounds=legend_bounds,
            dark_thresh=150
        )
        # ★変更: フィルタ範囲は較正基準点ではなく x_display_min/max を使う
        valid_tracks = [
            [(x, y) for (x, y) in track if x_display_min <= x <= x_display_max]
            for track in valid_tracks
        ]
        valid_tracks = [t for t in valid_tracks if len(t) > 0]
    else:
        sys.exit()

    # ── 結果の可視化 ─────────────────────────────────────────
    fig_res, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.imshow(mask_cleaned)
    ax1.set_title(f"Extracted Mask (Mode: {mode_choice})")
    ax1.axis('off')

    ax2.set_xlim(x_display_min, x_display_max)
    # ★変更: Y軸表示範囲は較正基準点の min_val/max_val をそのまま使わず、
    #        pixel_to_value で plot_area の上端・下端を変換した値を使う。
    #        plot_area_px が未指定の場合は較正のmin_val/max_valで代用。
    if 'plot_area_px' in calibration:
        y_disp_top = pixel_to_value(calibration['plot_area_px']['y_top'], y_cal)
        y_disp_bottom = pixel_to_value(calibration['plot_area_px']['y_bottom'], y_cal)
    else:
        y_disp_top, y_disp_bottom = y_cal['max_val'], y_cal['min_val']

    ax2.set_ylim(min(y_disp_top, y_disp_bottom), max(y_disp_top, y_disp_bottom))
    if y_cal.get('scale') == 'log':
        ax2.set_yscale('log')
    ax2.grid(True)
    ax2.set_title("Reconstructed Data")
    ax2.set_xlabel("X-axis Value")
    ax2.set_ylabel("Y-axis Value")

    if mode_choice == '1' and 'data_pts' in dir() and data_pts:
        dx = [p[0] for p in data_pts]
        dy = [p[1] for p in data_pts]
        ax2.scatter(dx, dy, color='red', marker='o', s=20, alpha=0.6)
        print(f"\n✅ {len(data_pts)} 個のデータ点を抽出しました。")

    elif mode_choice == '2' and valid_tracks:
        colors = plt.cm.tab10.colors
        valid_tracks.sort(key=lambda t: t[0][0])
        for i, track in enumerate(valid_tracks):
            dx = [p[0] for p in track]
            dy = [p[1] for p in track]
            ax2.plot(dx, dy, color=colors[i % 10],
                     label=f'Line_ID: {i}', linewidth=2, alpha=0.8)
        ax2.legend()
        print(f"\n✅ {len(valid_tracks)} 本の線を抽出しました。")

    plt.tight_layout()
    plt.show()

    # ── CSV保存 ──────────────────────────────────────────────
    print("\n" + "-"*50)
    save_choice = input("CSVに保存しますか？ (y/n): ").strip().lower()

    if save_choice == 'y':
        save_path = filedialog.asksaveasfilename(
            title="保存先を指定してください",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="extracted_data.csv"
        )
        if save_path:
            try:
                with open(save_path, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if mode_choice == '1':
                        writer.writerow(['X_Value', 'Y_Value'])
                        for pt in data_pts:
                            writer.writerow([f"{pt[0]:.5f}", f"{pt[1]:.5f}"])
                    elif mode_choice == '2':
                        target_id = input(
                            f"保存する Line_ID (0〜{len(valid_tracks)-1}) "
                            f"/ 全て保存なら 'all': "
                        ).strip()
                        if target_id.lower() == 'all':
                            writer.writerow(['Line_ID', 'X_Value', 'Y_Value'])
                            for i, track in enumerate(valid_tracks):
                                for pt in track:
                                    writer.writerow([i, f"{pt[0]:.5f}", f"{pt[1]:.5f}"])
                        else:
                            writer.writerow(['X_Value', 'Y_Value'])
                            for pt in valid_tracks[int(target_id)]:
                                writer.writerow([f"{pt[0]:.5f}", f"{pt[1]:.5f}"])
                print(f"\n✅ 保存完了: {save_path}")
            except Exception as e:
                print(f"\n❌ 保存エラー: {e}")