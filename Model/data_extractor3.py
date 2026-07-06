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
# モード1: 散布図マーカー抽出（変更なし）
# ============================================================
def extract_and_calibrate_points(img, hsv_ranges, calibration, legend_bounds=None,
                                 min_dist=15, param1=50, param2=12,
                                 min_radius=4, max_radius=15):
    """Hough円検出による散布図マーカー抽出"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lower, upper) in hsv_ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(
            hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)))

    x_cal, y_cal = calibration['x'], calibration['y']
    roi_mask = np.zeros_like(mask)
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
            cX, cY, r = i[0], i[1], i[2]
            pixel_points.append((cX, cY))
            data_x = (x_cal['min_val']
                      + (cX - x_cal['min_px'])
                      * (x_cal['max_val'] - x_cal['min_val'])
                      / (x_cal['max_px'] - x_cal['min_px']))
            data_y = (y_cal['min_val']
                      + (cY - y_cal['min_px'])
                      * (y_cal['max_val'] - y_cal['min_val'])
                      / (y_cal['max_px'] - y_cal['min_px']))
            real_data_points.append((data_x, data_y))
            cv2.circle(display_mask, (cX, cY), r, (0, 255, 0), 2)
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
                
                data_x = (x_cal['min_val']
                          + (ux - x_cal['min_px'])
                          * (x_cal['max_val'] - x_cal['min_val'])
                          / (x_cal['max_px'] - x_cal['min_px']))
                data_y = (y_cal['min_val']
                          + (mean_y - y_cal['min_px'])
                          * (y_cal['max_val'] - y_cal['min_val'])
                          / (y_cal['max_px'] - y_cal['min_px']))
                track_pts.append((data_x, data_y))
                
            valid_tracks.append(track_pts)

    debug_img = cv2.cvtColor(extracted_mask, cv2.COLOR_GRAY2RGB)
    return valid_tracks, debug_img


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

    print("\n" + "="*50)
    print("画像ウィンドウが開きます。以下の順番で4箇所をクリックしてください。")
    print("1. X軸の 最小値\n2. X軸の 最大値\n3. Y軸の 最小値\n4. Y軸の 最大値")
    print("="*50)

    fig_cal, ax_cal = plt.subplots(figsize=(10, 8))
    ax_cal.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax_cal.set_title("Click 4 points: X_min, X_max, Y_min, Y_max")
    pts = plt.ginput(4, timeout=0)
    plt.close(fig_cal)
    if len(pts) != 4:
        sys.exit()

    x_min_px, x_max_px = pts[0][0], pts[1][0]
    y_min_px, y_max_px = pts[2][1], pts[3][1]

    print("\n実際の数値を入力してください。")
    try:
        x_min_val = float(input("1. X軸の最小値: "))
        x_max_val = float(input("2. X軸の最大値: "))
        y_min_val = float(input("3. Y軸の最小値: "))
        y_max_val = float(input("4. Y軸の最大値: "))
    except ValueError:
        sys.exit()

    calibration = {
        'x': {'min_px': x_min_px, 'min_val': x_min_val,
               'max_px': x_max_px, 'max_val': x_max_val},
        'y': {'min_px': y_min_px, 'min_val': y_min_val,
               'max_px': y_max_px, 'max_val': y_max_val}
    }

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
    print(" [2] 黒系実線・破線トラッキング（スケルトン廃止・重心抽出）")
    mode_choice = input("番号を入力 (1 or 2): ").strip()

    color_ranges_hsv = [([0, 0, 0], [179, 255, 150])]  

    # ── 処理実行 ─────────────────────────────────────────────
    if mode_choice == '1':
        data_pts, _, mask_cleaned = extract_and_calibrate_points(
            img, color_ranges_hsv, calibration,
            legend_bounds=legend_bounds,
            min_dist=15, param2=12, min_radius=4, max_radius=15
        )
        data_pts = [pt for pt in data_pts if x_min_val <= pt[0] <= x_max_val]

    elif mode_choice == '2':
        valid_tracks, mask_cleaned = extract_curves_with_eraser_and_seeds(
            img, calibration,
            legend_bounds=legend_bounds,
            dark_thresh=150
        )
        valid_tracks = [
            [(x, y) for (x, y) in track if x_min_val <= x <= x_max_val]
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

    ax2.set_xlim(calibration['x']['min_val'], calibration['x']['max_val'])
    ax2.set_ylim(calibration['y']['min_val'], calibration['y']['max_val'])
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