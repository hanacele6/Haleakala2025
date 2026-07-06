import cv2
import numpy as np
import matplotlib.pyplot as plt
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

def extract_and_calibrate_points(img, hsv_ranges, calibration, legend_bounds=None, min_dist=15, param1=50, param2=12, min_radius=4, max_radius=15):
    """【モード1：Hough円検出版】散布図の「円形マーカー」を直接探す関数（凡例除外対応）"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lower, upper) in hsv_ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)))

    # ROI（対象領域）以外を黒塗りにして枠線を消す
    x_cal, y_cal = calibration['x'], calibration['y']
    roi_mask = np.zeros_like(mask)
    top, bottom = int(min(y_cal['min_px'], y_cal['max_px'])), int(max(y_cal['min_px'], y_cal['max_px']))
    left, right = int(min(x_cal['min_px'], x_cal['max_px'])), int(max(x_cal['min_px'], x_cal['max_px']))
    margin = 5
    roi_mask[top+margin:bottom-margin, left+margin:right-margin] = 1

    # ====== 【新規追加】凡例エリアを黒塗り（0）にして完全に無視させる ======
    if legend_bounds is not None:
        l_top, l_bottom = legend_bounds['y_min'], legend_bounds['y_max']
        l_left, l_right = legend_bounds['x_min'], legend_bounds['x_max']
        # 凡例の四角形の中を 0 (除外) にする
        roi_mask[l_top:l_bottom, l_left:l_right] = 0
    # ====================================================================

    mask = mask * roi_mask

    # ガウシアンブラーをかけて、線のギザギザを滑らかにする（円として認識されやすくする）
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)

    # === Hough Circle Transform による円検出 ===
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=min_dist,      # 円同士の最小距離
        param1=param1,         
        param2=param2,         # 円の検出感度
        minRadius=min_radius,  # 円の最小半径
        maxRadius=max_radius   # 円の最大半径
    )

    pixel_points, real_data_points = [], []
    display_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cX, cY, r = i[0], i[1], i[2]
            pixel_points.append((cX, cY))
            
            data_x = x_cal['min_val'] + (cX - x_cal['min_px']) * (x_cal['max_val'] - x_cal['min_val']) / (x_cal['max_px'] - x_cal['min_px'])
            data_y = y_cal['min_val'] + (cY - y_cal['min_px']) * (y_cal['max_val'] - y_cal['min_val']) / (y_cal['max_px'] - y_cal['min_px'])
            real_data_points.append((data_x, data_y))
            
            cv2.circle(display_mask, (cX, cY), r, (0, 255, 0), 2)
            cv2.circle(display_mask, (cX, cY), 2, (255, 0, 0), 3)

    return real_data_points, pixel_points, display_mask


def extract_and_calibrate_curves(img, hsv_ranges, calibration, x_bounds):
    """【モード2】輪郭グループ化による複数線トラッキング関数"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lower, upper) in hsv_ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)))

    x_cal, y_cal = calibration['x'], calibration['y']
    roi_mask = np.zeros_like(mask)
    top, bottom = int(min(y_cal['min_px'], y_cal['max_px'])), int(max(y_cal['min_px'], y_cal['max_px']))
    left, right = int(min(x_cal['min_px'], x_cal['max_px'])), int(max(x_cal['min_px'], x_cal['max_px']))
    
    padding = 5
    roi_mask[top+padding:bottom-padding, left+padding:right-padding] = 1
    mask = mask * roi_mask

    kernel = np.ones((5, 40), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_tracks = []
    x_min_val, x_max_val = x_bounds

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 50: 
            continue

        contour_mask = np.zeros_like(mask_closed)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

        real_pts = []
        for cX in range(x, x + w):
            y_indices = np.where(contour_mask[:, cX] > 0)[0]
            if len(y_indices) > 0:
                cY = int(np.mean(y_indices))
                
                data_x = x_cal['min_val'] + (cX - x_cal['min_px']) * (x_cal['max_val'] - x_cal['min_val']) / (x_cal['max_px'] - x_cal['min_px'])
                data_y = y_cal['min_val'] + (cY - y_cal['min_px']) * (y_cal['max_val'] - y_cal['min_val']) / (y_cal['max_px'] - y_cal['min_px'])
                
                if x_min_val <= data_x <= x_max_val:
                    real_pts.append((data_x, data_y))
        
        if len(real_pts) > 30:
            valid_tracks.append(real_pts)

    return valid_tracks, mask_closed


# ==========================================
# 実行ブロック
# ==========================================
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    
    image_path = filedialog.askopenfilename(
        title="解析するグラフ画像を選択してください",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp")]
    )
    if not image_path: sys.exit()

    img = imread_japanese(image_path)
    if img is None: sys.exit()

    print("\n" + "="*50)
    print("画像ウィンドウが開きます。以下の順番で4箇所をクリックしてください。")
    print("1. X軸の 最小値 の位置（例: 0）\n2. X軸の 最大値 の位置（例: 360）")
    print("3. Y軸の 最小値 の位置（例: 0.0）\n4. Y軸の 最大値 の位置（例: 1.0）")
    print("="*50 + "\n")

    fig_cal, ax_cal = plt.subplots(figsize=(10, 8))
    ax_cal.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax_cal.set_title("Click 4 points: X_min, X_max, Y_min, Y_max")
    
    pts = plt.ginput(4, timeout=0) 
    plt.close(fig_cal)

    if len(pts) != 4: sys.exit()

    x_min_px, x_max_px = pts[0][0], pts[1][0]
    y_min_px, y_max_px = pts[2][1], pts[3][1]

    print("\nクリックした位置の「実際の数値」を入力してください。")
    try:
        x_min_val = float(input("1. X軸の最小値: "))
        x_max_val = float(input("2. X軸の最大値: "))
        y_min_val = float(input("3. Y軸の最小値: "))
        y_max_val = float(input("4. Y軸の最大値: "))
    except ValueError:
        sys.exit()

    calibration = {
        'x': {'min_px': x_min_px, 'min_val': x_min_val, 'max_px': x_max_px, 'max_val': x_max_val},
        'y': {'min_px': y_min_px, 'min_val': y_min_val, 'max_px': y_max_px, 'max_val': y_max_val}
    }
    x_bounds = (x_min_val, x_max_val)

    # ====== 【新規追加】凡例エリアの除外設定 ======
    print("\n" + "="*50)
    print("凡例（テキスト部分など）を誤検知させないための【除外エリア】を設定しますか？")
    exclude_legend = input("設定するなら 'y'、しないなら 'n' を入力: ").strip().lower()

    legend_bounds = None
    if exclude_legend == 'y':
        print("\n画像ウィンドウが開きます。除外したい凡例全体を囲むように、")
        print("【左上】と【右下】の2箇所をクリックしてください。")
        
        fig_leg, ax_leg = plt.subplots(figsize=(10, 8))
        ax_leg.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax_leg.set_title("Click 2 points (Top-Left and Bottom-Right) to Exclude Legend Area")
        
        leg_pts = plt.ginput(2, timeout=0) 
        plt.close(fig_leg)

        if len(leg_pts) == 2:
            # ユーザーがどの順番でクリックしても大丈夫なように min/max を自動判別
            leg_x_min = min(leg_pts[0][0], leg_pts[1][0])
            leg_x_max = max(leg_pts[0][0], leg_pts[1][0])
            leg_y_min = min(leg_pts[0][1], leg_pts[1][1])
            leg_y_max = max(leg_pts[0][1], leg_pts[1][1])
            legend_bounds = {
                'x_min': int(leg_x_min), 'x_max': int(leg_x_max),
                'y_min': int(leg_y_min), 'y_max': int(leg_y_max)
            }
            print(f"✅ 凡例エリアを除外領域として設定しました。")
        else:
            print("⚠️ クリックがキャンセルされました。除外エリアなしで進めます。")
    # ==============================================

    print("\n" + "="*50)
    print("抽出モードを選択してください：")
    print(" [1] データ点抽出 (散布図のマーカーなど)")
    print(" [2] 複数線トラッキング (実線・破線など分離)")
    mode_choice = input("番号を入力 (1 or 2): ").strip()
    
    color_ranges = [
        ([0, 0, 0], [179, 255, 150])  
    ]

    # --- 処理の実行 ---
    if mode_choice == '1':
        data_pts, _, mask_cleaned = extract_and_calibrate_points(
            img, color_ranges, calibration, legend_bounds=legend_bounds,
            min_dist=15, param2=12, min_radius=4, max_radius=15
        )
        data_pts = [pt for pt in data_pts if x_min_val <= pt[0] <= x_max_val]
    elif mode_choice == '2':
        valid_tracks, mask_cleaned = extract_and_calibrate_curves(
            img, color_ranges, calibration, x_bounds
        )
    else:
        sys.exit()
    
    # --- 結果の可視化 ---
    fig_res, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # モード1の場合はカラーでマスクを表示（緑と赤の枠線を見るため）
    if mode_choice == '1':
        ax1.imshow(mask_cleaned)
    else:
        ax1.imshow(mask_cleaned, cmap='gray')
        
    ax1.set_title(f"Cleaned Mask (Mode: {mode_choice})")
    ax1.axis('off')
    
    ax2.set_xlim(calibration['x']['min_val'], calibration['x']['max_val'])
    ax2.set_ylim(calibration['y']['min_val'], calibration['y']['max_val'])
    ax2.grid(True)
    ax2.set_title("Reconstructed Data")
    ax2.set_xlabel("X-axis Value")
    ax2.set_ylabel("Y-axis Value")

    if mode_choice == '1' and data_pts:
        data_x = [p[0] for p in data_pts]
        data_y = [p[1] for p in data_pts]
        ax2.scatter(data_x, data_y, color='red', marker='o', s=20, alpha=0.6)
        print(f"\n✅ {len(data_pts)} 個のデータ点を抽出しました。")
    
    elif mode_choice == '2' and valid_tracks:
        colors = plt.cm.tab10.colors 
        valid_tracks.sort(key=lambda track: track[0][0])
        
        for i, track in enumerate(valid_tracks):
            data_x = [p[0] for p in track]
            data_y = [p[1] for p in track]
            ax2.plot(data_x, data_y, color=colors[i % 10], label=f'Line_ID: {i}', linewidth=2, alpha=0.8)
        ax2.legend()
        print(f"\n✅ {len(valid_tracks)} 本の線を分離抽出しました。グラフの凡例(Line_ID)を確認してください。")

    plt.tight_layout()
    plt.show()

    # --- データの保存 (CSV出力) ---
    print("\n" + "-" * 50)
    save_choice = input("データをCSVファイルとして保存しますか？ (y/n): ").strip().lower()
    
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
                        target_id = input(f"保存する Line_ID を入力してください (0 から {len(valid_tracks)-1} / 全て保存なら 'all'): ").strip()
                        
                        if target_id.lower() == 'all':
                            writer.writerow(['Line_ID', 'X_Value', 'Y_Value']) 
                            for i, track in enumerate(valid_tracks):
                                for pt in track:
                                    writer.writerow([i, f"{pt[0]:.5f}", f"{pt[1]:.5f}"])
                        else:
                            writer.writerow(['X_Value', 'Y_Value']) 
                            target_id = int(target_id)
                            for pt in valid_tracks[target_id]:
                                writer.writerow([f"{pt[0]:.5f}", f"{pt[1]:.5f}"])
                                
                print(f"\n✅ データを保存しました: {save_path}")
            except Exception as e:
                print(f"\n❌ 保存中にエラーが発生しました: {e}")