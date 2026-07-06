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

def extract_and_calibrate_points(img, hsv_ranges, calibration, min_area=5, max_area=500, noise_kernel=3):
    """【モード1】散布図の「点」を抽出し、重心を計算する関数"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lower, upper) in hsv_ranges:
        lower_bound = np.array(lower, dtype=np.uint8)
        upper_bound = np.array(upper, dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_bound, upper_bound))

    # オープニング（細い線を消す）
    kernel = np.ones((noise_kernel, noise_kernel), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pixel_points = []
    real_data_points = []
    x_cal, y_cal = calibration['x'], calibration['y']

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                pixel_points.append((cX, cY))
                
                data_x = x_cal['min_val'] + (cX - x_cal['min_px']) * (x_cal['max_val'] - x_cal['min_val']) / (x_cal['max_px'] - x_cal['min_px'])
                data_y = y_cal['min_val'] + (cY - y_cal['min_px']) * (y_cal['max_val'] - y_cal['min_val']) / (y_cal['max_px'] - y_cal['min_px'])
                
                real_data_points.append((data_x, data_y))

    return real_data_points, pixel_points, mask_cleaned


def extract_and_calibrate_curve(img, hsv_ranges, calibration):
    """【モード2】「曲線（実線等）」をX軸方向にスキャンして抽出する関数"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lower, upper) in hsv_ranges:
        lower_bound = np.array(lower, dtype=np.uint8)
        upper_bound = np.array(upper, dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_bound, upper_bound))

    # 曲線の場合は逆に「クロージング（穴埋め）」をして線が途切れないようにする
    kernel = np.ones((3, 3), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    pixel_points = []
    real_data_points = []
    x_cal, y_cal = calibration['x'], calibration['y']

    height, width = mask_cleaned.shape
    # X軸方向（画像の幅）に1ピクセルずつスキャン
    for cX in range(width):
        # そのX座標（列）にある、色が塗られているY座標のインデックスを取得
        y_indices = np.where(mask_cleaned[:, cX] > 0)[0]
        
        if len(y_indices) > 0:
            # 線の太さがある場合、その中心（平均値）をとる
            cY = int(np.mean(y_indices))
            pixel_points.append((cX, cY))

            data_x = x_cal['min_val'] + (cX - x_cal['min_px']) * (x_cal['max_val'] - x_cal['min_val']) / (x_cal['max_px'] - x_cal['min_px'])
            data_y = y_cal['min_val'] + (cY - y_cal['min_px']) * (y_cal['max_val'] - y_cal['min_val']) / (y_cal['max_px'] - y_cal['min_px'])
            
            real_data_points.append((data_x, data_y))

    return real_data_points, pixel_points, mask_cleaned


# ==========================================
# 実行ブロック
# ==========================================
if __name__ == "__main__":
    # --- 1. エクスプローラーでファイルを選択 ---
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    
    image_path = filedialog.askopenfilename(
        title="解析するグラフ画像を選択してください",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp")]
    )
    
    if not image_path:
        print("キャンセルされました。")
        sys.exit()

    img = imread_japanese(image_path)
    if img is None:
        print("画像の読み込みに失敗しました。")
        sys.exit()

    # --- 2. 画像をクリックして軸のピクセル座標を取得 ---
    print("\n" + "="*50)
    print("画像ウィンドウが開きます。以下の順番で4箇所をクリックしてください。")
    print("1. X軸の 最小値 の位置（例: 0）")
    print("2. X軸の 最大値 の位置（例: 100）")
    print("3. Y軸の 最小値 の位置（例: 0.0）")
    print("4. Y軸の 最大値 の位置（例: 1.0）")
    print("="*50 + "\n")

    fig_cal, ax_cal = plt.subplots(figsize=(10, 8))
    ax_cal.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax_cal.set_title("Click 4 points: X_min, X_max, Y_min, Y_max")
    
    pts = plt.ginput(4, timeout=0) 
    plt.close(fig_cal)

    if len(pts) != 4:
        print("4点クリックされませんでした。処理を中断します。")
        sys.exit()

    x_min_px, x_max_px = pts[0][0], pts[1][0]
    y_min_px, y_max_px = pts[2][1], pts[3][1]

    # --- 3. 実際の数値をターミナルから入力 ---
    print("\nクリックした位置の「実際の数値」を入力してください。")
    try:
        x_min_val = float(input("1. X軸の最小値: "))
        x_max_val = float(input("2. X軸の最大値: "))
        y_min_val = float(input("3. Y軸の最小値: "))
        y_max_val = float(input("4. Y軸の最大値: "))
    except ValueError:
        print("数値以外が入力されました。処理を中断します。")
        sys.exit()

    calibration = {
        'x': {'min_px': x_min_px, 'min_val': x_min_val, 'max_px': x_max_px, 'max_val': x_max_val},
        'y': {'min_px': y_min_px, 'min_val': y_min_val, 'max_px': y_max_px, 'max_val': y_max_val}
    }

    # --- 4. モード選択と抽出条件の設定 ---
    print("\n" + "="*50)
    print("抽出モードを選択してください：")
    print(" [1] データ点抽出 (散布図のマーカー)")
    print(" [2] 曲線抽出 (実線、破線などの理論曲線)")
    mode_choice = input("番号を入力 (1 or 2): ").strip()
    
    # --- 4. 抽出条件（色：赤色）の設定 ---
    # HSV空間における赤色の範囲。0付近と180付近の2つの範囲を指定します。
    # [Hue(色相 0-179), Saturation(彩度 0-255), Value(明度 0-255)]
    # 画像の色が薄い場合は、Saturationの下限（ここでは100）を50位に下げてください。
    #color_ranges = [
    #    ([0, 100, 100], [10, 255, 255]),   # 赤色範囲1（オレンジ寄り）
    #    ([170, 100, 100], [180, 255, 255]) # 赤色範囲2（紫寄り）
    #]
    #color_ranges = [
    #    ([80, 100, 100], [105, 255, 255])  # シアンの範囲 (H:80~105)
    #]
    #color_ranges = [
    #    ([140, 100, 100], [165, 255, 255]) # マゼンタの範囲 (H:140~165)
    #]
    color_ranges = [
        ([0, 0, 0], [179, 255, 50])  # 黒色の範囲 (明度 V が 0〜50 の範囲)
    ]
    
    # --- 5. 処理の実行 ---
    if mode_choice == '1':
        data_pts, px_pts, mask_cleaned = extract_and_calibrate_points(
            img, color_ranges, calibration, min_area=10, noise_kernel=3
        )
    elif mode_choice == '2':
        data_pts, px_pts, mask_cleaned = extract_and_calibrate_curve(
            img, color_ranges, calibration
        )
    else:
        print("無効な選択です。処理を終了します。")
        sys.exit()
    
    # グラフのX軸範囲外のデータを弾く（画像端の枠線などを拾ってしまうのを防ぐため）
    filtered_data_pts = []
    for pt in data_pts:
        if x_min_val <= pt[0] <= x_max_val:
            filtered_data_pts.append(pt)
    data_pts = filtered_data_pts

    print(f"\n抽出されたデータ数: {len(data_pts)} 件")
    print("-" * 50)
    for i, pt in enumerate(data_pts[:10]):
        print(f"データ {i+1:02d}: 実データ (X={pt[0]:.3f}, Y={pt[1]:.3f})")
    if len(data_pts) > 10:
        print("...")

    # --- 6. 結果の可視化 ---
    fig_res, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.imshow(mask_cleaned, cmap='gray')
    ax1.set_title(f"Cleaned Mask (Mode: {mode_choice})")
    ax1.axis('off')
    
    if data_pts:
        data_x = [p[0] for p in data_pts]
        data_y = [p[1] for p in data_pts]
        
        # モードに合わせてプロットの見た目を変更
        if mode_choice == '1':
            ax2.scatter(data_x, data_y, color='red', marker='o', s=20, alpha=0.6)
        else:
            # 曲線モードの場合は小さい点で連続的に描画
            ax2.plot(data_x, data_y, color='red', marker='.', markersize=2, linestyle='-', linewidth=1, alpha=0.8)
            
        ax2.set_xlim(calibration['x']['min_val'], calibration['x']['max_val'])
        ax2.set_ylim(calibration['y']['min_val'], calibration['y']['max_val'])
        ax2.grid(True)
        ax2.set_title("Reconstructed Data")
        ax2.set_xlabel("X-axis Value")
        ax2.set_ylabel("Y-axis Value")
        
    plt.tight_layout()
    plt.show()

    # --- 7. データの保存 (CSV出力) ---
    print("\n" + "-" * 50)
    save_choice = input("データをCSVファイルとして保存しますか？ (y/n): ").strip().lower()
    
    if save_choice == 'y':
        save_path = filedialog.asksaveasfilename(
            title="保存先を指定してください",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="extracted_curve.csv" if mode_choice == '2' else "extracted_points.csv"
        )
        
        if save_path:
            try:
                with open(save_path, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['X_Value', 'Y_Value']) 
                    for pt in data_pts:
                        writer.writerow([f"{pt[0]:.5f}", f"{pt[1]:.5f}"])
                print(f"\n✅ データを保存しました: {save_path}")
            except Exception as e:
                print(f"\n❌ 保存中にエラーが発生しました: {e}")
        else:
            print("\n保存がキャンセルされました。")
    else:
        print("\n保存せずに終了します。")