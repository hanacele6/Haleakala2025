import numpy as np


def debug_asin_comparison():
    """
    IDLとPythonのarcsin(asin)の計算結果を直接比較するためのデバッグ用スクリプト。
    惑星の北半球と南半球の対称な2点を選び、入力値と出力値を高精度で表示して、
    実装による系統的な誤差の有無を確認します。
    """
    # --- 1. IDLと同一の定数を設定 ---
    pi = np.pi
    r2d = np.rad2deg(1.0)
    R_pix = 7.362434 / 2 * 100  # 368.1217

    print("--- Python/NumPy arcsin() Debug ---")
    print(f"惑星半径 (R_pix): {R_pix:.17f}\n")

    # --- 2. デバッグ対象のy座標を設定 ---
    # 対称性を確認するため、同じ絶対値を持つ正と負のy座標を使う
    y_val = 150.5

    # --- 北半球の点 (y > 0) ---
    print("--- Case 1: 北半球 (y > 0) ---")
    input_val_north = y_val / R_pix
    result_rad_north = np.arcsin(input_val_north)
    result_deg_north = r2d * result_rad_north

    print(f"入力値 (y/R): {input_val_north:.17f}")
    print(f"出力 (radians): {result_rad_north:.17f}")
    print(f"出力 (degrees): {result_deg_north:.17f}\n")

    # --- 南半球の点 (y < 0) ---
    print("--- Case 2: 南半球 (y < 0) ---")
    input_val_south = -y_val / R_pix
    result_rad_south = np.arcsin(input_val_south)
    result_deg_south = r2d * result_rad_south

    print(f"入力値 (y/R): {input_val_south:.17f}")
    print(f"出力 (radians): {result_rad_south:.17f}")
    print(f"出力 (degrees): {result_deg_south:.17f}\n")

    # --- 差の確認 ---
    print("--- 差の絶対値 ( |North| - |South| ) ---")
    diff = np.abs(result_rad_north) - np.abs(result_rad_south)
    print(f"差 (radians): {diff:.17e}")


if __name__ == '__main__':
    debug_asin_comparison()
