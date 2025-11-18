#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
水星の二体問題（太陽＋水星）に基づいた軌道計算スクリプト

このスクリプトは、水星の軌道要素（公転周期 T, 離心率 e）と
太陽の重力定数 (GM_SUN) を基に、ケプラーの第三法則を用いて
軌道長半径 a を厳密に計算します。

その後、SciPy の `solve_ivp` を用いてニュートンの万有引力の法則に
従う微分方程式を数値積分し、水星の軌道上の位置と速度を時系列で
生成します。

★ 更新 (v6) ★
シミュレーションで必須となる「惑星固定座標系での太陽直下点経度」を
6列目として追加で計算・出力するように変更しました。

最終的に、以下の【6つ】の物理量を含むテキストファイルを出力します。
1. TAA (True Anomaly Angle) [deg]: 真近点角（公転の角度）
2. AU [-]: 太陽からの距離（天文単位）
3. Time [s]: シミュレーション開始（近日点通過）からの経過時間
4. V_radial_ms [m/s]: 視線速度
5. V_tangential_ms [m/s]: 接線速度
6. SubSolarLon_Fixed[deg]: 惑星固定座標系での太陽直下点経度
"""

# --- 必要なライブラリのインポート ---
import numpy as np  # 数値計算ライブラリ (配列操作、数学関数など)
from scipy.integrate import solve_ivp  # 常微分方程式(ODE)ソルバー
import sys  # システム関連（エラー終了用）

# --- 1. 物理定数と軌道要素の定義 ---

# 太陽の重力定数 (G * M_sun) [m^3/s^2]
GM_SUN = 1.32712440018e20

# 水星の離心率 (無次元)
e = 0.205630

# 水星の公転周期（恒星時周期）[秒]
# 87.969 [日] * 24 [時間/日] * 3600 [秒/時間]
T_sec = 87.969 * 24 * 3600

# ★ 追加 ★
# 水星の自転周期（恒星時周期）[秒]
# 58.646 [日] * 24 [時間/日] * 3600 [秒/時間]
ROTATION_PERIOD_SEC = 58.646 * 24 * 3600

# --- 2. 物理定数の一貫性を保証する計算 ---

# ケプラーの第三法則から軌道長半径 a [m] を逆算
a_metres = ((T_sec**2 * GM_SUN) / (4 * np.pi**2))**(1 / 3)

# このシミュレーションと一貫性のある AU の値 [m] を逆算
AU_consistent = a_metres / 0.387098

# --- 3. 軌道運動の微分方程式の定義 ---

def orbital_motion(t, y_state):
    """
    太陽の重力下での惑星の運動を記述する微分方程式（二体問題）。
    """
    # 状態ベクトルを展開
    x, y_pos, vx, vy = y_state

    # 太陽（原点 0,0）から水星までの距離 r を計算
    r_sq = x**2 + y_pos**2
    if r_sq == 0:
        return [vx, vy, 0, 0]
    r = np.sqrt(r_sq)

    # 加速度のx成分: ax = -(GM_SUN / r^3) * x
    ax = -GM_SUN * x / r**3
    # 加速度のy成分: ay = -(GM_SUN / r^3) * y_pos
    ay = -GM_SUN * y_pos / r**3

    # 状態ベクトルの時間微分 [vx, vy, ax, ay] を返す
    return [vx, vy, ax, ay]


# --- 4. 軌道データ生成とファイル出力のメイン関数 ---

def generate_orbit_file_new_order():
    """
    軌道積分を実行し、結果を指定された【6列】の形式でファイルに保存する。
    """

    # --- 4a. SciPyの存在チェック ---
    try:
        from scipy.integrate import solve_ivp
    except ImportError:
        print("エラー: このスクリプトの実行には SciPy が必要です。", file=sys.stderr)
        print("ターミナルで `pip install scipy` を実行してインストールしてください。", file=sys.stderr)
        sys.exit(1)

    print(f"水星の楕円軌道データを生成しています (使用する周期 T = {T_sec} s)...")
    print(f"ケプラーの法則と一致させた軌道長半径 a = {a_metres:.6e} m を使用します。")
    print(f"単位換算に使用する 1 AU = {AU_consistent:.6e} m")
    print(f"★ 6列目計算用の自転周期: {ROTATION_PERIOD_SEC} s")

    # --- 4b. 初期条件の設定 (t=0) ---
    # 近日点距離 r_perihelion = a * (1 - e)
    r_perihelion = a_metres * (1 - e)

    # 近日点での速度 v_perihelion
    v_perihelion = np.sqrt(GM_SUN * ((2 / r_perihelion) - (1 / a_metres)))

    # 初期状態ベクトル y0 = [x, y, vx, vy]
    y0 = [r_perihelion, 0, 0, v_perihelion]

    # --- 4c. 軌道積分（数値シミュレーション）の実行 ---

    # 積分期間: t=0 (近日点) から t=T_sec (1公転後) まで
    t_span = [0, T_sec]
    n_steps = 3601
    t_eval = np.linspace(t_span[0], t_span[1], n_steps)

    print("軌道積分を実行中 (solve_ivp)...")
    sol = solve_ivp(
        fun=orbital_motion,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-9,
        atol=1e-9
    )
    print("積分完了。")

    # --- 4d. 積分結果の物理量への変換 ---

    t = sol.t      # 時間 [s] (インデックス 2)
    x = sol.y[0]
    y = sol.y[1]
    vx = sol.y[2]
    vy = sol.y[3]

    # 1. 太陽からの距離 r
    r_metres = np.sqrt(x**2 + y**2)
    # 2. 距離を AU に単位換算 (インデックス 1)
    r_au = r_metres / AU_consistent

    # 3. 視線速度 (インデックス 3)
    v_radial_ms = (x * vx + y * vy) / r_metres

    # 4. 接線速度 (インデックス 4)
    v_tangential_ms = (x * vy - y * vx) / r_metres

    # 5. TAA (True Anomaly Angle) (インデックス 0)
    taa_rad = np.arctan2(y, x)
    taa_deg = np.rad2deg(taa_rad)
    taa_deg = np.unwrap(taa_deg, period=360)

    # 6. 惑星固定座標系での太陽直下点経度 [deg] を計算

    # 物理モデル:
    # 水星は「公転周期 T_sec (87.97日)」で1周(360°)し、
    # 「自転周期 ROTATION_PERIOD_SEC (58.65日)」で1回自転します。
    # この 87.97 / 58.65 ≒ 1.5 という比率が「3:2スピン軌道共鳴」です。
    #
    # シミュレーションの t=0 (近日点) で、
    # 1. 公転の角度(TAA) = 0°
    # 2. 自転の角度 = 0°
    # 3. 太陽直下点経度 = 0°
    # と、すべてを揃えて定義します。

    # (1) 各時刻 t における「自転の角度」[deg] を計算
    # (t / ROTATION_PERIOD_SEC) で自転回数を計算し、360を掛ける
    rotation_angle_deg = (t / ROTATION_PERIOD_SEC) * 360.0

    # (2) 太陽直下点経度 = 公転の角度(TAA) - 自転の角度
    #    (例: 公転が90°進み、自転が60°しか進んでいなければ、
    #     太陽は経度 30° の位置に見える)
    subsolar_lon_deg = taa_deg - rotation_angle_deg

    # (3) 角度を -180° から +180° の範囲に正規化(ラップ)する
    #    この計算により、値は常に -180 から +180 の範囲に収まる
    subsolar_lon_deg_wrapped = (subsolar_lon_deg + 180) % 360 - 180



    # --- 4e. ファイルへの保存 ---

    # 出力する【6つ】の列データをNumpy配列にまとめる
    output_data = np.column_stack([
        taa_deg,                   # 1列目 (インデックス 0)
        r_au,                      # 2列目 (インデックス 1)
        t,                         # 3列目 (インデックス 2)
        v_radial_ms,               # 4列目 (インデックス 3)
        v_tangential_ms,           # 5列目 (インデックス 4)
        subsolar_lon_deg_wrapped   # 6列目 (インデックス 5) ★追加★
    ])

    # 出力ファイル名 (v6 に変更)
    filename = 'orbit2025_v6.txt'

    # ファイルにテキストとして保存
    np.savetxt(
        fname=filename,
        X=output_data,
        fmt='%.6f',
        # ヘッダー（先頭行）に列名を追加 ★修正★
        header='TAA[deg]  AU[-]  Time[s]  V_radial_ms[m/s]  V_tangential_ms[m/s]  SubSolarLon_Fixed[deg]',
        comments='# '
    )

    print("-" * 30)
    print(f"'{filename}' を生成しました。")
    print("シミュレーションコードとプロットコードのファイル名をこれに変更し、")
    print("列インデックスが 0～5 になっていることを確認してください。")
    print("生成されたファイルは以下の形式です:")
    print("# TAA[deg]  AU[-]  Time[s]  V_radial_ms[m/s]  V_tangential_ms[m/s]  SubSolarLon_Fixed[deg]")
    print(f"データ例 (1行目):\n{output_data[0]}")
    print(f"データ例 (最終行):\n{output_data[-1]}")
    print("-" * 30)


# --- 5. スクリプトの実行 ---

if __name__ == '__main__':
    generate_orbit_file_new_order()