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

最終的に、以下の5つの物理量を含むテキストファイルを出力します。
1. TAA (True Anomaly Angle) [deg]: 真近点角（太陽から見た水星の角度）
2. AU [-]: 太陽からの距離（天文単位）
3. Time [s]: シミュレーション開始（近日点通過）からの経過時間
4. V_radial_ms [m/s]: 視線速度（太陽から遠ざかる方向が正）
5. V_tangential_ms [m/s]: 接線速度（軌道運動の反時計回りが正）

このバージョンでは、入力された物理定数間の矛盾を解消しています。
"""

# --- 必要なライブラリのインポート ---
import numpy as np  # 数値計算ライブラリ (配列操作、数学関数など)
from scipy.integrate import solve_ivp  # 常微分方程式(ODE)ソルバー
import sys  # システム関連（エラー終了用）

# --- 1. 物理定数と軌道要素の定義 ---

# 太陽の重力定数 (G * M_sun) [m^3/s^2]
# G: 万有引力定数, M_sun: 太陽質量
# この値は観測から非常に高い精度で求まっている
GM_SUN = 1.32712440018e20

# 水星の離心率 (無次元)
# 軌道の「潰れ具合」を示す。0なら真円、1なら放物線。
e = 0.205630

# 水星の公転周期（恒星時周期）[秒]
# 恒星時周期: 宇宙空間の固定された星々を基準に、水星が太陽を1周する時間
# 87.969 [日] * 24 [時間/日] * 3600 [秒/時間]
T_sec = 87.969 * 24 * 3600

# --- 2. 物理定数の一貫性を保証する計算 ---

# ★★★ 物理定数の矛盾解消 ★★★
#
# 以前のコードでは、`a` (軌道長半径) を `0.387098 * AU` から計算し、
# `AU` (天文単位) を `1.496e11` m という「丸められた値」で定義していました。
#
# しかし、`GM_SUN`, `T_sec`, `a` の3つは、ケプラーの第三法則によって
# T^2 = (4 * pi^2 * a^3) / GM
# という関係で厳密に結びついていなければなりません。
#
# `GM_SUN` と `T_sec` (観測された周期) は信頼性が高いため、
# この2つの値から、シミュレーションで使うべき `a` [m] を逆算します。
# これにより、シミュレーションのすべての定数が物理法則と一致します。
#
# a^3 = (T_sec^2 * GM_SUN) / (4 * pi^2)
# a = ( (T_sec^2 * GM_SUN) / (4 * pi^2) )^(1/3)
#
a_metres = ((T_sec**2 * GM_SUN) / (4 * np.pi**2))**(1 / 3)

# (参考) この厳密な `a_metres` (メートル単位の軌道長半径) と
# 水星の軌道長半径の定義 `0.387098 AU` を使って、
# このシミュレーションと一貫性のある `AU` の値 [m] を逆算します。
# この `AU_consistent` を、のちのメートルからAUへの単位換算で使用します。
AU_consistent = a_metres / 0.387098

# --- 3. 軌道運動の微分方程式の定義 ---

def orbital_motion(t, y_state):
    """
    太陽の重力下での惑星の運動を記述する微分方程式（二体問題）。
    `solve_ivp` から呼び出される関数。

    この関数は、ある時刻 t における状態ベクトル y_state (位置と速度) を受け取り、
    その瞬間の「状態の変化率」（速度と加速度）を返します。

    パラメータ:
    ----------
    t : float
        現在の時刻 [s] (このモデルではtに依存しないが、ソルバーには必要)
    y_state : array_like (要素数4)
        現在の状態ベクトル [x, y_pos, vx, vy]
        x: x位置 [m]
        y_pos: y位置 [m] (Pythonの予約語 'y' との衝突を避けるため 'y_pos' と命名)
        vx: x速度 [m/s]
        vy: y速度 [m/s]

    戻り値:
    -------
    list (要素数4)
        状態ベクトルの時間微分 [dx/dt, dy/dt, dvx/dt, dvy/dt]
        dx/dt = vx
        dy/dt = vy
        dvx/dt = ax (x方向の加速度)
        dvy/dt = ay (y方向の加速度)
    """
    # 状態ベクトルを展開
    x, y_pos, vx, vy = y_state

    # 太陽（原点 0,0）から水星までの距離 r を計算
    r_sq = x**2 + y_pos**2  # 距離の2乗 r^2

    # ゼロ除算の回避: 万が一、原点ピッタリにいた場合の安全装置
    if r_sq == 0:
        return [vx, vy, 0, 0]

    r = np.sqrt(r_sq)  # 距離 r

    # 万有引力の法則 F = G * M * m / r^2
    # 加速度 a = F / m = G * M / r^2
    # 加速度ベクトル a_vec = -(GM / r^3) * r_vec
    # (r_vec = [x, y_pos], マイナス符号は引力を示す)

    # 加速度のx成分: ax = -(GM_SUN / r^3) * x
    ax = -GM_SUN * x / r**3
    # 加速度のy成分: ay = -(GM_SUN / r^3) * y_pos
    ay = -GM_SUN * y_pos / r**3

    # 状態ベクトルの時間微分 [vx, vy, ax, ay] を返す
    return [vx, vy, ax, ay]


# --- 4. 軌道データ生成とファイル出力のメイン関数 ---

def generate_orbit_file_new_order():
    """
    軌道積分を実行し、結果を指定された5列の形式でファイルに保存する。
    """

    # --- 4a. SciPyの存在チェック ---
    try:
        # このスクリプトは scipy.integrate.solve_ivp に依存するため、
        # ライブラリが存在するかどうかを最初に確認する
        from scipy.integrate import solve_ivp
    except ImportError:
        # 存在しない場合はエラーメッセージを表示して終了
        print("エラー: このスクリプトの実行には SciPy が必要です。", file=sys.stderr)
        print("ターミナルで `pip install scipy` を実行してインストールしてください。", file=sys.stderr)
        sys.exit(1)  # エラーコード 1 で終了

    print(f"水星の楕円軌道データを生成しています (使用する周期 T = {T_sec} s)...")
    print(f"ケプラーの法則と一致させた軌道長半径 a = {a_metres:.6e} m を使用します。")
    print(f"単位換算に使用する 1 AU = {AU_consistent:.6e} m")

    # --- 4b. 初期条件の設定 (t=0) ---
    # シミュレーションの開始点 (t=0) を近日点（太陽に最も近い点）とする。
    # 近日点は TAA (真近点角) = 0 度 の点。
    # x軸の正の方向に水星が位置するように座標系を設定する。

    # 近日点距離 r_perihelion = a * (1 - e)
    r_perihelion = a_metres * (1 - e)

    # 近日点での速度 v_perihelion
    # エネルギー保存則（活力方程式） v^2 = GM * (2/r - 1/a) より
    v_perihelion = np.sqrt(GM_SUN * ((2 / r_perihelion) - (1 / a_metres)))

    # 初期状態ベクトル y0 = [x, y, vx, vy]
    # 位置: (r_perihelion, 0)
    # 速度: (0, v_perihelion)
    # ※近日点では、速度ベクトルは位置ベクトルと直交する（接線方向のみ）
    y0 = [r_perihelion, 0, 0, v_perihelion]

    # --- 4c. 軌道積分（数値シミュレーション）の実行 ---

    # 積分期間: t=0 (近日点) から t=T_sec (1公転後) まで
    t_span = [0, T_sec]

    # 結果を評価（保存）する時間点の数
    # TAA 0度から360度までを約0.1度未満の解像度で得るために 3601 点以上を指定
    n_steps = 3601

    # 結果を評価する時間点の配列を生成 [0, ..., T_sec]
    # `np.linspace` は指定した開始点、終了点、点数で等間隔の配列を生成する
    t_eval = np.linspace(t_span[0], t_span[1], n_steps)

    print("軌道積分を実行中 (solve_ivp)...")
    # 常微分方程式ソルバーを実行
    sol = solve_ivp(
        fun=orbital_motion,  # 呼び出す微分方程式
        t_span=t_span,       # 積分期間 [t_start, t_end]
        y0=y0,               # 初期条件
        t_eval=t_eval,       # 結果を出力する時間点
        method='RK45',       # 積分アルゴリズム（デフォルト、高精度）
        rtol=1e-9,           # 相対許容誤差（計算精度を上げるため小さめに設定）
        atol=1e-9            # 絶対許容誤差（同上）
    )
    print("積分完了。")

    # --- 4d. 積分結果の物理量への変換 ---

    # ソルバーの結果 (sol オブジェクト) から時間と状態ベクトルを取り出す
    t = sol.t      # 時間 [s] の配列 (t_eval とほぼ同じ)
    x = sol.y[0]   # x位置 [m] の配列
    y = sol.y[1]   # y位置 [m] の配列
    vx = sol.y[2]  # x速度 [m/s] の配列
    vy = sol.y[3]  # y速度 [m/s] の配列

    # 1. 太陽からの距離 r を計算
    r_metres = np.sqrt(x**2 + y**2)  # [m]
    # 2. 距離を AU に単位換算
    #    物理定数と一貫性のある AU_consistent を用いる
    r_au = r_metres / AU_consistent  # [AU]

    # 3. 視線速度 (Radial Velocity) を計算
    #    速度ベクトル v = (vx, vy) と
    #    位置ベクトル r = (x, y) の内積 (ドット積) を利用
    #    v_radial = (v・r) / |r|
    #    (v・r) = (vx * x + vy * y)
    #    |r| = r_metres
    v_radial_ms = (x * vx + y * vy) / r_metres  # [m/s]
    # (太陽から遠ざかる方向が正、近づく方向が負)

    # 4. 接線速度 (Tangential Velocity) を計算
    #    位置ベクトル r と速度ベクトル v の外積（クロス積）のz成分を利用
    #    (r × v)_z = x * vy - y * vx
    #    角運動量 L = m * (r × v)
    #    接線速度 v_tan = (r × v)_z / |r|
    v_tangential_ms = (x * vy - y * vx) / r_metres  # [m/s]
    # (軌道運動の反時計回りを正とする)

    # 5. TAA (True Anomaly Angle) を計算
    #    `arctan(y/x)` の代わりに `arctan2(y, x)` を使う
    #    `arctan2` はxとyの符号から正しい象限(0-360度)の角度を返す
    taa_rad = np.arctan2(y, x)  # [radian]
    taa_deg = np.rad2deg(taa_rad)  # [degree]

    # 角度のアンラップ処理
    # `arctan2` は -180 から +180 の範囲で角度を返すため、
    # 180度を超える瞬間に +179 -> -179 のように値が飛んでしまう。
    # `np.unwrap` は、角度が飛んだ箇所を検出し、360度(period=360)を
    # 足し引きして、角度が連続的に増加するように補正する。
    # (例: ... 358, 359, 360, 361 ...)
    taa_deg = np.unwrap(taa_deg, period=360)

    # --- 4e. ファイルへの保存 ---

    # 出力する5つの列データをNumpy配列にまとめる
    # `np.column_stack` は、1次元配列のリストを縦に束ねて2次元配列にする
    output_data = np.column_stack([
        taa_deg,          # 1列目
        r_au,             # 2列目
        t,                # 3列目
        v_radial_ms,      # 4列目
        v_tangential_ms   # 5列目
    ])

    # 出力ファイル名
    # (v5 = 5列バージョン、 consistent = 物理定数整合済み)
    filename = 'orbit2025_v5_consistent.txt'

    # ファイルにテキストとして保存
    np.savetxt(
        fname=filename,        # ファイル名
        X=output_data,         # 保存するデータ配列
        fmt='%.6f',            # 書式指定（浮動小数点数、小数点以下6桁）
        # ヘッダー（先頭行）に列名を追加
        header='TAA[deg]  AU[-]  Time[s]  V_radial_ms[m/s]  V_tangential_ms[m/s]',
        comments='# '          # ヘッダー行の先頭に付ける文字
    )

    print("-" * 30)
    print(f"'{filename}' を生成しました。")
    print("シミュレーションコードのファイル名をこれに変更し、")
    print("コード内の列インデックスも調整してください。")
    print("生成されたファイルは以下の形式です:")
    print("# TAA[deg]  AU[-]  Time[s]  V_radial_ms[m/s]  V_tangential_ms[m/s]")
    print(f"データ例 (1行目):\n{output_data[0]}")
    print(f"データ例 (最終行):\n{output_data[-1]}")
    print("-" * 30)


# --- 5. スクリプトの実行 ---

# この .py ファイルが直接実行された場合のみ、
# (`import` されてモジュールとして使われた場合は実行しない)
if __name__ == '__main__':
    # メイン関数を実行する
    generate_orbit_file_new_order()