# -*- coding: utf-8 -*-
"""
check_orbit_plot.py
==============================================================================
generate_orbit_spice.pyで生成された水星軌道データ
(orbit2025_spice.txt)を視覚的にチェックするスクリプト。

- 太陽中心の軌道アニメーションを表示します。
- 水星上にプライムメリディアン(経度0度)と太陽直下点をプロットし、
  TAAとSubSolarLonの関係が正しいかを確認します。
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os

# 水星の視覚的なサイズ（軌道に対する比率）
MERCURY_RADIUS_AU = 0.05  # 描画上の半径 (AUスケール)


def load_orbit_data(filename='orbit2025_spice.txt'):
    """生成された軌道データをロードする"""
    if not os.path.exists(filename):
        print(f"[Error] 必須ファイル '{filename}' が見つかりません。")
        print("先に 'generate_orbit_spice.py' を実行してファイルを生成してください。", file=sys.stderr)
        return None

    try:
        # TAA[deg], AU[-], Time[s], Vr[m/s], Vt[m/s], SubSolarLon[deg]
        data = np.loadtxt(filename, skiprows=1)
        print(f"'{filename}' から {len(data)} ステップのデータをロードしました。")
        return data

    except Exception as e:
        print(f"[Error] データファイルのロードまたは解析に失敗しました: {e}", file=sys.stderr)
        return None


def animate_orbit_check(data):
    """軌道アニメーションを生成する"""
    # データを分割
    taa_deg = data[:, 0]  # TAA [deg]
    r_au = data[:, 1]  # AU [-]
    ssl_deg = data[:, 5]  # SubSolarLon [deg]

    # ラジアンに変換
    taa_rad = np.radians(taa_deg)
    # ssl_rad = np.radians(ssl_deg) # SubSolarLonはそのまま角度として扱う

    # ----------------------------------------------
    # 太陽中心の軌道座標 (2D)
    # 軌道長軸をX軸に仮定 (簡略化されたチェック用)
    # ----------------------------------------------
    X_merc = r_au * np.cos(taa_rad)
    Y_merc = r_au * np.sin(taa_rad)

    # ----------------------------------------------
    # プロット設定
    # ----------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('black')
    ax.set_title("Orbit Data Checking (SPICE Data)")
    ax.set_xlabel("X (AU)")
    ax.set_ylabel("Y (AU)")
    ax.set_aspect('equal', adjustable='box')

    # 軸の範囲を設定 (軌道全体をカバー)
    max_r = np.max(r_au) * 1.1
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)

    # ----------------------------------------------
    # 描画要素の初期化
    # ----------------------------------------------
    # 1. 太陽 (中心)
    ax.plot(0, 0, 'o', color='yellow', markersize=15, label='太陽')

    # 2. 軌道全体 (参照用)
    ax.plot(X_merc, Y_merc, ':', color='gray', alpha=0.5, linewidth=0.5)

    # 3. 水星 (現在の位置)
    mercury_line, = ax.plot([], [], 'o', color='lightgray', markersize=20, zorder=5)

    # 4. 水星の自転マーカー（プライムメリディアン: 経度0度）
    # この線がSubSolarLonの計算結果によって回転する
    prime_meridian_line, = ax.plot([], [], '-', color='red', linewidth=3, zorder=6)

    # 5. 太陽直下点 (Sub-Solar Lon: SSL) がどこを向いているかを示すマーカー
    # SPICEの出力では、SubSolarLonはその点が太陽に向いているはずなので、
    # Prime Meridianの位置からSSLの角度を引いた位置に太陽が来る
    ssl_marker, = ax.plot([], [], 'o', color='cyan', markersize=5, zorder=7)

    # 6. テキスト表示 (TAAとSSLの値)
    info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, color='white',
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    # ----------------------------------------------
    # アニメーション更新関数
    # ----------------------------------------------
    def update(frame):
        r_f = r_au[frame]
        taa_f = taa_deg[frame]
        ssl_f = ssl_deg[frame]
        X_f, Y_f = X_merc[frame], Y_merc[frame]

        # ----------------------------------------------------
        # 水星の自転状態の計算
        # ----------------------------------------------------
        # TAA (軌道角度) を基準として、SubSolarLonが示す角度を調整する。
        # SubSolarLon: 太陽から見た水星の経度。これが0度なら、0度の点が太陽に向いている。
        #
        # Prime Meridian (経度0度) の、慣性座標系 (X, Y) における角度 (deg)
        # $\phi_{PM} = TAA - \lambda_{SSL}$
        # この角度が、水星が公転しながら自転している向きを示す
        # 水星から見て太陽は反対側にあるので +180度 する
        prime_meridian_angle_deg = (taa_f + 180.0 - ssl_f) % 360.0
        pm_rad = np.radians(prime_meridian_angle_deg)

        # ----------------------------------------------------
        # プライムメリディアンの描画
        # ----------------------------------------------------
        # 水星の中心からの線分の座標
        r_merc_vis = MERCURY_RADIUS_AU * 0.5  # 描画上のマーカー長さ
        pm_X_end = X_f + r_merc_vis * np.cos(pm_rad)
        pm_Y_end = Y_f + r_merc_vis * np.sin(pm_rad)

        prime_meridian_line.set_data([X_f, pm_X_end], [Y_f, pm_Y_end])

        # ----------------------------------------------------
        # 太陽直下点マーカーの描画
        # ----------------------------------------------------
        # SubSolarLonは「太陽に向いている点」の経度。
        # その点の慣性座標系における角度は、**太陽の方向** と一致するはず。
        # (TAAの角度 - 180度) の方向にマーカーを置くことで、
        # プライムメリディアンラインとの相対位置を確認する。
        # ただし、ここではシンプルに「太陽直下点の角度」を計算し、水星上に描く
        ssl_angle_in_inertial_deg = (taa_f + 180) % 360.0  # 常に太陽の方向(位置ベクトルの反対)

        # SSLはPrime Meridianからssl_f度離れている
        # 慣性座標系でのSSLの角度 $\phi_{SSL} = \phi_{PM} + \lambda_{SSL} = (TAA - \lambda_{SSL}) + \lambda_{SSL} = TAA$
        # これは間違っている。SSLは**太陽に向いている**ので、位置ベクトルとは逆向き。
        # 正しい：$\phi_{SSL} = TAA + 180^\circ$

        # SSLマーカーは、Prime Meridianから $\lambda_{SSL}$ だけずれた位置に描画
        ssl_marker_angle_rad = pm_rad + np.radians(ssl_f)
        ssl_X_end = X_f + r_merc_vis * np.cos(ssl_marker_angle_rad)
        ssl_Y_end = Y_f + r_merc_vis * np.sin(ssl_marker_angle_rad)

        ssl_marker.set_data([ssl_X_end], [ssl_Y_end])  # 点マーカーとして描画

        # ----------------------------------------------------
        # 描画要素の更新
        # ----------------------------------------------------
        mercury_line.set_data([X_f], [Y_f])


        info_text.set_text(f'Step: {frame}/{len(data) - 1}\n'
                           f'TAA: {taa_f:.2f} deg\n'
                           f'AU: {r_f:.6f}\n'
                           f'SSL: {ssl_f:.2f} deg\n'
                           f'PM Angle: {prime_meridian_angle_deg:.2f} deg')

        return mercury_line, prime_meridian_line, ssl_marker, info_text

    # アニメーションの生成と表示
    # 全ステップだと長すぎるので、1/20のデータを使用 (5年分を3日で1フレーム程度に)
    skip_frames = 20
    frames_to_use = np.arange(0, len(data), skip_frames)

    print(f"Generate Animation... (All {len(data)} Framing、{len(frames_to_use)} Using Frames)")

    ani = FuncAnimation(fig, update, frames=frames_to_use,
                        interval=100, blit=True)

    # plt.show() # アニメーションとして再生する場合

    # 最終的な確認画像を出力する場合 (ファイルとして保存)
    # ani.save('mercury_orbit_check.gif', writer='pillow', fps=10)

    # 最終的な確認画像を出力する場合 (最後のフレーム)
    update(len(data) - 1)
    print("最終フレームの状態を静止画として表示します。")
    plt.show()


# --- メイン処理 ---
if __name__ == '__main__':
    orbit_data = load_orbit_data()

    if orbit_data is not None:
        animate_orbit_check(orbit_data)