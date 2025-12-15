# -*- coding: utf-8 -*-
"""
generate_orbit_spice.py
==============================================================================
SPICEカーネル(de442.bsp等)を使用して、NASAの精密暦に基づいた
水星の軌道データ(2025年)を生成するスクリプト。

出力形式は従来の 'orbit2025_v6.txt' と完全互換の6列形式です。
1. TAA [deg]: 真近点角 (Osculating Elementsより算出)
2. AU [-]: 太陽距離
3. Time [s]: 近日点通過からの経過時間
4. V_radial [m/s]: 視線速度
5. V_tangential [m/s]: 接線速度
6. SubSolarLon [deg]: 太陽直下点経度 (物理的秤動を含む厳密値)
==============================================================================
"""

import numpy as np
import spiceypy as spice
import sys
import os


def generate_precise_orbit_file():
    # --- 1. SPICEカーネルのロード ---
    # 必要なファイルが揃っているか確認
    required_kernels = ['de442.bsp', 'naif0012.tls', 'pck00011.tpc']
    for k in required_kernels:
        if not os.path.exists(k):
            print(f"[Error] 必須ファイル '{k}' が見つかりません。", file=sys.stderr)
            print("NASA NAIFサイト等からダウンロードして同じフォルダに置いてください。", file=sys.stderr)
            return

    try:
        spice.furnsh(required_kernels)
    except Exception as e:
        print(f"[Error] カーネルのロードに失敗: {e}")
        return

    print("SPICEカーネルをロードしました。精密軌道計算を開始します...")

    # --- 2. 計算期間の設定 (2025年の近日点通過付近から1水星年) ---
    # 2025年の水星の近日点はおよそ 2025-03-12 付近ですが、
    # ここでは計算開始日を指定し、そこから幾何学的な近日点を探します。

    # 検索開始日
    search_start_str = '2025-01-01'
    et_start_search = spice.str2et(search_start_str)

    # おおまかな1周分(90日)のデータを取得して近日点(距離最小)の時刻を特定
    search_duration = 100 * 24 * 3600
    times_search = np.linspace(et_start_search, et_start_search + search_duration, 10000)

    dists = []
    for et in times_search:
        # 太陽(10)から見た水星(199)の位置
        pos, _ = spice.spkpos('MERCURY', et, 'ECLIPJ2000', 'NONE', 'SUN')
        dists.append(spice.vnorm(pos))

    # 最も太陽に近い時刻(近日点)を特定 -> ここを Time=0 とする
    idx_min = np.argmin(dists)
    et_perihelion = times_search[idx_min]
    t_peri_str = spice.et2utc(et_perihelion, 'C', 0)
    print(f"特定された近日点時刻(Time=0): {t_peri_str} UTC")

    # --- 3. データ生成 (1水星年分 + マージン) ---
    # 水星公転周期 約88日
    #MERCURY_YEAR_DAYS = 88.0
    # ステップ数 (360度を十分カバーするように)
    #N_STEPS = 3601

    # --- 変更後：5年分 (Earth Years) 確保しておく ---
    # 5年 x 365日 = 1825日
    DURATION_DAYS = 5 * 365.0

    # 時間分解能を維持するためにステップ数も増やす
    # (例: 1日4回 = 6時間ごとのデータなら十分)
    N_STEPS = int(DURATION_DAYS * 10)

    #t_array = np.linspace(0, MERCURY_YEAR_DAYS * 24 * 3600, N_STEPS)
    t_array = np.linspace(0, DURATION_DAYS * 24 * 3600, N_STEPS)
    et_array = et_perihelion + t_array  # 近日点からの経過秒を加算

    output_rows = []

    # 太陽のGM (Standard Gravitational Parameter) [km^3/s^2]
    # SPICEから取得しても良いが、一貫性のため定数定義
    # (de442などでは値がわずかに異なる場合があるため bodvrd 推奨だが簡略化)
    GM_SUN_KM = 132712440041.939400  # TDB scale

    for i, et in enumerate(et_array):
        # A. 状態ベクトル (位置・速度) [km, km/s]
        # Frame: ECLIPJ2000 (黄道座標系)
        state, _ = spice.spkezr('MERCURY', et, 'ECLIPJ2000', 'NONE', 'SUN')
        pos = state[0:3]
        vel = state[3:6]

        # 距離 [km] -> [AU]
        r_km = spice.vnorm(pos)
        r_au = spice.convrt(r_km, 'KM', 'AU')

        # B. 速度成分の分解 (Radial, Tangential) [m/s]
        # v_radial = (r . v) / |r|
        v_rad_km_s = np.dot(pos, vel) / r_km

        # v_tangential = sqrt(|v|^2 - v_rad^2)
        v_sq = np.dot(vel, vel)
        v_tan_km_s = np.sqrt(v_sq - v_rad_km_s ** 2)

        # 単位変換 km/s -> m/s
        v_rad_ms = v_rad_km_s * 1000.0
        v_tan_ms = v_tan_km_s * 1000.0

        # C. 真近点角 (TAA) [deg]
        # osculating elements (接触軌道要素) を計算
        # elts[0] = p (parameter), [1] = e, ... [8] = nu (True Anomaly)
        elts = spice.oscltx(state, et, GM_SUN_KM)
        nu_rad = elts[8]  # 0 ~ 2pi
        taa_deg = np.degrees(nu_rad)

        # 0-360に正規化
        if taa_deg < 0: taa_deg += 360.0

        # D. 太陽直下点経度 (Sub-Solar Longitude) [deg]
        # ★ここが重要: 物理的な秤動(Libration)が自動的に考慮される
        # method: "Intercept: ellipsoid" or "Near point: ellipsoid"
        # fixref: "IAU_MERCURY" (水星固定座標系)
        # abcorr: "LT+S" (光行差+光路時間補正: 実際に表面が見る太陽の位置)
        # obsrvr: "SUN" (太陽から見た水星中心の表面点 = 水星から見た太陽直下点)

        # spice.subslr は「観測者から見た天体の表面点」を返す
        # ここでは「太陽(SUN)から見た水星(MERCURY)の直下点」を計算する
        spoint, trgepc, srfvec = spice.subslr('INTERCEPT/ELLIPSOID',
                                              'MERCURY', et, 'IAU_MERCURY', 'NONE', 'SUN')

        # 直交座標(spoint)を 経度・緯度・半径 に変換
        # r, lon, lat
        _, sub_lon_rad, _ = spice.reclat(spoint)
        sub_lon_deg = np.degrees(sub_lon_rad)

        # -180 ~ +180 の範囲に収める (reclatは -pi ~ pi を返すのでそのままでOKだが念のため)
        # シミュレーションコード側は -180~180 を期待している

        # データ格納
        output_rows.append([
            taa_deg,  # 1. TAA
            r_au,  # 2. AU
            t_array[i],  # 3. Time [s]
            v_rad_ms,  # 4. Vr [m/s]
            v_tan_ms,  # 5. Vt [m/s]
            sub_lon_deg  # 6. SubSolarLon [deg]
        ])

    # --- 4. ファイル出力 ---
    output_data = np.array(output_rows)
    filename = 'orbit2025_spice.txt'

    header_str = 'TAA[deg]  AU[-]  Time[s]  V_radial_ms[m/s]  V_tangential_ms[m/s]  SubSolarLon_Fixed[deg]'

    np.savetxt(filename, output_data, fmt='%.6f', header=header_str, comments='# ')

    print("-" * 30)
    print(f"'{filename}' を生成しました。")
    print("【以前のコードとの違い】")
    print("1. 軌道が楕円近似ではなく、実際の摂動を含んだ軌道です。")
    print("2. 速度ベクトルが厳密になり、ドップラーシフト計算(Na D線)の精度が向上します。")
    print("3. SubSolarLonが、3:2共鳴の近似式ではなく、実測の「秤動(Libration)」を含んだ値になります。")
    print("-" * 30)

    # カーネルのアンロード
    spice.kclear()


if __name__ == '__main__':
    generate_precise_orbit_file()