import pandas as pd
import numpy as np
from astropy.io import fits
import spiceypy as spice
import os

# --- 設定項目 ---
# ご自身の環境に合わせて、SPICEカーネルをまとめたフォルダのパスに変更してください
SPICE_KERNEL_DIR = "C:/Users/hanac/University/Senior/Mercury/Haleakala2025/kernels"
CSV_FILE = "mcparams20251029.csv"


# ▼▼▼【拡張機能】明け方/夕方を判断するための関数 ▼▼▼
def get_terminator_side(et):
    """
    指定された時刻(et)において、地球から見た水星が「明け方」側か「夕方」側かを判断する。
    """
    try:
        # 地球から見た水星の中心点の経度を計算
        subp_vec, _, _ = spice.subpnt('INTERCEPT/ELLIPSOID', 'MERCURY', et, 'IAU_MERCURY', 'LT+S', 'EARTH')
        _, obs_lon_rad, _ = spice.reclat(subp_vec)

        # 水星で太陽が真上にある点の経度を計算
        subsol_vec, _, _ = spice.subslr('INTERCEPT/ELLIPSOID', 'MERCURY', et, 'IAU_MERCURY', 'LT+S', 'SUN')
        _, noon_lon_rad, _ = spice.reclat(subsol_vec)

        # 経度を度数法に変換
        obs_lon_deg = obs_lon_rad * spice.dpr()
        noon_lon_deg = noon_lon_rad * spice.dpr()

        # 観測面中心と正午地点の経度差を計算 (-180度から+180度の範囲)
        delta_lon = (obs_lon_deg - noon_lon_deg + 540) % 360 - 180

        # 経度差に基づいて方角を判断
        if delta_lon >= 0:
            side = "Dusk"
        else:
            side = "Dawn"

        # ラベルと、数値（経度差）の両方を返す
        return side, delta_lon

    except Exception:
        # エラー時はNA（Not a Number）を返す
        return np.nan, np.nan


def planazel(et, target='MERCURY', lon=203.742, lat=20.708, alt=3043.0):
    """地心位置から見た天体の方向を計算する"""
    earth_radii = spice.bodvrd('EARTH', 'RADII', 3)[1]
    equatorial_radius = earth_radii[0]
    flattening = (earth_radii[0] - earth_radii[2]) / earth_radii[0]
    obs_pos = spice.georec(lon * spice.rpd(), lat * spice.rpd(), alt / 1000.0, equatorial_radius, flattening)
    par, lt = spice.azlcpo("ELLIPSOID", target, et, 'LT+S', True, True, obs_pos, 'EARTH', 'ITRF93')
    az, el, ddist, ddot = par[1], par[2], par[0], par[3]
    return az * spice.dpr(), el * spice.dpr(), ddist, ddot


def update_csv_with_spice(csv_path, kernel_dir):
    """
    FITSファイルのパスが記載されたCSVを読み込み、SPICEを使って天文データを計算し、
    指定の列順で上書き保存する。
    """
    try:
        spice.furnsh(os.path.join(kernel_dir, "lsk/naif0012.tls"))
        spice.furnsh(os.path.join(kernel_dir, "pck/pck00011.tpc"))
        spice.furnsh(os.path.join(kernel_dir, "spk/planets/de430.bsp"))
        spice.furnsh(os.path.join(kernel_dir, "pck/earth_000101_251119_250824.bpc"))
    except Exception as e:
        print(f"SPICEカーネルの読み込みに失敗しました: {e}")
        return

    df = pd.read_csv(csv_path)

    if len(df.columns) < 2:
        print(f"エラー: CSVファイルには少なくとも2列（ファイルパス、説明）が必要です。")
        spice.kclear()
        return

    user_cols = df.columns[:2].tolist()
    fits_col_name = user_cols[0]

    date_col = ['DATE-OBS']
    numeric_cols = [
        'apparent_diameter_arcsec', 'mercury_sun_distance_au',
        'mercury_sun_radial_velocity_km_s', 'mercury_earth_radial_velocity_km_s',
        'phase_angle_deg', 'true_anomaly_deg',
        'ecliptic_longitude_deg', 'ecliptic_latitude_deg'
    ]
    # ▼▼▼【拡張機能】新しい列名を追加 ▼▼▼
    terminator_cols = ['terminator_side', 'terminator_lon_delta_deg']

    # スクリプトが管理する全ての列
    script_cols = date_col + numeric_cols + terminator_cols

    for col in script_cols:
        if col not in df.columns:
            df[col] = pd.NA

    rows_to_process = df[df[script_cols].isnull().any(axis=1)]
    if rows_to_process.empty:
        print("新しいFITSファイルの追加はありません。すべてのデータは計算済みです。")
        spice.kclear()
        return

    print(f"{len(rows_to_process)}件の新しいFITSファイルについて、データの計算を開始します...")

    AU = 149597870.7
    mu_sun = 1.32712440018e11

    try:
        radius_km = spice.bodvrd('MERCURY', 'RADII', 3)[1][0]
    except Exception:
        print("警告: 水星の半径をカーネルから取得できませんでした。固定値 2439.7 km を使用します。")
        radius_km = 2439.7

    for index, row in rows_to_process.iterrows():
        fits_path = row[fits_col_name]
        if not (isinstance(fits_path, str) and os.path.exists(fits_path)):
            print(f"ファイルが見つからないかパスが不正です: {fits_path}")
            continue

        try:
            with fits.open(fits_path) as hdul:
                header = hdul[0].header
                date_obs_str = header.get('EXPMID', header.get('DATE'))

            et = spice.str2et(date_obs_str)

            # --- 元の計算 ---
            az, el, ddist, ddot = planazel(et, 'MERCURY')
            apparent_diameter_arcsec = np.degrees(2 * np.arctan(radius_km / ddist)) * 3600
            posSP, _ = spice.spkezr('MERCURY', et, 'J2000', 'LT+S', 'SUN')
            posEP, _ = spice.spkezr('MERCURY', et, 'J2000', 'LT+S', 'EARTH')
            mercury_sun_dist_km = spice.vnorm(posSP[:3])
            mercury_sun_radial_velocity = spice.vdot(posSP[:3], posSP[3:]) / mercury_sun_dist_km
            phase_angle_deg = np.degrees(spice.vsep(posSP[:3], posEP[:3]))
            pos_ecliptic, _ = spice.spkezr('MERCURY', et, 'ECLIPJ2000', 'LT+S', 'SUN')
            _, lon_rad, lat_rad = spice.reclat(pos_ecliptic[:3])
            elon_deg = (lon_rad * spice.dpr()) % 360
            beta_deg = lat_rad * spice.dpr()
            orbital_elements = spice.oscelt(posSP, et, mu_sun)
            rp, ecc = orbital_elements[0], orbital_elements[1]
            p = rp * (1.0 + ecc)
            cos_nu = np.clip((p / mercury_sun_dist_km - 1.0) / ecc, -1.0, 1.0)
            nu_rad = np.arccos(cos_nu)
            if mercury_sun_radial_velocity < 0: nu_rad = (2 * np.pi) - nu_rad
            true_anomaly_deg = np.degrees(nu_rad)

            # ▼▼▼【拡張機能】明け方/夕方を計算 ▼▼▼
            side_label, lon_delta = get_terminator_side(et)

            # --- DataFrameへの書き込み ---
            df.loc[index, 'DATE-OBS'] = date_obs_str
            df.loc[index, 'apparent_diameter_arcsec'] = apparent_diameter_arcsec
            df.loc[index, 'mercury_sun_distance_au'] = mercury_sun_dist_km / AU
            df.loc[index, 'mercury_sun_radial_velocity_km_s'] = mercury_sun_radial_velocity
            df.loc[index, 'mercury_earth_radial_velocity_km_s'] = ddot
            df.loc[index, 'phase_angle_deg'] = phase_angle_deg
            df.loc[index, 'true_anomaly_deg'] = true_anomaly_deg
            df.loc[index, 'ecliptic_longitude_deg'] = elon_deg
            df.loc[index, 'ecliptic_latitude_deg'] = beta_deg

            # ▼▼▼【拡張機能】新しいデータを書き込み ▼▼▼
            df.loc[index, 'terminator_side'] = side_label
            df.loc[index, 'terminator_lon_delta_deg'] = lon_delta

            # コンソールにも表示
            print(f"処理完了: {os.path.basename(fits_path)} -> {side_label} (経度差: {lon_delta:.1f}°)")

        except Exception as e:
            print(f"ファイル '{os.path.basename(fits_path)}' の処理中にエラーが発生しました: {e}")

    final_column_order = user_cols + script_cols
    df_final = df[final_column_order]

    df_final.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\n計算と追記が完了し、'{csv_path}' を更新しました。")
    print("\n--- 更新後のファイルプレビュー ---")
    print(df_final.round(6))

    spice.kclear()


if __name__ == "__main__":
    update_csv_with_spice(CSV_FILE, SPICE_KERNEL_DIR)