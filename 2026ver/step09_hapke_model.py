import numpy as np
from astropy.io import fits
from pathlib import Path
import warnings
import pandas as pd
from datetime import datetime, timedelta


def run(run_info, config):
    """
    パイプラインから呼び出されるHapkeモデルの実行関数
    """
    output_dir = run_info["output_dir"]
    csv_file_path = run_info["csv_path"]
    date_str = run_info["date"]

    # config.yaml から設定を読み込む
    hapke_conf = config.get("hapke", {}).get("constants", {})
    JL = hapke_conf.get("JL", 5.18e+14)
    theta_deg = hapke_conf.get("theta_deg", 16.0)
    w = hapke_conf.get("w", 0.2)
    h = hapke_conf.get("h", 0.065)
    B0 = hapke_conf.get("B0", 2.4)
    b = hapke_conf.get("b", 0.20)
    c = hapke_conf.get("c", 0.18)

    force_rerun = config.get("pipeline", {}).get("force_rerun_hapke", False)

    print(f"\n--- Hapkeモデル シミュレーションを開始します ---")

    # 予想される出力ファイル
    fits_path_sr = output_dir / f'Hapke{date_str}.fits'
    fits_path_rr = output_dir / 'test_python.fit'
    dat_path = output_dir / f'{date_str}HapkeMRnm.dat'

    # ▼▼▼ 極限まで実行を減らすスキップ処理 ▼▼▼
    if fits_path_sr.exists() and dat_path.exists() and not force_rerun:
        print(f"  > 処理済みスキップ: {fits_path_sr.name}")
        print("--- Hapkeモデル処理完了 ---")
        return

    # --- CSVからの読み込み (日付誤差対応) ---
    try:
        params_df = pd.read_csv(csv_file_path)
        target_dt = datetime.strptime(date_str, '%Y%m%d')
        search_dates = [
            (target_dt - timedelta(days=1)).strftime('%Y-%m-%d'),
            target_dt.strftime('%Y-%m-%d'),
            (target_dt + timedelta(days=1)).strftime('%Y-%m-%d')
        ]

        date_part = params_df['DATE-OBS'].astype(str).str[:10]
        rows = params_df[date_part.isin(search_dates)]

        if rows.empty:
            print(
                f"  > エラー: {csv_file_path.name} に日付 '{target_dt.strftime('%Y-%m-%d')}' (±1日) のデータがありません。")
            return

        params = rows.iloc[0]
        g_deg = params['phase_angle_deg']
        AU = params['mercury_sun_distance_au']
        apparent_diameter = params['apparent_diameter_arcsec']
        R_pix = apparent_diameter / 2.0 * 100.0

        print(f"  > 観測時刻: {params['DATE-OBS']} | 位相角: {g_deg:.2f}° | 半径: {R_pix:.2f} pix")

    except Exception as e:
        print(f"  > エラー: パラメータの読み込みに失敗しました: {e}")
        return

    # --- 物理計算 ---
    pi = np.pi
    theta = np.deg2rad(theta_deg)
    dim = int(R_pix * 2)
    x_coords = np.arange(dim, dtype=np.float64) - R_pix + 0.5
    y_coords = np.arange(dim, dtype=np.float64) - R_pix + 0.5
    x, y = np.meshgrid(x_coords, y_coords)
    disk_mask = (x ** 2 + y ** 2) < R_pix ** 2

    with np.errstate(divide='ignore', invalid='ignore'):
        g = np.deg2rad(g_deg)
        LL = np.arcsin(y / R_pix)
        LambdaL = np.arcsin(x / (R_pix * np.cos(LL)))
        terminator_mask = LambdaL < (pi / 2.0 - g)
        valid_mask = disk_mask & terminator_mask

        LL_v = LL[valid_mask]
        LambdaL_v = LambdaL[valid_mask]

        cosi = np.clip(np.cos(LambdaL_v + g) * np.cos(LL_v), 0, 1)
        cose = np.clip(np.cos(LambdaL_v) * np.cos(LL_v), 0, 1)
        i = np.arccos(cosi)
        e = np.arccos(cose)
        sini = np.sin(i)
        sine = np.sin(e)
        cospsi = np.clip((np.cos(g) - cosi * cose) / (np.sin(i) * np.sin(e)), -1, 1)
        psi = np.arccos(cospsi)

    # Hapke式の計算
    tantheta = np.tan(theta)
    tani = np.tan(i)
    tane = np.tan(e)
    sinpsihalf = np.sin(psi / 2.0)

    kai = 1.0 / np.sqrt(1.0 + pi * (tantheta) ** 2)
    E1i = np.exp(-2.0 / pi / tantheta / tani)
    E2i = np.exp(-1.0 / pi / tantheta ** 2 / tani ** 2)
    E1e = np.exp(-2.0 / pi / tantheta / tane)
    E2e = np.exp(-1.0 / pi / tantheta ** 2 / tane ** 2)

    mu0e0 = kai * (cosi + sini * tantheta * E2i / (2.0 - E1i))
    cond = (i <= e)

    mu0e_true = kai * (cosi + sini * tantheta * (cospsi * E2e + sinpsihalf ** 2 * E2i) / (2 - E1e - (psi / pi) * E1i))
    mue_true = kai * (cose + sine * tantheta * (E2e - sinpsihalf ** 2 * E2i) / (2 - E1e - (psi / pi) * E1i))
    mue0_true = kai * (cose + sine * tantheta * E2e / (2 - E1e))
    fpsi_true = np.exp(-2.0 * np.tan(psi / 2.0))
    Siepsi_true = (mue_true / mue0_true) * (cosi / mu0e0) * kai / (1 - fpsi_true + fpsi_true * kai * cosi / mu0e0)

    mu0e_false = kai * (cosi + sini * tantheta * (E2i - sinpsihalf ** 2 * E2e) / (2 - E1i - (psi / pi) * E1e))
    mue_false = kai * (cose + sine * tantheta * (cospsi * E2i + sinpsihalf ** 2 * E2e) / (2 - E1i - (psi / pi) * E1e))
    mue0_false = kai * (cose + sine * tantheta * E2e / (2 - E1e))
    fpsi_false = np.exp(-2.0 * np.tan(psi / 2.0))
    Siepsi_false = (mue_false / mue0_false) * (cosi / mu0e0) * kai / (
                1 - fpsi_false + fpsi_false * kai * cose / mue0_false)

    mu0e = np.where(cond, mu0e_true, mu0e_false)
    mue = np.where(cond, mue_true, mue_false)
    Siepsi = np.where(cond, Siepsi_true, Siepsi_false)

    Bg = B0 / (1.0 + np.tan(g / 2.0) / h)
    pg = 1.0 + b * np.cos(g) + c * (3.0 * (np.cos(g)) ** 2 - 1.0) / 2.0
    gamma = np.sqrt(1.0 - w)
    Hmu0e = (1.0 + 2.0 * mu0e) / (1.0 + 2.0 * gamma * mu0e)
    Hmue = (1.0 + 2.0 * mue) / (1.0 + 2.0 * gamma * mue)
    rRieg = (w / (4.0 * pi)) * (mu0e / (mu0e + mue)) * ((1.0 + Bg) * pg + Hmu0e * Hmue - 1.0) * Siepsi
    SR_values = (JL / AU ** 2) * rRieg * (4.0 * pi / 1e12)

    RR = np.zeros((dim, dim), dtype=np.float64)
    SR = np.zeros((dim, dim), dtype=np.float64)
    RR[valid_mask] = rRieg.astype(np.float64)
    SR[valid_mask] = SR_values.astype(np.float64)

    SR2 = np.fliplr(SR)
    RR2 = np.fliplr(RR)

    # --- 結果の保存 ---
    SSR = SR.sum()
    ic = valid_mask.sum()
    avg_brightness = SSR / ic if ic > 0 else 0
    max_brightness = SR.max()

    with open(dat_path, 'w') as f:
        f.write(f'{avg_brightness} {max_brightness}\n')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fits.verify.VerifyWarning)
        fits.writeto(fits_path_sr, np.rot90(SR2, k=2), overwrite=True)
        fits.writeto(fits_path_rr, np.rot90(RR2, k=2), overwrite=True)

    print(f"  > 保存完了: {fits_path_sr.name}, {fits_path_rr.name}, {dat_path.name}")
    print("--- Hapkeモデル処理完了 ---")


if __name__ == '__main__':
    print("このスクリプトは main.py からモジュールとして呼び出してください。")