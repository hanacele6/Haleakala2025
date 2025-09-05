import numpy as np
from astropy.io import fits
from pathlib import Path
import warnings
import pandas as pd
from datetime import datetime, timedelta  # 日付計算のために追加


def run_hapke_model_corrected():
    """
    IDLの 'pro hapke' コードをPythonで再現・高速化します。
    mcparams202505.csvから観測パラメータを読み込むように修正。
    DATE-OBSの日付誤差(±1日)に対応します。
    """
    # --- 1. パスの設定と日付の定義 ---
    pi = np.pi
    d2r = np.deg2rad(1.0)
    r2d = np.rad2deg(1.0)

    try:
        #base_path = Path.cwd()
        base_path = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
        # パラメータを読み込む基準の日付を指定
        date = "20250823"
        out_dir = base_path / 'output' / date
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"出力ディレクトリ: {out_dir.resolve()}")
    except Exception as e:
        print(f"パスの設定中にエラーが発生しました: {e}")
        return

    # --- 2. CSVファイルから観測パラメータを読み込む (日付誤差対応) ---
    try:
        csv_path = base_path / "2025ver" / f"mcparams{date}.csv"
        #csv_path = Path("mcparams20250711.csv")
        params_df = pd.read_csv(csv_path)

        # --- 日付の誤差に対応 ---
        # 指定された日付(date)とその前後1日を検索対象とする
        target_dt = datetime.strptime(date, '%Y%m%d')
        search_dates = [
            (target_dt - timedelta(days=1)).strftime('%Y-%m-%d'),  # 前日
            target_dt.strftime('%Y-%m-%d'),  # 当日
            (target_dt + timedelta(days=1)).strftime('%Y-%m-%d')  # 翌日
        ]

        # 'DATE-OBS'列の先頭10文字(YYYY-MM-DD)が検索日付リストに含まれる行を探す
        date_part = params_df['DATE-OBS'].astype(str).str[:10]
        rows = params_df[date_part.isin(search_dates)]

        if rows.empty:
            print(
                f"エラー: {csv_path.name} に日付 '{target_dt.strftime('%Y-%m-%d')}' (およびその前後1日) のデータが見つかりません。")
            return

        # 該当する最初の行を使用
        params = rows.iloc[0]

        # CSVから読み込んだ値で変数を設定
        g_deg = params['phase_angle_deg']
        AU = params['mercury_sun_distance_au']
        apparent_diameter = params['apparent_diameter_arcsec']
        R_pix = apparent_diameter / 2.0 * 100.0

        print("\n--- CSVから読み込んだパラメータ ---")
        print(f"検索日付(中心): {target_dt.strftime('%Y-%m-%d')}")
        print(f"使用した観測時刻 (DATE-OBS): {params['DATE-OBS']}")
        print(f"位相角 (g_deg): {g_deg:.4f}°")
        print(f"太陽距離 (AU): {AU:.6f}")
        print(f"視直径 (arcsec): {apparent_diameter:.6f}")
        print(f"惑星半径 (R_pix): {R_pix:.4f} pix")
        print("---------------------------------\n")

    except FileNotFoundError:
        print(f"エラー: パラメータファイル {csv_path.resolve()} が見つかりません。")
        return
    except KeyError as e:
        print(f"エラー: CSVファイルに列 '{e}' が見つかりません。列名を確認してください。")
        return
    except Exception as e:
        print(f"パラメータの読み込み中にエラーが発生しました: {e}")
        return

    # --- 3. 物理・モデル定数の設定 ---
    # (ここから下の部分は変更ありません)
    JL = 5.18e+14
    theta = np.deg2rad(16.0)
    w = 0.2
    h = 0.065
    B0 = 2.4
    b = 0.20
    c = 0.18

    print(f"シミュレーション開始: 位相角={g_deg:.2f}°, 水星半径={R_pix:.2f} pix")

    # --- 4. 計算グリッドとマスクの準備 ---
    dim = int(R_pix * 2)
    x_coords = np.arange(dim, dtype=np.float64) - R_pix + 0.5
    y_coords = np.arange(dim, dtype=np.float64) - R_pix + 0.5
    x, y = np.meshgrid(x_coords, y_coords)
    disk_mask = (x ** 2 + y ** 2) < R_pix ** 2

    # --- 5. 幾何学的な角度の計算 (ベクトル化) ---
    with np.errstate(divide='ignore', invalid='ignore'):
        g = np.deg2rad(g_deg)
        LL = np.arcsin(y / R_pix)
        LambdaL = np.arcsin(x / (R_pix * np.cos(LL)))
        terminator_mask = LambdaL < (pi / 2.0 - g)
        valid_mask = disk_mask & terminator_mask
        LL_v = LL[valid_mask]
        LambdaL_v = LambdaL[valid_mask]
        cosi = np.cos(LambdaL_v + g) * np.cos(LL_v)
        cose = np.cos(LambdaL_v) * np.cos(LL_v)
        cosi = np.clip(cosi, 0, 1)
        cose = np.clip(cose, 0, 1)
        i = np.arccos(cosi)
        e = np.arccos(cose)
        sini = np.sin(i)
        sine = np.sin(e)
        cospsi = (np.cos(g) - cosi * cose) / (np.sin(i) * np.sin(e))
        cospsi = np.clip(cospsi, -1, 1)
        psi = np.arccos(cospsi)

    # --- 6. ハプケモデルの計算 (ベクトル化) ---
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

    # --- 7. 結果の保存 ---
    SSR = SR.sum()
    ic = valid_mask.sum()
    avg_brightness = SSR / ic if ic > 0 else 0
    max_brightness = SR.max()

    dat_path = out_dir / f'{date}HapkeMRnm.dat'
    with open(dat_path, 'w') as f:
        f.write(f'{avg_brightness} {max_brightness}\n')
    print(f"平均/最大輝度を保存しました: {dat_path}")

    fits_path_sr = out_dir / f'Hapke{date}.fits'
    fits_path_rr = out_dir / 'test_python.fit'

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fits.verify.VerifyWarning)
        fits.writeto(fits_path_sr, np.rot90(SR2, k=2), overwrite=True)
        print(f"輝度画像 (MR/nm) を保存しました: {fits_path_sr}")
        fits.writeto(fits_path_rr, np.rot90(RR2, k=2), overwrite=True)
        print(f"反射輝度画像 (reflectivity/sr) を保存しました: {fits_path_rr}")

    print("\nend")
    print("処理が完了しました。")


# --- スクリプトの実行 ---
if __name__ == '__main__':
    run_hapke_model_corrected()