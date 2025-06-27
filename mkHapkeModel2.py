import numpy as np
from astropy.io import fits
from pathlib import Path
import warnings


def run_hapke_model_corrected():
    """
    IDLの 'pro hapke' コードをPythonで再現・高速化します。
    (mu0e0_false の未定義エラーを修正)
    """
    # --- 1. 定数と物理パラメータの設定 ---
    # ... (この部分は変更ありません) ...
    pi = np.pi
    d2r = np.deg2rad(1.0)
    r2d = np.rad2deg(1.0)

    try:
        base_path = Path.cwd()
        out_dir = base_path / 'output' / 'test'
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"出力ディレクトリ: {out_dir.resolve()}")
    except Exception as e:
        print(f"パスの設定中にエラーが発生しました: {e}")
        return

    g_deg = 90.5452
    AU = 0.3658675989
    date = 'test'
    R_pix = 7.362434 / 2 * 100
    JL = 5.18e+14
    theta = np.deg2rad(16.0)

    w = 0.2
    h = 0.065
    B0 = 2.4
    b = 0.20
    c = 0.18

    print(f"シミュレーション開始: 位相角={g_deg:.2f}°, 水星半径={R_pix} pix")

    # --- 2. 計算グリッドとマスクの準備 ---
    dim = int(R_pix * 2)
    x_coords = np.arange(dim, dtype=np.float64) - R_pix + 0.5
    y_coords = np.arange(dim, dtype=np.float64) - R_pix + 0.5
    x, y = np.meshgrid(x_coords, y_coords)
    #x, y = x.T, y.T
    disk_mask = (x ** 2 + y ** 2) < R_pix ** 2

    # --- 3. 幾何学的な角度の計算 (ベクトル化) ---
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


    # --- 4. ハプケモデルの計算 (ベクトル化) ---
    tantheta = np.tan(theta)
    tani = np.tan(i)
    tane = np.tan(e)
    sinpsihalf = np.sin(psi / 2.0)

    kai = 1.0 / np.sqrt(1.0 + pi * (tantheta) ** 2)
    E1i = np.exp(-2.0 / pi / tantheta / tani)
    E2i = np.exp(-1.0 / pi / tantheta ** 2 / tani ** 2)
    E1e = np.exp(-2.0 / pi / tantheta / tane)
    E2e = np.exp(-1.0 / pi / tantheta ** 2 / tane ** 2)

    # shikigamachigai らしい
    mu0e0 = kai * (cosi + sini * tantheta * E2i / (2.0 - E1i))  # (12.49)

    cond = (i <= e)

    # i <= e の場合の式
    mu0e_true = kai * (cosi + sini * tantheta * (cospsi * E2e + sinpsihalf ** 2 * E2i) / (2 - E1e - (psi / pi) * E1i))
    mue_true = kai * (cose + sine * tantheta * (E2e - sinpsihalf ** 2 * E2i) / (2 - E1e - (psi / pi) * E1i))
    mue0_true = kai * (cose + sine * tantheta * E2e / (2 - E1e))
    fpsi_true = np.exp(-2.0 * np.tan(psi / 2.0))
    # 修正: mu0e0_true の代わりに共通の mu0e0 を使う
    Siepsi_true = (mue_true / mue0_true) * (cosi / mu0e0) * kai / (1 - fpsi_true + fpsi_true * kai * cosi / mu0e0)

    # i > e の場合の式
    mu0e_false = kai * (cosi + sini * tantheta * (E2i - sinpsihalf ** 2 * E2e) / (2 - E1i - (psi / pi) * E1e))
    mue_false = kai * (cose + sine * tantheta * (cospsi * E2i + sinpsihalf ** 2 * E2e) / (2 - E1i - (psi / pi) * E1e))
    mue0_false = kai * (cose + sine * tantheta * E2e / (2 - E1e))
    fpsi_false = np.exp(-2.0 * np.tan(psi / 2.0))
    # 修正: 未定義だった mu0e0_false の代わりに共通の mu0e0 を使う
    Siepsi_false = (mue_false / mue0_false) * (cosi / mu0e0) * kai / (
                1 - fpsi_false + fpsi_false * kai * cose / mue0_false)

    # np.whereで結合
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

    SSR = SR.sum()
    ic = valid_mask.sum()
    avg_brightness = SSR / ic if ic > 0 else 0
    max_brightness = SR.max()

    dat_path = out_dir / f'{date}HapkeMRnm_python.dat'
    with open(dat_path, 'w') as f:
        f.write(f'{avg_brightness} {max_brightness}\n')
    print(f"平均/最大輝度を保存しました: {dat_path}")

    fits_path_sr = out_dir / f'Hapke{date}_python.fits'
    fits_path_rr = out_dir / 'test_python.fit'

    fits.writeto(fits_path_sr, np.rot90(SR2, k=2), overwrite=True)
    print(f"輝度画像 (MR/nm) を保存しました: {fits_path_sr}")

    fits.writeto(fits_path_rr, np.rot90(RR2, k=2), overwrite=True)
    print(f"反射輝度画像 (reflectivity/sr) を保存しました: {fits_path_rr}")

    print("\nend")
    print("処理が完了しました。")


# --- スクリプトの実行 ---
if __name__ == '__main__':
    run_hapke_model_corrected()