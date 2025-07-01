import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
from pathlib import Path
import warnings


def reproduce_pix2spec():
    """
    IDLの 'pro pix2spec' コードをPythonで再現します。

    処理の概要:
    1. 複数のFITSファイル（比較光源のスペクトル画像）を読み込み、合成します。
    2. 合成した2D画像を1Dスペクトルに変換します。
    3. スペクトル中の2つの既知の輝線をガウス関数でフィッティングし、
       正確な中心ピクセル位置を求めます。
    4. 輝線のピクセル位置と既知の波長から、ピクセルを波長に変換する
       一次式（波長校正式）を決定します。
    5. 計算結果（波長校正係数、波長校正済みスペクトル、FWHM）を
       テキストファイルに出力します。
    """

    # --- 1. ファイルパスの設定 ---
    # 元のIDLコードのパス構造を再現します。
    # ご自身の環境に合わせてこのパスを変更してください。
    # 例: file_f = Path('C:/Users/hanac/University/Senior/Mercury/Haleakala2025/')
    try:
        # このスクリプトが置かれているディレクトリを基準にします
        base_path = file_f = Path('C:/Users/hanac/University/Senior/Mercury/Haleakala2025/')
        file_f3 = base_path / 'output' / 'test'
        # 出力ディレクトリが存在しない場合は作成
        file_f3.mkdir(parents=True, exist_ok=True)
        print(f"入力・出力ディレクトリ: {file_f3.resolve()}")
    except Exception as e:
        print(f"パスの設定中にエラーが発生しました: {e}")
        return

    # --- 2. FITSファイルの読み込みと合成 ---
    comp = None
    # ループで 10009.fit から 10012.fit までを読み込む
    for i in range(10009, 10013):
        file_path = file_f3 / f'{i}_Na_python.fit'
        try:
            # fits.getdataはデータを直接NumPy配列として返します
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                data = fits.getdata(file_path).astype(np.float64)

            if comp is None:
                comp = data
            else:
                comp += data
            print(f"読み込み成功: {file_path}")

        except FileNotFoundError:
            print(f"警告: ファイルが見つかりません {file_path}。")
            # 最初のファイルが見つからない場合はダミーデータを生成
            if comp is None and i == 10009:
                print("デモ用にダミーデータを生成します。")
                ixm_dummy, iym_dummy = 1024, 512
                x_dummy = np.arange(ixm_dummy)
                # 輝線を模擬した1Dデータを作成
                c00p_true, c10p_true = 553, 801  # 0-indexed
                width_true = 4.5
                comp_1d = (1500 * np.exp(-(x_dummy - c00p_true) ** 2 / (2 * width_true ** 2)) +
                           1800 * np.exp(-(x_dummy - c10p_true) ** 2 / (2 * width_true ** 2)))
                comp_1d += 50 + np.random.randn(ixm_dummy) * 10  # ノイズとベースラインを追加
                # 2D画像に拡張
                comp = np.tile(comp_1d, (iym_dummy, 1)).astype(np.float64)
                break  # ダミーデータ生成後はループを抜ける

    if comp is None:
        print("エラー: FITSファイルを一つも読み込めませんでした。処理を終了します。")
        return

    # --- 3. 画像サイズの取得 ---
    # NumPyのshapeは (rows, columns) = (iym, ixm) の順
    iym, ixm = comp.shape
    print(f"合成後の画像サイズ: (ixm={ixm}, iym={iym})")

    # --- 4. スペクトルの1次元化 ---
    # 各列（波長軸）の値を合計して1Dスペクトルを作成
    comp2 = np.sum(comp, axis=0)

    # --- 5. 輝線中心のフィッティング ---
    dw = 15  # フィッティング領域の半分の幅
    c00p = 554 - 1  # 輝線1のおおよその中心ピクセル (0-indexed)
    c10p = 802 - 1  # 輝線2のおおよその中心ピクセル (0-indexed)

    # ガウス関数＋線形バックグラウンドのモデル関数
    def gaussian_with_background(x, const, slope, amplitude, center, sigma):
        return const + slope * x + amplitude * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))

    # 結果を格納する配列
    c = np.zeros(2)
    w = np.zeros(2)
    width = np.zeros(2)

    # --- 輝線1のフィッティング (g00) ---
    x_fit_00 = np.arange(2 * dw + 1)
    y_fit_00 = comp2[c00p - dw: c00p + dw + 1]

    # パラメータの初期値 [const, slope, amplitude, center, sigma]
    p0_00 = [np.min(y_fit_00), (y_fit_00[-1] - y_fit_00[0]) / (2 * dw), np.max(y_fit_00) - np.min(y_fit_00), dw, 5]

    try:
        popt_00, _ = curve_fit(gaussian_with_background, x_fit_00, y_fit_00, p0=p0_00)
        fit_center_00, fit_sigma_00 = popt_00[3], abs(popt_00[4])  # sigmaは正の値
        c[0] = fit_center_00 + c00p - dw
        w[0] = 589.1582 - 589.0
        width[0] = fit_sigma_00
        print(f"輝線1 フィッティング成功: 中心={c[0]:.2f} pix, 幅(σ)={width[0]:.2f} pix")
    except RuntimeError:
        print("エラー: 輝線1のガウスフィッティングに失敗しました。")
        c[0], w[0], width[0] = np.nan, np.nan, np.nan

    # --- 輝線2のフィッティング (g10) ---
    x_fit_10 = np.arange(2 * dw + 1)
    y_fit_10 = comp2[c10p - dw: c10p + dw + 1]
    p0_10 = [np.min(y_fit_10), (y_fit_10[-1] - y_fit_10[0]) / (2 * dw), np.max(y_fit_10) - np.min(y_fit_10), dw, 5]

    try:
        popt_10, _ = curve_fit(gaussian_with_background, x_fit_10, y_fit_10, p0=p0_10)
        fit_center_10, fit_sigma_10 = popt_10[3], abs(popt_10[4])
        c[1] = fit_center_10 + c10p - dw
        w[1] = 589.7558 - 589.0
        width[1] = fit_sigma_10
        print(f"輝線2 フィッティング成功: 中心={c[1]:.2f} pix, 幅(σ)={width[1]:.2f} pix")
    except RuntimeError:
        print("エラー: 輝線2のガウスフィッティングに失敗しました。")
        c[1], w[1], width[1] = np.nan, np.nan, np.nan

    # --- 6. 波長校正式の決定 ---
    if not np.isnan(c).any():
        aa1 = (w[1] - w[0]) / (c[1] - c[0])  # 傾き (dispersion)
        aa0 = w[0] - aa1 * c[0]  # 切片
    else:
        print("フィッティング失敗のため、波長校正式を計算できません。")
        aa0, aa1 = np.nan, np.nan

    # --- 7. 結果のファイル出力 (pix2speccoeff.txt) ---
    if not np.isnan(aa0):
        coeff_path = file_f3 / 'pix2speccoeff_python.txt'
        with open(coeff_path, 'w') as f:
            f.write(f'{aa0} {aa1}\n')
        print(f"\n波長校正係数を保存しました: {coeff_path}")

    # --- 8. フィッティング結果の確認 ---
    print("\n--- 計算結果 ---")
    if not np.isnan(aa0):
        print("フィッティング残差:")
        for i in range(2):
            # (計算された波長 - 真の波長) * 1000 [mÅ]
            residual = (aa0 + aa1 * c[i] - w[i]) * 1000.0
            print(f"  輝線 {i}: {residual:.4f} mÅ")

    print("\n波長校正係数:")
    print(f"  aa0 (切片): {aa0}")
    print(f"  aa1 (傾き, nm/pix): {aa1}")

    # --- 9. 波長校正済みスペクトルの出力 (comp.txt) ---
    indx = np.arange(ixm)
    if not np.isnan(aa0):
        wl = (aa0 + aa1 * indx)
        data_to_save = np.vstack((wl, comp2)).T
        comp_path = file_f3 / 'comp_python.txt'
        np.savetxt(comp_path, data_to_save, fmt='%.6f %f', header='Wavelength(nm) Intensity')
        print(f"\n波長校正済みスペクトルを保存しました: {comp_path}")

    # --- 10. ピクセル-カウント値スペクトルの出力 (comp2.txt) ---
    data_to_save_2 = np.vstack((indx, comp2)).T
    comp2_path = file_f3 / 'comp2_python.txt'
    np.savetxt(comp2_path, data_to_save_2, fmt='%d %f', header='Pixel Intensity')
    print(f"ピクセルースペクトルを保存しました: {comp2_path}")

    # --- 11. FWHM（半値全幅）の計算 ---
    if not np.isnan(c[0]) and not np.isnan(width[0]) and not np.isnan(aa1):
        ic = 0  # 1本目の輝線について計算

        # 波長分散 (nm/pixel) は線形なので傾きaa1に等しい
        nmppix = aa1

        # FWHM = 2 * sqrt(2 * ln(2)) * sigma
        fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))

        fwhm_pix = fwhm_factor * width[ic]
        fwhm_nm = fwhm_pix * nmppix

        print("\nFWHM (半値全幅) 計算結果 (輝線1):")
        print(f"  FWHM = {fwhm_nm:.4f} nm")
        print(f"  FWHM = {fwhm_pix:.4f} pixels")

        fwhm_path = file_f3 / 'FWHM_python.txt'
        with open(fwhm_path, 'w') as f:
            f.write(f'{fwhm_nm} {fwhm_pix}\n')
        print(f"FWHMの値を保存しました: {fwhm_path}")
    else:
        print("\nフィッティング失敗のため、FWHMを計算できません。")

    print("\n処理が完了しました。")


# --- スクリプトの実行 ---
if __name__ == '__main__':
    reproduce_pix2spec()