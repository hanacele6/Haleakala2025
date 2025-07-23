import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.signal import convolve
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from astropy.io import fits
from astropy.stats import sigma_clip
from pathlib import Path
import pandas as pd
import os
import sys

# ==============================================================================
# 補助関数の定義
# ==============================================================================
plt.rcParams.update({'font.size': 8})
nterm = 4


def gauss1d(x, A):
    z = (x - A[1]) / A[2]
    if nterm == 3:
        return A[0] * np.exp(-(z ** 2) / 2)
    elif nterm == 4:
        return A[0] * np.exp(-(z ** 2) / 2) + A[3]
    elif nterm == 5:
        return A[0] * np.exp(-(z ** 2) / 2) + A[3] + A[4] * (x - A[1])


def residuals4(A, x, y):
    z = (x - A[1]) / A[2]
    model = A[0] * np.exp(-(z ** 2) / 2) + A[3]
    return model - y


def gaussian_kernel(size, sigma=1):
    x = np.linspace(-size // 2, size // 2, size)
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    return kernel / np.sum(kernel)


# ==============================================================================
# メインの波長校正関数
# ==============================================================================
def mkWavMap4b(flgPause=True):
    # --- 1. 設定項目 ---
    # ★★★★★ ご自身の環境に合わせて、以下のパスを修正してください ★★★★★
    base_dir = "C:/Users/hanac/University/Senior/Mercury/Haleakala2025/"
    csv_path = Path( "mcparams202505.csv")
    output_base_dir = os.path.join(base_dir, "output/20250501/")  # FITSやPNGの保存先
    solar_spec_path = os.path.join(base_dir, "psg/")  # 参照スペクトルがある場所
    # ★★★★★ ここまで ★★★★★

    # --- 2. CSVを読み込み、処理対象('sky')を抽出 ---
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"エラー: CSVファイルが見つかりません: {csv_path}")
        return

    df_sky = df[df['Type'].str.strip().str.upper() == 'SKY'].copy()
    print(f"'sky'タイプのファイルが {len(df_sky)} 個見つかりました。")

    # 畳み込みに使うカーネル
    kernel = gaussian_kernel(size=181, sigma=61)

    # --- 3. 抽出したファイルごとにループ処理 ---
    for index, row in df_sky.iterrows():
        fileFSp = row['fits']
        base_filename = os.path.basename(fileFSp)
        fileWMP = os.path.join(output_base_dir, base_filename.replace("_f.fits", ".wmp.fits"))

        print("\n" + "=" * 60)
        print(f"処理開始: {base_filename}")

        if not os.path.exists(fileFSp):
            print(f"  -> エラー: ファイルが存在しません。スキップします。")
            continue

        try:
            with fits.open(fileFSp) as hdu1:
                spDat = hdu1[0].data.astype(np.float64)
                hd = hdu1[0].header
                ny, nx = spDat.shape
                # 仮に全ファイバーを処理対象とする
                iFibAct = np.arange(ny)
        except Exception as e:
            print(f"  -> ファイル読み込みエラー: {e}")
            continue



        # --- 4. ★★★ シグマクリッピングによるノイズ除去 ★★★ ---
        print("  -> シグマクリッピングでノイズ除去を実行...")
        for j in iFibAct:
            # 宇宙線などのスパイクノイズをσ=3でクリップ
            clipped_spec = sigma_clip(spDat[j, :], sigma=3, maxiters=5, masked=True)
            # マスクされたピクセル(ノイズ)を、マスクされていない部分の中央値で置き換える
            spDat[j, clipped_spec.mask] = np.ma.median(clipped_spec)

        # --- 5. 波長校正の基準値と参照スペクトルを設定 ---
        wlinesM = np.array([588.39,588.995, 589.3, 589.592,590.560,591.002,591.417])
        #wlinesM = np.array([588.544, 589.157, 589.449, 589.756, 590.732, 591.580, 591.789])
        pxlinesD0 = np.array([769, 949, 1035, 1127,1423, 1557,1685])
        pixwav1, wavstep1, calwav1 = 1024, 0.00293, 588.9

        xpix = np.arange(nx)
        wpix_initial = (xpix - pixwav1) * wavstep1 + calwav1
        wDwav_initial = (pxlinesD0 - pixwav1) * wavstep1 + calwav1

        if (calwav1 >= 586 and calwav1 <= 596):
            fileCal = 'psgrad586-596.txt'
        else:
            print(f"エラー: 波長域 {calwav1} nm に対応する参照スペクトルが設定されていません。")
            continue

        try:
            spMdl = np.loadtxt(os.path.join(solar_spec_path, fileCal), skiprows=14)
            print(f"  -> 参照スペクトルを読み込みました: {fileCal}")
        except FileNotFoundError:
            print(f"  -> エラー: 参照スペクトルが見つかりません: {os.path.join(solar_spec_path, fileCal)}")
            continue

        # さらにメディアンフィルタで滑らかにする
        for iJ in range(ny):
            spDat[iJ, :] = median_filter(spDat[iJ, :], size=3)

        # --- 6. 初期状態の確認プロット ---
        if flgPause:
            fig, axes = plt.subplots(3, 1, figsize=(9, 10), dpi=96)
            plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.08, right=0.92, top=0.95, bottom=0.05)

            # 参照スペクトルのプロット
            ax1 = axes[0]
            wavair2vac = 1.000276
            #wavair2vac = 1.000000

            xm = spMdl[:, 0] / wavair2vac
            cv = convolve(spMdl[:, 1], kernel, mode='same')
            ax1.plot(xm, cv / np.median(cv), label='Reference Solar Spectrum')
            for i in wlinesM: ax1.axvline(x=i, color='r', linestyle='--', label=f'{i} nm')
            ax1.set_title("1. Reference Spectrum & Lines")
            ax1.set_xlabel("Wavelength [nm]");
            ax1.set_ylabel("Normalized Intensity")
            ax1.set_xlim(wpix_initial.min(), wpix_initial.max());
            ax1.legend()

            # 観測データの2Dイメージ
            ax2 = axes[1]
            vrng = [np.percentile(spDat, 2), np.percentile(spDat, 98)]
            im = ax2.imshow(spDat, cmap='jet', vmin=vrng[0], vmax=vrng[1], origin='lower', aspect='auto')
            ax2.set_title(f"2. Observed Data (2D Image): {base_filename}")
            fig.colorbar(im, ax=ax2, label="Counts")

            # 観測データと参照スペクトルの重ね描き
            ax3 = axes[2]
            j_mid = ny // 2
            ax3.plot(wpix_initial, spDat[j_mid, :] / np.median(spDat[j_mid, :]), label=f'Observed Data (fiber {j_mid})')
            ax3.plot(xm, cv / np.median(cv), linewidth=1, label='Reference (convolved)')
            ax3.set_title("3. Initial Alignment Check")
            ax3.set_xlabel("Approx. Wavelength [nm]");
            ax3.set_ylabel("Normalized Intensity")
            ax3.set_xlim(wpix_initial.min(), wpix_initial.max());
            ax3.legend()

            plt.show()
            val = input("初期状態を確認し、Enterを押して続行 (qで終了): ")
            if val.lower() == 'q': sys.exit()



        # --- 7. 波長校正の実行 ---
        print("  -> 波長校正のフィッティングを開始...")
        wavs = np.zeros_like(wlinesM)
        # (ここでは簡略化のため、参照スペクトルの精密化は省略し、wlinesMをそのまま正解値として使用)
        wavs = wlinesM

        wmp = np.zeros_like(spDat)
        dltWMP = np.zeros((ny, len(wlinesM)))
        dltFibY = -12.5
        y0even = pxlinesD0[0]
        y0odd = y0even + dltFibY

        for j in iFibAct:
            if j % 2 == 0:
                pxlinesD_current = pxlinesD0 - pxlinesD0[0] + y0even
            else:
                pxlinesD_current = pxlinesD0 - pxlinesD0[0] + y0odd

            #pxlinesD_current = pxlinesD0

            wpix_fit = np.zeros_like(pxlinesD_current)

            for idx, p_approx in enumerate(pxlinesD_current):
                xrng = p_approx + np.array([-0.5, 0.5]) * 0.12 / wavstep1
                print(f"--- Fiber:{j}, Line:{idx}, p_approx:{p_approx:.2f} ---")
                print(f"    wavstep1 = {wavstep1}")
                print(f"    Calculated xrng = [{xrng[0]:.2f}, {xrng[1]:.2f}]")

                #xrng = p_approx + np.array([-10, 10])
                xrng = p_approx + np.array([-0.5, 0.5]) * 0.12 / wavstep1
                ixd = (xpix >= max(0, xrng[0])) & (xpix < min(nx, xrng[1]))
                xd = xpix[ixd]
                yd = spDat[j, ixd]

                if yd.size == 0:
                    print(f"  -> 警告: Fiber {j}, Line {idx} でデータが見つかりません。スキップします。")
                    wpix_fit[idx] = np.nan
                    continue

                    # 5. ★★★ 堅牢な「動的Aini計算」★★★
                    # 正規化はせず、生のデータの統計量から初期値を計算する
                amp_ini = np.min(yd) - np.median(yd)
                offset_ini = np.median(yd)
                Aini = [amp_ini, p_approx, 2.0, offset_ini]

                # 6. フィッティングを実行
                res = least_squares(residuals4, Aini, args=(xd, yd), loss='huber')
                wpix_fit[idx] = res.x[1]

            #Aini = [-1, np.mean(xd[ixd]), 5, 1, 0];
                #bounds = ([-1.5, np.min(xd[ixd]), 3, 0, -1], [0, np.max(xd[ixd]), 20, np.inf, 1])
                #res = least_squares(residuals4, Aini, args=(xd[ixd], yd[ixd]), bounds=bounds, loss='huber');
                #par = res.x

                #Aini = [np.min(yd) - np.median(yd), p_approx, 2.0, np.median(yd)]
                #res = least_squares(residuals4, Aini, args=(xd, yd), loss='huber')
                #wpix_fit[idx] = res.x[1]

            # 5次多項式でフィッティング
            coef = np.polyfit(wpix_fit, wavs, 5)
            pfit = np.poly1d(coef)
            dltWMP[j, :] = wavs - pfit(wpix_fit)
            wmp[j, :] = pfit(xpix)

            # 次のファイバーの初期位置を更新
            ifunc = interp1d(wmp[j, :], xpix, kind='linear', fill_value='extrapolate')
            xpix0 = ifunc(wavs[0])
            if j % 2 == 0:
                y0even = xpix0
                y0odd = y0even + dltFibY
            else:
                y0odd = xpix0
                y0even = y0odd - dltFibY

        print(f"  -> フィッティング完了。残差の標準偏差: {np.std(dltWMP) * 1000:.3f} pm")

        # --- 8. 結果の保存 ---
        hd_out = hd.copy()
        hd_out['HISTORY'] = 'Wavelength calibrated'
        hd_out['BUNIT'] = ('nm', 'Wavelength unit')

        hdu = fits.PrimaryHDU(data=wmp.astype(np.float32), header=hd_out)
        hdul = fits.HDUList([hdu])
        os.makedirs(os.path.dirname(fileWMP), exist_ok=True)
        hdul.writeto(fileWMP, overwrite=True)
        print(f"  -> 波長マップを保存しました: {os.path.basename(fileWMP)}")

    print("\n" + "=" * 60)
    print("すべての処理が完了しました。")


# ==============================================================================
# スクリプトの実行
# ==============================================================================
if __name__ == "__main__":
    mkWavMap4b(flgPause=True)