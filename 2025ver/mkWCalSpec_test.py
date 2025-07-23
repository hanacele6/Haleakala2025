### Last update on 21-JUL-2025 (リファクタリング版)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from scipy.interpolate import interp1d
from astropy.io import fits
from pathlib import Path
import os
import sys


def mkWcalSpec_final(input_fsp_path, wavmap_path, wl_flat_path,
                     sky_flat_fsp_path=None, params=None,
                     save_plots=False, representative_fiber_plot=None,
                     apply_wl_flat=True):
    """
    2Dスペクトルを波長校正し、等間隔の波長軸に再サンプリングします。

    Args:
        input_fsp_path (str): 入力となる2Dスペクトルファイル(*.fsp.fits)のパス。
        wavmap_path (str): 波長マップファイル(*.wmp.fits)のパス。
        wl_flat_path (str): ホワイトライトフラットの2Dスペクトルファイル(*.fsp.fits)のパス。
        sky_flat_fsp_path (str, optional): スカイフラット用の2Dスペクトルファイル(*.fsp.fits)のパス。Defaults to None.
        params (dict, optional): 'wavshift', 'interpolation_kind' を含むパラメータ辞書。Defaults to None.
        save_plots (bool, optional): 各ファイバーの詳細なプロットを保存するかどうか。Defaults to False.
        representative_fiber_plot (int, optional): 指定した場合、そのファイバー番号のプロットのみを表示する。Defaults to None.
    """
    print("\n" + "=" * 80)
    print(f"Starting Wavelength Resampling for: {os.path.basename(input_fsp_path)}")

    # --- パラメータとデフォルト値の設定 ---
    if params is None:
        params = {}
    wavshift = params.get('wavshift', 0.0)
    interp_kind = params.get('interpolation_kind', 'quadratic')
    header_info = params.get('header_info', None)

    # --- 出力ファイル名の設定 ---
    base_filename = os.path.basename(input_fsp_path).replace(".fits", "")
    output_dir = os.path.dirname(input_fsp_path)
    file_wc = os.path.join(output_dir, f"{base_filename}.wc.fits")
    file_dcb = os.path.join(output_dir, f"{base_filename}.dcb.fits")
    file_img = os.path.join(output_dir, f"{base_filename}.img.fits")
    plot_output_dir = os.path.join(output_dir, "wcal_plots", base_filename)

    if save_plots:
        os.makedirs(plot_output_dir, exist_ok=True)
        print(f"  -> Diagnostic plots will be saved to '{plot_output_dir}'")

    # --- FITSファイルの読み込み ---
    try:
        # 1. ファイルを1つずつ読み込み、成功したかを確認する
        print(f"  -> Reading input spectrum: {os.path.basename(input_fsp_path)}")
        with fits.open(input_fsp_path) as hdul:
            spDat = hdul[0].data.astype(np.float64)
            hd = hdul[0].header

        print(f"  -> Reading wavelength map: {os.path.basename(wavmap_path)}")
        with fits.open(wavmap_path) as hdul:
            wmp = hdul[0].data.astype(np.float64)

        spFlt = None
        if apply_wl_flat and wl_flat_path and os.path.exists(wl_flat_path):
            print(f"  -> Reading white-light flat: {os.path.basename(wl_flat_path)}")
            with fits.open(wl_flat_path) as hdul:
                spFlt = hdul[0].data.astype(np.float64)

        spSky = None
        if sky_flat_fsp_path and os.path.exists(sky_flat_fsp_path):
            print(f"  -> Reading sky flat: {os.path.basename(sky_flat_fsp_path)}")
            with fits.open(sky_flat_fsp_path) as hdul:
                spSky = hdul[0].data.astype(np.float64)

        # 2. ヘッダー関連の情報を整理する
        ny, nx = spDat.shape
        if header_info:
            print("  -> Using hardcoded header parameters.")
            nFibX = header_info['NFIBX']
            nFibY = header_info['NFIBY']
            iFibAct = header_info['iFibAct']
            iFib = header_info['iFib']
        else:
            print("  -> Reading header info from FITS file.")
            nFibX, nFibY = hd.get('NFIBX', 1), hd.get('NFIBY', ny)
            with fits.open(input_fsp_path) as hdul:
                try:
                    iFibAct = hdul['IFIBERS'].data
                    iFib = hdul['FIBERS'].data
                except KeyError:
                    iFibAct = np.arange(ny)
                    iFib = np.arange(ny)

    except Exception as e:
        print("\n" + "!" * 80)
        print(f"  -> FATAL ERROR: FITSファイルの読み込み中に問題が発生しました。")
        print(f"  -> エラー内容: {e}")
        print("!" * 80 + "\n")
        return
    # --- 波長軸の再定義（リニアな等間隔グリッドを作成） ---
    wmp_shifted = wmp + wavshift
    wav_min_end = np.nanmax(wmp_shifted[iFibAct, 0])
    wav_max_start = np.nanmin(wmp_shifted[iFibAct, nx - 1])

    # 昇順か降順かを判定
    if wav_min_end < wav_max_start:
        rwmin, rwmax_orig = wav_min_end, wav_max_start
    else:
        rwmin, rwmax_orig = wav_max_start, wav_min_end

    rwstep = np.abs((rwmax_orig - rwmin) / (nx - 1))
    rwmin_f = float(f"{rwmin:.8g}")
    rwstep_f = float(f"{rwstep:.4g}")
    wavs = rwmin_f + rwstep_f * np.arange(nx, dtype=np.float64)
    rwmax_f = wavs[-1]
    rwmid_f = (rwmin_f + rwmax_f) / 2

    print(f"  -> Resampling to new linear wavelength axis:")
    print(f"     WAV_MIN : {rwmin_f:.4f} nm")
    print(f"     WAV_MAX : {rwmax_f:.4f} nm")
    print(f"     WAV_STEP: {rwstep_f:.5f} nm/pix")

    # --- 各ファイバーの処理 ---
    spDatWC = np.zeros_like(spDat)
    spDatFlt = np.zeros_like(spDat)
    spDatImg = np.zeros((nFibY, nFibX))
    spDatDcb = np.zeros((nx, nFibY, nFibX))

    # スカイフラット関連の配列
    FibFF = np.ones(ny)
    spSkyWC = np.zeros_like(spDat) if spSky is not None else None
    spSkyImg = np.zeros((nFibY, nFibX)) if spSky is not None else None

    print("  -> Processing and resampling each fiber...")
    # 1. ホワイトフラット補正と再サンプリング
    for j in iFibAct:
        # --- 各ファイバーの処理（最終診断モード） ---
        print("  -> Processing and resampling each fiber (ULTIMATE DIAGNOSTIC MODE)...")
        # このループは最初の有効なファイバーを処理した後にbreakで終了します
        for j in iFibAct:
            wmp_j = wmp_shifted[j, :]
            if np.all(np.isnan(wmp_j)):
                # 無効なファイバーは通常通りスキップ
                continue

            # フラット補正がOFFなので、入力データをそのまま使う
            spDat_to_resample = spDat[j, :]

            # --- ここからが最終診断の本丸です ---
            print("\n" + "#" * 15 + f" ファイバー {j} の詳細診断 " + "#" * 15)

            # ステップ1: 補間オブジェクトを作成
            print("ステップ1: interp1d オブジェクトを作成します...")
            try:
                ifunct = interp1d(wmp_j, spDat_to_resample, kind=interp_kind, fill_value="extrapolate",
                                  bounds_error=False)
                print(" -> 成功。")
            except Exception as e:
                print(f" -> 失敗。エラー: {e}")
                break

            # ステップ2: 新しい波長軸で値を評価（補間）
            print("ステップ2: 新しい波長軸(wavs)で値を評価します...")
            try:
                resampled_data_row = ifunct(wavs)
                print(" -> 成功。")
            except Exception as e:
                print(f" -> 失敗。エラー: {e}")
                break

            # ステップ3: 評価結果の統計情報を表示
            print("ステップ3: 評価結果の統計情報を表示します...")
            print(f"  -> 補間後のデータ(1行分)の平均値: {np.nanmean(resampled_data_row)}")
            print(f"  -> 補間後のデータ(1行分)の最大値: {np.nanmax(resampled_data_row)}")
            print(f"  -> サンプルデータ [1020:1030]: {resampled_data_row[1020:1030]}")

            # ステップ4: このデータをspDatWCに代入
            print(f"ステップ4: 結果を spDatWC の {j} 行目に代入します...")
            spDatWC[j, :] = resampled_data_row
            print(f" -> 成功。spDatWCの{j}行目の平均値: {np.nanmean(spDatWC[j, :])}")

            print("\n最初の有効なファイバーの処理が完了しました。ここで診断を終了します。")
            # 最初の有効なファイバーだけで強制終了して確認
            break

            # (以降の処理は変更なし)
    # 3. 最終的な画像とデータキューブを作成
    for j in iFibAct:
        iy, ix = j // nFibX, j % nFibX
        spDatImg[iy, ix] = np.median(spDatWC[j, 512:1536])
        spDatDcb[:, iy, ix] = spDatWC[j, :]

    # --- FITSファイルの保存 ---
    def create_header(base_hd):
        hd_out = base_hd.copy()
        hd_out['HISTORY'] = 'Resampled to linear wavelength grid with mkWcalSpec_final.py'
        hd_out['INTERPK'] = (interp_kind, 'Interpolation kind for resampling')
        if spSky is not None:
            hd_out['SKYFLAT'] = 'Applied'
        else:
            hd_out['SKYFLAT'] = 'None'
        # WCSキーワードを追加
        hd_out['CTYPE1'] = 'WAVE'
        hd_out['CRPIX1'] = 1.0
        hd_out['CRVAL1'] = rwmin_f
        hd_out['CDELT1'] = rwstep_f
        hd_out['CUNIT1'] = 'nm'
        hd_out['CTYPE2'] = 'FIBERID'
        hd_out['CRPIX2'] = 1.0
        hd_out['CRVAL2'] = 0.0
        hd_out['CDELT2'] = 1.0
        return hd_out

    # .wc.fits (波長校正済み2Dスペクトル)
    hd_wc = create_header(hd)
    fits.HDUList([
        fits.PrimaryHDU(data=spDatWC.astype(np.float32), header=hd_wc),
        fits.ImageHDU(data=iFib, name='FIBERS'),
        fits.ImageHDU(data=iFibAct, name='IFIBERS')
    ]).writeto(file_wc, overwrite=True)
    print(f"  -> Saved wavelength-calibrated spectra to: {os.path.basename(file_wc)}")

    # .dcb.fits (データキューブ)
    hd_dcb = create_header(hd)
    hd_dcb['NAXIS'] = 3
    hd_dcb['NAXIS1'] = nx
    hd_dcb['NAXIS2'] = nFibY
    hd_dcb['NAXIS3'] = nFibX
    # WCSも3次元用に更新
    hd_dcb['CTYPE3'] = 'FIBER_X'
    hd_dcb['CRPIX3'] = 1.0
    hd_dcb['CRVAL3'] = 0.0
    hd_dcb['CDELT3'] = 1.0

    fits.HDUList([
        fits.PrimaryHDU(data=spDatDcb.astype(np.float32), header=hd_dcb),
        fits.ImageHDU(data=iFib, name='FIBERS'),
        fits.ImageHDU(data=iFibAct, name='IFIBERS')
    ]).writeto(file_dcb, overwrite=True)
    print(f"  -> Saved data cube to: {os.path.basename(file_dcb)}")

    # .img.fits (ファイバーバンドル再構成像)
    hd_img = create_header(hd)
    fits.HDUList([
        fits.PrimaryHDU(data=spDatImg.astype(np.float32), header=hd_img),
        fits.ImageHDU(data=iFib, name='FIBERS'),
        fits.ImageHDU(data=iFibAct, name='IFIBERS')
    ]).writeto(file_img, overwrite=True)
    print(f"  -> Saved reconstructed image to: {os.path.basename(file_img)}")

    # --- プロットの作成 ---
    if save_plots or representative_fiber_plot is not None:
        fibers_to_plot = iFibAct if save_plots and representative_fiber_plot is None else [representative_fiber_plot]
        for j in fibers_to_plot:
            if j not in iFibAct: continue
            iy, ix = j // nFibX, j % nFibX

            fig = plt.figure(figsize=(10, 12), dpi=100)
            gs = gridspec.GridSpec(5, 1, height_ratios=[2, 2, 2, 3, 3])

            # (1) 波長校正済みスペクトル
            ax1 = fig.add_subplot(gs[0])
            vrng = np.percentile(spDatWC[iFibAct, :], [1, 99])
            im = ax1.imshow(spDatWC, aspect='auto', origin='lower', vmin=vrng[0], vmax=vrng[1],
                            extent=[wavs[0], wavs[-1], 0, ny])
            ax1.set_title(f"Wavelength Calibrated Spectra ({os.path.basename(file_wc)})", fontsize=10)
            ax1.axhline(y=j, color='w', ls='--', lw=1)

            # (2) フラットフィールド済みスペクトル
            ax2 = fig.add_subplot(gs[1])
            vrng = np.percentile(spDatFlt[iFibAct, :], [1, 99])
            im = ax2.imshow(spDatFlt, aspect='auto', origin='lower', vmin=vrng[0], vmax=vrng[1])
            ax2.set_title(f"Flat-Fielded Spectra (before resampling)", fontsize=10)
            ax2.axhline(y=j, color='w', ls='--', lw=1)

            # (3) 1次元スペクトルプロット
            ax3 = fig.add_subplot(gs[2])
            ax3.plot(wavs, spDatWC[j, :], label=f"Fiber {j} (Object)")
            if spSkyWC is not None:
                ax3.plot(wavs, spSkyWC[j, :], label="Sky", alpha=0.7)
            ax3.set_title(f"1D Spectrum for Fiber {j}", fontsize=10)
            ax3.set_xlabel("Wavelength (nm)")
            ax3.set_ylabel("Intensity")
            ax3.grid(True, ls=':', alpha=0.5)
            ax3.legend()

            # (4) & (5) 再構成像と感度補正係数
            gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[3:])
            ax4 = fig.add_subplot(gs_bottom[0])
            vrng = np.percentile(spDatImg, [1, 99])
            im = ax4.imshow(spDatImg, origin='lower', vmin=vrng[0], vmax=vrng[1])
            ax4.set_title(f"Reconstructed Image ({os.path.basename(file_img)})", fontsize=10)
            rect = patches.Rectangle((ix - 0.5, iy - 0.5), 1, 1, lw=2, ec='r', fc='none')
            ax4.add_patch(rect)
            fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

            if spSky is not None:
                ax5 = fig.add_subplot(gs_bottom[1])
                ax5.plot(iFibAct, FibFF[iFibAct], '.-')
                ax5.set_title("Sky Flat Correction Factor", fontsize=10)
                ax5.set_xlabel("Fiber ID")
                ax5.set_ylabel("Correction Factor")
                ax5.grid(True, ls=':', alpha=0.5)

            fig.suptitle(f"Diagnostics for {base_filename} | Fiber {j}", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            if save_plots:
                png_filename = os.path.join(plot_output_dir, f"wcal_diag_{j:03d}.png")
                fig.savefig(png_filename)
                print(f"    -> Saved plot: {os.path.basename(png_filename)}")

            if representative_fiber_plot is not None:
                plt.show()

            plt.close(fig)

    print("=" * 80)
    print("All processing finished.")
    print("=" * 80)


# ==============================================================================
# スクリプトの実行 (CSVからタイプを読み込み、自動で連番を振ってバッチ処理)
# ==============================================================================
if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    # --- 基本設定（ユーザーが環境に合わせて変更する部分） ---

    # 1. ベースディレクトリとCSVファイルのパス
    base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
    csv_file_path = base_dir / "2025ver" / "mcparams202505.csv"

    # 2. データが格納されているディレクトリ
    data_dir = base_dir / "output/20250501/"

    # 3. 使用するマスターフラットファイル（拡張子なし）
    master_wl_flat_name = ("LED1_f")
    master_sky_flat_name = "master_sky_f"

    # 4. 処理したいデータのタイプをリストで指定
    TYPES_TO_PROCESS = ['MERCURY']

    # 5. ★★★ ここを実際のCSVに合わせて修正 ★★★
    # CSVの列名を指定します (ファイル番号の列は不要になりました)
    type_col = 'Type'  # 'SKY', 'MERCURY'などが書かれた列のヘッダー名

    header_params = {
        'NFIBX': 10,
        'NFIBY': 12,
        # iFibAct: 有効なファイバー番号のリスト。以下は0から119まで全て有効な場合の例
        'iFibAct': np.arange(120),
        # iFib: 全ファイバーのリスト。通常はiFibActと同じで良い
        'iFib': np.arange(120)
    }

    # 6. その他のパラメータ
    params = {
        'wavshift': 0.0,
        'interpolation_kind': 'quadratic'
    }


    print("--- 波長校正・再サンプリング処理を開始します ---")
    print(f"読み込むCSV: {csv_file_path}")

    APPLY_WL_FLAT = False

    # --- CSVの読み込み ---
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"エラー: CSVファイルが見つかりません: {csv_file_path}")
        sys.exit()

    # --- 各タイプに対してループ処理 ---
    for process_type in TYPES_TO_PROCESS:
        print("\n" + "=" * 25 + f" 処理タイプ: {process_type} " + "=" * 25)

        try:
            target_df = df[df[type_col] == process_type].copy()
        except KeyError:
            print(f"エラー: CSVに列 '{type_col}' が見つかりません。スクリプトの`type_col`の設定を確認してください。")
            continue

        if target_df.empty:
            print(f"-> CSV内にタイプ '{process_type}' のデータは見つかりませんでした。スキップします。")
            continue

        print(f"-> {len(target_df)}個の '{process_type}' ファイルを処理します...")

        # ★★★ タイプごとに連番を振るためのループ ★★★
        # enumerateを使い、カウンター(i)を1から始める
        for i, (index, row) in enumerate(target_df.iterrows(), start=1):

            # 'MERCURY'とカウンター'1'から "MERCURY1_f" というベース名を作成
            target_base_name = f"{process_type}{i}_f"
            print(f"\n  ({i}/{len(target_df)}) 処理中: {target_base_name}")

            # 処理に必要なファイルパスを構築
            input_fsp = data_dir / f"{target_base_name}.fits"
            wavmap = data_dir / f"{master_sky_flat_name}.wmp.fits"
            wl_flat = data_dir / f"{master_wl_flat_name}.fits"
            sky_flat = data_dir / f"{master_sky_flat_name}.fits"

            if not all([input_fsp.exists(), wavmap.exists(), wl_flat.exists()]):
                print(f"    -> スキップ: {target_base_name} の関連ファイルが見つかりません。")
                continue


            # メインの処理関数を呼び出し
            mkWcalSpec_final(
                input_fsp_path=str(input_fsp),
                wavmap_path=str(wavmap),
                wl_flat_path=str(wl_flat),
                sky_flat_fsp_path=str(sky_flat) if sky_flat.exists() else None,
                #sky_flat_fsp_path=None,
                params=params,
                save_plots=False,
                representative_fiber_plot=None
            )

    print("\n--- 全ての処理が完了しました ---")