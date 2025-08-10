import numpy as np
from scipy.optimize import curve_fit
import os
import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt


def gaussian_linear_baseline(x, height, center, sigma, const, linear):
    """
    curve_fitで使用する、ガウス関数 + 線形ベースラインのモデル。
    IDLのgaussfit(nterms=5)に相当します。
    """
    # sigmaが負の値にならないように絶対値をとる
    return height * np.exp(-(x - center) ** 2 / (2 * abs(sigma) ** 2)) + const + linear * x


def fit_spectrum_and_get_counts(file_paths, fit_config, plot_config, target_wavelength):
    """
    3つのスペクトルファイル（主、プラス誤差、マイナス誤差）を読み込み、
    それぞれにガウスフィットを行い、積分強度と誤差を計算する関数。
    ★★★ プロット機能強化版 ★★★
    """
    try:
        # 3つの異なるファイルを読み込む
        wl, cts_main = np.loadtxt(file_paths['main'], unpack=True)
        _, cts_plus = np.loadtxt(file_paths['plus'], unpack=True)
        _, cts_minus = np.loadtxt(file_paths['minus'], unpack=True)

        spectra_to_fit = {'main': cts_main, 'plus': cts_plus, 'minus': cts_minus}
        iim = len(wl)  # データ点数を動的に取得

    except FileNotFoundError as e:
        print(f"    -> 警告: データファイルが見つかりません ({e})。このセットをスキップします。")
        return None

    # --- フィッティングとプロットの準備 ---
    if plot_config['create_plots']:
        plt.figure(figsize=(12, 8))
        plot_colors = {'main': 'blue', 'plus': 'green', 'minus': 'red'}

    fit_results = {}
    integration_range_to_plot = {}  # プロット用の積分範囲を保持

    # --- 3つのスペクトルそれぞれにフィットを実行 ---
    for name, spectrum in spectra_to_fit.items():
        # フィット範囲を決定
        center_idx_abs = np.argmin(np.abs(wl - target_wavelength))
        dw = fit_config['fit_half_width_pix']
        start_idx_abs = center_idx_abs - dw
        end_idx_abs = center_idx_abs + dw

        y_data = spectrum[start_idx_abs: end_idx_abs + 1]
        x_data = np.arange(len(y_data))

        # フィットの初期値を設定
        center_guess_rel = np.argmax(y_data)
        initial_guess = [
            np.max(y_data) - np.min(y_data),  # 高さ
            center_guess_rel,  # 中心
            5.0,  # 幅 (シグマ)
            np.min(y_data),  # ベースライン切片
            0  # ベースライン傾き
        ]

        try:
            popt, _ = curve_fit(gaussian_linear_baseline, x_data, y_data, p0=initial_guess)

            # --- 積分範囲を計算 ---
            center_rel, sigma_fit = popt[1], popt[2]
            center_abs_fit = center_rel + start_idx_abs
            sigma_abs = abs(sigma_fit)

            # 3シグマの範囲で積分
            start_int = int(round(center_abs_fit - 3 * sigma_abs))
            end_int = int(round(center_abs_fit + 3 * sigma_abs))
            start_int = max(0, start_int)
            end_int = min(iim - 1, end_int)

            # 'main'スペクトルの積分範囲をプロット用に保存
            if name == 'main':
                integration_range_to_plot = {'start': start_int, 'end': end_int}

            # ベースラインを引いてから積分
            baseline_in_range = popt[3] + popt[4] * (
                        np.arange(start_int, end_int + 1) - start_idx_abs)  # 修正: xの参照点をフィット範囲の開始点に合わせる
            integrated_counts = np.sum(spectrum[start_int: end_int + 1] - baseline_in_range)
            width = end_int - start_int + 1

            fit_results[name] = {'counts': integrated_counts, 'width': width}
            print(f"    -> フィット '{name}': 積分幅 = {width} ピクセル")

            # --- プロット処理 ---
            if plot_config['create_plots']:
                # 元データをプロット
                plt.scatter(wl[start_idx_abs: end_idx_abs + 1], y_data, color=plot_colors[name], label=f'{name} data',
                            s=15, alpha=0.7)

                # フィット曲線とベースラインをプロットするための滑らかなx軸を生成
                x_smooth_abs = np.linspace(wl[start_idx_abs], wl[end_idx_abs], 300)
                x_smooth_rel = np.linspace(x_data.min(), x_data.max(), 300)  # フィット関数用の相対座標

                fit_curve = gaussian_linear_baseline(x_smooth_rel, *popt)
                baseline_curve = popt[3] + popt[4] * x_smooth_rel

                plt.plot(x_smooth_abs, fit_curve, color=plot_colors[name], label=f'{name} fit')
                plt.plot(x_smooth_abs, baseline_curve, color=plot_colors[name], linestyle='--',
                         label=f'{name} baseline')


        except RuntimeError:
            print(f"    -> 警告: '{name}' スペクトルのガウスフィットに失敗しました。")
            fit_results[name] = {'counts': 0, 'width': 0}

    # --- グラフの仕上げと保存 ---
    if plot_config['create_plots']:
        # ★★★ ここから追加機能 ★★★
        # 'main'フィットの積分範囲をハイライト表示
        if integration_range_to_plot:
            start_wl = wl[integration_range_to_plot['start']]
            end_wl = wl[integration_range_to_plot['end']]
            plt.axvspan(start_wl, end_wl, color='orange', alpha=0.3, label='Integration Range (main, 3σ)')
        # ★★★ ここまで追加機能 ★★★

        plt.title(f'Gaussian Fit for {plot_config["base_name"]}')
        plt.xlabel('Wavelength (nm)',fontsize=22)  # X軸をピクセルから波長に変更
        plt.ylabel('Intensity (photons/s)',fontsize=22)
        plt.legend(fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.grid(True, linestyle=':')
        # 表示範囲をフィット領域に限定して見やすくする
        fit_start_wl = wl[np.argmin(np.abs(wl - target_wavelength)) - fit_config['fit_half_width_pix']]
        fit_end_wl = wl[np.argmin(np.abs(wl - target_wavelength)) + fit_config['fit_half_width_pix']]
        plt.xlim(fit_start_wl, fit_end_wl)

        plot_path = plot_config['output_dir'] / f'{plot_config["base_name"]}_fit.png'
        plt.savefig(plot_path)
        print(f"    -> フィット結果のプロットを保存: {plot_path.name}")
        plt.close()

    # --- 誤差を計算 ---
    cts2 = fit_results.get('main', {}).get('counts', 0)
    ctsp2 = fit_results.get('plus', {}).get('counts', 0)
    ctsm2 = fit_results.get('minus', {}).get('counts', 0)

    # ノイズレベルを計算 (フィット範囲外のウイング部分から)
    wing_width = fit_config['noise_wing_width_pix']
    wing1 = cts_main[0:wing_width]
    wing2 = cts_main[-wing_width:]
    wings_combined = np.concatenate((wing1, wing2))
    sigma_noise = np.std(wings_combined, ddof=1)

    width_main = fit_results.get('main', {}).get('width', 1)
    err_stat = sigma_noise * np.sqrt(width_main)

    errp = abs(ctsp2 - cts2)
    errm = abs(ctsm2 - cts2)
    err_fuse = max(errp, errm)

    return {'counts': cts2, 'stat_error': err_stat, 'sys_error': err_fuse}

# ==============================================================================
# スクリプトの実行部
# ==============================================================================
if __name__ == '__main__':
    # --- 基本設定 ---
    day = "20250703"
    base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
    data_dir = base_dir / "output" / day
    csv_file_path = base_dir / "2025ver" / f"mcparams{day}.csv"

    # sftの値（ファイル名から探すために使用）
    #sft_map = {'main': '002','minus': '001','plus': '003'}
    #sft_map = {'main': '02','minus': '01','plus': '03'}#dawn
    sft_map = {'main': '-05', 'minus': '005', 'plus': '015'}#dusk

    # --- フィッティングとプロットに関する設定 ---
    FIT_CONFIG = {
        'fit_half_width_pix': 30,#70,#14
        'noise_wing_width_pix': 90#300#60
    }
    PLOT_CONFIG = {
        'create_plots': True,
        'output_dir': data_dir
    }

    # --- 物理定数 ---
    PI = np.pi
    NaD1_nm = 589.7558
    #NaD1_nm = 589.594  #空気中
    c_cms = 299792.458 * 1e5
    me_g = 9.1093897e-28
    #e_esu = 4.80320425e-10
    e_esu = 1.60217733e-19 * 2.99792458e+8 * 10
    JL_nm = 5.18e+14
    f1 = 0.327

    # --- メイン処理 ---
    print(f"--- 原子数密度計算を開始します (日付: {day}) ---")

    # ガンマ値の読み込み
    try:
        gamma_path = base_dir / 'output' / 'gamma_factor' / f'gamma_{day}.txt'
        _, gamma = np.loadtxt(gamma_path)
    except FileNotFoundError:
        print(f"エラー: ガンマ値ファイルが見つかりません: {gamma_path}")
        sys.exit()

    # g-factorのAUに依存しない部分を先に計算
    F_lambda_cgs = JL_nm * 1e7
    lambda_cm = NaD1_nm * 1e-7
    JL_nu = F_lambda_cgs * (lambda_cm ** 2 / c_cms)
    sigma_D1_nu = PI * e_esu ** 2 / me_g / c_cms * f1
    gfac_base = sigma_D1_nu * JL_nu * gamma

    # CSVファイルの読み込み
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"エラー: CSVファイルが見つかりません: {csv_file_path}")
        sys.exit()

    # 結果を保存するリスト
    final_results = []

    # CSVファイルの各行に対してループ
    target_df = df[df['Type'] == 'MERCURY'].copy()
    for idx, (row_index, row) in enumerate(target_df.iterrows(), start=1):
        base_name = f"MERCURY{idx}_tr"
        print(f"\n-> {base_name} の処理を開始...")

        # ★★★ 1. CSVからAUを読み込む ★★★
        try:
            # ↓↓↓ CSVのヘッダー名に合わせてこの行を修正してください ↓↓↓
            AU = row['mercury_sun_distance_au']
        except KeyError:
            print(f"    -> エラー: CSVファイルに 'mercury_sun_distance_au' 列が見つかりません。スキップします。")
            continue

        # ★★★ 2. 観測ごとのg-factorを計算 ★★★
        gfac1 = gfac_base / AU ** 2

        # 必要な3つのファイルパスを構築
        try:
            #file_paths = {
            #    'main': next(data_dir.glob(f"{base_name}.totfib_sft{sft_map['main']}.exos.dat")),
            #    'plus': next(data_dir.glob(f"{base_name}.totfib_sft{sft_map['plus']}.exos.dat")),
            #    'minus': next(data_dir.glob(f"{base_name}.totfib_sft{sft_map['minus']}.exos.dat"))
            #}
            file_paths = {
                'main': next(data_dir.glob(f"{base_name}.totfib_orig_sft{sft_map['main']}.exos.dat")),
                'plus': next(data_dir.glob(f"{base_name}.totfib_orig_sft{sft_map['plus']}.exos.dat")),
                'minus': next(data_dir.glob(f"{base_name}.totfib_orig_sft{sft_map['minus']}.exos.dat"))
            }
        except StopIteration:
            print(
                f"    -> 警告: {base_name} に対応する3つのexos.datファイルセットが見つかりませんでした。スキップします。")
            continue

        PLOT_CONFIG['base_name'] = f"{base_name}_sft{sft_map['main']}"

        result = fit_spectrum_and_get_counts(file_paths, FIT_CONFIG, PLOT_CONFIG, NaD1_nm) # 修正後


        if result:
            naatm = result['counts'] / gfac1
            naerr = (result['stat_error'] + result['sys_error']) / gfac1
            final_results.append([idx, naatm, naerr])
            print(f"    -> 計算結果 (AU={AU:.4f}): 原子数密度 = {naatm:.4e}, 誤差 = {naerr:.4e}")

    # --- 最終結果をファイルに書き出し ---
    if final_results:
        output_filename = data_dir / 'Na_atoms_final.dat'
        np.savetxt(output_filename, final_results, fmt='%d %.6e %.6e',
                   header="Index Na_Atoms_cm-2 Error")
        print(f"\n処理が完了しました。結果を {output_filename} に保存しました。")
    else:
        print("\n処理対象のファイルが見つからなかったため、結果ファイルは作成されませんでした。")
