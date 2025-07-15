import numpy as np
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot as plt


def gaussian_linear_baseline(x, height, center, sigma, const, linear):
    """
    curve_fitで使用する、ガウス関数 + 線形ベースラインのモデル。
    IDLのgaussfit(nterms=5)に相当する。
    """
    return height * np.exp(-(x - center) ** 2 / (2 * abs(sigma) ** 2)) + const + linear * x


def ptn2atm_new():
    """
    IDLコード 'pro ptn2atm_new' のPython翻訳版。
    *** ガウスフィットの結果をプロットする機能を追加 ***
    """
    # --- 基本設定 ---
    day = 'test'
    fileF = 'C:/Users/hanac/University/Senior/Mercury/Haleakala2025/'
    file_dir = os.path.join(fileF, 'output', day)

    is_loop = 10001
    ie_loop = 10004
    num_files = ie_loop - is_loop + 1
    iim = 401
    dw = 14

    AU = 0.3658675989
    pi = np.pi

    # --- 結果を格納する配列 ---
    naatm = np.zeros(num_files)
    naerr = np.zeros(num_files)

    # --- ガンマ値の読み込み ---
    try:
        gamma_path = os.path.join(fileF, 'output', 'gamma_factor', f'gamma_{day}.txt')
        _, gamma = np.loadtxt(gamma_path)
    except FileNotFoundError:
        print(f"Error: Gamma factor file not found at {gamma_path}")
        return

    # --- 出力ファイルの準備 ---
    output_filename = os.path.join(file_dir, 'Na_atoms2_python_orig.dat')
    with open(output_filename, 'w') as f_out:

        # --- メインループ ---
        for i in range(is_loop, ie_loop + 1):
            print(f"Processing file index: {i}")
            file_index = i - is_loop

            # --- ★★★ プロット用の準備 ★★★ ---
            plt.figure(figsize=(12, 8))
            plot_colors = {'main': 'blue', 'plus': 'green', 'minus': 'red'}

            # --- スペクトルファイルの読み込み ---
            try:
                # 3つの異なるファイルを読み込むように修正
                wl_main, cts_main = np.loadtxt(os.path.join(file_dir, f'{i}exos.txt'), unpack=True)
                _, cts_plus = np.loadtxt(os.path.join(file_dir, f'{i}exos+1.txt'), unpack=True)
                _, cts_minus = np.loadtxt(os.path.join(file_dir, f'{i}exos-1.txt'), unpack=True)

                wl = wl_main  # 波長は共通と仮定
                spectra_to_fit = {'main': cts_main, 'plus': cts_plus, 'minus': cts_minus}

            except FileNotFoundError as e:
                print(f"Warning: Data file not found for index {i} ({e}). Skipping.")
                continue

            # --- ガウスフィッティングと強度計算 ---
            fit_results = {}

            for name, spectrum in spectra_to_fit.items():
                center_idx_abs = iim // 2
                start_idx_abs = center_idx_abs - dw
                end_idx_abs = center_idx_abs + dw

                y_data = spectrum[start_idx_abs: end_idx_abs + 1]
                x_data = np.arange(len(y_data))

                #initial_guess = [np.max(y_data) - np.min(y_data), dw, 2.0, np.min(y_data), 0]
                center_guess_rel = np.argmax(y_data)

                # その位置を、ピーク中心の初期値として与える
                initial_guess = [
                    np.max(y_data) - np.min(y_data),  # 高さ
                    center_guess_rel,  # ★中心 (より賢い初期値)
                    5.0,  # 幅 (少し広めに見積もる)
                    np.min(y_data),  # ベースラインの切片
                    0  # ベースラインの傾き
                ]

                try:
                    popt, pcov = curve_fit(gaussian_linear_baseline, x_data, y_data, p0=initial_guess)

                    # --- ★★★ プロット処理 ★★★ ---
                    # 元のデータ点をプロット
                    plt.scatter(x_data + start_idx_abs, y_data,
                                color=plot_colors[name],
                                label=f'{name} data', s=10, alpha=0.6)

                    # フィット曲線を滑らかに描画するためのX軸
                    x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
                    # フィット曲線とベースラインを計算
                    fit_curve = gaussian_linear_baseline(x_smooth, *popt)
                    baseline = popt[3] + popt[4] * x_smooth

                    # フィット曲線とベースラインをプロット
                    plt.plot(x_smooth + start_idx_abs, fit_curve,
                             color=plot_colors[name],
                             label=f'{name} fit')
                    plt.plot(x_smooth + start_idx_abs, baseline,
                             color=plot_colors[name], linestyle='--',
                             label=f'{name} baseline')

                    # --- 計算処理（変更なし） ---
                    center_rel, sigma_fit = popt[1], popt[2]
                    center_abs_fit = center_rel + start_idx_abs
                    sigma_abs = abs(sigma_fit)

                    start_int = int(round(center_abs_fit - 3 * sigma_abs))
                    end_int = int(round(center_abs_fit + 3 * sigma_abs))
                    start_int = max(0, start_int)
                    end_int = min(iim - 1, end_int)

                    integrated_counts = np.sum(spectrum[start_int: end_int + 1])
                    width = end_int - start_int + 1

                    fit_results[name] = {'counts': integrated_counts, 'width': width}
                    print(f"  Fit '{name}': width = {width}")

                except RuntimeError:
                    print(f"Warning: Gaussian fit for '{name}' spectrum (index {i}) failed.")
                    fit_results[name] = {'counts': 0, 'width': 0}

            # --- ★★★ グラフの仕上げ ★★★ ---
            plt.title(f'Gaussian Fit Comparison for Index {i}')
            plt.xlabel('Pixel Index')
            plt.ylabel('Counts')
            plt.legend()
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.show()

            # --- 物理量の計算（変更なし） ---
            cts2 = fit_results.get('main', {}).get('counts', 0)
            ctsp2 = fit_results.get('plus', {}).get('counts', 0)
            ctsm2 = fit_results.get('minus', {}).get('counts', 0)

            wing1 = cts_main[0:60]
            wing2 = cts_main[100:160]
            wings_combined = np.concatenate((wing1, wing2))
            sigma_noise = np.std(wings_combined, ddof=1)
            width_main = fit_results.get('main', {}).get('width', 1)
            err_stat = sigma_noise * np.sqrt(width_main)

            errp = abs(ctsp2 - cts2)
            errm = abs(ctsm2 - cts2)
            err_fuse = max(errp, errm)

            NaD1_nm = 589.7558
            c_cms = 299792.458 * 1e5
            me_g = 9.1093897e-28
            e_esu = 4.80320425e-10
            JL_nm = 5.18e+14
            f1 = 0.327

            F_lambda_cgs = JL_nm * 1e7
            lambda_cm = NaD1_nm * 1e-7
            JL_nu = F_lambda_cgs * (lambda_cm ** 2 / c_cms)

            sigma_D1_nu = pi * e_esu ** 2 / me_g / c_cms * f1
            gfac1 = sigma_D1_nu * JL_nu / AU ** 2 * gamma

            naatm[file_index] = cts2 / gfac1
            naerr[file_index] = (err_stat + err_fuse) / gfac1

            f_out.write(f"{i - 10000} {naatm[file_index]} {naerr[file_index]}\n")

        print(f"gfac1={gfac1}, err_stat={err_stat}, err_fuse={err_fuse}")

    print(f"Processing finished. Results saved to {output_filename}")
    print('end')


# --- スクリプトの実行 ---
if __name__ == '__main__':
    ptn2atm_new()
