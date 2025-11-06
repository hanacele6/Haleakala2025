import numpy as np
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot as plt


def gaussian_linear_baseline(x, height, center, sigma, const, linear):
    """
    curve_fitで使用する、ガウス関数 + 線形ベースラインのモデル。
    """
    return height * np.exp(-(x - center) ** 2 / (2 * abs(sigma) ** 2)) + const + linear * x


def ptn2atm_new():
    """
    IDLコード 'pro ptn2atm_new' のPython翻訳版。
    *** アプローチ1を適用し、誤差計算の安定性を向上させたバージョン ***
    """
    # --- 基本設定 ---
    day = 'test'
    fileF = 'C:/Users/hanac/University/Senior/Mercury/Haleakala2025/'
    file_dir = os.path.join(fileF, 'output', 'test')

    is_loop = 10001
    ie_loop = 10004
    num_files = ie_loop - is_loop + 1
    iim = 401
    dw = 50

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
    output_filename = os.path.join(file_dir, 'Na_atoms2_python_test1.dat')
    with open(output_filename, 'w') as f_out:

        # --- メインループ ---
        for i in range(is_loop, ie_loop + 1):
            print(f"Processing file index: {i}")
            file_index = i - is_loop

            # --- スペクトルファイルの読み込み ---
            try:
                wl_main, cts_main = np.loadtxt(os.path.join(file_dir, f'{i}exos_IDL.txt'), unpack=True)
                _, cts_plus = np.loadtxt(os.path.join(file_dir, f'{i}exos_IDL+1.txt'), unpack=True)
                _, cts_minus = np.loadtxt(os.path.join(file_dir, f'{i}exos_IDL-1.txt'), unpack=True)
            except FileNotFoundError as e:
                print(f"Warning: Data file not found for index {i} ({e}). Skipping.")
                continue

            # --- ★★★ アプローチ1の実装 ★★★ ---

            # 1. 'main'データのみをフィッティングする
            center_idx_abs = iim // 2
            start_idx_abs = center_idx_abs - dw
            end_idx_abs = center_idx_abs + dw

            y_data = cts_main[start_idx_abs: end_idx_abs + 1]
            x_data = np.arange(len(y_data))

            try:
                center_guess_rel = np.argmax(y_data)
                initial_guess = [np.max(y_data) - np.min(y_data), center_guess_rel, 5.0, np.min(y_data), 0]
                popt, _ = curve_fit(gaussian_linear_baseline, x_data, y_data, p0=initial_guess)
            except RuntimeError:
                print(f"Warning: Main Gaussian fit failed for index {i}. Skipping.")
                continue

            # 2. 'main'のフィット結果から積分範囲を決定する
            center_rel, sigma_fit = popt[1], popt[2]
            center_abs_fit = center_rel + start_idx_abs
            sigma_abs = abs(sigma_fit)

            start_int = int(round(center_abs_fit - 3 * sigma_abs))
            end_int = int(round(center_abs_fit + 3 * sigma_abs))
            start_int = max(0, start_int)
            end_int = min(iim - 1, end_int)
            width = end_int - start_int + 1
            print(f"  Integration range: {start_int} to {end_int} (width={width})")

            # 3. 全てのスペクトルを、同じ固定範囲で合計する
            cts2 = np.sum(cts_main[start_int: end_int + 1])
            ctsp2 = np.sum(cts_plus[start_int: end_int + 1])
            ctsm2 = np.sum(cts_minus[start_int: end_int + 1])

            # --- プロット処理 (簡略化) ---
            plt.figure(figsize=(12, 8))
            # データ点をプロット
            plt.scatter(np.arange(iim), cts_main, color='blue', s=10, alpha=0.7, label='main data')
            plt.scatter(np.arange(iim), cts_plus, color='green', s=10, alpha=0.7, label='plus data')
            plt.scatter(np.arange(iim), cts_minus, color='red', s=10, alpha=0.7, label='minus data')

            # フィット曲線とベースラインをプロット
            x_fit_range = np.arange(start_idx_abs, end_idx_abs + 1)
            fit_curve = gaussian_linear_baseline(x_fit_range - start_idx_abs, *popt)
            plt.plot(x_fit_range, fit_curve, color='black', linewidth=2, label='main fit')

            # 積分範囲を可視化
            plt.axvspan(start_int, end_int, color='gray', alpha=0.2, label='Integration Range')

            plt.title(f'Gaussian Fit for Index {i}')
            plt.xlabel('Pixel Index')
            plt.ylabel('Counts')
            plt.legend()
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.xlim(start_idx_abs, end_idx_abs)  # フィット範囲を拡大表示
            plt.show()

            # --- 誤差評価（変更なし） ---
            wing1 = cts_main[0:60]
            wing2 = cts_main[100:160]
            wings_combined = np.concatenate((wing1, wing2))
            sigma_noise = np.std(wings_combined, ddof=1)
            err_stat = sigma_noise * np.sqrt(width)

            errp = abs(ctsp2 - cts2)
            errm = abs(ctsm2 - cts2)
            err_fuse = max(errp, errm)

            # --- 物理量の計算（変更なし） ---
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
