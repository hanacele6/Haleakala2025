import numpy as np
from scipy.optimize import curve_fit
import os


def gaussian_linear_baseline(x, height, center, sigma, const, linear):
    """
    curve_fitで使用する、ガウス関数 + 線形ベースラインのモデル。
    IDLのgaussfit(nterms=5)に相当する。
    """
    return height * np.exp(-(x - center) ** 2 / (2 * sigma ** 2)) + const + linear * x


def ptn2atm_new():
    """
    IDLコード 'pro ptn2atm_new' のPython翻訳版。
    スペクトルから輝線強度を測定し、原子数に変換する。
    """
    # --- 基本設定 ---
    day = 'test'
    fileF = 'C:/Users/hanac/University/Senior/Mercury/Haleakala2025/'
    filef0 = os.path.join(fileF, 'output', 'test')
    # IDLコードではfilef, filefp, filefmが同じパスを指しているため、一つにまとめる
    file_dir = os.path.join(fileF, 'output', 'test')

    is_loop = 10001
    ie_loop = 10004
    num_files = ie_loop - is_loop + 1
    iim = 401  # スペクトルの要素数
    dw = 14  # フィットに使う中心からの幅

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
    output_filename = os.path.join(file_dir, 'Na_atoms2_python_test.dat')
    with open(output_filename, 'w') as f_out:
        # ヘッダーなどを書き込む場合はここ
        # f_out.write("# Index, Na_Atoms, Na_Error\n")

        # --- メインループ ---
        for i in range(is_loop, ie_loop + 1):
            print(f"Processing file index: {i}")
            file_index = i - is_loop

            # --- スペクトルファイルの読み込み ---
            try:
                # 3つのファイルを同時に読み込む
                wl, cts = np.loadtxt(os.path.join(file_dir, f'{i}exos_IDL.txt'), unpack=True)
                _, ctsp = np.loadtxt(os.path.join(file_dir, f'{i}exos_IDL+1.txt'), unpack=True)  # IDLコードでは同じファイル
                _, ctsm = np.loadtxt(os.path.join(file_dir, f'{i}exos_IDL-1.txt'), unpack=True)  # IDLコードでは同じファイル
            except FileNotFoundError:
                print(f"Warning: Data file not found for index {i}. Skipping.")
                continue

            # --- ガウスフィッティングと強度計算 ---
            fit_results = {}
            spectra_to_fit = {'main': cts, 'plus': ctsp, 'minus': ctsm}

            for name, spectrum in spectra_to_fit.items():
                # フィット対象のデータ範囲を切り出す
                center_idx_rel = dw
                start_idx_rel = center_idx_rel - dw
                end_idx_rel = center_idx_rel + dw

                # Pythonのスライス仕様に合わせて+1する
                y_data = spectrum[start_idx_rel: end_idx_rel + 1]
                x_data = np.arange(len(y_data))

                # フィッティングの初期値を設定
                #initial_guess = [np.max(y_data), center_idx_rel, 5.0, np.min(y_data), 0]
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
                    popt, _ = curve_fit(gaussian_linear_baseline, x_data, y_data, p0=initial_guess )

                    # 絶対座標での中心位置
                    center_abs = popt[1] + (iim // 2 - dw)
                    sigma_abs = abs(popt[2])

                    # 3σの範囲で強度を合計
                    start_int = int(round(center_abs - 3 * sigma_abs))
                    end_int = int(round(center_abs + 3 * sigma_abs))

                    # 配列の範囲内に収める
                    start_int = max(0, start_int)
                    end_int = min(iim - 1, end_int)

                    # Pythonのスライス仕様に合わせて+1する
                    integrated_counts = np.sum(spectrum[start_int: end_int + 1])
                    width = end_int - start_int + 1


                    fit_results[name] = {'counts': integrated_counts, 'width': width}
                    print(f"  Fit '{name}': width = {width}")

                except RuntimeError:
                    print(f"Warning: Gaussian fit for '{name}' spectrum (index {i}) failed.")
                    fit_results[name] = {'counts': 0, 'width': 0}

            cts2 = fit_results.get('main', {}).get('counts', 0)
            ctsp2 = fit_results.get('plus', {}).get('counts', 0)
            ctsm2 = fit_results.get('minus', {}).get('counts', 0)

            # --- 誤差評価 ---
            # 1. スペクトルのウィング（端）の標準偏差から評価
            wing1 = cts[0:60]
            wing2 = cts[100:160]
            wings_combined = np.concatenate((wing1, wing2))
            # ddof=1はN-1で割ることを意味し、IDLの挙動と一致
            sigma_noise = np.std(wings_combined, ddof=1)
            width_main = fit_results.get('main', {}).get('width', 1)
            err_stat = sigma_noise * np.sqrt(width_main)

            # 2. Fusegawa法（+/-スペクトルとの差から評価）
            errp = abs(ctsp2 - cts2)
            errm = abs(ctsm2 - cts2)
            err_fuse = max(errp, errm)

            # --- 物理定数とg-factorの計算 ---
            NaD1 = 589.7558  # nm
            c_kms = 299792.458  # km/s
            c_cms = c_kms * 1e5  # cm/s
            me_g = 9.1093897e-31 * 1e3  # g
            e_esu = 1.60217733e-19 * 2.99792458e+9  # esu (cgs)
            JL_nm = 5.18e+14  # ph/cm2/nm/s @1AU
            f1 = 0.327

            # 単位をcgs系に揃えて計算
            #JL_nu = JL_nm * (1e9) * ((NaD1 * 1e-9) ** 2 / c_ms)  # ph/cm2/Hz/s
            #sigma_D1_nu = np.pi * e_esu ** 2 / (me_g * 1e-3 * c_ms * 1e2) * f1  # cm^2 Hz
            #JL_nu = JL_nm * (NaD1 * 1e-7) ** 2 / c_cms
            F_lambda_cgs = JL_nm * 1e7
            lambda_cm = NaD1 * 1e-7
            JL_nu = F_lambda_cgs * (lambda_cm ** 2 / c_cms)
            sigma_D1_nu = pi * e_esu ** 2 / me_g / c_cms * f1

            gfac1 = sigma_D1_nu * JL_nu / AU ** 2 * gamma

            # --- Na原子数の計算 ---
            naatm[file_index] = cts2 / gfac1
            naerr[file_index] = (err_stat + err_fuse) / gfac1

            # --- ファイルへの書き出し ---
            f_out.write(f"{i - 10000} {naatm[file_index]} {naerr[file_index]}\n")

        print(f"gfac1={gfac1}, err_stat={err_stat}, err_fuse={err_fuse}")

    print(f"Processing finished. Results saved to {output_filename}")
    print('end')


# --- スクリプトの実行 ---
if __name__ == '__main__':
    ptn2atm_new()
