import numpy as np
from astropy.io import fits
import os

from numexpr.necompiler import double


def pro_subsurfnew2():
    """
    IDLコード 'pro subsurfnew2' のPython翻訳版。
    水星のスペクトルデータから太陽光成分を除去する。
    *** 配列スライスの不整合を修正し、根本的なエラー解決を行ったバージョン ***
    """
    day = 'test'

    # --- 基本設定 ---
    fileF = 'C:/Users/hanac/University/Senior/Mercury/Haleakala2025/'
    fileF1 = os.path.join(fileF, 'output', 'test')
    fileF2 = os.path.join(fileF, 'output', 'test')

    is_loop = 10001
    ie_loop = 10004
    # D2 = 209  # この固定値は使わず、ixm//2を基準にすることで不整合を解消

    # --- FITSファイルと基本情報の読み込み ---
    try:
        a = fits.getdata(os.path.join(fileF1, f'{is_loop}_sf22_python.fit'))
        iym, ixm = a.shape
    except FileNotFoundError:
        print(f"Error: FITS file not found at {os.path.join(fileF1, f'{is_loop}_sf22.fit')}")
        return

    # --- 物理定数とパラメータ ---
    Vme = -38.3179619
    Vms = -10.0384175
    Rmn = 7.362434
    sft = 0.002
    Rmc = 4.879e+8
    Rca = Rmc / Rmn
    Shap = Rca ** 2 / 1e+4
    c = 299792.458

    # --- 入力ファイルの読み込み ---
    try:
        wl = np.loadtxt(os.path.join(fileF1, 'wl_python.txt'))
        dwl = wl[ixm // 2] - wl[ixm // 2 - 1]
        sol_data = np.loadtxt(os.path.join(fileF, 'SolarSpectrum.txt'))
    except FileNotFoundError:
        print(f"Error: Input data files not found.")
        return

    # --- 理論スペクトル(surf)の生成 ---
    sol = np.zeros((sol_data.shape[0], 3))
    sol[:, 0] = sol_data[:, 0] - 589.0 - sft
    sol[:, 1] = sol_data[:, 2]
    sol[:, 2] = sol_data[:, 1]
    wlsurf = (sol[:, 0] + 589.0) * (1 + Vms / c) * (1 + Vme / c) - 589.0
    surf = np.zeros((ixm, 2))
    surf[:, 0] = np.interp(wl, sol[:, 0], sol[:, 1])
    surf[:, 1] = np.interp(wl, wlsurf, sol[:, 2])

    # --- メインループ ---
    for i in range(is_loop, ie_loop + 1):
        print(f"Processing file index: {i}")

        try:
            Nat = np.loadtxt(os.path.join(fileF2, f'{i}totfib_python.dat'), usecols=1)
        except FileNotFoundError:
            print(f"Warning: Data file not found for index {i}. Skipping.")
            continue

        # --- スライスルールの統一 ---
        # 観測データ(Nat)と理論モデル(surf)で、同じルールで中心部分を切り出す
        center_pix = ixm // 2
        d2 = 209
        slice_end1 = center_pix - 7
        slice_start1 = center_pix + 6
        slice_end2 = d2 - 7
        slice_start2 = d2 + 6

        # 統一されたルールでNat2を生成
        Nat2 = np.concatenate((Nat[0:slice_end2], Nat[slice_start2:ixm]))

        zansamin = 1.0e+32
        best_params = {}

        # --- 最適化ループ ---
        for iFWHM in range(30, 101):
            FWHM = iFWHM * 0.1
            sigma = FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))

            # PSFと畳み込み

            psf = np.exp(-((np.arange(ixm) - double(ixm) / 2)/sigma) ** 2/2.0)
            psf2 = psf / np.sum(psf)
            fft_surf0 = np.fft.fft(surf[:, 0])
            fft_surf1 = np.fft.fft(surf[:, 1])
            fft_psf2 = np.fft.fft(psf2)
            conv_surf0 = np.fft.fftshift(np.real(np.fft.ifft(fft_surf0 * fft_psf2)))
            conv_surf1 = np.fft.fftshift(np.real(np.fft.ifft(fft_surf1 * fft_psf2)))
            surf1 = np.column_stack([conv_surf0, conv_surf1])

            for iairm in range(51):
                airm = 0.1 * iairm
                surf2 = surf1[:, 0] ** airm * surf1[:, 1]

                # 統一されたルールでsurf3とpix_rangeを生成
                surf3 = np.concatenate((surf2[0:slice_end1], surf2[slice_start1:ixm]))
                pix_range = np.concatenate((np.arange(slice_end1), np.arange(slice_start1, ixm)))

                # これで全ての配列の長さが一致する
                ratioa = Nat2 / surf3
                aa0 = np.polyfit(pix_range, ratioa, 2)[::-1]
                ratiof = aa0[0] + aa0[1] * pix_range + aa0[2] * pix_range ** 2
                zansa = np.sum((Nat2 - surf3 * ratiof) ** 2)

                if iFWHM == 30 and iairm == 10:
                    print(f"Python FWHM=3.0, airm=1.0")
                    #print(f"psf2: {psf2}")
                    print(f"  surf sum: {np.sum(surf[:,0])}")
                    print(f"  surf sum: {np.sum(surf[:,1])}")
                    print(f"  surf2 sum: {np.sum(surf2)}")
                    print(f"  surf3 sum: {np.sum(surf3)}")
                    print(f"  ratioa[100]: {ratioa[100]}")
                    print(f"  aa0: {aa0}")
                    print(f"  zansa: {zansa}")

                    # 最初のファイル(10001)の時だけ書き出す
                    if i == is_loop:
                        # 小数点以下20桁で出力フォーマットを指定
                        output_format = '%.20f'

                        # surf1 の出力
                        np.savetxt(os.path.join(fileF1, 'python_surf1_output.txt'),
                                   surf1, fmt=output_format)

                        # surf2 の出力
                    #    np.savetxt(os.path.join(fileF1, 'python_surf2_output.txt'),
                    #               surf2, fmt=output_format)

                        # surf3 の出力
                    #    np.savetxt(os.path.join(fileF1, 'python_surf3_output.txt'),
                    #               surf3, fmt=output_format)

                if zansa <= zansamin:
                    zansamin = zansa
                    best_params = {
                        'ratiof_full': aa0[0] + aa0[1] * np.arange(ixm) + aa0[2] * np.arange(ixm) ** 2,
                        'aa0s': aa0,
                        'airms': airm,
                        'FWHMs': FWHM,
                        'surf2s': surf2
                    }

        print(
            f"  Best fit params: airm={best_params['airms']:.2f}, FWHM={best_params['FWHMs']:.2f}, zansa={zansamin:.4e}")

        # --- ループ後の最終計算 ---
        # ここでもループ内と全く同じ統一ルールで配列を生成する
        Nat2_final = np.concatenate((Nat[0:slice_end1], Nat[slice_start2:ixm]))
        surf3s_final = np.concatenate((best_params['surf2s'][0:slice_end1], best_params['surf2s'][slice_start2:ixm]))

        # IDLのロジックに合わせ、単純なスケーリングファクターで補正
        ratio2 = np.sum(Nat2_final * surf3s_final) / np.sum(surf3s_final * surf3s_final)
        Natb = Nat - ratio2 * best_params['surf2s']

        # --- 結果の保存 ---
        # test2.txtには多項式フィットの結果を出力
        output_data1 = np.column_stack(
            [wl, Nat / np.mean(Nat2_final), best_params['surf2s'] * best_params['ratiof_full'] / np.mean(Nat2_final)])
        np.savetxt(os.path.join(fileF1, f'{i}test2_python.txt'), output_data1, fmt='%f')
        np.savetxt(os.path.join(fileF1, f'{i}test2_python.dat'), output_data1, fmt='%f')

        # ratio2.txtの保存
        err = np.sqrt(np.mean((Nat2_final - ratio2 * surf3s_final) ** 2)) / np.sqrt(len(Nat2_final))
        with open(os.path.join(fileF1, f'{i}ratio2_python.txt'), 'w') as f:
            f.write(f'{ratio2} {err}\n')

        # Hapkeモデルを使ったフラックス変換
        try:
            hap_path = os.path.join(fileF1, f'Hapke{day}_python.fits')
            hap = fits.getdata(hap_path)
            tothap = np.sum(hap) * Shap * dwl * 1e+12
            cts2MR = tothap / ratio2

            print(f"Python cts2MR: {cts2MR}")
            print(f"Python Natb sum: {np.sum(Natb)}")

            # exos.txt / exos.datの保存
            exos_data = np.column_stack([wl, Natb * cts2MR])
            np.savetxt(os.path.join(fileF1, f'{i}exos_python.txt'), exos_data, fmt='%f')
            np.savetxt(os.path.join(fileF1, f'{i}exos_python.dat'), exos_data, fmt='%f')

        except FileNotFoundError:
            print(f"Warning: Hapke file not found at {hap_path}. Skipping final conversion.")
            continue

    print('end')


# --- スクリプトの実行 ---
if __name__ == '__main__':
    pro_subsurfnew2()
