import numpy as np
from astropy.io import fits
import os


def pro_subsurfnew2():
    """
    IDLコード 'pro subsurfnew2' のPython翻訳版。
    水星のスペクトルデータから太陽光成分を除去する。
    """
    day = 'test'

    # --- 基本設定 ---
    # ファイルパスは環境に合わせて変更してください
    fileF = 'C:/Users/hanac/University/Senior/Mercury/Haleakala2025/'
    fileF1 = os.path.join(fileF, 'output', 'test')
    fileF2 = os.path.join(fileF, 'output', 'test')

    is_loop = 10001
    ie_loop = 10004
    D2 = 209  # IDLの1-based indexを想定すると209番目。Pythonは0-basedなので208。


    # --- FITSファイルと基本情報の読み込み ---
    # is_loopの最初のファイルから画像サイズを取得
    try:
        a = fits.getdata(os.path.join(fileF1, f'{is_loop}_sf22_IDL.fit'))
        # NumPyのshapeは (iym, ixm) の順になる
        iym, ixm = a.shape
    except FileNotFoundError:
        print(f"Error: FITS file not found at {os.path.join(fileF1, f'{is_loop}_sf22.fit')}")
        print("Please check the file path and file name.")
        return

    # --- 物理定数とパラメータ ---
    Vme = -38.3179619
    Vms = -10.0384175
    Rmn = 7.362434  # arcsec
    sft = 0.002

    Rmc = 4.879e+8  # cm
    Rca = Rmc / Rmn  # cm/arcsec
    Shap = Rca ** 2 / 1e+4  # cm^2/pix
    c = 299792.458
    NaD2 = 0.1582
    NaD1 = 0.7558

    # --- 波長データ(wl.txt)の読み込み ---
    try:
        wl_path = os.path.join(fileF1, 'wl_IDL.txt')
        wl = np.loadtxt(wl_path)
        dwl = wl[ixm // 2] - wl[ixm // 2 - 1]
    except FileNotFoundError:
        print(f"Error: Wavelength file not found at {wl_path}")
        return

    # --- 太陽スペクトル(SolarSpectrum.txt)の読み込み ---
    try:
        sol_path = os.path.join(fileF, 'SolarSpectrum.txt')
        # IDLコードに合わせて3つの列を読み込む
        sol_data = np.loadtxt(sol_path)
        iwm2 = sol_data.shape[0]

        sol = np.zeros((iwm2, 3))
        sol[:, 0] = sol_data[:, 0] - 589.0 - sft
        sol[:, 1] = sol_data[:, 2]  # a2 -> sol[*,1]
        sol[:, 2] = sol_data[:, 1]  # a1 -> sol[*,2]

    except FileNotFoundError:
        print(f"Error: Solar spectrum file not found at {sol_path}")
        return

    # ドップラーシフトした太陽スペクトルの波長を計算
    wlsurf = (sol[:, 0] + 589.0) * (1 + Vms / c) * (1 + Vme / c) - 589.0

    # --- 観測波長グリッドに太陽スペクトルを内挿 ---
    surf = np.zeros((ixm, 2))



    # np.interpを使い、IDLのループによる線形補間を置き換える
    # surf[*,0]は地球大気の吸収線 (Telluric)
    surf[:, 0] = np.interp(wl, sol[:, 0], sol[:, 1], left=np.nan, right=np.nan)
    # surf[*,1]は太陽表面の吸収線 (Fraunhofer)
    surf[:, 1] = np.interp(wl, wlsurf, sol[:, 2], left=np.nan, right=np.nan)

    # NaNができてしまった場合、最も近い値で埋める (Edge case handling)
    surf[:, 0] = np.nan_to_num(surf[:, 0], nan=sol[0, 1])
    surf[:, 1] = np.nan_to_num(surf[:, 1], nan=sol[0, 2])

    # debug1
    print(f"Python surf[0] sum: {np.sum(surf[:, 0])}")
    print(f"Python surf[1] sum: {np.sum(surf[:, 1])}")
    print(f"Python surf[0, 100]: {surf[100, 0]}")
    print(f"Python surf[1, 100]: {surf[100, 1]}")

    Nat = np.zeros(ixm)
    surf1 = surf.copy()

    # --- メインループ ---
    for i in range(is_loop, ie_loop + 1):
        print(f"Processing file index: {i}")

        # --- 合計スペクトルデータの読み込み ---
        try:
            totfib_path = os.path.join(fileF2, f'{i}totfib_IDL.dat')
            # IDLコードではnat1, nat2を読み、nat2のみを使用
            Nat = np.loadtxt(totfib_path, usecols=1)
        except FileNotFoundError:
            print(f"Warning: Data file not found for index {i} at {totfib_path}. Skipping.")
            continue

        # D2輝線の周辺を除いたスペクトルを作成
        Nat2 = np.concatenate((Nat[0: D2 - 7], Nat[D2 + 7: ixm]))

        zansamin = 1.0e+32
        best_params = {}

        # FWHMとairmassの最適値をグリッドサーチ
        for iFWHM in range(30, 101):
            FWHM = iFWHM * 0.1
            sigma = FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))

            # PSF (Point Spread Function) の作成
            psf_x = np.arange(ixm) - float(ixm) / 2
            psf = np.exp(-psf_x ** 2 / (2.0 * sigma ** 2))
            psf2 = psf / np.sum(psf)

            # FFTによるコンボリューション
            # IDLの `shift(real_part(FFT(FFT(A,-1)*FFT(B,-1),1)),-ixm/2)*ixm` に相当
            # NumPyのFFTは規格化が異なる場合があるが、ここではIDLの振る舞いを模倣
            fft_surf0 = np.fft.fft(surf[:, 0])
            fft_surf1 = np.fft.fft(surf[:, 1])
            fft_psf2 = np.fft.fft(psf2)

            # ifft(fft(A)*fft(B))でコンボリューションを計算し、シフトして中央に合わせる
            conv_surf0 = np.fft.fftshift(np.real(np.fft.ifft(fft_surf0 * fft_psf2)))
            conv_surf1 = np.fft.fftshift(np.real(np.fft.ifft(fft_surf1 * fft_psf2)))

#debug2
            if iFWHM == 30:
                print(f"Python FWHM=3.0, sigma: {sigma}")
                print(f"Python FWHM=3.0, psf2 sum: {np.sum(psf2)}")
                # Pythonコードで *ixm をコメントアウトしている場合、それも揃える
                # surf1[:, 0] = conv_surf0
                # surf1[:, 1] = conv_surf1
                print(f"Python FWHM=3.0, surf1[*,0] sum: {np.sum(surf1[:, 0])}")
                print(f"Python FWHM=3.0, surf1[*,1] sum: {np.sum(surf1[:, 1])}")

            # IDLのコードは *ixm のスケーリングがあったが、IDLと規格化のタイミングが逆なので不要
            # 必要であれば以下の行のコメントを外す
            # conv_surf0 *= ixm
            # conv_surf1 *= ixm

            surf1[:, 0] = conv_surf0
            surf1[:, 1] = conv_surf1

            for iairm in range(51):
                airm = 0.1 * iairm
                surf2 = surf1[:, 0] ** airm * surf1[:, 1]

                # フィッティングに使用する領域を切り出す
                surf3 = np.concatenate((surf2[0: ixm // 2 - 7], surf2[ixm // 2 + 7: ixm]))
                pix_range = np.concatenate((np.arange(ixm // 2 - 7), np.arange(ixm // 2 + 7, ixm)))

                # 多項式フィッティング

                ratioa = Nat2 / surf3

                # np.polyfitは係数を高次から返すので、IDLのaa0[0]..[2]に合わせるため逆順にする
                aa0 = np.polyfit(pix_range, ratioa, 2)[::-1]

                ratiof = aa0[0] + aa0[1] * pix_range + aa0[2] * pix_range ** 2
                zansa = np.sum((Nat2 - surf3 * ratiof) ** 2)

#debug3
                if iFWHM == 30 and iairm == 10:
                    print(f"Python FWHM=3.0, airm=1.0")
                    print(f"  surf2 sum: {np.sum(surf2)}")
                    print(f"  surf3 sum: {np.sum(surf3)}")
                    print(f"  ratioa[100]: {ratioa[100]}")
                    # NumPyの係数は高次からなので、表示前に逆順にしてIDLと比較しやすくする
                    print(f"  aa0: {aa0[::-1]}")
                    print(f"  zansa: {zansa}")
                    # --- ここからデバッグ用コード ---
                   # try:
                   #    # IDLが書き出したデータを読み込む
                   #     idl_input = np.loadtxt('idl_polyfit_input.txt')
                   #     idl_pix_range = idl_input[:, 0]
                   #     idl_ratioa = idl_input[:, 1]

                   #     print("--- Polyfit Implementation Test ---")

                        # 1. Pythonが生成したデータでフィッティング
                   #     py_aa0 = np.polyfit(pix_range, ratioa, 2)[::-1]
                   #     print(f"Python data -> Python fit: {py_aa0[::-1]}")

                        # 2. IDLが生成したデータでフィッティング
                   #     idl_aa0_by_python = np.polyfit(idl_pix_range, idl_ratioa, 2)[::-1]
                   #     print(f"IDL data    -> Python fit: {idl_aa0_by_python[::-1]}")
                   #     print("Compare this with the IDL output for `aa0`")

                   # except FileNotFoundError:
                   #     print("`idl_polyfit_input.txt` not found. Please run the IDL script first.")

                if zansa <= zansamin:
                    zansamin = zansa
                    ratio = np.sum(Nat2 * surf3) / np.sum(surf3 * surf3)
                    best_params = {
                        'ratios': ratio,
                        'ratiof': aa0[0] + aa0[1] * np.arange(ixm) + aa0[2] * np.arange(ixm) ** 2,
                        'aa0s': aa0,
                        'airms': airm,
                        'FWHMs': FWHM,
                        'surf2s': surf2
                    }

        print(
            f"  Best fit params: airm={best_params['airms']:.2f}, FWHM={best_params['FWHMs']:.2f}, zansa={zansamin:.4e}")

        #最適パラメータだった時のsurf3sを再計算
        surf3s = np.concatenate((best_params['surf2s'][0: ixm // 2 - 7], best_params['surf2s'][ixm // 2 + 7: ixm]))
        # IDLと同様に、単純なスケーリングファクター ratio2 を計算
        ratio2 = np.sum(Nat2 * surf3s) / np.sum(surf3s * surf3s)
        Natb = Nat - ratio2 * best_params['surf2s']

        # --- 結果の保存 ---
        # test2.txt / test2.dat : 補正前とモデルスペクトル
        output_data1 = np.column_stack(
            [wl, Nat / np.mean(Nat2), best_params['surf2s'] * best_params['ratiof'] / np.mean(Nat2)])
        np.savetxt(os.path.join(fileF1, f'{i}test2_python_test.txt'), output_data1, fmt='%f')
        np.savetxt(os.path.join(fileF1, f'{i}test2_python_test.dat'), output_data1, fmt='%f')

        # ratio2.txt :
        surf3s = np.concatenate((best_params['surf2s'][0: ixm // 2 - 7], best_params['surf2s'][ixm // 2 + 7: ixm]))
        ratio2 = np.sum(Nat2 * surf3s) / np.sum(surf3s * surf3s)
        # 誤差の計算
        err = np.sqrt(np.mean((Nat2 - ratio2 * surf3s) ** 2)) / np.sqrt(len(Nat2))

#debug4
        print(f"Python Best airms: {best_params['airms']}")
        print(f"Python Best FWHMs: {best_params['FWHMs']}")
        print(f"Python ratio2: {ratio2}")


        with open(os.path.join(fileF1, f'{i}ratio2_python_test.txt'), 'w') as f:
            f.write(f'{ratio2} {err}\n')

        # Hapkeモデルを使ったフラックス変換
        try:
            hap_path = os.path.join(fileF1, f'Hapke{day}_IDL.fits')
            hap = fits.getdata(hap_path)
            tothap = np.sum(hap) * Shap * dwl * 1e+12
            cts2MR = tothap / ratio2
            print(f"Python cts2MR: {cts2MR}")
            print(f"  cts2MR = {cts2MR}")
            print(f"Python Natb sum: {np.sum(Natb)}")

            # exos.txt / exos.dat : 最終的なスペクトル
            exos_data = np.column_stack([wl, Natb * cts2MR])

            np.savetxt(os.path.join(fileF1, f'{i}exos_python_test.txt'), exos_data, fmt='%f')
            np.savetxt(os.path.join(fileF1, f'{i}exos_python_test.dat'), exos_data, fmt='%f')

            # --- writefitsに相当する処理 (コメントアウト) ---
            # b = a # この'b'は元コードでは定義が複雑なので、ここでは単純化
            # fits.writeto(f'{i}_sus2.fit', b, overwrite=True)
            # final_natb_data = (Natb * cts2MR).reshape(1, -1) # 1Dデータを2D FITSとして保存する場合
            # fits.writeto(f'{i}_test2.fit', final_natb_data, overwrite=True)

        except FileNotFoundError:
            print(f"Warning: Hapke file not found at {hap_path}. Skipping final conversion.")
            continue

    print('end')


# --- スクリプトの実行 ---
if __name__ == '__main__':
    pro_subsurfnew2()