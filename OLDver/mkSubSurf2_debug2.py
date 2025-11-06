import numpy as np
from astropy.io import fits
import os


def pro_subsurfnew2_use_idl_polyfit_input():
    """
    最終版のPythonコード。
    polyfitの入力データ(ratioa)をIDLのファイルから読み込むバージョン。
    """
    day = 'test'
    # !! 注意: ご自身の環境に合わせてパスを修正してください !!
    fileF = 'C:/Users/hanac/University/Senior/Mercury/Haleakala2025/'
    fileF1 = os.path.join(fileF, 'output', 'test')
    fileF2 = os.path.join(fileF, 'output', 'test')

    is_loop = 10001
    ie_loop = 10004

    try:
        a_template = fits.getdata(os.path.join(fileF1, '10001_sf22_IDL.fit'))
        iym, ixm = a_template.shape
    except FileNotFoundError:
        print(f"エラー: テンプレートFITSファイル '10001_sf22_IDL.fit' が見つかりません。")
        return

    # --- 物理定数とパラメータ ---
    Vme = -38.3179619
    Vms = -10.0384175
    c = 299792.458
    sft = 0.002

    # --- 共有データの読み込み ---
    try:
        wl = np.loadtxt(os.path.join(fileF1, 'wl_IDL.txt'))
        sol_data = np.loadtxt(os.path.join(fileF, 'SolarSpectrum.txt'))
    except FileNotFoundError:
        print(f"エラー: wl_IDL.txt または SolarSpectrum.txt が見つかりません。")
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
        print(f"\nProcessing file index: {i}")
        Nat = np.loadtxt(os.path.join(fileF2, f'{i}totfib_IDL.dat'), usecols=1)

        center_pix = ixm // 2
        d2 = 209
        slice_end1 = center_pix - 7
        slice_start1 = center_pix + 6
        slice_end2 = d2 - 7
        slice_start2 = d2 + 6

        Nat2 = np.concatenate((Nat[0:slice_end2], Nat[slice_start2:ixm]))
        pix_range = np.concatenate((np.arange(slice_end1), np.arange(slice_start1, ixm)))

        zansamin = 1.0e+32
        best_params = {}

        # 最初のファイルの時だけ、zansaをファイルに書き出す
        if i == is_loop:
            output_filename = os.path.join(fileF1, 'all_zansa_python.txt')
            with open(output_filename, 'w') as f:
                f.write("FWHM,airm,zansa\n")

                for iFWHM in range(30, 101):
                    FWHM = iFWHM * 0.1
                    try:
                        # IDLが生成したsurf1ファイルを指定
                        idl_surf1_path = os.path.join(fileF1, 'idl_surf1_output.txt')
                        surf1 = np.loadtxt(idl_surf1_path)
                    except Exception as e:
                        print(f"IDLのsurf1ファイル読み込みに失敗しました: {e}")
                        print("先にIDLを実行して idl_surf1_output.txt を生成してください。")
                        return

                    for iairm in range(51):
                        airm = 0.1 * iairm
                        surf2 = surf1[:, 0] ** airm * surf1[:, 1]
                        surf3 = np.concatenate((surf2[0:slice_end1], surf2[slice_start1:ixm]))

                        ### ★★★ 変更点 ★★★ ###
                        # ratioaを計算する代わりに、IDLの出力ファイルから読み込む
                        try:
                            idl_input_path = os.path.join(fileF1, 'idl_polyfit_input.txt')
                            ratioa = np.loadtxt(idl_input_path, usecols=1)
                        except Exception as e:
                            print(f"idl_polyfit_input.txt の読み込みに失敗: {e}")
                            return


                        aa0 = np.polyfit(pix_range, ratioa, 2)[::-1]
                        ratiof = aa0[0] + aa0[1] * pix_range + aa0[2] * pix_range ** 2
                        zansa = np.sum((Nat2 - surf3 * ratiof) ** 2)

                        f.write(f"{FWHM:.1f},{airm:.1f},{zansa:.6e}\n")

                        if zansa <= zansamin:
                            zansamin = zansa
                            best_params = {'aa0s': aa0, 'airms': airm, 'FWHMs': FWHM, 'surf2s': surf2}
        else:
            # 2番目以降のファイルは通常通り計算のみ行う
            for iFWHM in range(30, 101):
                FWHM = iFWHM * 0.1
                sigma = FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))
                psf_x = np.arange(ixm) - float(ixm) / 2
                psf = np.exp(-psf_x ** 2 / (2.0 * sigma ** 2))
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
                    surf3 = np.concatenate((surf2[0:slice_end1], surf2[slice_start1:ixm]))

                    ### ★★★ 変更点 ★★★ ###
                    # ratioaを計算する代わりに、IDLの出力ファイルから読み込む
                    try:
                        idl_input_path = os.path.join(fileF1, 'idl_polyfit_input.txt')
                        ratioa = np.loadtxt(idl_input_path, usecols=1)
                    except Exception as e:
                        print(f"idl_polyfit_input.txt の読み込みに失敗: {e}")
                        return
                    ### ★★★ここまで★★★ ###

                    aa0 = np.polyfit(pix_range, ratioa, 2)[::-1]
                    ratiof = aa0[0] + aa0[1] * pix_range + aa0[2] * pix_range ** 2
                    zansa = np.sum((Nat2 - surf3 * ratiof) ** 2)
                    if zansa <= zansamin:
                        zansamin = zansa
                        best_params = {'aa0s': aa0, 'airms': airm, 'FWHMs': FWHM, 'surf2s': surf2}

        print(
            f"  Best fit params for {i}: airm={best_params['airms']:.1f}, FWHM={best_params['FWHMs']:.1f}, zansa_min={zansamin:.4e}")

        aa0s = best_params['aa0s']
        surf2s = best_params['surf2s']
        Natb = Nat - (aa0s[0] + aa0s[1] * np.arange(ixm) + aa0s[2] * np.arange(ixm) ** 2) * surf2s
        # (以降のファイル保存処理など)

    print('end')


if __name__ == '__main__':
    pro_subsurfnew2_use_idl_polyfit_input()