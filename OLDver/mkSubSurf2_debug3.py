import numpy as np
from astropy.io import fits
import os


def pro_subsurfnew2_with_specified_params():
    """
    ユーザーが指定した特定の最適パラメータを使って計算を行うバージョン。
    パラメータ探索ループは行わない。
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

    # --- 物理定数と共有データの準備 (変更なし) ---
    Vme = -38.3179619
    Vms = -10.0384175
    Rmn = 7.362434
    sft = 0.002
    Rmc = 4.879e+8
    Rca = Rmc / Rmn
    Shap = Rca ** 2 / 1e+4
    c = 299792.458
    try:
        wl = np.loadtxt(os.path.join(fileF1, 'wl_IDL.txt'))
        dwl = wl[ixm // 2] - wl[ixm // 2 - 1]
        sol_data = np.loadtxt(os.path.join(fileF, 'SolarSpectrum.txt'))
        hap_path = os.path.join(fileF1, f'Hapke{day}_IDL.fits')
        hap = fits.getdata(hap_path)
    except FileNotFoundError as e:
        print(f"エラー: 入力ファイルが見つかりません。({e.filename})")
        return

    # --- メインループ ---
    for i in range(is_loop, ie_loop + 1):
        print(f"\nProcessing file index: {i}")
        Nat = np.loadtxt(os.path.join(fileF2, f'{i}totfib_IDL.dat'), usecols=1)

        # --- ★★★ ユーザーが指定する最適パラメータ ★★★ ---
        # 例として、過去の探索で見つかった値を設定
        specified_FWHM = 10.0
        specified_airm = 1.3
        print(f"Using specified parameters: FWHM={specified_FWHM}, airm={specified_airm}")
        # ----------------------------------------------------

        # --- スライス処理 ---
        center_pix = ixm // 2
        d2 = 209
        slice_end1 = center_pix - 7
        slice_start1 = center_pix + 6
        slice_end2 = d2 - 7
        slice_start2 = d2 + 6
        Nat2 = np.concatenate((Nat[0:slice_end2], Nat[slice_start2:ixm]))
        pix_range = np.concatenate((np.arange(slice_end1), np.arange(slice_start1, ixm)))

        # --- 指定されたパラメータで単一計算を実行 ---

        # 1. surf1 (畳み込み) の計算
        try:
            # IDLが生成したsurf1ファイルを指定
            idl_surf1_path = os.path.join(fileF1, 'python_surf1_output.txt')
            surf1 = np.loadtxt(idl_surf1_path)
        except Exception as e:
            print(f"IDLのsurf1ファイル読み込みに失敗しました: {e}")
            print("先にIDLを実行して idl_surf1_output.txt を生成してください。")
            return

        # 2. surf2 の計算
        surf2s = surf1[:, 0] ** specified_airm * surf1[:, 1]

        # 3. polyfit係数 (aa0s) の計算
        surf3s = np.concatenate((surf2s[0:slice_end1], surf2s[slice_start1:ixm]))
        ratioa = Nat2 / surf3s
        aa0s = np.polyfit(pix_range, ratioa, 2)[::-1]

        # 4. 最終的なスペクトル (Natb) の計算
        ratiofts = aa0s[0] + aa0s[1] * np.arange(ixm) + aa0s[2] * np.arange(ixm) ** 2
        Natb = Nat - ratiofts * surf2s

        print(f"  Calculation complete for file {i}.")

        # 1. ratio2 の計算
        ratio2 = np.sum(Nat2 * surf3s) / np.sum(surf3s * surf3s)

        # 2. tothap の計算
        tothap = np.sum(hap) * Shap * dwl * 1e12

        # 3. cts2MR の計算
        cts2MR = tothap / ratio2
        print(f"  Conversion factor (cts2MR): {cts2MR}")

        # 4. exos.txt / exos.dat ファイルへの書き出し
        exos_data = np.column_stack([wl, Natb * cts2MR])

        exos_txt_path = os.path.join(fileF1, f'{i}exos_python_debug.txt')
        exos_dat_path = os.path.join(fileF1, f'{i}exos_python_debug.dat')

        np.savetxt(exos_txt_path, exos_data, fmt='%f')
        np.savetxt(exos_dat_path, exos_data, fmt='%f')

        print(f"  Result saved to {exos_txt_path}")



print('end')


if __name__ == '__main__':
    pro_subsurfnew2_with_specified_params()