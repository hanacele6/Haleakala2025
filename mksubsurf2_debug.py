import numpy as np
from astropy.io import fits
import os


def pro_subsurfnew2_load_idl_surf1():
    """
    IDLコード 'pro subsurfnew2' のPython翻訳版。
    原因切り分けのため、IDLで計算したsurf1を読み込んで
    その後の計算をPythonで実行するテストバージョン。
    """
    day = 'test'

    # --- 基本設定 ---
    # !! 注意: ご自身の環境に合わせてパスを修正してください !!
    fileF = 'C:/Users/hanac/University/Senior/Mercury/Haleakala2025/'
    fileF1 = os.path.join(fileF, 'output', 'test')
    fileF2 = os.path.join(fileF, 'output', 'test')

    is_loop = 10001
    ie_loop = 10004
    D2 = 209  # IDLのロジックに合わせる

    # --- FITSファイルと基本情報の読み込み ---
    try:
        # FITSファイル名は適宜修正してください
        a = fits.getdata(os.path.join(fileF1, f'{is_loop}_sf22_IDL.fit'))
        iym, ixm = a.shape
    except FileNotFoundError:
        print(
            f"エラー: FITSファイルが見つかりません。パスを確認してください: {os.path.join(fileF1, f'{is_loop}_sf22.fit')}")
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
        wl = np.loadtxt(os.path.join(fileF1, 'wl_IDL.txt'))
        dwl = wl[ixm // 2] - wl[ixm // 2 - 1]
        sol_data = np.loadtxt(os.path.join(fileF, 'SolarSpectrum.txt'))
    except FileNotFoundError:
        print("エラー: wl_python.txt または SolarSpectrum.txt が見つかりません。")
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
            # DATファイル名は適宜修正してください
            Nat = np.loadtxt(os.path.join(fileF2, f'{i}totfib_IDL.dat'), usecols=1)
        except FileNotFoundError:
            print(f"警告: データファイルが見つかりません (index: {i})。スキップします。")
            continue

        # --- スライスルールの統一（D2を使用） ---
        center_pix = ixm // 2
        d2 = 209
        slice_end1 = center_pix - 7  # Pythonのスライス仕様に合わせて調整
        slice_start1 = center_pix + 6
        slice_end2 = d2 - 7  # Pythonのスライス仕様に合わせて調整
        slice_start2 = d2 + 6
        Nat2 = np.concatenate((Nat[0:slice_end2], Nat[slice_start2:ixm]))

        zansamin = 1.0e+32
        best_params = {}

        # --- 最適化ループ ---
        for iFWHM in range(30, 101):
            FWHM = iFWHM * 0.1

            # -------------------- 変更点 --------------------
            # surf1をPythonで計算する代わりに、IDLの出力ファイルを読み込む
            try:
                # IDLが生成したsurf1ファイルを指定
                idl_surf1_path = os.path.join(fileF1, 'idl_surf1_output.txt')
                surf1 = np.loadtxt(idl_surf1_path)
            except Exception as e:
                print(f"IDLのsurf1ファイル読み込みに失敗しました: {e}")
                print("先にIDLを実行して idl_surf1_output.txt を生成してください。")
                return

            # 元のPythonによる畳み込み計算は実行しない
            # ------------------------------------------------

            for iairm in range(51):
                airm = 0.1 * iairm
                surf2 = surf1[:, 0] ** airm * surf1[:, 1]

                surf3 = np.concatenate((surf2[0:slice_end1], surf2[slice_start1:ixm]))
                pix_range = np.concatenate((np.arange(slice_end1), np.arange(slice_start1, ixm)))

                    # -------------------- ★★★ 変更点 ★★★ --------------------
                    # ratioaを計算する代わりに、IDLの出力ファイルから読み込む
                try:
                    idl_input_path = os.path.join(fileF1, 'idl_polyfit_input.txt')
                        # ファイルの2列目（IDLが計算したratioa）のみを読み込む
                    ratioa = np.loadtxt(idl_input_path, usecols=1)
                except Exception as e:
                    print(f"idl_polyfit_input.txt の読み込みに失敗: {e}")
                    return
                aa0 = np.polyfit(pix_range, ratioa, 2)[::-1]
                ratiof = aa0[0] + aa0[1] * pix_range + aa0[2] * pix_range ** 2
                zansa = np.sum((Nat2 - surf3 * ratiof) ** 2)+1e+8

                if i == is_loop:
                    # 指数表記の方が見やすいので :.4e を使います
                    print(f"FWHM={FWHM:.1f}, airm={airm:.1f}, zansa={zansa:.4e}")


                if iFWHM == 30 and iairm == 10:
                    print(f"\n--- Python (IDLのsurf1を使用) ---")
                    print(f"FWHM=3.0, airm=1.0")
                    print(f"  surf2 sum: {np.sum(surf2)}")
                    print(f"  surf3 sum: {np.sum(surf3)}")
                    print(f"  ratioa[100]: {ratioa[100]}")
                    print(f"  aa0: {aa0}")
                    print(f"  zansa: {zansa}")

                    if i == is_loop:
                        # 小数点以下20桁で出力フォーマットを指定
                        output_format = '%.20f'

                        # surf1 の出力
                        #np.savetxt(os.path.join(fileF1, 'python_surf1_output.txt'),
                        #           surf1, fmt=output_format)

                        # surf2 の出力
                        np.savetxt(os.path.join(fileF1, 'python_surf2_output_test.txt'),
                                   surf2, fmt=output_format)

                        # surf3 の出力
                        np.savetxt(os.path.join(fileF1, 'python_surf3_output_test.txt'),
                                   surf3, fmt=output_format)

                        # polyfitの入力データ(x, y)をファイルに出力
                        polyfit_input_data = np.column_stack([pix_range, ratioa])
                        np.savetxt(os.path.join(fileF1, 'python_polyfit_input.txt'),
                                   polyfit_input_data, fmt=output_format)

                        # polyfitの出力係数をファイルに出力
                        np.savetxt(os.path.join(fileF1, 'python_polyfit_coeffs.txt'),
                                   aa0, fmt=output_format)

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
            f"  Best fit params: airm={best_params.get('airms', 'N/A'):.2f}, FWHM={best_params.get('FWHMs', 'N/A'):.2f}, zansa={zansamin:.4e}")

        # --- ループ後の最終計算 ---
        Nat2_final = np.concatenate((Nat[0:slice_end1], Nat[slice_start2:ixm]))
        surf2s = best_params.get('surf2s')
        if surf2s is None:
            print("エラー: 最適なパラメータが見つかりませんでした。")
            continue

        surf3s_final = np.concatenate((surf2s[0:slice_end1], surf2s[slice_start2:ixm]))

        ratio2 = np.sum(Nat2_final * surf3s_final) / np.sum(surf3s_final * surf3s_final)
        Natb = Nat - ratio2 * surf2s

        # test2.txtには多項式フィットの結果を出力
        output_data1 = np.column_stack(
            [wl, Nat / np.mean(Nat2_final), best_params['surf2s'] * best_params['ratiof_full'] / np.mean(Nat2_final)])
        np.savetxt(os.path.join(fileF1, f'{i}test2_python_test.txt'), output_data1, fmt='%f')
        np.savetxt(os.path.join(fileF1, f'{i}test2_python_test.dat'), output_data1, fmt='%f')

        # ratio2.txtの保存
        err = np.sqrt(np.mean((Nat2_final - ratio2 * surf3s_final) ** 2)) / np.sqrt(len(Nat2_final))
        with open(os.path.join(fileF1, f'{i}ratio2_python_test.txt'), 'w') as f:
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
            np.savetxt(os.path.join(fileF1, f'{i}exos_python_test.txt'), exos_data, fmt='%f')
            np.savetxt(os.path.join(fileF1, f'{i}exos_python_test.dat'), exos_data, fmt='%f')

        except FileNotFoundError:
            print(f"Warning: Hapke file not found at {hap_path}. Skipping final conversion.")
            continue

    print('end')


if __name__ == '__main__':
    pro_subsurfnew2_load_idl_surf1()