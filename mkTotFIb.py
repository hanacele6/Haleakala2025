import numpy as np
from astropy.io import fits
from pathlib import Path
import warnings


def total_fibers():
    """
    IDLの 'pro totfib' コードをPythonで再現します。

    2Dスペクトル画像からファイバーをグループ化し、信号を合計、
    背景光を除去して最終的な1次元スペクトルを生成します。
    """
    # --- ファイルパスとパラメータの設定 ---
    try:
        base_path = Path.cwd()
        date = "20241119"
        data_dir = base_path / 'output' / date
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"データディレクトリ: {data_dir.resolve()}")
    except Exception as e:
        print(f"パスの設定中にエラーが発生しました: {e}")
        return

    is_frame = 10001
    ie_frame = 10004

    # --- ファイバーのグループ分け定義 ---
    # 10x12の仮想的なファイバー配列を想定
    X_FIBERS = 10

    # 水星のファイバーが含まれる行の範囲 (0-indexed)
    jyms = 4 - 1
    jyme = 8 - 1

    # 背景光のファイバーが含まれる行の範囲 (2つの領域)
    jys1 = 1 - 1
    jye1 = 3 - 1
    jys2 = 9 - 1
    jye2 = 12 - 1

    # --- 波長データを読み込む ---
    wl_txt_path = data_dir / 'wl_python.txt'
    try:
        wl = np.loadtxt(wl_txt_path)
        ixm = len(wl)
        print(f"波長データを {wl_txt_path.name} から読み込みました。 ({ixm}点)")
    except (FileNotFoundError, IOError):
        print(f"エラー: 波長ファイル {wl_txt_path} が見つかりません。処理を中断します。")
        return

    # --- デッドファイバーのインデックスを定義 ---
    # IDLの `if(k ge ...)` のロジックを整理。
    # これらは、マッピング後のグリッドにおける「デッドファイバー」の開始インデックスです。
    dead_fiber_indices = sorted(list(set([
        0, 6, 9, 82, 89, 94, 97, 100, 102, 103, 104, 105, 106, 107,
        108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119
    ])))

    # --- メイン処理ループ ---
    print("\nファイバー合成処理を開始...")
    for i in range(is_frame, ie_frame + 1):
        in_path = data_dir / f'{i}_sf22_python.fit'
        out_path = data_dir / f'{i}totfib_python.dat'

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                mc = fits.getdata(in_path).astype(np.float64)
            ifibm = mc.shape[0]  # 入力画像の行数を取得

            # --- ステップ1: デッドファイバーを考慮したリマッピング ---
            # mc2: 120ファイバー分のスペースを持つ配列
            mc2 = np.zeros((120, ixm), dtype=np.float64)

            for ifib in range(ifibm):  # 入力画像の各ファイバー(行)をループ
                k = ifib
                # kがデッドファイバーのインデックス以上になるたびにkを1つ増やす
                for dead_idx in dead_fiber_indices:
                    if k >= dead_idx:
                        k += 1

                if k < 120:
                    # NumPyの配列は[行, 列]なので、mcの行全体をmc2のk行目にコピー
                    mc2[k, :] = mc[ifib, :]

            # --- ステップ2 & 3: ファイバーのグループ分けと信号の合計 ---
            # 水星と背景光のファイバーインデックス(jfib)のリストを作成
            mercury_jfibs = [jx + jy * X_FIBERS for jy in range(jyms, jyme + 1) for jx in range(X_FIBERS)]
            bg_jfibs1 = [jx + jy * X_FIBERS for jy in range(jys1, jye1 + 1) for jx in range(X_FIBERS)]
            bg_jfibs2 = [jx + jy * X_FIBERS for jy in range(jys2, jye2 + 1) for jx in range(X_FIBERS)]
            background_jfibs = bg_jfibs1 + bg_jfibs2

            # NumPyの機能で、リストにあるインデックスの行だけを抜き出して合計
            d_mercury = mc2[mercury_jfibs, :].sum(axis=0)
            e_background = mc2[background_jfibs, :].sum(axis=0)

            # --- ステップ4: 背景光の除去 ---
            # 50/45 は、水星と背景光で合計したファイバー数の比率による補正係数
            scaling_factor = 50.0 / 45.0
            final_spectrum = d_mercury - e_background * scaling_factor

            # --- ステップ5: 最終スペクトルの保存 ---
            data_to_save = np.vstack((wl, final_spectrum)).T
            np.savetxt(out_path, data_to_save, fmt='%.8e')
            print(f"  {in_path.name} -> {out_path.name}")

        except FileNotFoundError:
            print(f"警告: ファイル {in_path} が見つかりません。スキップします。")
            continue

    print("\nend")
    print("処理が完了しました。")


# --- スクリプトの実行 ---
if __name__ == '__main__':
    total_fibers()