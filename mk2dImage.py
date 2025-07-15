import numpy as np
from astropy.io import fits
from pathlib import Path
import warnings


def process_image():
    """
    IDLの 'pro image' コードをPythonで再現します。

    処理の概要:
    1. 各 `_sf22.fit` 画像を読み込みます。
    2. 画像の各行について、特定の列範囲 (71-91) のピクセル値を合計し、
       1次元のプロファイル `b` を作成します。
    3. `b` の値を元に、奇妙なロジックを経て1次元配列 `b2` を生成します。
    4. `b2` を10x12の2次元画像 `c` に再構成します。
    5. 最終的な画像 `c` を `_im.fit` として保存します。
    """
    # --- ファイルパスとパラメータの設定 ---
    try:
        base_path = Path.cwd()
        date = "20241107"
        data_dir = base_path / 'output' / date
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"データディレクトリ: {data_dir.resolve()}")
    except Exception as e:
        print(f"パスの設定中にエラーが発生しました: {e}")
        return

    is_frame = 10001
    ie_frame = 10003

    # 出力画像のサイズ
    X, Y = 10, 12

    # --- 最初のファイルから画像サイズを取得 ---
    try:
        first_path = data_dir / f'{is_frame}_sf22_python.fit'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            header_data = fits.getdata(first_path)
        # NumPyのshapeは (行, 列)
        ifibm, ixm = header_data.shape
        print(f"入力画像サイズを {first_path.name} から取得: (ixm={ixm}, ifibm={ifibm})")
    except FileNotFoundError:
        print(f"エラー: 基準ファイル {first_path} が見つかりません。")
        return

    # --- メイン処理ループ ---
    print("\n画像再構成処理を開始...")
    for ifile in range(is_frame, ie_frame + 1):
        in_path = data_dir / f'{ifile}_sf22_python.fit'
        out_path = data_dir / f'{ifile}_im_python.fit'

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                a = fits.getdata(in_path).astype(np.float64)

            # --- ステップ1 & 2: 空間プロファイル(b)の作成とb2への代入 ---

            # b: 各行の列71-91の合計強度 (NumPyなら1行で計算可能)
            # IDLの a[71:91,ifib] に相当。Pythonスライスは終点を含まないので+1
            b = a[:, 71:92].sum(axis=1)

            # b2: おそらくデッドファイバーの処理部分
            b2 = np.arange(X * Y, dtype=np.float64)
            j = 0

            for ifib in range(ifibm):  # 入力画像の各行をループ
                # このif文の羅列は、jが単調増加するため、j=0 の場合を除き、
                # ほとんどの条件は決して満たされていない。
                if j == 1 - 1:  # j=0
                    b2[j] = 0;
                    j += 1
                if j == 7 - 1:  # j=6
                    b2[j] = 0;
                    j += 1
                if j == 83 - 1:  # j=82
                    b2[j] = 0;
                    j += 1
                if j == 10 - 1:  # j=9
                    b2[j] = 0;
                    j += 1
                if j == 10 - 1:  # j=9
                    b2[j] = 0;
                    j += 1

                # bの値をb2に代入
                # 配列の範囲外アクセスを防ぐチェック
                if j < len(b2):
                    b2[j] = b[ifib]
                j += 1

            # --- ステップ3: 1D配列(b2)から2D画像(c)への再構成 ---
            c = b2.reshape((Y, X))

            # 補正後のファイルを保存
            fits.writeto(out_path, c, overwrite=True)
            print(f"  {in_path.name} -> {out_path.name} (10x12画像)")

        except FileNotFoundError:
            print(f"警告: ファイル {in_path} が見つかりません。スキップします。")
            continue

    print("\nend")
    print("処理が完了しました。")


# --- スクリプトの実行 ---
if __name__ == '__main__':
    process_image()