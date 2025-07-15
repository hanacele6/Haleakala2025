import numpy as np
from astropy.io import fits
from pathlib import Path
import warnings


def perform_skyflat_correction():
    """
    IDLの 'pro skyflat1' コードをPythonで再現します。

    処理の概要:
    1. スカイフレーム（夜空を撮影した画像）を複数読み込み、合成して
       マスタースカイフレームを作成します。
    2. マスターフレームから、検出器の空間方向（スリット方向）の
       明るさのムラを表す1次元の規格化プロファイルを作成します。
    3. このプロファイルを使い、各観測フレームの明るさのムラを補正します。
    """
    # --- ファイルパスとフレーム番号の設定 ---
    is_frame = 10001  # 処理を開始するフレーム番号
    ie_frame = 10008  # 処理を終了するフレーム番号
    skys_frame = 10005  # スカイフレームの開始番号
    skye_frame = 10008  # スカイフレームの終了番号

    # ご自身の環境に合わせてパスを変更してください
    try:
        base_path = Path.cwd()
        #base_path = Path('C:/Users/hanac/University/Senior/Mercury/Haleakala2025/')
        # 入力と出力のディレクトリを同じ場所に設定
        date = "20241119"
        in_dir = base_path / 'output' / date
        out_dir = base_path / 'output' / date
        in_dir.mkdir(parents=True, exist_ok=True)
        print(f"入力ディレクトリ: {in_dir.resolve()}")
        print(f"出力ディレクトリ: {out_dir.resolve()}")
    except Exception as e:
        print(f"パスの設定中にエラーが発生しました: {e}")
        return

    # --- 画像サイズの取得 ---
    try:
        # 最初のフレームを読み込み、画像の大きさを取得
        first_frame_path = in_dir / f'{is_frame}_Na_python.fit'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            header_data = fits.getdata(first_frame_path)
        iym, ixm = header_data.shape  # NumPyのshapeは (行, 列) = (iym, ixm)
        print(f"画像サイズを {first_frame_path.name} から取得: (ixm={ixm}, iym={iym})")
    except FileNotFoundError:
        print(f"エラー: 基準ファイル {first_frame_path} が見つかりません。")
        print("デモ用にダミーデータを生成して処理を続行します。")
        ixm, iym = 1024, 512

        # デモ用のダミーファイルを作成する関数
        def create_dummy_fits(filepath, shape):
            iym_d, ixm_d = shape
            # y方向に緩やかに暗くなるプロファイル(周辺減光)を模擬
            y_profile = 1.0 - 0.5 * (np.linspace(-1, 1, iym_d)) ** 2
            # 2D画像に拡張
            image_data = 1000 * y_profile[:, np.newaxis]
            # ノイズを追加
            image_data += np.random.normal(0, 10, size=shape)
            # FITSファイルとして保存
            hdu = fits.PrimaryHDU(image_data.astype(np.float32))
            hdu.writeto(filepath, overwrite=True)
            print(f"ダミーファイルを作成: {filepath.name}")

        for i in range(is_frame, ie_frame + 1):
            create_dummy_fits(in_dir / f'{i}_Na_python.fit', (iym, ixm))

    # --- 1. マスタースカイフレームの作成 ---
    sky = None
    print("\nマスタースカイフレームを作成中...")
    for i in range(skys_frame, skye_frame + 1):
        file_path = in_dir / f'{i}_Na_python.fit'
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                data = fits.getdata(file_path).astype(np.float64)

            if sky is None:
                sky = data
            else:
                sky += data  # 画像を加算（スタッキング）
            print(f"  読み込み・合成: {file_path.name}")
        except FileNotFoundError:
            print(f"警告: スカイフレーム {file_path} が見つかりません。スキップします。")

    if sky is None:
        print("エラー: スカイフレームを1枚も読み込めませんでした。処理を終了します。")
        return

    # --- 2. 空間方向の明るさプロファイルの計算と規格化 ---
    # sky2: 各行の平均強度を計算 (IDLの total(sky[*,iy])/ixm と等価)
    # axis=1 は行ごとに計算することを意味します
    sky2 = sky.mean(axis=1)

    # 計算結果をテキストファイルに保存
    sky_txt_path = base_path / 'sky_python.txt'
    sky2_data_to_save = np.vstack((np.arange(iym), sky2)).T
    np.savetxt(sky_txt_path, sky2_data_to_save, fmt='%d %.6f', header='iy average_intensity')
    print(f"各行の平均強度プロファイルを保存しました: {sky_txt_path}")

    # sky4: 中央の行の強度でプロファイルを規格化（ノーマライズ）する
    center_row_intensity = sky2[iym // 2]  # iym//2 は中央の行インデックス
    sky4 = sky2 / center_row_intensity
    print(f"プロファイルを中央行(iy={iym // 2})の強度({center_row_intensity:.2f})で規格化しました。")

    # (参考) IDLコードのsky3の作成と保存
    # これは各ピクセルの値を、そのピクセルが属する行の規格化係数で割る処理
    # NumPyのブロードキャスト機能を使うと、forループなしで効率的に計算できます
    # sky4[:, np.newaxis] は (iym,) の1D配列を (iym, 1) の2D配列に変換し、
    # (iym, ixm) の sky 画像と正しく割り算できるようにします。
    sky3 = sky / sky4[:, np.newaxis]
    sky3_path = Path.cwd() / 'sky_python.fit'
    fits.writeto(sky3_path, sky3, overwrite=True)
    print(f"補正済みのマスタースカイフレームを保存しました: {sky3_path}")

    # --- 3. 各観測フレームへの補正適用 ---
    print("\n各観測フレームに補正を適用中...")
    for i in range(is_frame, ie_frame + 1):
        in_path = in_dir / f'{i}_Na_python.fit'
        out_path = out_dir / f'{i}_sf_python.fit'  # `_sf` は skyflat の略

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                mc = fits.getdata(in_path).astype(np.float64)

            # 画像の各行を、対応するsky4の値で割る
            # ここでもNumPyのブロードキャストが活躍します
            mc2 = mc / sky4[:, np.newaxis]

            # 補正後のファイルを保存
            fits.writeto(out_path, mc2, overwrite=True)
            print(f"  {in_path.name} -> {out_path.name}")

        except FileNotFoundError:
            print(f"警告: 観測フレーム {in_path} が見つかりません。スキップします。")

    print("\nOK")
    print("処理が完了しました。")


# --- スクリプトの実行 ---
if __name__ == '__main__':
    perform_skyflat_correction()