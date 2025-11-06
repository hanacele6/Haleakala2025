import numpy as np
from astropy.io import fits
from pathlib import Path
import warnings


def process_skyflat2():
    """
    IDLの 'pro skyflat2' コードをPythonで再現します。

    処理の概要:
    1. 'skyflat1'で補正された各画像 (`_sf.fit`) を読み込みます。
    2. 画像から、指定された中心ピクセル周りの特定の列範囲だけを切り出します。
    3. 切り出した部分画像を、新しいFITSファイル (`_sf22.fit`) として保存します。
    4. 波長データファイル (`comp.txt`) を読み込み、切り出した画像の列範囲に
       対応する波長リストを `wl.txt` として保存します。
    """
    # --- ファイルパスとパラメータの設定 ---
    # ご自身の環境に合わせてパスを変更してください
    try:
        base_path = Path.cwd()
        date = "20241119"
        data_dir = base_path / 'output' / date
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"データディレクトリ: {data_dir.resolve()}")
    except Exception as e:
        print(f"パスの設定中にエラーが発生しました: {e}")
        return

    is_frame = 10001  # 処理を開始するフレーム番号
    ie_frame = 10004  # 処理を終了するフレーム番号

    # 切り出しの中心列と幅
    ixc = 761 - 1  # 中心の列番号 (0-indexed)
    dw1 = 200  # 中心からの切り出し幅 (左側)
    dw2 = 200  # 中心からの切り出し幅 (右側) ※IDLコードではwl.txtの書き出しにdw2が使われていましたが、同じ値です

    # --- メイン処理ループ (画像の切り出し) ---
    print("\n画像の切り出し処理を開始...")
    for i in range(is_frame, ie_frame + 1):
        in_path = data_dir / f'{i}_sf_python.fit'
        out_path = data_dir / f'{i}_sf22_python.fit'

        try:
            # 補正済み画像を読み込む
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                mc = fits.getdata(in_path)

            # --- 元のIDLコードでコメントアウトされていた部分 (参考) ---
            # これらは、画像の特定領域の平均値で規格化（ノーマライズ）し、
            # スカイ画像を引き算する、という背景光除去の処理です。
            # 今回のコードでは実行されません。
            # ixs=100; ixe=350; iys=80; iye=100
            # sky=readfits('10001_sf.fit')
            # sky1=sky/mean(sky[ixs:ixe,iys:iye])
            # Mc1=Mc/mean(Mc[ixs:ixe,iys:iye])
            # Mc2=(Mc1-sky1)*mean(Mc[ixs:ixe,iys:iye])

            # --- 画像の切り出し ---
            # NumPyの配列スライス [行, 列] を使います
            # IDL: Mc[ixc-dw1:ixc+dw1, *]
            # NumPy: mc[:, ixc-dw1:ixc+dw1+1]
            start_col = ixc - dw1
            end_col = ixc + dw1
            # Pythonのスライスは終点を含まないので、+1 が必要です
            mc_cut = mc[:, start_col: end_col + 1]

            # 切り出した画像を新しいFITSファイルとして保存
            fits.writeto(out_path, mc_cut, overwrite=True)
            print(f"  {in_path.name} から一部を切り出し -> {out_path.name}")

        except FileNotFoundError:
            print(f"警告: ファイル {in_path} が見つかりません。スキップします。")
            continue

    # --- 波長データの切り出し ---
    print("\n波長データの処理を開始...")
    comp_txt_path = data_dir / 'comp_python.txt'
    wl_txt_path = data_dir / 'wl_python.txt'

    try:
        # np.loadtxtを使ってcomp.txtの1列目（波長データ）だけを効率的に読み込む
        wl = np.loadtxt(comp_txt_path, usecols=0)

        # 画像を切り出したのと同じ範囲の波長データをスライスする
        start_idx = ixc - dw1
        end_idx = ixc + dw2  # dw2はdw1と同じ
        wl_cut = wl[start_idx: end_idx + 1]

        # 切り出した波長リストをテキストファイルに保存
        np.savetxt(wl_txt_path, wl_cut, fmt='%.8f', header='Wavelength(nm)')
        print(f"対応する波長リストを保存しました: {wl_txt_path}")

    except (FileNotFoundError, IOError):
        print(f"警告: 波長ファイル {comp_txt_path} が見つからないため、wl.txt は作成されませんでした。")

    print("\nOK")
    print("処理が完了しました。")


# --- スクリプトの実行 ---
if __name__ == '__main__':
    process_skyflat2()