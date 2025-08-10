import numpy as np
from pathlib import Path  # ファイルパスを扱うためのライブラリをインポート
import pandas as pd      # CSVファイルを扱うためのpandasライブラリをインポート


def pro_gamma_convolution_custom_path():
    """
    IDLのpro gamma_convolutionをPythonに変換したものです。
    入力・出力のファイルパスを環境に合わせて指定しています。
    mcparams*.csvから視線速度Vmsを読み込むように変更。
    """
    # --- 0. ファイルパスの設定 (Path Setup) ---
    day = '20250720'  # ここ忘れずに！！！！
    # 作業の基点となるディレクトリを指定
    # Windowsのパスは、先頭に 'r' を付けるとバックスラッシュ(\)を正しく扱えます
    base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
    output_dir = base_dir / "output"
    csv_filename = base_dir / "2025ver" / f"mcparams{day}.csv"

    # 入力ファイルのフルパスを構築
    input_filename = base_dir / 'solarspectrum.txt'
    # Vmsを読み込むCSVファイルのパス
    #csv_filename = Path("mcparams20250711.csv")#下にもあるよ

    # 出力ディレクトリのパスを構築
    output_dir = output_dir / 'gamma_factor'

    # 出力ディレクトリが存在しない場合は作成する
    # exist_ok=True は、ディレクトリが既に存在していてもエラーにしないオプションです
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. 初期化 (Initialization) ---


    # --- Vmsの読み込み (Load Vms from CSV) ---
    try:
        # CSVファイルを読み込む (1行目がヘッダーとして扱われる)
        params_df = pd.read_csv(csv_filename)
        # 'mercury_sun_radial_velocity_km_s'列の最初の値(0番目の行)を取得
        Vms = params_df['mercury_sun_radial_velocity_km_s'].iloc[0]
        print(f"CSVから視線速度を読み込みました: Vms = {Vms:.6f} km/s")
        print("-" * 30)
    except FileNotFoundError:
        print(f"エラー: パラメータファイルが見つかりません。")
        print(f"指定されたパスを確認してください: {csv_filename}")
        return
    except KeyError:
        print(f"エラー: CSVファイルに 'mercury_sun_radial_velocity_km_s' の列が見つかりません。")
        return
    except Exception as e:
        print(f"パラメータファイルの読み込み中にエラーが発生しました: {e}")
        return

    im = 5757  # データ点数

    # 波長関連のパラメータ (nm)
    d1s = 589.7571833  # NaD1線付近の太陽スペクトルの最小値
    fwhm = 0.005  # ガウス関数の半値全幅 (FWHM)

    # 物理定数
    c = 299792.458  # 光速 (km/s)

    # FWHMからガウス関数の標準偏差sigmaを計算
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    print(f"sigma={sigma}")
    # --- 2. データの読み込み (Data Loading) ---
    # 指定されたフルパスからファイルを読み込む
    try:
        data = np.loadtxt(input_filename)
        wl0 = data[:, 0]  # 元の波長 (nm)
        sol = data[:, 1]  # スペクトル強度
    except FileNotFoundError:
        print(f"エラー: 入力ファイルが見つかりません。")
        print(f"指定されたパスを確認してください: {input_filename}")
        return
    except Exception as e:
        print(f"ファイルの読み込み中にエラーが発生しました: {e}")
        return

    # 読み込んだデータ数が 'im' と一致するか確認
    if len(wl0) != im:
        print(f"警告: データ点数({len(wl0)})がim({im})と一致しません。")
        im = len(wl0)  # 実際のデータ数でimを更新

    # ドップラーシフトを適用した波長を計算
    wl = wl0 * (1.0 + Vms / c)

    # --- 3. 最近傍波長の探索 (Finding the Closest Wavelength) ---
    id1 = np.argmin(np.abs(wl - d1s))
    print(f"基準波長に最も近い点を探索:")
    print(f"  - wl[id1]      : {wl[id1]:.8f} nm")
    print(f"  - index (id1)  : {id1}")
    print(f"  - wl[id1]-d1s  : {wl[id1] - d1s:.4e} nm")
    print("-" * 30)

    print(f"Python id1: {id1}")
    print(f"Python wl[id1]: {wl[id1]:.15e}")
    print(f"Python raw wl0[3680]: {wl0[3680]:.15e}")

    # --- 4. 畳み込みウィンドウの決定 (Determining the Convolution Window) ---
    c10, c20 = 100.0, 100.0
    hw1, hw2 = 0, 0
    for j in range(1001):
        if id1 + j >= im or id1 - j < 0: break
        c1 = abs((wl[id1 + j] - wl[id1]) - 5 * sigma)
        if c1 < c10:
            c10 = c1
            hw1 = j
        c2 = abs((wl[id1] - wl[id1 - j]) - 5 * sigma)
        if c2 < c20:
            c20 = c2
            hw2 = j
    hw = (hw1 + hw2) / 2.0
    print("畳み込みウィンドウを計算:")
    print(f"  - hw (平均)    : {hw:.1f}, hw1 (右側): {hw1}, hw2 (左側): {hw2}")
    print(f"  - 中心波長(元) : wl0[id1] = {wl0[id1]:.8f} nm")
    print("-" * 30)

    print(f"Python hw1: {hw1}, hw2: {hw2}")


    # --- 5. 畳み込み計算 (Convolution Calculation) ---
    start_index = id1 - hw2
    end_index = id1 + hw1 + 1
    sol2 = sol[start_index:end_index]
    phi = np.exp(-((wl - wl[id1]) / sigma) ** 2 / 2.0)
    phi_slice = phi[start_index:end_index]
    phi2 = phi_slice / np.sum(phi_slice)
    gamma0 = np.sum(sol2 * phi2)
    print(f"畳み込み計算結果:")
    print(f"phi[3649]={phi[3649]}")
    print(f"phi[3710]={phi[3710]}")
    print(f"np.sum(phi[3649:3710])={np.sum(phi[3649:3710])}")
    print(f"  - total(gamma0) = {gamma0:.8f}")
    print("-" * 30)

    # --- 6. ファイル出力 (Output) ---
    # 出力ファイル名を構築 (gamma_factorフォルダ内に作成)
    output_filename = output_dir / f'gamma_{day}.txt'
    with open(output_filename, 'w') as f:
        f.write(f'{wl0[id1]:.8f} {gamma0:.8f}\n')

    print(f"結果を '{output_filename}' に書き込みました。")
    print('end')



# --- メイン処理 ---
if __name__ == '__main__':
    pro_gamma_convolution_custom_path()
