import os
import spiceypy as spice
import pandas as pd
import numpy as np

def verify_mercury_taa(csv_path, kernel_dir, output_csv_path, tolerance=2.0):
    """
    CSVファイルから日付とTAAを読み込み、SPICEで再計算して妥当性を判定する
    
    :param csv_path: 読み込むCSVファイルのパス
    :param kernel_dir: カーネルが保存されているディレクトリパス
    :param output_csv_path: 判定結果を書き出すCSVファイルのパス
    :param tolerance: 許容するTAAの誤差（度）。デフォルトは2.0度。
    """
    
    # 1. カーネルのロード
    print(f"Loading SPICE kernels from {kernel_dir}...")
    spice.furnsh(os.path.join(kernel_dir, "lsk/naif0012.tls"))
    spice.furnsh(os.path.join(kernel_dir, "pck/pck00011.tpc"))
    spice.furnsh(os.path.join(kernel_dir, "spk/planets/de430.bsp"))
    spice.furnsh(os.path.join(kernel_dir, "pck/earth_000101_260527_260228.bpc"))

    # 太陽の重力定数(GM) [km^3/s^2]
    # gm_de430.tpc等のカーネル不足によるエラーを回避するため、DE430の定数を直接指定
    gm_sun = 132712440041.93938

    # 2. CSVの読み込み
    print(f"Reading {csv_path}...")
    # Shift-JIS(cp932)を指定して、文字コード由来の読み込みエラーを回避
    try:
        df = pd.read_csv(csv_path, header=None, encoding='cp932')
    except Exception as e:
        print(f"CSV読み込みエラー: {e}")
        spice.kclear()
        return
    
    # 判定結果を格納するリスト
    calculated_taas = []
    differences = []
    judgements = []
    
    # 3. 各行ごとのTAA計算と判定
    # 2列目(インデックス1)が日付、3列目(インデックス2)がTAAと仮定
    for index, row in df.iterrows():
        try:
            # 日付文字列を取得 (例: "2024/05/10" または "2024-05-10T12:00:00")
            date_str = str(row[1]).strip()
            
            # UTC文字列からエフェメリスタイム(ET)へ変換
            et = spice.utc2et(date_str)
            
            # 太陽に対する水星の状態ベクトルを取得
            # 座標系: J2000, 光行差補正: NONE (幾何学的位置)
            state, lt = spice.spkezr("MERCURY", et, "J2000", "NONE", "SUN")
            
            # 状態ベクトルから軌道要素(osculating elements)を計算
            elts = spice.oscltx(state, et, gm_sun)
            
            # elts[8] が True Anomaly（真近点角）
            # ※elts[5]はMean Anomaly（平均近点角）なので使用しない
            calc_taa_rad = elts[8]
            calc_taa_deg = np.degrees(calc_taa_rad) % 360.0
            
            # CSV記載のTAAを取得
            csv_taa = float(row[2])
            
            # 差分を計算（359度と1度のような、360度の境界をまたぐ場合の処理を含む）
            diff = abs(calc_taa_deg - csv_taa)
            diff = min(diff, 360.0 - diff)
            
            # 判定
            judgement = "OK" if diff <= tolerance else "NG"
            
            calculated_taas.append(calc_taa_deg)
            differences.append(diff)
            judgements.append(judgement)
                
        except Exception as e:
            print(f"Row {index} error: {e}")
            calculated_taas.append(np.nan)
            differences.append(np.nan)
            judgements.append("ERROR")

    # 4. 結果をデータフレームに追加して保存
    df['Calculated_TAA'] = calculated_taas
    df['Difference'] = differences
    df['Judgement'] = judgements
    
    df.to_csv(output_csv_path, index=False, header=False, encoding='cp932')
    print(f"Check complete. Results saved to {output_csv_path}\n")
    
    # カーネルのアンロード (メモリ解放)
    spice.kclear()

# ==========================================
# 実行部分
# ==========================================
if __name__ == "__main__":
    # カーネルディレクトリのパス
    KERNEL_DIR = "C:/Users/hanac/univ/Mercury/Haleakala2025/kernels/"
    
    # 判定したいCSVファイルのパス (適宜書き換えてください)
    INPUT_CSV_DUSK = "C:/Users/hanac/univ/Mercury/DUSK.csv"
    INPUT_CSV_DAWN = "C:/Users/hanac/univ/Mercury/DAWN.csv"
    
    # 結果を出力するCSVファイルのパス
    OUTPUT_CSV_DUSK = "C:/Users/hanac/univ/Mercury//dusk_checked.csv"
    OUTPUT_CSV_DAWN = "C:/Users/hanac/univ/Mercury/dawn_checked.csv"
    
    # 許容誤差（度）。時刻が含まれず00:00:00で計算される場合のズレを考慮して5.0度に設定
    TOLERANCE = 5.0
    
    # DUSKデータの判定実行 (ファイルが存在する場合のみ実行)
    if os.path.exists(INPUT_CSV_DUSK):
        verify_mercury_taa(INPUT_CSV_DUSK, KERNEL_DIR, OUTPUT_CSV_DUSK, tolerance=TOLERANCE)
    else:
        print(f"File not found: {INPUT_CSV_DUSK}")
        
    # DAWNデータの判定実行 (ファイルが存在する場合のみ実行)
    if os.path.exists(INPUT_CSV_DAWN):
        verify_mercury_taa(INPUT_CSV_DAWN, KERNEL_DIR, OUTPUT_CSV_DAWN, tolerance=TOLERANCE)
    else:
        print(f"File not found: {INPUT_CSV_DAWN}")