import os
import re
import glob
import pandas as pd
import spiceypy as spice

def check_idl_au_updates(pro_dir, kernel_dir):
    """
    指定フォルダ内のIDL(.pro)ファイルをスキャンし、
    記載されている日付とAUが正しいか(コピペ忘れがないか)をSPICEで検証する
    """
    print(f"Loading SPICE kernels from {kernel_dir}...")
    spice.furnsh(os.path.join(kernel_dir, "lsk/naif0012.tls"))
    spice.furnsh(os.path.join(kernel_dir, "spk/planets/de430.bsp"))
    
    AU_KM = 149597870.7 # 1天文単位(km)
    
    # .proファイルのパスを取得
    pro_files = glob.glob(os.path.join(pro_dir, "*.pro"))
    if not pro_files:
        print(f"エラー: {pro_dir} に .pro ファイルが見つかりません。")
        spice.kclear()
        return

    results = []
    
    for file_path in pro_files:
        filename = os.path.basename(file_path)
        day_str = None
        idl_au = None
        
        # IDLファイルの読み込み (文字化け回避のため errors='ignore')
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # day='20150517' のような記述を探す
            day_match = re.search(r"day\s*=\s*'(\d{8})'", content, re.IGNORECASE)
            if day_match:
                day_str = day_match.group(1) # '20150517'
                
            # AU=0.42945883d0 のような記述を探す (d0はIDLの倍精度表現)
            au_match = re.search(r"AU\s*=\s*([0-9\.]+)d0", content, re.IGNORECASE)
            if au_match:
                idl_au = float(au_match.group(1)) # 0.42945883

        # 日付とAUの両方が見つかった場合のみ計算
        if day_str and idl_au is not None:
            # YYYYMMDD を YYYY-MM-DD に変換 (時刻は便宜上 00:00:00 とする)
            utc_date = f"{day_str[:4]}-{day_str[4:6]}-{day_str[6:8]}T00:00:00"
            
            try:
                et = spice.utc2et(utc_date)
                # 水星と太陽の距離(km)を取得
                state, _ = spice.spkezr("MERCURY", et, "J2000", "NONE", "SUN")
                true_dist_km = spice.vnorm(state[:3])
                true_au = true_dist_km / AU_KM
                
                # IDLのAUと本物のAUの差分を計算
                diff = abs(idl_au - true_au)
                
                # 誤差が 0.01 AU (約150万km) 以上あれば、明らかに別の日付の値を使い回していると判定
                status = "OK" if diff < 0.01 else "NG (コピペ忘れの可能性大)"
                
                results.append({
                    "File": filename,
                    "Date": utc_date[:10],
                    "IDL_AU": idl_au,
                    "True_AU": true_au,
                    "Diff_AU": diff,
                    "Judgement": status
                })
            except Exception as e:
                print(f"  > {filename} ({utc_date}) の計算エラー: {e}")

    spice.kclear()

    # 結果をデータフレーム化して表示
    df = pd.DataFrame(results)
    if not df.empty:
        # コンソールにわかりやすく表示
        print("\n=== IDLコード AU検証結果 ===")
        print(df.to_string(index=False))
        
        # CSVにも保存しておく
        output_csv = os.path.join(pro_dir, "idl_au_check_result.csv")
        df.to_csv(output_csv, index=False, encoding='cp932')
        print(f"\n結果を保存しました: {output_csv}")
    else:
        print("検証できるデータが見つかりませんでした。")

# ==========================================
# 実行部分
# ==========================================
if __name__ == "__main__":
    KERNEL_DIR = "C:/Users/hanac/univ/Mercury/Haleakala2025/kernels/"
    
    # 過去のIDLコード (.proファイル) がたくさん入っているフォルダのパスを指定してください
    PRO_DIR = "C:/Users/hanac/univ/Mercury/Haleakala2025/old_idl_scripts/" 
    
    check_idl_au_updates(PRO_DIR, KERNEL_DIR)