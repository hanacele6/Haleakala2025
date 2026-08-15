import spiceypy as spice

def calculate_mercury_phase_angle(date_str, bsp_kernel, lsk_kernel):
    """
    指定した日時の水星の位相角を計算する
    
    Parameters:
        date_str (str): 計算したい日時 (例: '2026-05-01T12:00:00')
        bsp_kernel (str): 惑星の軌道情報カーネルのパス (例: 'de442.bsp')
        lsk_kernel (str): うるう秒カーネルのパス (例: 'naif0012.tls')
        
    Returns:
        float: 位相角（度）
    """
    # カーネルの読み込み
    spice.furnsh(bsp_kernel)
    spice.furnsh(lsk_kernel)
    
    try:
        # 指定された日時文字列を暦表時(ET: Ephemeris Time)に変換
        et = spice.str2et(date_str)
        
        # 位相角の計算
        # target: MERCURY (対象)
        # illmn: SUN (光源)
        # obsr: EARTH (観測者)
        # abcorr: 'LT+S' (光行差と光の到達時間の補正を含める。観測データとの比較に最適)
        phase_rad = spice.phaseq(et, 'MERCURY', 'SUN', 'EARTH', 'LT+S')
        
        # ラジアンから度へ変換
        phase_deg = phase_rad * spice.dpr()
        
        return phase_deg
        
    finally:
        # メモリ解放・カーネルのアンロード（安全のため必ず実行）
        spice.kclear()

# --- 実行例 ---
if __name__ == "__main__":
    # カーネルのパス（環境に合わせて書き換えてください）
    BSP_FILE = "de442.bsp"
    LSK_FILE = "naif0012.tls"
    
    # 調べたい日付 (UTC)
    target_date = "2016-06-13T00:00:00"
    
    try:
        angle = calculate_mercury_phase_angle(target_date, BSP_FILE, LSK_FILE)
        print(f"日時 (UTC): {target_date}")
        print(f"水星の位相角: {angle:.2f} 度")
    except Exception as e:
        print(f"エラーが発生しました: {e}")