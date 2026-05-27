import os
import datetime
import spiceypy as spice
import numpy as np

def find_true_observation_date(kernel_dir, target_taa, target_phase, target_au=None, target_side=None, tolerance_deg=1.0):
    print(f"Loading SPICE kernels from {kernel_dir}...")
    spice.furnsh(os.path.join(kernel_dir, "lsk/naif0012.tls"))
    spice.furnsh(os.path.join(kernel_dir, "spk/planets/de430.bsp"))
    spice.furnsh(os.path.join(kernel_dir, "pck/pck00011.tpc"))
    spice.furnsh(os.path.join(kernel_dir, "pck/earth_000101_260527_260228.bpc"))

    gm_sun = 132712440041.93938
    AU_KM = 149597870.7

    # 検索範囲：2012年1月1日 〜 2015年12月31日
    start_date = datetime.datetime(2020, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    end_date = datetime.datetime(2024, 12, 31, 0, 0, 0, tzinfo=datetime.timezone.utc)
    
    # 1時間刻みで総当たり検索
    step_hours = 1
    current_date = start_date
    
    side_str = target_side if target_side else "指定なし"
    print(f"\n検索開始: 目標 TAA={target_taa}°, Phase={target_phase}°, 観測面={side_str}")
    print("--------------------------------------------------")
    
    candidates = []

    while current_date <= end_date:
        time_str = current_date.strftime('%Y-%m-%dT%H:%M:%S')
        et = spice.utc2et(time_str)
        
        try:
            # --- TAAとAUの計算 ---
            state_merc_sun, _ = spice.spkezr("MERCURY", et, "J2000", "NONE", "SUN")
            elts = spice.oscltx(state_merc_sun, et, gm_sun)
            calc_taa = np.degrees(elts[8]) % 360.0
            calc_au = spice.vnorm(state_merc_sun[:3]) / AU_KM
            
            # --- 位相角(Phase Angle)の計算 ---
            pos_sun_from_merc = -state_merc_sun[:3] 
            state_merc_earth, _ = spice.spkezr("EARTH", et, "J2000", "LT+S", "MERCURY")
            pos_earth_from_merc = state_merc_earth[:3]
            calc_phase = np.degrees(spice.vsep(pos_sun_from_merc, pos_earth_from_merc))
            
            # --- DAWN/DUSKの厳密判定 ---
            # 地球直下点の経度
            subp_vec, _, _ = spice.subpnt('INTERCEPT/ELLIPSOID', 'MERCURY', et, 'IAU_MERCURY', 'LT+S', 'EARTH')
            _, obs_lon_rad, _ = spice.reclat(subp_vec)
            # 太陽直下点(正午)の経度
            subsol_vec, _, _ = spice.subslr('INTERCEPT/ELLIPSOID', 'MERCURY', et, 'IAU_MERCURY', 'LT+S', 'SUN')
            _, noon_lon_rad, _ = spice.reclat(subsol_vec)

            # 経度差を計算し、DAWNかDUSKかを判定
            delta_lon = (np.degrees(obs_lon_rad) - np.degrees(noon_lon_rad) + 540) % 360 - 180
            current_side = "DUSK" if delta_lon >= 0 else "DAWN"

            # ターゲットの観測面が指定されていて、かつ合致しない場合はスキップ
            if target_side and current_side != target_side.upper():
                current_date += datetime.timedelta(hours=step_hours)
                continue

            # --- 誤差判定 ---
            diff_taa = min(abs(calc_taa - target_taa), 360.0 - abs(calc_taa - target_taa))
            diff_phase = abs(calc_phase - target_phase)
            
            # TAAとPhase Angleが許容誤差以内に収まったら候補として保存
            if diff_taa < tolerance_deg and diff_phase < tolerance_deg:
                total_diff = diff_taa + diff_phase
                candidates.append((current_date, calc_taa, calc_phase, calc_au, current_side, total_diff))
                
        except Exception as e:
            pass
            
        current_date += datetime.timedelta(hours=step_hours)

    spice.kclear()

    # 結果の表示（誤差が小さい順にソート）
    if candidates:
        candidates.sort(key=lambda x: x[5]) # total_diffでソート
        print("★ 候補が見つかりました！最も可能性が高い日付順：\n")
        for i, cand in enumerate(candidates[:5]): # 上位5件を表示
            date_utc, c_taa, c_phase, c_au, c_side, diff = cand
            print(f"[{i+1}] {date_utc.strftime('%Y-%m-%d %H:00')} UTC")
            print(f"    観測面     : {c_side}")
            print(f"    TAA        : {c_taa:.2f}° (誤差 {c_taa-target_taa:+.2f}°)")
            print(f"    Phase Angle: {c_phase:.2f}° (誤差 {c_phase-target_phase:+.2f}°)")
            if target_au:
                print(f"    AU         : {c_au:.5f} AU (誤差 {c_au-target_au:+.5f} AU)")
            print("-" * 50)
    else:
        print("指定された許容誤差の範囲内に一致する日付は見つかりませんでした。")

# ==========================================
# 実行部分
# ==========================================
if __name__ == "__main__":
    KERNEL_DIR = "C:/Users/hanac/univ/Mercury/Haleakala2025/kernels/"
    
    # ！！ここにCSVに記録されていた謎のデータの値を入力してください！！
    TARGET_TAA = 214
    TARGET_PHASE_ANGLE = 104
    TARGET_AU = 0.44
    
    # "DAWN" または "DUSK" を指定 (Noneにすれば両方検索します)
    TARGET_SIDE = "DAWN"

    # 検索の実行
    find_true_observation_date(KERNEL_DIR, TARGET_TAA, TARGET_PHASE_ANGLE, target_au=TARGET_AU, target_side=TARGET_SIDE, tolerance_deg=2.0)