import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice


# ==============================================================================
# 1. SPICEカーネルを利用した物理計算関数
# ==============================================================================
def get_mercury_data_from_spice(start_date_str, duration_days=88, step_hours=6):
    """
    SPICEカーネル(de442.bsp)を使用して、指定期間の水星の
    「太陽距離(AU)」と「真近点角(TAA)」を計算して返す。
    """
    # カーネルのロード (パスは適宜変更してください)
    # naif0012.tls は時間を扱うために必須です
    try:
        spice.furnsh('de442.bsp')
        spice.furnsh('naif0012.tls')
    except Exception as e:
        print("Error: カーネルのロードに失敗しました。ファイルパスを確認してください。")
        print("必要なファイル: de442.bsp, naif0012.tls")
        raise e

    # 観測対象と観測者
    target = 'MERCURY'
    observer = 'SUN'
    frame = 'ECLIPJ2000'  # 黄道座標系
    abcorr = 'NONE'  # 光行差補正なし（幾何学的重力中心）

    # 太陽の重力定数 (GM) [km^3/s^2] (標準的な値)
    # PCKファイルがあれば spice.bodvrd('SUN', 'GM', 1) で取得可能だが今回は直打ち
    GM_SUN = 132712440018.0

    # 時間ループの作成
    et_start = spice.str2et(start_date_str)
    et_end = et_start + duration_days * 24 * 3600
    times = np.arange(et_start, et_end, step_hours * 3600)

    taa_list = []
    dist_list = []

    for et in times:
        # 1. 状態ベクトル（位置・速度）を取得 [km, km/s]
        state, _ = spice.spkezr(target, et, frame, abcorr, observer)
        pos = state[0:3]  # 位置ベクトル r
        vel = state[3:6]  # 速度ベクトル v

        # 2. 距離の計算
        r_km = np.linalg.norm(pos)
        r_au = spice.convrt(r_km, 'KM', 'AU')

        # 3. 真近点角 (TAA) の計算 (状態ベクトルから算出)
        # 離心率ベクトル e = ((v^2 - mu/r)*r - (r*v)*v) / mu
        v_sq = np.dot(vel, vel)
        r_dot_v = np.dot(pos, vel)

        vec_e = ((v_sq - GM_SUN / r_km) * pos - (r_dot_v) * vel) / GM_SUN
        eccentricity = np.linalg.norm(vec_e)

        # TAA (nu) の計算: cos(nu) = (e・r) / (|e||r|)
        cos_nu = np.dot(vec_e, pos) / (eccentricity * r_km)
        # 数値誤差で範囲外に行かないようクリップ
        cos_nu = np.clip(cos_nu, -1.0, 1.0)
        nu_rad = np.arccos(cos_nu)

        # 0〜360度の判定 (r・vが正なら遠ざかっているので0-180、負なら近づくので180-360)
        if r_dot_v < 0:
            nu_rad = 2 * np.pi - nu_rad

        taa_deg = np.degrees(nu_rad)

        taa_list.append(taa_deg)
        dist_list.append(r_au)

    # データのクリア
    spice.kclear()

    return np.array(taa_list), np.array(dist_list)


# ==============================================================================
# 2. 物理モデル (Leblanc Model) - 修正版
# ==============================================================================
def calculate_surface_temperature_leblanc_corrected(AU, ref_au_actual):
    """
    修正されたLeblancモデル
    """
    T0 = 600.0
    T1 = 100.0

    # 実際のデータに基づく近日点距離を基準にする
    scaling = np.sqrt(ref_au_actual / AU)

    # 全体をスケーリング
    return T0  * scaling + T1


# ==============================================================================
# 3. メイン処理
# ==============================================================================
def main():
    # 2025年の水星軌道を計算 (88日間)
    # ※ naif0012.tls が必要です
    try:
        taa_spice, r_spice = get_mercury_data_from_spice('2025-01-01', duration_days=88)
    except Exception:
        return

    # データのソート (プロット時に線がぐちゃぐちゃにならないようにTAA順に並べる)
    sort_idx = np.argsort(taa_spice)
    taa_sorted = taa_spice[sort_idx]
    r_sorted = r_spice[sort_idx]

    # この期間の「実際の」近日点距離を取得 (最小距離)
    min_dist_actual = np.min(r_sorted)
    print(f"Simulation Perihelion Distance: {min_dist_actual:.5f} AU")

    # 温度計算 (基準距離には今回のシミュレーションの最小値を使用することで、最大値を確実に700Kにする)
    temps = calculate_surface_temperature_leblanc_corrected(r_sorted, min_dist_actual)

    # --- グラフ描画 ---
    plt.figure(figsize=(10, 6))

    # データプロット
    plt.plot(taa_sorted, temps, label='2025 Orbit (SPICE/de442)', color='crimson', linewidth=2)

    # 近日点 (最大温度)
    max_temp = np.max(temps)
    plt.text(0, max_temp + 5, f'Perihelion\n{max_temp:.2f} K', ha='center', color='crimson', fontweight='bold')

    # 遠日点 (TAA=180付近のデータを探す)
    # SPICEデータは離散的なので、180度に最も近い点を探す
    apo_idx = (np.abs(taa_sorted - 180)).argmin()
    apo_temp = temps[apo_idx]
    apo_taa = taa_sorted[apo_idx]
    plt.scatter(apo_taa, apo_temp, color='blue', zorder=5)
    plt.text(180, apo_temp - 25, f'Aphelion\n{apo_temp:.1f} K', ha='center', color='blue', fontweight='bold')

    plt.title("Mercury Surface Temperature (2025 Orbit via SPICE)", fontsize=14)
    plt.xlabel("True Anomaly Angle (TAA) [deg]", fontsize=12)
    plt.ylabel("Temperature [K]", fontsize=12)
    plt.xlim(0, 360)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(np.arange(0, 361, 30))
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()