import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sys

# --- 物理定数 (元のコードから引用) ---
PHYSICAL_CONSTANTS = {
    'C': 299792458.0,  # 光速 [m/s]
    'H': 6.62607015e-34,  # プランク定数 [J・s]
    'MASS_NA': 22.98976928 * 1.66054e-27,  # Na原子の質量 [kg]
    'RM': 2439.7e3,  # 水星の半径 [m]
    'E_CHARGE': 1.602176634e-19,  # 素電荷 [C]
    'ME': 9.1093897e-31,  # 電子の質量 [kg]
    'EPSILON_0': 8.854187817e-12  # 真空の誘電率 [F/m]
}


def calculate_radiation_acceleration(Vms_ms, AU, spec_data, x_position):
    """
    指定された軌道条件と表面位置における、静止したNa原子の放射圧加速度を計算する。
    Args:
        Vms_ms (float): 水星の公転速度 [m/s]
        AU (float): 太陽からの距離 [AU]
        spec_data (dict): 太陽スペクトルと物理定数を含む辞書
        x_position (float): 太陽方向を+xとしたときの粒子のx座標 [m]
    Returns:
        float: 放射圧による加速度 [m/s^2]
    """
    # 粒子は表面で静止しているため、速度ベクトルは [0, 0, 0] とする
    # ドップラーシフトに寄与するのは水星の公転速度のみ
    velocity_for_doppler = Vms_ms

    # ドップラーシフトした波長の計算 (SI単位)
    w_na_d2 = 589.1582e-9 * (1.0 - velocity_for_doppler / PHYSICAL_CONSTANTS['C'])
    w_na_d1 = 589.7558e-9 * (1.0 - velocity_for_doppler / PHYSICAL_CONSTANTS['C'])

    wl, gamma, sigma0_perdnu1, sigma0_perdnu2, JL = spec_data.values()

    # 波長がスペクトルデータ範囲外なら加速度は0
    if not (wl[0] <= w_na_d2 * 1e9 < wl[-1] and wl[0] <= w_na_d1 * 1e9 < wl[-1]):
        return 0.0

    # 水星の影に入っている場合は加速度は0 (太陽方向が-xなので、x > 0が昼側)
    if x_position < 0:
        return 0.0

    # スペクトルデータからガンマ値を内挿
    gamma2 = np.interp(w_na_d2 * 1e9, wl, gamma)
    gamma1 = np.interp(w_na_d1 * 1e9, wl, gamma)

    # --- 散乱率と加速度の計算 (元のコードのロジックをそのまま使用) ---
    F_lambda_1AU_m = JL * 1e9
    # 距離とスペクトル形状を反映した波長あたりのフラックス
    F_lambda_d1_at_Mercury = F_lambda_1AU_m / (AU ** 2) * gamma1
    F_lambda_d2_at_Mercury = F_lambda_1AU_m / (AU ** 2) * gamma2

    # 周波数あたりのフラックスに変換
    F_nu_d1 = F_lambda_d1_at_Mercury * (w_na_d1 ** 2) / PHYSICAL_CONSTANTS['C']
    F_nu_d2 = F_lambda_d2_at_Mercury * (w_na_d2 ** 2) / PHYSICAL_CONSTANTS['C']

    # 散乱率 J [photons/s]
    J1 = sigma0_perdnu1 * F_nu_d1
    J2 = sigma0_perdnu2 * F_nu_d2

    # 加速度 b [m/s^2]
    b = (1 / PHYSICAL_CONSTANTS['MASS_NA']) * (
            (PHYSICAL_CONSTANTS['H'] / w_na_d1) * J1 + (PHYSICAL_CONSTANTS['H'] / w_na_d2) * J2)

    return b


def main():
    """
    メイン実行関数
    """
    print("シミュレーションを開始します...")

    # --- 1. 外部データの読み込み ---
    try:
        spec_data_np = np.loadtxt('SolarSpectrum_Na0.txt', usecols=(0, 3))
        orbit_lines = open('orbit360.txt', 'r').readlines()
    except FileNotFoundError as e:
        print(f"エラー: データファイルが見つかりません - {e}", file=sys.stderr)
        sys.exit(1)

    # スペクトルデータを波長でソート
    wl, gamma = spec_data_np[:, 0], spec_data_np[:, 1]
    if not np.all(np.diff(wl) > 0):
        sort_indices = np.argsort(wl)
        wl, gamma = wl[sort_indices], gamma[sort_indices]

    # --- 2. 物理定数の準備 ---
    sigma_const = PHYSICAL_CONSTANTS['E_CHARGE'] ** 2 / (
            4 * PHYSICAL_CONSTANTS['ME'] * PHYSICAL_CONSTANTS['C'] * PHYSICAL_CONSTANTS['EPSILON_0'])
    spec_data_dict = {
        'wl': wl, 'gamma': gamma,
        'sigma0_perdnu1': sigma_const * 0.320,
        'sigma0_perdnu2': sigma_const * 0.641,
        'JL': 5.18e14 * 1e4  # ph/s/m^2/nm @ 1AU
    }

    # --- 3. TAAごとのループ計算 ---
    taa_list = []
    accel_subsolar = []  # 太陽直下点 (SZA=0)
    accel_60deg = []  # 太陽天頂角60度
    accel_terminator = []  # 昼夜境界線 (SZA=90)

    print("各TAAでの放射圧を計算中...")
    for line in orbit_lines:
        try:
            TAA, AU, _, _, Vms_ms = map(float, line.split())
        except ValueError:
            continue  # 空行などをスキップ

        # 座標系：太陽方向を+x軸とする
        # 1. 太陽直下点 (x = R, y = 0)
        pos_x_subsolar = PHYSICAL_CONSTANTS['RM']
        b_subsolar = calculate_radiation_acceleration(Vms_ms, AU, spec_data_dict, pos_x_subsolar)

        # 2. 太陽天頂角60度 (x = R * cos(60))
        pos_x_60deg = PHYSICAL_CONSTANTS['RM'] * np.cos(np.deg2rad(60))
        b_60deg = calculate_radiation_acceleration(Vms_ms, AU, spec_data_dict, pos_x_60deg)

        # 3. 昼夜境界線 (x = R * cos(90) = 0)
        pos_x_terminator = 0.0
        b_terminator = calculate_radiation_acceleration(Vms_ms, AU, spec_data_dict, pos_x_terminator)

        # 結果をリストに格納
        taa_list.append(TAA)
        accel_subsolar.append(b_subsolar)
        accel_60deg.append(b_60deg)
        accel_terminator.append(b_terminator)

    print("計算完了。結果をプロットします。")

    # --- 4. 結果のプロット ---
    try:
        # 日本語フォントの設定（ご自身の環境に合わせてパスを変更してください）
        font_path = 'C:/Windows/Fonts/meiryo.ttc'
        jp_font = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = jp_font.get_name()
    except FileNotFoundError:
        print("警告: 日本語フォントが見つかりません。グラフのラベルが文字化けする可能性があります。")

    plt.figure(figsize=(12, 7))
    plt.plot(taa_list, accel_subsolar, label='太陽直下点 (SZA 0°)', lw=2.5)
    plt.plot(taa_list, accel_60deg, label='SZA 60°', lw=2.5, linestyle='--')
    plt.plot(taa_list, accel_terminator, label='昼夜境界線 (SZA 90°)', lw=2.5, linestyle=':')

    plt.title('水星軌道上の各地点におけるナトリウム原子への初期放射圧加速度', fontsize=16)
    plt.xlabel('真近点角 (TAA) [度]', fontsize=12)
    plt.ylabel('放射圧加速度 [m/s$^2$]', fontsize=12)
    plt.xticks(np.arange(0, 361, 30))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()