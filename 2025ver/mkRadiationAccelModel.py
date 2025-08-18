import numpy as np
import matplotlib.pyplot as plt


def calculate_radiation_acceleration():
    """
    水星のナトリウム原子に対する太陽放射圧による加速度を、
    真近点離角(TAA)の関数として計算し、テキストファイルとグラフで出力する関数。
    論文の図21を再現することを目的とします。
    """

    # --- 物理定数と天文定数 ---
    # 物理定数
    h = 6.62607015e-34  # プランク定数 (J·s)
    c = 299792458  # 光速 (m/s)
    e = 1.60217663e-19  # 素電荷 (C)
    m_e = 9.1093837e-31  # 電子の質量 (kg)
    m_Na = 22.989770 * 1.660539e-27  # ナトリウム原子の質量 (kg)
    AU = 1.496e11  # 天文単位 (m)

    # ナトリウムD線のデータ
    # D2線
    lambda_D2 = 588.995e-9  # D2線の波長 (m)
    f_D2 = 0.641  # D2線の振動子強度
    # D1線
    lambda_D1 = 589.592e-9  # D1線の波長 (m)
    f_D1 = 0.320  # D1線の振動子強度

    lines = [
        {'lambda': lambda_D2, 'f': f_D2, 'name': 'D2'},
        {'lambda': lambda_D1, 'f': f_D1, 'name': 'D1'},
    ]

    # --- 太陽と水星の軌道パラメータ ---
    # 太陽
    G = 6.67430e-11  # 万有引力定数 (m^3 kg^-1 s^-2)
    M_sun = 1.989e30  # 太陽質量 (kg)
    mu = G * M_sun  # 太陽の重力定数 (m^3/s^2)

    # 水星
    a = 0.3871 * AU  # 軌道長半径 (m)
    ecc = 0.2056  # 離心率

    # --- 太陽スペクトルモデル (フラウンホーファー線) ---
    # 地球軌道(1AU)における連続光のスペクトル流束密度を仮定
    # 単位: W / m^2 / m
    #TODO ここの値が分からないので計算できない
    F_lambda_continuum_1AU = 1.712e13

    # ガウス関数で吸収線をモデル化
    # F(λ) = F_cont * (1 - depth * exp(-((λ - λ_center)^2) / (2 * sigma^2)))
    solar_spectrum_params = {
        'D2': {'center': 588.995e-9, 'depth': 0.9, 'sigma': 0.01e-9},
        'D1': {'center': 589.592e-9, 'depth': 0.85, 'sigma': 0.01e-9}
    }

    def get_solar_flux(wavelength):
        """
        与えられた波長における太陽スペクトル流束密度を返す。
        フラウンホーファー線をガウス関数で近似します。
        """
        flux = F_lambda_continuum_1AU
        for line_name, params in solar_spectrum_params.items():
            flux *= (1 - params['depth'] * np.exp(
                -((wavelength - params['center']) ** 2) / (2 * params['sigma'] ** 2)
            ))
        return flux

    # --- 計算の実行 ---
    TAA_degrees = np.arange(0, 361, 1)
    TAA_radians = np.deg2rad(TAA_degrees)
    accelerations = []

    for theta in TAA_radians:
        # 1. 太陽-水星間の距離 R を計算
        R = a * (1 - ecc ** 2) / (1 + ecc * np.cos(theta))

        # 2. 水星の太陽に対する視線速度 v_r を計算
        # 特定の角運動量 h = sqrt(μ * a * (1 - e^2))
        h_momentum = np.sqrt(mu * a * (1 - ecc ** 2))
        v_r = (mu / h_momentum) * ecc * np.sin(theta)

        total_scattering_rate = 0

        # 3. D1線、D2線それぞれについて光子散乱率を計算
        for line in lines:
            # ドップラーシフトした波長を計算
            lambda_shifted = line['lambda'] * (1 + v_r / c)

            # 水星の位置での太陽スペクトル流束密度を計算
            F_lambda_at_mercury = get_solar_flux(lambda_shifted) * (AU / R) ** 2

            # 光子散乱率 g (g-factor) を計算
            g_factor = (np.pi * e ** 2 / (m_e * c)) * line['f'] * (F_lambda_at_mercury * lambda_shifted / (h * c))
            total_scattering_rate += g_factor

        # 4. 全散乱率から加速度を計算
        acceleration = total_scattering_rate * (h / c) / m_Na
        accelerations.append(acceleration)

    # 結果をcm/s^2に変換
    accelerations_cms2 = np.array(accelerations) * 100

    # --- 結果をテキストファイルに保存 ---
    file_name = 'radiation_pressure_results.txt'
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write("--- TAA別 太陽放射圧による加速度 ---\n")
        f.write(f"{'TAA (度)':<12} | {'加速度 (cm/s²)'}\n")
        f.write("-" * 32 + "\n")
        # 10度おきに結果を書き込み
        for i in range(0, len(TAA_degrees), 10):
            f.write(f"{TAA_degrees[i]:<12} | {accelerations_cms2[i]:.2f}\n")
        f.write("-" * 32 + "\n")

    print(f"計算結果を '{file_name}' に保存しました。")

    # --- グラフの描画 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(TAA_degrees, accelerations_cms2, lw=2, color='royalblue')

    # グラフの装飾
    ax.set_title('Solar Radiation Acceleration on Mercury\'s Sodium Atoms', fontsize=16)
    ax.set_xlabel('True Anomaly Angle (TAA) [degree]', fontsize=12)
    ax.set_ylabel('Acceleration [cm/s²]', fontsize=12)
    ax.set_xlim(0, 360)
    ax.set_ylim(0, max(accelerations_cms2) * 1.1)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # TAAの主要な点にラベルを追加
    special_taa = {
        0: 'Perihelion',
        90: 'v_r max',
        180: 'Aphelion',
        270: 'v_r min'
    }
    for angle, label in special_taa.items():
        ax.axvline(x=angle, color='gray', linestyle=':', linewidth=1)

    plt.tight_layout()
    plt.show()


# 関数を実行
calculate_radiation_acceleration()
