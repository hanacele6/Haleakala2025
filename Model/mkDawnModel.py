import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 日本語フォントの設定 (お使いの環境に合わせてフォントパスを指定してください)
# 例: Windows 'C:/Windows/Fonts/meiryo.ttc', macOS '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'
# ColabやLinuxの場合は !apt-get -y install fonts-ipafont-gothic などのインストールが必要な場合があります。
try:
    font_path = 'C:/Windows/Fonts/meiryo.ttc'  # Windowsの例
    jp_font = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = jp_font.get_name()
except FileNotFoundError:
    print("日本語フォントが見つかりません。グラフのラベルが文字化けする可能性があります。")

# --- 物理定数とパラメータ ---
# これらの値を1か所にまとめることで、管理しやすくなります。
G = 6.67384e-5  # 万有引力定数 [cm^3/kg/s^2]
MS = 1.9884e30  # 太陽質量 [kg]
MM = 3.3e23  # 水星質量 [kg]
AU = 1.495978707e13  # 天文単位 [cm/AU]
RM = 2440e5  # 水星半径 [cm]

# 軌道要素
E = 0.2056 # 離心率
A = 0.3871  # 軌道長半径 [AU]
#L = A * (1.0 - E ** 2)  #半直弦 [AU]
L = 0.37078

# モデルパラメータ
TAU0 = 169200.0  # 光脱離の基準タイムスケール [s]
PHI0 = 4.6e7  # 基準光子フラックス [atoms/cm^2/s]
D_PARAM = 4.6e7  # 蓄積モデルのフリーパラメータ なお、リサイクリングレートα、C0、およびwに不確定性を持つため、本研究ではフリーパラメータとしている。
ROT_ANGLE_ACCUM = 90  # 蓄積計算を行う回転角 [deg]


# --- 物理モデルの関数 ---
# IDLの関数をPythonで再定義。分かりやすいように関数名を英語に変更。

def get_sun_distance(taa_deg):
    """太陽からの距離(R)を計算する [AU]"""
    return L / (1.0 + E * np.cos(np.deg2rad(taa_deg)))


def get_orbital_angular_velocity(r_au):
    """公転角速度を計算する [deg/s]"""
    # sqrt((G * Ms * l) / R^2) を AU単位系で計算
    return np.sqrt((G / AU ** 3) * MS * L) / r_au ** 2 * np.rad2deg(1)


def get_relative_angular_velocity(taa_deg):
    """太陽に対する相対的な自転角速度(ω)を計算する [deg/s]"""
    # 惑星の公転周期 T_orbit [s]
    t_orbit = 2.0 * np.pi * np.sqrt(A ** 3 / ((G / AU ** 3) * (MS + MM)))
    # 惑星の自転周期 T_rot [s] (3:2の軌道共鳴)
    t_rot = t_orbit * (2.0 / 3.0)
    # 自転角速度 rot_dot [deg/s]
    rot_dot = 360.0 / t_rot

    r = get_sun_distance(taa_deg)
    taa_dot = get_orbital_angular_velocity(r)

    return rot_dot - taa_dot


def calculate_sun_rotation(taa_range):
    """各TAAまでの太陽の総自転角(rot)を計算する [deg]"""
    rot = 0.0
    step = 10.0  # 積分ステップ
    rot_history = []

    for taa in taa_range:
        # TAAが1度進む間に、小さなステップで積分
        for i in range(int(step)):
            taa_current = taa + i / step

            r = get_sun_distance(taa_current)
            taa_dot = get_orbital_angular_velocity(r)
            omega = get_relative_angular_velocity(taa_current)

            # d(rot) = (d(rot)/d(TAA)) * d(TAA)
            # d(rot)/d(TAA) = (d(rot)/dt) / (d(TAA)/dt) = omega / taa_dot
            # 1ステップあたりの回転角増分
            drot = (omega / taa_dot) / step
            rot += drot
        rot_history.append(rot)

    return np.array(rot_history)


# --- メイン処理 ---
if __name__ == "__main__":
    # 1. 入力ファイルの読み込み
    # TAA_SRP.txt は、1列目にTAA(0-359)、2列目にSRP(太陽放射圧)が
    # 書かれていると仮定します。
    try:
        data = np.loadtxt('TAA_SRP.txt')
        taa_obs = data[:, 0]
        srp_obs = data[:, 1]
    except FileNotFoundError:
        print("エラー: 'TAA_SRP.txt' が見つかりません。")
        print("ダミーデータを生成して処理を続行します。")
        # ダミーデータを生成
        taa_obs = np.arange(360)
        srp_obs = 1.0 + 0.5 * np.sin(np.deg2rad(taa_obs) * 2)  # 適当な変動

    # --- Part 1: 蓄積量の計算 ---
    print("Part 1: 蓄積量の計算を開始...")

    taa_range = np.arange(360)

    # 各TAAにおける太陽の総自転角(rot)と相対角速度(omega)を事前に計算
    total_rotation = calculate_sun_rotation(taa_range)
    omega_values = get_relative_angular_velocity(taa_range)

    # 蓄積量 (s_accumulated) を初期化
    s_accumulated = np.zeros(360)

    for taa in taa_range:
        # 現在の太陽の総自転角
        current_rot = total_rotation[taa]

        # 過去90度分の回転についてループして蓄積量を計算
        for i in range(ROT_ANGLE_ACCUM):
            # 過去の自転角 (rot_accu)
            rot_to_check = current_rot - ROT_ANGLE_ACCUM + i

            # 角度が負になった場合、180度加算 (元のコードのロジックを再現)
            if rot_to_check < 0:
                rot_to_check += 180.0

            # 過去の自転角(rot_to_check)に対応するSRPとomegaを線形補間で求める
            # NumPyの interp を使うと、IDLの自作関数より高速で簡潔
            srp_at_rot = np.interp(rot_to_check, total_rotation, srp_obs)
            omega_at_rot = np.interp(rot_to_check, total_rotation, omega_values)

            # 1ステップあたりの蓄積量を計算し、加算
            # omegaが0に近いと発散するため、微小値を加えて安定させる
            s_rate = np.sqrt(srp_at_rot) / (np.abs(omega_at_rot) + 1e-9)
            s_accumulated[taa] += s_rate

    print("Part 1: 完了。")
    # これで、中間ファイルなしに `m` または `st` のデータが s_accumulated に得られた

    # --- Part 2: 明け方モデルの計算 (dawn_model1 の処理) ---
    print("Part 2: 明け方モデルの計算を開始...")

    dawn_model = np.zeros(360)
    n_dusk_term_history = np.zeros(360)  #
    n_add_term_history = np.zeros(360)  #

    for taa in taa_range:
        r_au = get_sun_distance(taa)

        # 距離に依存するパラメータの計算
        #ph = PHI0 * (A * (1 - E) / r_au) ** 2  # 論文でよく使われる 1/R^2 の形
        ph = PHI0 * (0.306 / r_au) ** 2
        #tau_val = TAU0 * (r_au / A) ** 2  # R^2 に比例
        tau_val = TAU0 * r_au ** 2

        # 移動時間 tm の計算
        tm = np.sqrt(2.0 * RM / srp_obs[taa])

        # --- 数式(19)の各項を計算 ---
        # 第1項 (N_dusk)
        n_dusk_term = ph * tau_val * (1.0 - np.exp(-tm / tau_val))

        # 第2項 (N_add)
        n_add_term = D_PARAM * s_accumulated[taa] * omega_values[taa]

        dawn_model[taa] = n_dusk_term + n_add_term

        # 可視化のために各項を保存
        n_dusk_term_history[taa] = n_dusk_term
        n_add_term_history[taa] = n_add_term

    print("Part 2: 完了。")

    # --- Part 3: 結果の可視化 ---
    print("Part 3: 結果をプロット中...")

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(taa_range, dawn_model, 'r-', lw=2.5, label='最終モデル ($N_{dawn}$)')
    ax.plot(taa_range, n_dusk_term_history, 'b--', lw=1.5, label='Duskからの供給項 ($N_{dusk}$)')
    ax.plot(taa_range, n_add_term_history, 'g:', lw=1.5, label='蓄積からの放出項 ($N_{add}$)')

    ax.set_title('水星ナトリウム大気 Dawnの計算結果', fontsize=16)
    ax.set_xlabel('真近点角 (TAA) [度]', fontsize=12)
    ax.set_ylabel('ナトリウム原子柱密度 [atoms/cm$^2$]', fontsize=12)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.show()

    # 結果をファイルに保存
    output_data = np.vstack((taa_range, dawn_model, n_dusk_term_history, n_add_term_history)).T
    np.savetxt(
        'dawn_model_output.csv',
        output_data,
        delimiter=',',
        header='TAA,Total_N_dawn,N_dusk_term,N_add_term',
        fmt='%.4f'
    )
    print("計算結果を 'dawn_model_python_output.csv' に保存しました。")