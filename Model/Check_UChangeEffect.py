import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 簡易予測用シミュレーション設定
# ==============================================================================
# 比較したいUの値のリスト
U_LIST_EV = [1.85, 2.05]

# 物理定数 (ユーザーコードより抜粋・簡略化)
KB = 1.380649e-23
EV_TO_J = 1.602e-19
VIB_FREQ = 1e13
AU_DIST = 0.387  # 水星の平均距離 (AU)
ROTATION_PERIOD = 58.6462 * 86400

# 温度モデル設定
TEMP_BASE = 100.0
TEMP_AMP = 600.0
TEMP_NIGHT = 100.0

# 拡散供給などのベース供給率 (適当な値を仮定: 夜間に回復する量)
# 実際は拡散モデルですが、ここでは「枯渇のしやすさ」を見るため一定供給と仮定
BASE_SUPPLY_FLUX = 1.0e8 * (100 ** 2)


# ==============================================================================
# 関数定義 (ユーザーコードのロジックを流用)
# ==============================================================================
def calculate_temp(cos_theta, au):
    scaling = np.sqrt(0.306 / au)
    if cos_theta <= 0:
        return TEMP_NIGHT
    return TEMP_BASE + TEMP_AMP * (cos_theta ** 0.25) * scaling


def calc_desorption_rate(temp_k, u_ev):
    if temp_k < 10.0: return 0.0
    u_joule = u_ev * EV_TO_J
    exponent = -u_joule / (KB * temp_k)
    # オーバーフロー防止
    if exponent < -700: return 0.0
    return VIB_FREQ * np.exp(exponent)


# ==============================================================================
# メイン計算: 1地点の「夜明け前」から「日没後」までの追跡
# ==============================================================================
def run_1d_simulation():
    # 時間ステップ (ローカルタイムの移動)
    # -180度(深夜) -> -90度(Dawn) -> 0度(Noon) -> 90度(Dusk) -> 180度
    longitudes = np.linspace(-180, 180, 1000)

    # 水星の自転角速度 [rad/s]
    omega = 2 * np.pi / ROTATION_PERIOD

    # グラフ描画準備
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for u_val in U_LIST_EV:
        densities = []
        fluxes = []
        temps = []

        # 初期密度 (夜明け前に十分溜まっていると仮定)
        current_density = 5.0e14 * (100 ** 2)

        for lon_deg in longitudes:
            # 1. 温度計算
            lon_rad = np.deg2rad(lon_deg)
            cos_theta = np.cos(lon_rad)  # Subsolar point経度0と仮定
            temp = calculate_temp(cos_theta, AU_DIST)

            # 2. 脱離率計算
            rate = calc_desorption_rate(temp, u_val)

            # 3. フラックス計算 (Flux = Density * Rate)
            flux = current_density * rate

            # 4. 密度更新 (オイラー法)
            # 次のステップまでの時間 dt
            # d(lon) / dt = omega  => dt = d(lon) / omega
            d_lon = np.deg2rad(longitudes[1] - longitudes[0])
            dt = d_lon / omega

            # dN/dt = Supply - Loss
            # 日中は供給より損失が圧倒的なのでSupplyは簡易的に定数
            loss = flux
            supply = BASE_SUPPLY_FLUX

            # クランプ拡散っぽく、暑いと供給が止まる挙動を入れるならここですが、
            # 今回はUの効果を見たいのでシンプルにします。

            current_density += (supply - loss) * dt
            if current_density < 0: current_density = 0

            densities.append(current_density)
            fluxes.append(flux)
            temps.append(temp)

        # プロット
        # スケール調整: 見やすいように対数軸などを考慮
        ax1.plot(longitudes, densities, label=f'U = {u_val} eV', linewidth=2)
        ax2.plot(longitudes, fluxes, label=f'U = {u_val} eV', linewidth=2)

    # --- グラフ装飾 ---
    # 表面密度
    ax1.set_title("Surface Density Evolution (1-Day Cycle)", fontsize=14)
    ax1.set_ylabel("Surface Density [m^-2]", fontsize=12)
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(fontsize=12)

    # 朝と夕方にラインを引く
    ax1.axvline(-90, color='gray', linestyle=':', alpha=0.8, label='Dawn')
    ax1.axvline(90, color='gray', linestyle=':', alpha=0.8, label='Dusk')

    # 放出フラックス
    ax2.set_title("Thermal Desorption Flux", fontsize=14)
    ax2.set_ylabel("Flux [atoms m^-2 s^-1]", fontsize=12)
    ax2.set_xlabel("Local Time (Longitude deg) [0=Noon, -90=Dawn, 90=Dusk]", fontsize=12)
    ax2.set_yscale('log')
    ax2.set_ylim(1e10, 1e16)  # 範囲は適宜調整
    ax2.grid(True, which="both", ls="--", alpha=0.5)

    ax2.axvline(-90, color='gray', linestyle=':', alpha=0.8)
    ax2.axvline(90, color='gray', linestyle=':', alpha=0.8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_1d_simulation()