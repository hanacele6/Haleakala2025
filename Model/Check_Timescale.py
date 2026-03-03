import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 物理定数・パラメータ設定
# ==========================================
# 物理定数
KB_EV = 8.617e-5  # ボルツマン定数 [eV/K]
NU = 1.0e13  # 振動数因子 [Hz]
AU_M = 1.496e11  # 1天文単位 [m]

# 軌道パラメータ (水星)
MERCURY_A_AU = 0.387098
MERCURY_E = 0.205630

# 温度モデル (Leblanc et al. based on code)
T_BASE = 100.0
T_AMP = 600.0
SCALING_REF_AU = 0.306

# PSDパラメータ
FLUX_1AU_ORIGINAL = 1.5e14  # [photons/cm2/s]
Q_PSD_ORIGINAL = 2.7e-21  # [cm2]

# 比較する活性化エネルギー [eV] (1.4 を追加)
# 1.4 eV : 実験値の下限付近
# 1.85 eV: Yakshinskiy et al. (2000) などでよく見る値
# 2.7 eV : 非常に強い結合
U_VALUES = [1.4, 1.85, 2.7]

# シミュレーションのタイムステップ [s]
DT_STEP = 100.0


# ==========================================
# 2. 計算関数
# ==========================================
def get_dist_au(taa_deg):
    """TAAから太陽距離(AU)を計算"""
    rad = np.deg2rad(taa_deg)
    r = MERCURY_A_AU * (1 - MERCURY_E ** 2) / (1 + MERCURY_E * np.cos(rad))
    return r


def get_subsolar_temp(r_au):
    """日下点温度を計算 (コード準拠)"""
    scaling = np.sqrt(SCALING_REF_AU / r_au)
    return T_BASE + T_AMP * (1.0 ** 0.25) * scaling


def get_td_timescale(temp, u_ev):
    """熱脱離(TD)の滞留時間 [s]"""
    # expの中身が大きすぎるとOverflowするのでクリップ
    exponent = u_ev / (KB_EV * temp)
    if exponent > 200: return 1e50  # 実質無限 (グラフ描画のため適当な巨大数に)
    return (1.0 / NU) * np.exp(exponent)


def get_psd_timescale(r_au):
    """PSDの寿命 [s]"""
    flux = FLUX_1AU_ORIGINAL / (r_au ** 2)
    rate = flux * Q_PSD_ORIGINAL
    return 1.0 / rate


# ==========================================
# 3. メイン計算ループ
# ==========================================
taa_list = np.arange(0, 360, 1)
r_au_list = []
temp_list = []

# リストを用意
tau_td_140 = []  # 追加
tau_td_185 = []
tau_td_270 = []
tau_psd = []

for taa in taa_list:
    r = get_dist_au(taa)
    t = get_subsolar_temp(r)

    r_au_list.append(r)
    temp_list.append(t)

    # 各Uに対する滞留時間を計算
    tau_td_140.append(get_td_timescale(t, 1.4))
    tau_td_185.append(get_td_timescale(t, 1.85))
    tau_td_270.append(get_td_timescale(t, 2.7))

    tau_psd.append(get_psd_timescale(r))

# ==========================================
# 4. プロット作成
# ==========================================
fig, ax1 = plt.subplots(figsize=(10, 6))

# 背景色とグリッド
ax1.set_facecolor('#f0f0f0')
ax1.grid(True, which='both', linestyle='--', alpha=0.7)

# --- プロット (対数軸) ---

# U=2.7 (赤: ほぼ動かない)
ax1.plot(taa_list, tau_td_270, color='red', linewidth=2, label='TD (U=2.7 eV)')

# U=1.85 (青: 一般的な値)
ax1.plot(taa_list, tau_td_185, color='blue', linewidth=2, label='TD (U=1.85 eV)')

# U=1.4 (マゼンタ: 今回の検証値) -> 一番動きやすい
ax1.plot(taa_list, tau_td_140, color='magenta', linewidth=3, linestyle='-', label='TD (U=1.4 eV)')

# PSD (緑点線: 光脱離)
ax1.plot(taa_list, tau_psd, color='green', linewidth=2, linestyle='--', label='PSD')

# 基準線の追加
ax1.axhline(y=DT_STEP, color='black', linestyle=':', linewidth=2, label='Step (100s)')
# ax1.axhline(y=88 * 24 * 3600, color='gray', linestyle='-.', alpha=0.5, label='1 Mercury Year')

# 近日点・遠日点の注釈
peri_idx = 0
apo_idx = 180

# 軸設定
ax1.set_yscale('log')
ax1.set_xlabel('True Anomaly (TAA) [deg]', fontsize=12)
ax1.set_ylabel('Timescale [seconds]', fontsize=12)
#ax1.set_title('Desorption Timescales vs TAA (Effect of U=1.4eV)', fontsize=14)
ax1.set_xlim(0, 360)
# 見やすい範囲にY軸を制限 (1e-4秒 〜 1e12秒)
ax1.set_ylim(1e-4, 1e12)

# 凡例
ax1.legend(loc='upper right', fontsize=10, framealpha=1.0)

# レイアウト調整と表示
plt.tight_layout()
plt.show()

# ==========================================
# 5. 数値出力 (代表点)
# ==========================================
print(f"{'TAA':<5} | {'Temp(K)':<8} | {'TD(1.4eV)':<12} | {'TD(1.85eV)':<12} | {'TD(2.7eV)':<12}")
print("-" * 75)
indices = [0, 90, 180, 270]  # 近日点, ... 遠日点, ...
for i in indices:
    print(
        f"{taa_list[i]:<5} | {temp_list[i]:.1f}    | {tau_td_140[i]:.2e} s   | {tau_td_185[i]:.2e} s   | {tau_td_270[i]:.2e} s")