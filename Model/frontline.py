# -*- coding: utf-8 -*-
"""
高密度領域が「運ばれる」か「消費される」かを分ける境界の可視化

--- 確かめたいこと ---
近日点付近でターミネーター付近に形成された高密度領域が、遠日点に向かうにつれて
ターミネーターに留まらず太陽直下点(SSP)側へ移動してくる。
これは「消費されるより先に自転で運ばれる」ためと考えられる。
ではどこまで運ばれ、どこで消費に追いつかれるのか。

--- 図の内容 ---
(TAA, 太陽時角) 平面
  ・背景: その場所での消費タイムスケール tau = 1/max(r_psd, r_td)
          PSD と TD のうち「速い方」を使う(合成ではない)
  ・細線: 本体固定の区画の軌跡 (ターミネーターから出発)
  ・太線: 累積消費 ∫r dt = 1 に達する点をつないだ「消費フロント」
          Q ごとに描くので、Q による前進距離の差が見える

--- 消費フロントとタイムスケールの逆転は別物 ---
・タイムスケールの逆転 : ある瞬間に移動と消費のどちらが速いか(局所的)
・消費フロント         : ターミネーターを出てから積算した消費量が1に達した位置(履歴依存)
局所的な逆転位置は高eff_cosのTDで決まるためQにほぼ依存しないが、
累積では低eff_cosを長時間かけて通過する間のPSDの積み上げが効くのでQ依存が出る。

--- 注意 ---
本スクリプトは放出率の式と軌道力学だけから計算しており、
地下からの拡散供給とホップによる再堆積を含まない。
そのため実際にはここで示すより遠くまで在庫が生き残る。
向きと構造は正しいが、絶対位置は過小評価であることに留意すること。
"""

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm

# ==========================================
# 設定
# ==========================================
Q_LIST = [2.0, 0.3]                      # 比較する Q_psd_base
Q_COLORS = {2.0: 'crimson', 0.3: 'royalblue'}
START_TAAS = [0, 30, 60, 90, 120, 150]   # 区画がターミネーターを通過するTAA
ACC_THRESHOLD = 1.0                      # 消費フロントの定義: ∫r dt がこの値に達する
DT = 1800.0                              # 積分刻み [s]

# --- 物理定数(本体 mkNaColumnDensity と一致) ---
KB = 1.380649e-23
EV = 1.602e-19
F_UV_1AU = 1.5e14 * (100 ** 2)
A_AU = 0.387098
ECC = 0.205630
U_EV = 1.85
TEMP_BASE = 100.0
TEMP_AMP = 600.0
TD_PREFACTOR = 1e13
T_ORB = 87.969 * 86400
N_MEAN = 2 * np.pi / T_ORB
SPIN_RATIO = 1.5                         # 3:2 共鳴


# ==========================================
# 物理式
# ==========================================
def au_of_taa(taa):
    return A_AU * (1 - ECC**2) / (1 + ECC * np.cos(np.deg2rad(taa)))


def dtaa_dt(taa):
    """公転角速度 [rad/s]"""
    return N_MEAN * np.sqrt(1 - ECC**2) / (au_of_taa(taa) / A_AU) ** 2


def dh_dt(taa):
    """太陽時角の進む速さ [rad/s]。負なら太陽が逆行(近日点付近)。"""
    return SPIN_RATIO * N_MEAN - dtaa_dt(taa)


def rate_psd(eff_cos, au, q_base):
    q = (q_base * 1.0e-20) / (100 ** 2)
    return (F_UV_1AU / au**2) * q * np.clip(eff_cos, 0, None)


def rate_td(eff_cos, au):
    ec = np.clip(eff_cos, 0, None)
    T = TEMP_BASE + TEMP_AMP * (ec ** 0.25) * np.sqrt(0.306 / au)
    expo = -(U_EV * EV) / (KB * T)
    return np.where(expo > -700, TD_PREFACTOR * np.exp(np.clip(expo, -700, None)), 0.0)


# ==========================================
# 区画の追跡
# ==========================================
def loss_rate_fastest(eff_cos, au, q_base):
    """PSD と TD のうち「速い方」のレートを返す(合成ではない)。
    tau = 1/rate が最も短い過程が、その場所の消費を代表する。"""
    return np.maximum(rate_psd(eff_cos, au, q_base), rate_td(eff_cos, au))


def dominant_process(eff_cos, au, q_base):
    """その場所で速いのが PSD か TD かを返す"""
    return 'PSD' if rate_psd(eff_cos, au, q_base) >= rate_td(eff_cos, au) else 'TD'


def track_parcel(taa0, q_base, h_start=-90.0, max_steps=400000):
    """ターミネーター(h=h_start)から区画を追い、累積消費が閾値に達する点を返す。
    消費レートは PSD と TD のうち速い方を使う(合成ではない)。
    戻り値: (軌跡のTAA配列, 軌跡のh配列, フロント到達点 or None)"""
    h = float(h_start)
    taa = float(taa0)
    acc = 0.0
    taas, hs = [], []
    front = None
    for _ in range(max_steps):
        au = au_of_taa(taa)
        ec = np.cos(np.deg2rad(h))
        taas.append(taa % 360.0)
        hs.append(h)
        acc += float(loss_rate_fastest(ec, au, q_base)) * DT
        if front is None and acc >= ACC_THRESHOLD:
            front = (taa % 360.0, h)
        h += np.rad2deg(dh_dt(taa)) * DT
        taa += np.rad2deg(dtaa_dt(taa)) * DT
        if h >= 0.0 or (taa - taa0) > 360.0:
            break
    return np.array(taas), np.array(hs), front


# ==========================================
# 描画
# ==========================================
def main():
    fig, ax2 = plt.subplots(figsize=(11, 7))

    # ---------- (TAA, 太陽時角) 平面 ----------
    tg = np.linspace(0, 360, 361)
    hg = np.linspace(-90, 0, 181)
    TG, HG = np.meshgrid(tg, hg)
    AUG = au_of_taa(TG)
    ECG = np.cos(np.deg2rad(HG))
    # 背景: 速い方の過程の消費タイムスケール(代表として Q=2.0)
    tau = 1.0 / np.maximum(loss_rate_fastest(ECG, AUG, 2.0), 1e-30) / 86400.0

    im = ax2.pcolormesh(tg, hg, tau, norm=LogNorm(vmin=1e-2, vmax=1e3),
                        cmap='YlGnBu', shading='auto')
    cbar = fig.colorbar(im, ax=ax2, pad=0.02)
    cbar.set_label('消費タイムスケール τ = 1/max(r_PSD, r_TD) [日]  ※速い方')

    # 区画の軌跡と消費フロント
    fronts = {q: [] for q in Q_LIST}
    for taa0 in START_TAAS:
        t_tr, h_tr, _ = track_parcel(taa0, Q_LIST[0])
        # 軌跡は Q によらない(運動学のみ)ので1本だけ描く
        jump = np.where(np.abs(np.diff(t_tr)) > 180)[0]
        t_plot, h_plot = t_tr.copy(), h_tr.copy()
        for j in jump:
            t_plot = np.insert(t_plot, j + 1, np.nan)
            h_plot = np.insert(h_plot, j + 1, np.nan)
        ax2.plot(t_plot, h_plot, color='gray', lw=1.0, alpha=0.8)
        ax2.plot(t_tr[0], h_tr[0], 'o', ms=4, color='gray')

        for q in Q_LIST:
            _, _, fr = track_parcel(taa0, q)
            if fr is not None:
                fronts[q].append(fr)

    for q in Q_LIST:
        if fronts[q]:
            ft = np.array(fronts[q])
            ax2.plot(ft[:, 0], ft[:, 1], '-o', color=Q_COLORS[q], lw=2.5, ms=7,
                     markeredgecolor='k', markeredgewidth=0.5,
                     label=f'消費フロント (Q={q})  ∫r dt = {ACC_THRESHOLD:g}')

    ax2.set_xlabel('TAA [deg]')
    ax2.set_ylabel('太陽時角 h [deg]   (-90=ターミネーター, 0=SSP)')
    ax2.set_title('本体固定区画の軌跡(灰)と、累積消費が1に達する位置(色つき)\n'
                  '軌跡がフロントに達したところで在庫が食い尽くされる', fontsize=12)

    ax2.axvline(180, color='green', linestyle='--', lw=1.5, alpha=0.6)
    #ax2.text(182, -85, '遠日点 (180°)', color='green', fontsize=9, fontweight='bold')
    ax2.set_xlim(0, 360)
    ax2.set_ylim(-90, 0)
    ax2.xaxis.set_major_locator(MultipleLocator(60))
    ax2.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.show()

    # ---------- 数値サマリー ----------
    print("\n" + "=" * 76)
    print("=== 消費フロント (累積 ∫r dt が %.1f に達する位置) ===" % ACC_THRESHOLD)
    print("=" * 76)
    header = f"{'出発TAA':>9}"
    for q in Q_LIST:
        header += f"{'Q=%.1f 到達h' % q:>15}{'そのTAA':>10}"
    if len(Q_LIST) == 2:
        header += f"{'h の差':>9}"
    print(header)
    print("-" * 76)
    for i, taa0 in enumerate(START_TAAS):
        line = f"{taa0:>9}"
        hv = []
        for q in Q_LIST:
            if i < len(fronts[q]):
                t, h = fronts[q][i]
                hv.append(h)
                line += f"{h:>15.0f}{t:>10.0f}"
            else:
                line += f"{'到達せず':>15}{'-':>10}"
        if len(hv) == 2:
            line += f"{hv[1]-hv[0]:>+9.0f}"
        print(line)
    print("=" * 76)
    print("\n[注意] 地下からの拡散供給とホップによる再堆積を含まないため、")
    print("       実際にはここで示すより遠く(SSP側)まで在庫が生き残る。")
    print("       向きと構造は正しいが、絶対位置は過小評価。")


if __name__ == "__main__":
    main()