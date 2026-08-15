# -*- coding: utf-8 -*-
"""
==============================================================================
PSD vs TD 理論放出率 比較プロット
  Window 1: TAA に対する放出率の比較 (天頂角 eff_cos スライダー付き)
  Window 2: 天頂角 eff_cos に対する放出率の比較 (TAA スライダー付き)   ★追加
==============================================================================

概要:
    mkNaColumnDensity9_9.py 内の物理モデル式をそのまま用いて、
    「表面密度が一定」の場合の PSD (光刺激脱離) と TD (熱脱離) の
    理論的な脱離レート係数 [1/s] を比較する。

      PSD: r_psd = f_uv(AU) * Q_PSD * eff_cos
             f_uv(AU) = F_UV_1AU / AU^2   (太陽UVフラックスは日心距離の2乗に反比例)

      TD : r_td  = 1e13 * exp( -U / (kB * T_day(AU, eff_cos)) )
             T_day = TEMP_BASE + TEMP_AMP * (eff_cos^0.25) * sqrt(0.306 / AU)
                     (Leblanc et al. の表面温度モデル)

      AU(TAA) は水星の楕円軌道(離心率e)から Kepler の式で解析的に計算する。
      eff_cos は太陽天頂角 z の cos (1.0=直下点/正午, 0.0=終端線)。

    ------------------------------------------------------------------
    2つの見方 (2ウィンドウ):
    ------------------------------------------------------------------
      [Window 1]  横軸 = TAA、スライダー = eff_cos
          → 「ある天頂角のとき、軌道上(日心距離)でレートがどう変わるか」

      [Window 2]  横軸 = eff_cos (=cos(天頂角))、スライダー = TAA   ★今回追加
          → 「ある TAA(=ある日心距離)のとき、直下点→終端線に向かって
             天頂角が変わるとレートがどう変わるか」
          横軸下 = eff_cos、横軸上 = 天頂角 z [deg] を併記する。
          この向きで見ると、PSD は eff_cos に単純比例して直線的に落ちるのに対し、
          TD は eff_cos^0.25 経由で T_day に効くため、直下点付近で急峻に立ち上がり
          天頂角が大きくなると急激に弱まる様子(PSD/TD のクロスオーバー)が分かる。

    パラメータは下の CONFIG セクションにまとめてある。

作成者: Claude (Anthropic) - Koki Masaki氏のシミュレーションコード用に作成
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ==============================================================================
# CONFIG: ここのパラメータだけ変えれば挙動を変更できる
# ==============================================================================
CONFIG = {
    # --- PSDパラメータ ---
    'Q_PSD_BASE': 0.27,        # Q_PSDのベース係数 (例: 2.0 -> 2.0e-20/100^2, mkNaColumnDensity9_9.py と同じ定義)

    # --- TDパラメータ ---
    # TD計算に使う束縛エネルギーU [eV] のリスト（複数指定すると感度比較になる）
    'U_VALUES_EV': [1.85],

    # --- 天頂角(eff_cos)の範囲 ---
    # Window1 ではスライダーの範囲、Window2 では横軸(x)の範囲として共用する。
    'EFF_COS_INIT': 1.0,       # 起動時の初期値 (1.0=直下点=最大照度)
    'EFF_COS_MIN': 0.01,       # 下限 (0だとTD/PSDが厳密に0になるため僅かに正の値から)
    'EFF_COS_MAX': 1.0,        # 上限
    'EFF_COS_STEP': 0.005,     # Window2 横軸の分解能

    # --- TAAの計算範囲・分解能 ---
    'TAA_MIN_DEG': 0.0,
    'TAA_MAX_DEG': 360.0,
    'TAA_STEP_DEG': 0.5,
    'TAA_INIT_DEG': 0.0,       # Window2 の TAA スライダー初期値 (0=近日点)
    'TAA_SLIDER_STEP': 1.0,    # Window2 の TAA スライダーのステップ

    # --- グラフの縦軸を対数スケールにするか ---
    'LOGSCALE': True,
}

# ------------------------------------------------------------------
# 物理定数 (mkNaColumnDensity9_9.py の PHYSICAL_CONSTANTS と同一の値)
# ------------------------------------------------------------------
K_BOLTZMANN = 1.380649e-23      # [J/K]
EV_TO_JOULE = 1.602e-19         # [J/eV]

MERCURY_SEMI_MAJOR_AXIS_AU = 0.387098
MERCURY_ECCENTRICITY = 0.205630

# ------------------------------------------------------------------
# ソースプロセスの物理定数 (mkNaColumnDensity9_9.py と同一の値)
# ------------------------------------------------------------------
F_UV_1AU = 1.5e14 * (100 ** 2)   # 1AUでのUVフラックス [photons/m^2/s]

TEMP_BASE = 100.0   # [K]
TEMP_AMP = 600.0    # [K]


# ==============================================================================
# 物理モデル関数 (mkNaColumnDensity9_9.py と同一の式)
# ==============================================================================
def calculate_au_at_taa(taa_deg):
    """TAA [deg] から水星の日心距離 AU を Kepler の式で解析的に計算する。
    (mkNaColumnDensity9_9.py の calculate_au_at_taa と同一)"""
    a = MERCURY_SEMI_MAJOR_AXIS_AU
    e = MERCURY_ECCENTRICITY
    rad = np.deg2rad(taa_deg)
    return a * (1 - e ** 2) / (1 + e * np.cos(rad))


def calc_day_temp(AU, eff_cos):
    """Leblancモデルによる昼面の表面温度 [K]
    (mkNaColumnDensity9_9.py の update_surface_maps_numba 内の t_day 計算と同一)
    AU / eff_cos はどちらもスカラ・配列どちらでも可 (broadcast)。"""
    scaling = np.sqrt(0.306 / AU)
    return TEMP_BASE + TEMP_AMP * (np.asarray(eff_cos) ** 0.25) * scaling


def calc_r_psd(AU, q_psd, eff_cos):
    """PSDの理論脱離レート係数 [1/s] (表面密度に依存しない)"""
    f_uv = F_UV_1AU / (AU ** 2)
    return f_uv * q_psd * np.asarray(eff_cos)


def calc_r_td(t_day, u_ev):
    """TDの理論脱離レート係数 [1/s] (表面密度に依存しない, Arrhenius型)"""
    u_j = u_ev * EV_TO_JOULE
    exponent = -u_j / (K_BOLTZMANN * t_day)
    return 1e13 * np.exp(exponent)


# 共通のTD曲線カラー
TD_COLORS = ["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]


def _autoscale_log_y(ax, y_arrays):
    """対数軸のY範囲をデータに合わせて自動調整する共通処理。"""
    all_y = np.concatenate([np.atleast_1d(y) for y in y_arrays])
    all_y = all_y[all_y > 0]
    if all_y.size > 0:
        ax.set_ylim(all_y.min() * 0.5, all_y.max() * 2.0)


# ==============================================================================
# Window 1: 横軸 TAA、スライダー eff_cos  (既存)
# ==============================================================================
def build_window_vs_taa(cfg, q_psd):
    u_values = cfg['U_VALUES_EV']

    taa = np.arange(cfg['TAA_MIN_DEG'], cfg['TAA_MAX_DEG'], cfg['TAA_STEP_DEG'])
    AU = calculate_au_at_taa(taa)

    def compute(eff_cos):
        r_psd = calc_r_psd(AU, q_psd, eff_cos)
        t_day = calc_day_temp(AU, eff_cos)
        r_td_list = [calc_r_td(t_day, u) for u in u_values]
        return r_psd, t_day, r_td_list

    eff_cos0 = cfg['EFF_COS_INIT']
    r_psd0, t_day0, r_td_list0 = compute(eff_cos0)

    fig, ax = plt.subplots(figsize=(11, 7), num="Window 1: rate vs TAA")
    plt.subplots_adjust(bottom=0.22)

    (line_psd,) = ax.plot(taa, r_psd0, color="tab:blue", lw=2.2,
                          label=f"PSD rate (Q_psd_base={cfg['Q_PSD_BASE']})")

    lines_td = []
    for k, (u, r_td0) in enumerate(zip(u_values, r_td_list0)):
        (line,) = ax.plot(taa, r_td0, lw=1.8, ls="--",
                          color=TD_COLORS[k % len(TD_COLORS)],
                          label=f"TD rate (U={u} eV)")
        lines_td.append(line)

    if cfg['LOGSCALE']:
        ax.set_yscale("log")
    ax.set_xlabel("TAA (True Anomaly Angle) [deg]")
    ax.set_ylabel("Theoretical Desorption Rate [1/s]\n(density-independent)")
    ax.set_xlim(cfg['TAA_MIN_DEG'], cfg['TAA_MAX_DEG'] - cfg['TAA_STEP_DEG'])
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    title = ax.set_title("")

    info_text = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=9,
                        va="bottom", ha="left",
                        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

    def update_display(eff_cos):
        r_psd, t_day, r_td_list = compute(eff_cos)
        line_psd.set_ydata(r_psd)
        for line, r_td in zip(lines_td, r_td_list):
            line.set_ydata(r_td)

        title.set_text(f"PSD vs TD  |  x=TAA, slider=zenith  (eff_cos = {eff_cos:.2f})")
        info_text.set_text(
            f"AU range: {AU.min():.3f} - {AU.max():.3f}\n"
            f"T_day range (this eff_cos): {t_day.min():.1f} - {t_day.max():.1f} K"
        )
        if cfg['LOGSCALE']:
            _autoscale_log_y(ax, [r_psd] + r_td_list)
        fig.canvas.draw_idle()

    update_display(eff_cos0)

    ax_slider = plt.axes([0.15, 0.08, 0.7, 0.04])
    slider = Slider(ax_slider, "eff_cos\n(zenith angle)",
                    cfg['EFF_COS_MIN'], cfg['EFF_COS_MAX'],
                    valinit=eff_cos0, valstep=0.01)
    slider.on_changed(update_display)

    # スライダー参照を保持するため返す
    return fig, slider


# ==============================================================================
# Window 2: 横軸 eff_cos(天頂角)、スライダー TAA   ★追加
# ==============================================================================
def build_window_vs_effcos(cfg, q_psd):
    u_values = cfg['U_VALUES_EV']

    # 横軸: eff_cos = cos(天頂角)
    eff_cos_axis = np.arange(cfg['EFF_COS_MIN'],
                             cfg['EFF_COS_MAX'] + 1e-9,
                             cfg['EFF_COS_STEP'])

    def compute(taa_deg):
        AU = calculate_au_at_taa(taa_deg)              # スカラ
        r_psd = calc_r_psd(AU, q_psd, eff_cos_axis)    # eff_cos に線形
        t_day = calc_day_temp(AU, eff_cos_axis)        # eff_cos^0.25 経由
        r_td_list = [calc_r_td(t_day, u) for u in u_values]
        return AU, r_psd, t_day, r_td_list

    taa0 = cfg['TAA_INIT_DEG']
    AU0, r_psd0, t_day0, r_td_list0 = compute(taa0)

    fig, ax = plt.subplots(figsize=(11, 7), num="Window 2: rate vs zenith")
    plt.subplots_adjust(bottom=0.22, top=0.86)

    (line_psd,) = ax.plot(eff_cos_axis, r_psd0, color="tab:blue", lw=2.2,
                          label=f"PSD rate (Q_psd_base={cfg['Q_PSD_BASE']})")

    lines_td = []
    for k, (u, r_td0) in enumerate(zip(u_values, r_td_list0)):
        (line,) = ax.plot(eff_cos_axis, r_td0, lw=1.8, ls="--",
                          color=TD_COLORS[k % len(TD_COLORS)],
                          label=f"TD rate (U={u} eV)")
        lines_td.append(line)

    if cfg['LOGSCALE']:
        ax.set_yscale("log")
    ax.set_xlabel("eff_cos = cos(solar zenith angle)   [1.0 = subsolar / 0 = terminator]")
    ax.set_ylabel("Theoretical Desorption Rate [1/s]\n(density-independent)")
    ax.set_xlim(cfg['EFF_COS_MIN'], cfg['EFF_COS_MAX'])
    ax.legend(loc="lower right", ncol=2)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    title = ax.set_title("")

    # --- 上側の副軸: 天頂角 z [deg] を併記 (eff_cos = cos z の非線形写像) ---
    def effcos_to_zdeg(ec):
        ec = np.clip(np.asarray(ec, dtype=float), 0.0, 1.0)
        return np.degrees(np.arccos(ec))

    def zdeg_to_effcos(z):
        return np.cos(np.radians(np.asarray(z, dtype=float)))

    secax = ax.secondary_xaxis('top', functions=(effcos_to_zdeg, zdeg_to_effcos))
    secax.set_xlabel("solar zenith angle z [deg]")
    secax.set_xticks([0, 30, 45, 60, 75, 85])

    info_text = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=9,
                        va="bottom", ha="left",
                        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

    def update_display(taa_deg):
        AU, r_psd, t_day, r_td_list = compute(taa_deg)
        line_psd.set_ydata(r_psd)
        for line, r_td in zip(lines_td, r_td_list):
            line.set_ydata(r_td)

        title.set_text(f"PSD vs TD  |  x=zenith(eff_cos), slider=TAA  (TAA = {taa_deg:.1f} deg)")
        info_text.set_text(
            f"TAA = {taa_deg:.1f} deg  ->  AU = {AU:.4f}\n"
            f"T_day range (this AU): {t_day.min():.1f} - {t_day.max():.1f} K"
        )
        if cfg['LOGSCALE']:
            _autoscale_log_y(ax, [r_psd] + r_td_list)
        fig.canvas.draw_idle()

    update_display(taa0)

    ax_slider = plt.axes([0.15, 0.08, 0.7, 0.04])
    slider = Slider(ax_slider, "TAA [deg]",
                    cfg['TAA_MIN_DEG'], cfg['TAA_MAX_DEG'],
                    valinit=taa0, valstep=cfg['TAA_SLIDER_STEP'])
    slider.on_changed(update_display)

    return fig, slider


def main():
    cfg = CONFIG

    # Q_PSDの単位換算 (mkNaColumnDensity9_9.py: base_q = q_psd_base*1e-20 / (100**2) [m^2])
    q_psd = (cfg['Q_PSD_BASE'] * 1.0e-20) / (100 ** 2)

    # 2つのウィンドウを生成 (スライダー参照はGCされないよう保持しておく)
    fig1, slider1 = build_window_vs_taa(cfg, q_psd)
    fig2, slider2 = build_window_vs_effcos(cfg, q_psd)

    plt.show()


if __name__ == "__main__":
    main()