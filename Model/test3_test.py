# -*- coding: utf-8 -*-
"""
解析的トイモデル: PSD強度による放出ピークのシフトを最小構成で確認する。

--- アイデア ---
1つの表面在庫パケット(渋滞域の塊を模す)を、水星の自転による見かけの太陽運動に沿って
「ターミネーター(eff_cos=0) → 正午(eff_cos=1)」へ運びながら、
各時刻でその場の温度・日照に応じたPSDとTDに消費させる。
PSD強度(Q)を変えたとき、放出がどのTAA/局所時刻でピークになるかを見る。

本体シミュレーション(mkNaColumnDensity9_9_region.py)から抜き出した式:
  温度:   t_day = TEMP_BASE + TEMP_AMP * eff_cos^0.25 * sqrt(0.306/AU)
  PSD率:  r_psd = F_UV_1AU/AU^2 * q_psd * eff_cos          [1/s]
  TD率:   r_td  = 1e13 * exp(-U*eV/(kB*t_day))              [1/s]
  在庫減衰: dN/dt = -(r_psd + r_td) * N

--- 局所時刻(eff_cos)の時間発展 ---
簡単のため、パケットはターミネーター通過後、見かけの自転角速度で
サブソーラー点へ近づく = eff_cos が 0→1 へ単調増加する、とモデル化する。
eff_cos(t) = sin( omega_app * t )   (0 <= omega_app*t <= pi/2)
omega_app は「見かけの自転角速度」で、TAAに依存(近日点で遅く、遠日点で速い)。
"""

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401
from matplotlib.ticker import MultipleLocator

# ==========================================
# 本体から抜き出した物理定数・パラメータ
# ==========================================
KB = 1.380649e-23          # J/K
EV = 1.602e-19             # J/eV
TEMP_BASE = 100.0          # K
TEMP_AMP = 600.0           # K
F_UV_1AU = 1.5e14 * (100 ** 2)   # 本体の F_UV_1AU (cm->m 換算込みの値)
ECC = 0.20563593
A_AU = 0.387098            # 軌道長半径 [AU]
ROT_PERIOD = 58.6462 * 86400.0   # 自転周期 [s]

U_EV = 1.85               # 脱離エネルギー (UGモデルの代表値)

# 検証したい PSD 強度(Q) のリスト [cm^2 相当を本体と同じ /100^2 で持つ]
Q_LIST = {
    "Q=2.0 (強PSD)": 2.0e-20 / (100 ** 2),
    "Q=0.9":         0.9e-20 / (100 ** 2),
    "Q=0.3 (弱PSD)": 0.3e-20 / (100 ** 2),
}
Q_COLORS = {"Q=2.0 (強PSD)": "crimson", "Q=0.9": "green", "Q=0.3 (弱PSD)": "steelblue"}


# ==========================================
# 軌道: TAA -> AU, 見かけの自転角速度
# ==========================================
def au_of_taa(taa_deg):
    theta = np.deg2rad(taa_deg)
    return A_AU * (1 - ECC**2) / (1 + ECC * np.cos(theta))


def orbital_angular_velocity(taa_deg):
    """公転角速度 dθ/dt [rad/s]。ケプラー: r^2 dθ/dt = 一定 = sqrt(GM a(1-e^2))。
    ここでは相対値だけ効くので、近日点基準で規格化した形を使う。"""
    r = au_of_taa(taa_deg)
    # dθ/dt ∝ 1/r^2 。平均公転角速度 n で規格化
    n = 2 * np.pi / (87.969 * 86400.0)  # 平均公転角速度 [rad/s]
    # 角運動量 L = r^2 dθ/dt = sqrt(a(1-e^2)) * n * a  (規格化)
    L = np.sqrt(1 - ECC**2) * n * A_AU**2
    return L / (r * A_AU)**2 * A_AU**2  # [rad/s] 近似(相対値として使う)


def apparent_rotation_rate(taa_deg):
    """見かけの自転角速度 = |自転角速度 - 公転角速度|。
    3:2共鳴なので自転角速度 ω_spin = 1.5 * 平均公転角速度 n。"""
    n = 2 * np.pi / (87.969 * 86400.0)
    omega_spin = 1.5 * n
    omega_orb = orbital_angular_velocity(taa_deg)
    return abs(omega_spin - omega_orb)


# ==========================================
# 温度とレート
# ==========================================
def temperature(eff_cos, au):
    scaling = np.sqrt(0.306 / au)
    return TEMP_BASE + TEMP_AMP * (max(eff_cos, 0.0) ** 0.25) * scaling


def rate_psd(eff_cos, au, q):
    f_uv = F_UV_1AU / au**2
    return f_uv * q * max(eff_cos, 0.0)   # [1/s]


def rate_td(eff_cos, au, u_ev):
    t = temperature(eff_cos, au)
    if t < 10.0:
        return 0.0
    expo = -(u_ev * EV) / (KB * t)
    if expo < -700.0:
        return 0.0
    return 1e13 * np.exp(expo)   # [1/s]


# ==========================================
# 1パケットを運びながら消費させる
# ==========================================
def simulate_packet(taa_start_deg, q, u_ev, n_steps=2000):
    """
    TAA=taa_start_deg でターミネーターに現れた在庫パケット(N=1に規格化)を、
    見かけの自転で eff_cos 0->1 へ運びながら PSD/TD に消費させる。
    各ステップの放出量を (TAA, eff_cos) に紐づけて返す。
    """
    au = au_of_taa(taa_start_deg)
    omega_app = apparent_rotation_rate(taa_start_deg)
    if omega_app <= 0:
        return None

    # eff_cos = sin(omega_app * t) が 0->1 になるまで(t: 0 -> pi/2/omega_app)
    t_max = (np.pi / 2) / omega_app
    dt = t_max / n_steps

    N = 1.0   # パケットの在庫(規格化)
    records = []

    for step in range(n_steps):
        t = step * dt
        phase = omega_app * t
        if phase > np.pi / 2:
            break
        eff_cos = np.sin(phase)

        # この間のTAAの進み(公転)。運搬中もTAAは進むが、簡単のため近似で加算
        omega_orb = orbital_angular_velocity(taa_start_deg)
        taa_now = taa_start_deg + np.rad2deg(omega_orb * t)

        r_p = rate_psd(eff_cos, au, q)
        r_t = rate_td(eff_cos, au, u_ev)
        r_tot = r_p + r_t

        # このステップでの放出量(在庫の減少分)
        decay = np.exp(-r_tot * dt)
        lost = N * (1.0 - decay)
        lost_psd = lost * (r_p / r_tot) if r_tot > 0 else 0.0
        lost_td = lost * (r_t / r_tot) if r_tot > 0 else 0.0

        records.append({
            'taa': taa_now % 360.0,
            'eff_cos': eff_cos,
            'lost': lost,
            'lost_psd': lost_psd,
            'lost_td': lost_td,
            'N_before': N,
        })

        N *= decay
        if N < 1e-6:
            break

    return records


# ==========================================
# 多数のTAA始点でパケットを流し、放出量マップを作る
# ==========================================
def sweep_in_weight(taa0_deg, mode="uniform"):
    """各TAA始点でターミネーターに現れる在庫パケットの大きさ(掃き込み量)の重み。
    mode:
      "uniform"     : 全TAA一定(素の効果を見る)
      "backlog_taa100": 近日点付近で見かけの自転が遅く、渋滞在庫が溜まる
                        → TAA=100付近でピークになるガウス的な重み
    """
    if mode == "uniform":
        return 1.0
    elif mode == "backlog_taa100":
        # 見かけの自転が遅いほど渋滞が溜まる = apparent_rotation_rate の逆数に比例
        omega_app = apparent_rotation_rate(taa0_deg)
        if omega_app <= 0:
            return 0.0
        return 1.0 / omega_app
    else:
        return 1.0


def routine_emission(taa_deg, q, u_ev):
    """日常域成分: Dawn半球の広い昼側に常在する在庫からの定常放出。
    渋滞パケットと違い「その場で供給されその場で放出される定常在庫」なので、
    昼側の各局所時刻(eff_cos)にわたって放出率を積分し、
    それに dt/dθ 重み(遠日点で滞在時間が長い)を掛ける。
    近日点でのTD暴走を防ぐため、単一eff_cos固定ではなく eff_cos∈(0,1] を積分する。"""
    au = au_of_taa(taa_deg)
    theta = np.deg2rad(taa_deg)
    dtdtheta = 1.0 / (1.0 + ECC * np.cos(theta))**2

    # 昼側の局所時刻にわたる放出率の積分(定常在庫=1と仮定、供給律速)
    ec_grid = np.linspace(0.05, 1.0, 20)
    emit = 0.0
    emit_psd = 0.0
    emit_td = 0.0
    for ec in ec_grid:
        r_p = rate_psd(ec, au, q)
        r_t = rate_td(ec, au, u_ev)
        r_tot = r_p + r_t
        # 供給律速の定常在庫からの放出率(在庫=1なので放出率=消費率)
        de = r_tot
        emit += de
        emit_psd += r_p
        emit_td += r_t
    emit *= dtdtheta
    emit_psd *= dtdtheta
    emit_td *= dtdtheta
    return emit, emit_psd, emit_td


def build_emission_map(q, u_ev, taa_starts, sweep_mode="uniform",
                       routine_weight=0.0):
    """各TAA始点でパケットを流し、放出をTAAビンに集計する。
    routine_weight: 日常域成分を足す強さ(0で渋滞パケットのみ)。"""
    taa_bins = np.arange(0, 360, 1.0)
    emit_total = np.zeros(len(taa_bins))
    emit_psd = np.zeros(len(taa_bins))
    emit_td = np.zeros(len(taa_bins))

    # --- 渋滞パケット成分 ---
    for taa0 in taa_starts:
        w = sweep_in_weight(taa0, sweep_mode)
        if w <= 0:
            continue
        recs = simulate_packet(taa0, q, u_ev)
        if recs is None:
            continue
        for r in recs:
            b = int(r['taa']) % 360
            emit_total[b] += r['lost'] * w
            emit_psd[b] += r['lost_psd'] * w
            emit_td[b] += r['lost_td'] * w

    # --- 日常域成分 ---
    if routine_weight > 0.0:
        # 渋滞成分と日常成分のスケールを揃えるため、それぞれの総量で規格化してから混合
        pk_total = emit_total.sum()
        rout_tot = np.zeros(len(taa_bins))
        rout_psd = np.zeros(len(taa_bins))
        rout_td = np.zeros(len(taa_bins))
        for bi, taa in enumerate(taa_bins):
            e, ep, et = routine_emission(taa, q, u_ev)
            rout_tot[bi] = e
            rout_psd[bi] = ep
            rout_td[bi] = et
        rsum = rout_tot.sum()
        if rsum > 0 and pk_total > 0:
            scale = routine_weight * pk_total / rsum
            emit_total += rout_tot * scale
            emit_psd += rout_psd * scale
            emit_td += rout_td * scale

    # --- 軽い平滑化(ビン集計のギザギザ除去) ---
    def smooth(y, w=5):
        k = np.ones(w) / w
        ext = np.concatenate([y[-w:], y, y[:w]])
        return np.convolve(ext, k, mode='same')[w:-w]
    emit_total = smooth(emit_total)
    emit_psd = smooth(emit_psd)
    emit_td = smooth(emit_td)

    return taa_bins, emit_total, emit_psd, emit_td


# ==========================================
# メイン
# ==========================================
if __name__ == "__main__":
    # ターミネーターに在庫が現れるTAA始点(全周)。細かくしてギザギザを抑える
    taa_starts = np.arange(0, 360, 0.5)

    # 掃き込み量のTAA依存モード: "uniform" か "backlog_taa100"
    SWEEP_MODE = "backlog_taa100"
    # 日常域成分の強さ(0で渋滞パケットのみ。1で渋滞成分と同程度の総量)
    ROUTINE_WEIGHT = 1.0

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

    peak_summary = {}
    for label, q in Q_LIST.items():
        taa_bins, e_tot, e_psd, e_td = build_emission_map(
            q, U_EV, taa_starts, SWEEP_MODE, ROUTINE_WEIGHT)
        c = Q_COLORS[label]

        axes[0].plot(taa_bins, e_tot, color=c, lw=2, label=label)
        axes[1].plot(taa_bins, e_psd, color=c, lw=2, label=label)
        axes[2].plot(taa_bins, e_td, color=c, lw=2, label=label)

        peak_summary[label] = {
            'total': taa_bins[np.argmax(e_tot)],
            'psd': taa_bins[np.argmax(e_psd)],
            'td': taa_bins[np.argmax(e_td)],
        }

    titles = ['総放出量 (PSD+TD)', 'PSD放出量', 'TD放出量']
    for ax, t in zip(axes, titles):
        ax.set_ylabel('放出量 [規格化]')
        ax.set_title(t)
        ax.axvline(180, color='gray', ls=':', alpha=0.6, label='遠日点')
        ax.grid(True, ls='--', alpha=0.4)
        ax.legend(fontsize=9)
    axes[-1].set_xlabel('TAA [deg]')
    axes[-1].set_xlim(0, 360)
    axes[-1].xaxis.set_major_locator(MultipleLocator(60))
    fig.suptitle('解析トイモデル: PSD強度(Q)による放出ピークのシフト', fontsize=13)
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("=== ピークTAA サマリー ===")
    print("=" * 60)
    for label, p in peak_summary.items():
        print(f"[{label}]  総放出ピーク: {p['total']:.0f}°,  "
              f"PSDピーク: {p['psd']:.0f}°,  TDピーク: {p['td']:.0f}°")
    print("=" * 60)
    print("\n注: このモデルは掃き込み量を全TAAで一定と仮定した最小構成。")
    print("    掃き込み量のTAA依存(近日点で大)を入れる場合は build_emission_map の重みを変更。")