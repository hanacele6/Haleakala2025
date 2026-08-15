# -*- coding: utf-8 -*-
"""
放出フラックスと「大気(カラム密度)寄与」のTAAピークを、
Dawn / Dusk / 全体 に分けて Q比較で数値化するスクリプト。

--- 核心 ---
カラム密度のピークは:
  ・全体(Dawn+Dusk)     → 常に遠日点(≈180°)、Qで動かない
  ・Dawn(明け方)側       → Qを下げると 180°→165° へ左シフト
  ・Dusk(夕方)側         → ほぼ動かない(U,Q感度が低い)
つまりピーク移動は「放出の総量」ではなく「Dawn/Dusk配分」の問題。
Dawnが下がるとDuskがやや上がる補償関係も確認する。

放出量 × 電離寿命(τ_ion = T1AU·AU²) を「大気寄与」の指標とする。
budget_statistics_per_taa.csv のみ使用。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401
from matplotlib.ticker import MultipleLocator
import os

# ==========================================
# 設定
# ==========================================
BASE_DIR = r"./SimulationResult_202607"

MODELS = {
    "Q2.0 (Standard)": "ParabolicHop_72x36_NoEq_DT100_0716_BD0.8_UG1.85_Q2.0_A2.0e+07_LT190k_15yr_test2",
    "Q0.3 (Weak PSD)": "ParabolicHop_72x36_NoEq_DT100_0716_BD0.8_UG1.85_Q0.3_A2.0e+07_LT190k_15yr_test2",
}
COLORS = {"Q2.0 (Standard)": "crimson", "Q0.3 (Weak PSD)": "steelblue"}

# 見たい成分: 'TD' / 'PSD' / 'Total'
COMPONENT = "Total"

# --- 物理定数 ---
T_ORBIT_SEC = 87.969 * 86400
ECC = 0.205630
A_AU = 0.387098
T1AU = 190000.0
RM_CM = 2.440e8

SMOOTH_WINDOW_DEG = 3


# ==========================================
# ヘルパー
# ==========================================
def au_of_taa(taa_deg):
    rad = np.deg2rad(taa_deg)
    return A_AU * (1 - ECC**2) / (1 + ECC * np.cos(rad))


def dwell_time_per_deg(taa_deg):
    rad = np.deg2rad(taa_deg)
    return (T_ORBIT_SEC / 360.0) * ((1.0 - ECC**2)**1.5 / (1.0 + ECC * np.cos(rad))**2)


def circular_smooth(y, w):
    if w <= 0:
        return y
    w = int(w)
    k = np.ones(w) / w
    ext = np.concatenate([y[-w:], y, y[:w]])
    return np.convolve(ext, k, mode='same')[w:-w]


def load_budget(model_dir):
    path = os.path.join(model_dir, "budget_statistics_per_taa.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"見つかりません: {path}")
    return pd.read_csv(path).sort_values('TAA_Bin').reset_index(drop=True)


def get_gen(df, component, region):
    """component: TD/PSD/Total, region: Dawn/Dusk/Total を Gen列から取り出す"""
    if region == "Total":
        if component == "Total":
            return df['Gen_TD_Dawn'] + df['Gen_TD_Dusk'] + df['Gen_PSD_Dawn'] + df['Gen_PSD_Dusk']
        else:
            return df[f'Gen_{component}_Dawn'] + df[f'Gen_{component}_Dusk']
    else:
        if component == "Total":
            return df[f'Gen_TD_{region}'] + df[f'Gen_PSD_{region}']
        else:
            return df[f'Gen_{component}_{region}']


def peak_taa(taa, y, smooth_deg):
    ys = circular_smooth(np.asarray(y, dtype=float), smooth_deg)
    return taa[np.nanargmax(ys)], ys


# ==========================================
# 本体
# ==========================================
def analyze(models, component, smooth_deg=3):
    global_area_cm2 = 4 * np.pi * (RM_CM ** 2)
    regions = ["Dawn", "Dusk", "Total"]
    region_jp = {"Dawn": "明け方(Dawn)", "Dusk": "夕方(Dusk)", "Total": "全体(Dawn+Dusk)"}

    # 上段=放出フラックス, 下段=電離寿命重み大気寄与。列=Dawn/Dusk/全体
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
    peaks = {}

    for label, subdir in models.items():
        full = os.path.join(BASE_DIR, subdir)
        try:
            df = load_budget(full)
        except Exception as e:
            print(f"[エラー] {label}: {e}")
            continue

        taa = df['TAA_Bin'].values
        au = au_of_taa(taa)
        tau_ion = T1AU * au**2
        dwell = dwell_time_per_deg(taa)
        conv = global_area_cm2 * dwell
        c = COLORS.get(label)

        for col, region in enumerate(regions):
            gen = get_gen(df, component, region).values

            # 上段: 放出フラックス
            flux = gen / conv
            pk_f, ys_f = peak_taa(taa, flux, smooth_deg)
            axes[0, col].plot(taa, ys_f, '-', color=c, lw=2, label=f'{label} (pk={pk_f:.0f}°)')
            peaks.setdefault(f'放出_{region}', {})[label] = pk_f

            # 下段: 大気寄与 = 放出 × 電離寿命
            contrib = gen * tau_ion
            pk_c, ys_c = peak_taa(taa, contrib, smooth_deg)
            axes[1, col].plot(taa, ys_c, '-', color=c, lw=2, label=f'{label} (pk={pk_c:.0f}°)')
            peaks.setdefault(f'寄与_{region}', {})[label] = pk_c

    for col, region in enumerate(regions):
        axes[0, col].set_title(f'{region_jp[region]} — {component} 放出フラックス')
        axes[1, col].set_title(f'{region_jp[region]} — {component} 大気寄与(放出×寿命)')
        for row in [0, 1]:
            ax = axes[row, col]
            ax.set_yscale('log')
            ax.axvline(180, color='gray', ls=':', alpha=0.6)
            ax.axvline(165, color='green', ls='--', alpha=0.5)
            ax.grid(True, which='both', ls='--', alpha=0.4)
            ax.legend(fontsize=8)
            ax.set_xlim(0, 360)
            ax.xaxis.set_major_locator(MultipleLocator(60))
        axes[1, col].set_xlabel('TAA [deg]')
    axes[0, 0].set_ylabel('放出フラックス [atoms/cm²/s]')
    axes[1, 0].set_ylabel('大気寄与 [相対]')

    fig.suptitle(f'{component}: Dawn / Dusk / 全体 の TAA分布 (Q比較)\n'
                 f'緑破線=Dawn柱密度ピーク実測(165°)  灰点線=遠日点(180°)', fontsize=13)
    plt.tight_layout()
    plt.show()

    # ==========================================
    # ピークTAA サマリー
    # ==========================================
    labels = list(models.keys())
    print("\n" + "=" * 80)
    print(f"=== ピーク TAA サマリー ({component}, 平滑化後 argmax) ===")
    print("=" * 80)
    print(f"{'量':<14}", end='')
    for label in labels:
        print(f"{label:<20}", end='')
    print("シフト量Δ")
    print("-" * 80)
    for qty, dd in peaks.items():
        print(f"{qty:<14}", end='')
        vals = [dd.get(l, np.nan) for l in labels]
        for v in vals:
            print(f"{v:<20.0f}", end='')
        if len(vals) == 2 and not any(np.isnan(vals)):
            print(f"{vals[1]-vals[0]:+.0f}°")
        else:
            print("")
    print("=" * 80)
    print("\n[期待される結果]")
    print("  ・寄与_Dawn : Qを下げると左シフト(Δ<0, 例 180→165)")
    print("  ・寄与_Dusk : ほぼ動かない(Δ≈0)")
    print("  ・寄与_Total: ほぼ動かない(Dawnのシフトが全体では埋もれる)")


# ==========================================
# 補償関係の確認: Dawn量が下がるとDusk量が上がるか
# ==========================================
def check_compensation(models, component, smooth_deg=3):
    """各Qで Dawn/Dusk の大気寄与の総量(TAA積分)を出し、
    Dawnが下がった分Duskが上がる補償関係を数値で確認。"""
    print("\n" + "=" * 80)
    print(f"=== Dawn/Dusk 総量の補償関係 ({component}, 大気寄与のTAA積分) ===")
    print("=" * 80)
    results = {}
    for label, subdir in models.items():
        try:
            df = load_budget(os.path.join(BASE_DIR, subdir))
        except Exception as e:
            print(f"[エラー] {label}: {e}")
            continue
        taa = df['TAA_Bin'].values
        tau_ion = T1AU * au_of_taa(taa)**2
        dawn = (get_gen(df, component, "Dawn").values * tau_ion).sum()
        dusk = (get_gen(df, component, "Dusk").values * tau_ion).sum()
        results[label] = (dawn, dusk, dawn + dusk)
        print(f"  [{label}]  Dawn={dawn:.3e}  Dusk={dusk:.3e}  合計={dawn+dusk:.3e}  "
              f"Dawn/Dusk比={dawn/dusk:.3f}")

    labels = list(results.keys())
    if len(labels) == 2:
        d0, d1 = results[labels[0]], results[labels[1]]
        print("-" * 80)
        print(f"  {labels[0]} → {labels[1]} の変化:")
        print(f"    Dawn : {d1[0]-d0[0]:+.3e} ({(d1[0]/d0[0]-1)*100:+.1f}%)")
        print(f"    Dusk : {d1[1]-d0[1]:+.3e} ({(d1[1]/d0[1]-1)*100:+.1f}%)")
        print(f"    合計 : {d1[2]-d0[2]:+.3e} ({(d1[2]/d0[2]-1)*100:+.1f}%)")
        if (d1[0] < d0[0]) and (d1[1] > d0[1]):
            print("    → Dawnが減りDuskが増える『補償関係』を確認 ✓")
        elif abs(d1[2]/d0[2]-1) < 0.05:
            print("    → 合計はほぼ保存(配分だけが変化)")
    print("=" * 80)


if __name__ == "__main__":
    analyze(MODELS, COMPONENT, smooth_deg=SMOOTH_WINDOW_DEG)
    check_compensation(MODELS, COMPONENT, smooth_deg=SMOOTH_WINDOW_DEG)