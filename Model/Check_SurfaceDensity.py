# -*- coding: utf-8 -*-
"""
放出量の Q 依存 — 記録された Gen_* を使う版

budget_statistics_per_taa.csv には、シミュレーション本体が実際にカウントした
放出数 Gen_PSD / Gen_TD / Gen_SWS / Gen_MMV が TAA ビンごとに入っている。
表面密度からの逆算 (StockVsQ.py) は DT キャップの影響を受けて過小評価になるため、
放出量を論じるときは必ずこちらを使うこと。

出力
  1. 放出量の TAA 依存 (Q ごと)
  2. 放出量の Q 依存 — P ∝ Q なら光子律速、頭打ちなら供給律速
  3. PSD / TD の内訳の Q 依存
  4. 放出量ピーク TAA の Q 依存 (柱密度ピークと比較するため)
"""

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401
from matplotlib.ticker import MultipleLocator

# ==========================================
# 設定
# ==========================================
BASE_DIR = r"./SimulationResult_202607"

MODELS = {
    0.1:  "New_ParabolicHop_72x36_NoEq_DT100_0804_BD0.8_UG1.85_Q0.1_A2.0e+07_LT190k_15yr",
    0.27: "Def2_ParabolicHop_72x36_NoEq_DT100_0729_BD0.8_UG1.85_Q0.3_A2.0e+07_LT190k_15yr",
    1.0:  "ParabolicHop_72x36_NoEq_DT100_0710_BD0.8_UG1.85_Q1.0_A2.0e+07_LT190k_15yr",
    2.0:  "Def2_ParabolicHop_72x36_NoEq_DT100_0729_BD0.8_UG1.85_Q2.0_A2.0e+07_LT190k_15yr",
    3.0:  "ParabolicHop_72x36_NoEq_DT100_0710_BD0.8_UG1.85_Q3.0_A2.0e+07_LT190k_15yr",
    5.0:  "New_ParabolicHop_72x36_NoEq_DT100_0804_BD0.8_UG1.85_Q5.0_A2.0e+07_LT190k_15yr",
    10.0: "New_ParabolicHop_72x36_NoEq_DT100_0804_BD0.8_UG1.85_Q10.0_A2.0e+07_LT190k_15yr",
}

CSV_NAME = "budget_statistics_per_taa.csv"

# Dawn / Dusk 別の列がある場合に使う ('total' / 'dawn' / 'dusk')
SIDE = 'dawn'

# ピーク探索の窓とフィット範囲 (PeakTAA_vs_Q.py と揃える)
PEAK_WINDOW = (90.0, 270.0)
FIT_HALFWIDTH_DEG = 50.0

SHOW_PLOT = True
SAVE_PNG = True

PROC_COLORS = {'PSD': 'royalblue', 'TD': 'crimson',
               'SWS': 'darkorange', 'MMV': 'seagreen'}


# ==========================================
# 読み込み
# ==========================================
def gen_columns(df, side):
    """side に応じた Gen_* の列名を返す。無ければ total にフォールバック。"""
    suf = {'total': '', 'dawn': '_Dawn', 'dusk': '_Dusk'}[side]
    cols = {}
    for p in ('PSD', 'TD', 'SWS', 'MMV'):
        c = f'Gen_{p}{suf}'
        if c not in df.columns:
            c = f'Gen_{p}'
        cols[p] = c if c in df.columns else None
    return cols


def load_budget(subdir, side=SIDE):
    path = os.path.join(BASE_DIR, subdir, CSV_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path).sort_values('TAA_Bin')
    taa = df['TAA_Bin'].to_numpy(dtype=float)

    cols = gen_columns(df, side)
    gen = {}
    for p, c in cols.items():
        gen[p] = df[c].to_numpy(dtype=float) if c else np.zeros_like(taa)
    gen['Total'] = sum(gen[p] for p in ('PSD', 'TD', 'SWS', 'MMV'))
    return taa, gen


# ==========================================
# ピーク検出 (PeakTAA_vs_Q.py と同じ方式)
# ==========================================
def _angdiff(a, b):
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def window_mask(taa, window):
    if window is None:
        return np.ones(len(taa), dtype=bool)
    lo, hi = window
    return (taa >= lo) & (taa <= hi) if lo <= hi else (taa >= lo) | (taa <= hi)


def peak_taa(taa, y, window=PEAK_WINDOW, half=FIT_HALFWIDTH_DEG):
    m = window_mask(taa, window)
    idx = np.where(m)[0]
    if len(idx) == 0 or not np.isfinite(y).any():
        return np.nan
    p0 = float(taa[idx[np.argmax(y[idx])]])
    d = _angdiff(taa, p0)
    sel = np.abs(d) <= half
    if sel.sum() < 4:
        return p0
    c = np.polyfit(d[sel], y[sel], 2)
    if c[0] >= 0:
        return p0
    return float((p0 - c[1] / (2 * c[0])) % 360.0)


# ==========================================
# メイン
# ==========================================
def main():
    results = {}
    for q, sub in sorted(MODELS.items()):
        if not sub:
            continue
        try:
            results[q] = load_budget(sub)
            print(f"[Q={q}] 読み込み OK  ({sub})")
        except Exception as e:
            print(f"[Q={q}] エラー: {e}")

    if not results:
        print("有効なデータがありませんでした。")
        return

    qs = np.array(sorted(results.keys()))

    tot = np.array([results[q][1]['Total'].sum() for q in qs])
    psd = np.array([results[q][1]['PSD'].sum() for q in qs])
    td = np.array([results[q][1]['TD'].sum() for q in qs])
    pk = np.array([peak_taa(results[q][0], results[q][1]['Total']) for q in qs])
    pk_psd = np.array([peak_taa(results[q][0], results[q][1]['PSD']) for q in qs])
    pk_td = np.array([peak_taa(results[q][0], results[q][1]['TD']) for q in qs])

    print("\n" + "=" * 88)
    print(f"{'Q':>6} {'総放出量':>12} {'総/Q(規格化)':>13} {'PSD割合':>8} "
          f"{'ピークTAA':>10} {'PSDピーク':>10} {'TDピーク':>9}")
    print("-" * 88)
    for i, q in enumerate(qs):
        norm = (tot[i] / q) / (tot[0] / qs[0])
        frac = psd[i] / tot[i] * 100 if tot[i] else np.nan
        print(f"{q:>6} {tot[i]:>12.4e} {norm:>13.3f} {frac:>7.1f}% "
              f"{pk[i]:>10.1f} {pk_psd[i]:>10.1f} {pk_td[i]:>9.1f}")
    print("=" * 88)
    print("総/Q が 1 のまま → 放出量は Q に比例 (光子フラックス律速)")
    print("総/Q が減少      → 放出量が Q に対して鈍る (供給律速)")

    with open('emission_vs_Q.csv', 'w', encoding='utf-8') as f:
        f.write('Q,total,PSD,TD,peak_total,peak_PSD,peak_TD\n')
        for i, q in enumerate(qs):
            f.write(f"{q},{tot[i]:.6e},{psd[i]:.6e},{td[i]:.6e},"
                    f"{pk[i]:.2f},{pk_psd[i]:.2f},{pk_td[i]:.2f}\n")
    print("  -> emission_vs_Q.csv")

    # ---------- 図 ----------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    a1, a2, a3, a4 = axes.ravel()

    cmap = plt.get_cmap('viridis')
    lo, hi = np.log10(qs.min()), np.log10(qs.max())
    col = {q: cmap((np.log10(q) - lo) / (hi - lo) if hi > lo else 0.5) for q in qs}

    for q in qs:
        taa, gen = results[q]
        a1.plot(taa, gen['Total'], '-', color=col[q], lw=2.0, label=f'Q={q}')
    a1.set_xlabel('TAA [deg]')
    a1.set_ylabel('放出量 [atoms]')
    a1.set_title('総放出量の TAA 依存 (記録値)', fontsize=12)
    a1.set_xlim(0, 360)
    a1.xaxis.set_major_locator(MultipleLocator(60))
    a1.axvline(180, color='gray', ls=':', alpha=0.6)
    a1.grid(True, ls='--', alpha=0.4)
    a1.legend(fontsize=8, ncol=2)

    a2.plot(qs, tot / tot[0], 'o-', color='crimson', lw=2.0, ms=7, label='総放出量')
    a2.plot(qs, psd / psd[0], 's--', color=PROC_COLORS['PSD'], lw=1.6, ms=6, label='PSD')
    a2.plot(qs, td / td[0], '^--', color=PROC_COLORS['TD'], lw=1.6, ms=6, label='TD')
    a2.plot(qs, qs / qs[0], ':', color='gray', lw=1.5, label='∝ Q の場合')
    a2.set_xscale('log')
    a2.set_yscale('log')
    a2.set_xlabel('Q  [×10⁻²⁰ cm²]')
    a2.set_ylabel('Q最小の値で規格化')
    a2.set_title('放出量の Q 依存', fontsize=12)
    a2.grid(True, which='both', ls='--', alpha=0.4)
    a2.legend(fontsize=9)

    for p in ('PSD', 'TD', 'SWS', 'MMV'):
        v = np.array([results[q][1][p].sum() for q in qs])
        s = np.array([results[q][1]['Total'].sum() for q in qs])
        a3.plot(qs, v / s * 100, 'o-', color=PROC_COLORS[p], lw=2.0, ms=6, label=p)
    a3.set_xscale('log')
    a3.set_xlabel('Q  [×10⁻²⁰ cm²]')
    a3.set_ylabel('放出量に占める割合 [%]')
    a3.set_title('過程別の寄与', fontsize=12)
    a3.grid(True, which='both', ls='--', alpha=0.4)
    a3.legend(fontsize=9)

    a4.plot(qs, pk, 'o-', color='black', lw=2.0, ms=7, label='総放出量のピーク')
    a4.plot(qs, pk_psd, 's--', color=PROC_COLORS['PSD'], lw=1.6, ms=6, label='PSD のピーク')
    a4.plot(qs, pk_td, '^--', color=PROC_COLORS['TD'], lw=1.6, ms=6, label='TD のピーク')
    a4.axhline(180, color='gray', ls=':', lw=1.2, label='遠日点')
    a4.set_xscale('log')
    a4.set_xlabel('Q  [×10⁻²⁰ cm²]')
    a4.set_ylabel('放出量ピークの TAA [deg]')
    a4.set_title('放出量ピークの Q 依存\n(柱密度ピークと比べる)', fontsize=12)
    a4.grid(True, which='both', ls='--', alpha=0.4)
    a4.legend(fontsize=9)

    plt.tight_layout()
    if SAVE_PNG:
        fig.savefig('emission_vs_Q.png', dpi=140, bbox_inches='tight')
        print("  -> emission_vs_Q.png")
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    main()