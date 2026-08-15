# -*- coding: utf-8 -*-
"""実測ピークの確認 — 何のピークが何度なのかを一覧にする"""
import numpy as np, pandas as pd, os

BASE_DIR = r"./SimulationResult_202607"
MODELS = {
    "Q2.0": "Def2_ParabolicHop_72x36_NoEq_DT100_0729_BD0.8_UG1.85_Q2.0_A2.0e+07_LT190k_15yr",
    "Q0.3": "Def2_ParabolicHop_72x36_NoEq_DT100_0729_BD0.8_UG1.85_Q0.3_A2.0e+07_LT190k_15yr",
}
SIDE = "Dawn"
SMOOTH = 5
PEAK_RANGE = (60, 300)

def smooth(y, w):
    y = np.asarray(y, float)
    if w <= 1: return y
    k = np.ones(w)/w
    return np.convolve(np.r_[y[-w:], y, y[:w]], k, 'same')[w:-w]

def peak(taa, y, rng=PEAK_RANGE):
    m = (taa >= rng[0]) & (taa <= rng[1])
    return taa[m][int(np.nanargmax(smooth(y, SMOOTH)[m]))]

for label, sub in MODELS.items():
    path = os.path.join(BASE_DIR, sub, "band_statistics_per_taa.csv")
    if not os.path.exists(path):
        print(f"[skip] {label}: {path} がありません"); continue
    df = pd.read_csv(path)
    sc = df['Step_Count'].replace(0, np.nan)
    df['Stock'] = (df[f'Stock_{SIDE}'] / sc).fillna(0.0)
    g = lambda c: df.pivot(index='TAA_Bin', columns='Band_Index', values=c).sort_index().values
    taa = np.sort(df['TAA_Bin'].unique())
    bands = df[['Band_Index','EffCos_Lo','EffCos_Hi']].drop_duplicates().sort_values('Band_Index')
    lo, hi = bands['EffCos_Lo'].values, bands['EffCos_Hi'].values

    gen_td, gen_psd, stock = g(f'Gen_TD_{SIDE}'), g(f'Gen_PSD_{SIDE}'), g('Stock')

    print(f"\n=== {label} ({SIDE}側) ===")
    print(f"{'量':<28}{'全バンド':>10}{'0.55-1.01':>12}{'0.70-0.85':>12}")
    print("-"*62)
    wins = [("全バンド", np.ones(len(lo), bool)),
            ("0.55-1.01", (lo>=0.55-1e-9)&(hi<=1.01+1e-9)),
            ("0.70-0.85", (lo>=0.70-1e-9)&(hi<=0.85+1e-9))]
    for name, arr in [("Gen_TD (TD放出)", gen_td),
                      ("Gen_PSD (PSD放出)", gen_psd),
                      ("Gen_TD+PSD (合計放出)", gen_td+gen_psd),
                      ("Stock (在庫)", stock)]:
        row = f"{name:<28}"
        for _, sel in wins:
            row += f"{peak(taa, arr[:, sel].sum(axis=1)):>10.0f}°" if sel.any() else f"{'--':>11}"
        print(row)
print("\n注) 窓 0.70-0.85 は band_statistics のバンド境界に一致しないため")
print("    該当バンドが無ければ '--' になります。")