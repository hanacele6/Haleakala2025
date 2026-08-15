# -*- coding: utf-8 -*-
"""
TD/PSD主力境界 eff_cos* の定量化 (ピーク左シフトの主犯の特定)

--- 背景 ---
Qを下げる(PSDを弱める)と、Dawn側の放出ピークが左(近日点寄り: 180°→165°付近)にシフトする。
この原因を「夜側から供給される在庫をTDが大量消費し始めるタイミング(eff_cos)が早まるため」
と仮定し、各TAAにおいて TD放出量 > PSD放出量 となる境界となる eff_cos* を算出する。

--- このスクリプトで測るもの ---
各TAAビンにおいて、eff_cosバンドごとの Gen_TD と Gen_PSD を比較。
Gen_TD - Gen_PSD が負(PSD優位)から正(TD優位)に反転する交差点(eff_cos*)を線形補間で取得。
Q2.0(強PSD) と Q0.3(弱PSD) で、この主力境界がどう推移するかを比較する。
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

SIDE = "Dawn"
SMOOTH_WINDOW_DEG = 3

# ==========================================
# 読み込み処理 (既存ロジックを流用)
# ==========================================
def load_stock_and_gen(model_dir, side):
    path = os.path.join(model_dir, "band_statistics_per_taa.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"見つかりません: {path}")
    df = pd.read_csv(path)

    stock_col = f"Stock_{side}" if f"Stock_{side}" in df.columns else f"Sweep_{side}"
    if 'Step_Count' in df.columns:
        sc = df['Step_Count'].replace(0, np.nan)
        df['Stock'] = df[stock_col] / sc
    else:
        df['Stock'] = df[stock_col]
    df['Stock'] = df['Stock'].fillna(0.0)
    df['Gen_TD'] = df[f'Gen_TD_{side}']
    df['Gen_PSD'] = df[f'Gen_PSD_{side}']

    bands = df[['Band_Index', 'EffCos_Lo', 'EffCos_Hi']].drop_duplicates().sort_values('Band_Index')
    bc = ((bands['EffCos_Lo'] + bands['EffCos_Hi']) / 2.0).values

    def grid(col):
        return df.pivot(index='TAA_Bin', columns='Band_Index', values=col).sort_index().values

    return {
        'stock': grid('Stock'),
        'gen_td': grid('Gen_TD'),
        'gen_psd': grid('Gen_PSD'),
        'band_centers': bc,
        'taa': np.sort(df['TAA_Bin'].unique()),
        'n_bands': len(bc),
    }

def circular_smooth(y, w):
    if w <= 0:
        return y
    w = int(w)
    k = np.ones(w) / w
    ext = np.concatenate([y[-w:], y, y[:w]])
    return np.convolve(ext, k, mode='same')[w:-w]

# ==========================================
# 解析：主力境界 eff_cos* の抽出
# ==========================================
def analyze_crossover_eff_cos(models, side, smooth_deg=3):
    fig, ax = plt.subplots(figsize=(11, 6))

    print("\n" + "=" * 70)
    print(f"=== TD/PSD 主力境界 eff_cos* の推移 ({side}側) ===")
    print("=" * 70)

    for label, subdir in models.items():
        try:
            d = load_stock_and_gen(os.path.join(BASE_DIR, subdir), side)
        except Exception as e:
            print(f"[エラー] {label}: {e}")
            continue
            
        bc = d['band_centers']
        taa = d['taa']
        c = COLORS.get(label)

        cross_eff_cos = np.full(len(taa), np.nan)
        
        for ti in range(len(taa)):
            td_vals = d['gen_td'][ti, :]
            psd_vals = d['gen_psd'][ti, :]
            
            # バンドを下(ターミネーター際)から走査し、TDがPSDを追い抜くポイントを探す
            for b in range(len(bc) - 1):
                # 在庫ゼロ等で放出が全くないバンドはスキップ
                if (td_vals[b] + psd_vals[b]) == 0 and (td_vals[b+1] + psd_vals[b+1]) == 0:
                    continue
                
                y0 = td_vals[b] - psd_vals[b]
                y1 = td_vals[b+1] - psd_vals[b+1]
                
                # y0が負(PSD優位)で、y1が正(TD優位)に切り替わった場合、線形補間で交差点を求める
                if y0 < 0 and y1 >= 0:
                    x0, x1 = bc[b], bc[b+1]
                    eff_cos_star = x0 - y0 * (x1 - x0) / (y1 - y0)
                    cross_eff_cos[ti] = eff_cos_star
                    break

        # TAA方向への平滑化
        cross_eff_cos_sm = circular_smooth(cross_eff_cos, smooth_deg)
        
        ax.plot(taa, cross_eff_cos_sm, '-', color=c, lw=2.5, label=f'{label}')
        
        # コンソール出力用に、160°〜180°付近の平均主力境界を計算
        target_mask = (taa >= 160) & (taa <= 180)
        mean_eff_cos_target = np.nanmean(cross_eff_cos_sm[target_mask])
        print(f"[{label}]")
        print(f"  TAA 160-180°での平均主力境界 eff_cos* : {mean_eff_cos_target:.3f}")

    # TD単独ピーク位置(162°)と遠日点(180°)を強調
    ax.axvline(180, color='gray', ls=':', alpha=0.8, label='遠日点 (180°)')
    ax.axvline(162.5, color='green', ls='--', alpha=0.8, label='TD単独ピーク位置 (162.5°)')
    ax.axvspan(160, 180, color='yellow', alpha=0.15, label='左シフト注目領域 (TAA 160-180°)')

    ax.set_xlabel('TAA [deg]', fontsize=12)
    ax.set_ylabel('主力境界 $eff\_cos^*$ (TD放出 > PSD放出 となる閾値)', fontsize=12)
    ax.set_title(f'TDとPSDの放出率逆転境界のTAA推移 ({side}側)\n'
                 'Qが低いほど、より低いeff_cos(ターミネーター寄り)からTDが主役に躍り出る', fontsize=14)
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(MultipleLocator(60))
    ax.grid(True, ls='--', alpha=0.6)
    ax.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    plt.show()
    print("=" * 70)

if __name__ == "__main__":
    analyze_crossover_eff_cos(MODELS, SIDE, smooth_deg=SMOOTH_WINDOW_DEG)