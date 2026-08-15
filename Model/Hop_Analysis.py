# -*- coding: utf-8 -*-
"""
ホップ輸送の統合解析 (hop_transport_integrated.py)

1つ目の図で Matplotlib の Slider / RadioButtons を使用したインタラクティブ表示に対応。
第1コードで消えていた解析（正味変位分布、2D伝達マップ、TAA別平均変位）を復活させ、
第2コードの改良解析（外向きフラックス、放射圧検証 fit、夜側着地率）と完全統合しています。

--- 描画一覧 ---
[図1] 【インタラクティブ】TAA・過程・モデル選択プロット (Slider + Buttons)
[図2] 正味変位(LT差)の分布と統計情報 (旧図2)
[図3] ターミネーターを越えて夜側に着地した割合の TAA 依存 (補完図B)
[図4] 外向き正味フラックス F_out(LT) と その発散 -dF/dLT (補完図C)
[図5] 放出LT → 着地LT の2次元マップ (旧図4)
[図6] 放出LTごとの平均変位 — 放射圧の検証 [sin(h) fit] (補完図D)
[図7] 平均正味変位の TAA 依存 (旧図5)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider, RadioButtons

# ==========================================
# 設定
# ==========================================
BASE_DIR = r"./SimulationResult_202607"

MODELS = {
    "Q2.0 (Standard)": "HL_ParabolicHop_72x36_NoEq_DT100_0801_BD0.8_UG1.85_Q2.0_A2.0e+07_LT190k_15yr",
    "Q0.3 (Weak PSD)": "HL_ParabolicHop_72x36_NoEq_DT100_0731_BD0.8_UG1.85_Q0.3_A2.0e+07_LT190k_15yr",
}

MODEL_COLORS = {"Q2.0 (Standard)": "crimson", "Q0.3 (Weak PSD)": "royalblue"}
MODEL_LS = {"Q2.0 (Standard)": '-', "Q0.3 (Weak PSD)": '--'}

PROC_NAMES = ['PSD', 'TD', 'SWS', 'MMV']
PROC_COLORS = {'PSD': 'royalblue', 'TD': 'crimson', 'SWS': 'darkorange', 'MMV': 'seagreen'}
PROC_TO_PLOT = ['PSD', 'TD']

R_BODY_KM = 2439.7     # 水星半径 [km]
NIGHT_LT = (6.0, 18.0) # 夜側範囲 (LT < 6 または LT > 18)


# ==========================================
# 読み込み & 補助関数
# ==========================================
def load_hop(model_dir, target_year=None):
    """target_year に数値を指定するとその年のデータを、Noneなら最終結果を読み込む"""
    if target_year is not None:
        filename = f"hop_transport_hist_yr{target_year:02d}.npy"
    else:
        filename = "hop_transport_hist.npy"
        
    path = os.path.join(model_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"見つかりません: {path}")
    hop = np.load(path)
    if hop.ndim != 4:
        raise ValueError(f"形状が想定外です: {hop.shape} (期待: 4次元)")
    n_taa, n_proc, n_lon, n_lon2 = hop.shape
    if n_lon != n_lon2:
        raise ValueError(f"放出/着地のビン数が不一致: {n_lon} vs {n_lon2}")
    
    lt_centers = (np.arange(n_lon) + 0.5) * (24.0 / n_lon)
    taa_centers = (np.arange(n_taa) + 0.5) * (360.0 / n_taa)
    return hop, lt_centers, taa_centers


def wrap12(x):
    """ローカルタイムの時間差を -12..12 時間に畳む"""
    return (x + 12.0) % 24.0 - 12.0


def disp_matrix(ltc):
    """disp[i, j] = 放出ビンi から 着地ビンj への変位 [hour]"""
    return wrap12(ltc[None, :] - ltc[:, None])


def taa_index(taa_centers, target):
    return int(np.argmin(np.abs(taa_centers - target)))


def is_night(ltc):
    return (ltc < NIGHT_LT[0]) | (ltc > NIGHT_LT[1])


def compute_flux(sub, n):
    """境界ごとの正味通過量 (LTが増える向きを正)"""
    flux = np.zeros(n + 1)
    for i in range(n):
        row = sub[i]
        for j in range(n):
            v = row[j]
            if v <= 0 or i == j:
                continue
            if j > i:
                flux[i+1:j+1] += v
            else:
                flux[j+1:i+1] -= v
    return flux


# ==========================================
# [1] インタラクティブ TAA・過程切り替えプロット
# ==========================================
def plot_interactive_taa_proc(models_data):
    fig = plt.figure(figsize=(11, 6.5))
    
    # 描画領域と UI 配置
    ax = fig.add_axes([0.25, 0.20, 0.70, 0.72])
    ax_slider = fig.add_axes([0.25, 0.06, 0.65, 0.04])
    ax_radio_proc = fig.add_axes([0.02, 0.55, 0.18, 0.25])
    ax_radio_model = fig.add_axes([0.02, 0.20, 0.18, 0.25])
    
    first_label = list(models_data.keys())[0]
    _, ltc, tc = models_data[first_label]
    n_taa = len(tc)
    step_taa = 360.0 / n_taa
    
    slider = Slider(ax_slider, 'TAA [deg]', 0, 360 - step_taa, valinit=tc[0], valstep=step_taa)
    
    proc_options = ['PSD', 'TD', 'ALL']
    radio_proc = RadioButtons(ax_radio_proc, proc_options, active=0)
    
    model_options = ['Both'] + list(models_data.keys())
    radio_model = RadioButtons(ax_radio_model, model_options, active=0)
    
    def update(val=None):
        ax.clear()
        
        target_taa = slider.val
        ti = taa_index(tc, target_taa)
        actual_taa = tc[ti]
        
        sel_proc = radio_proc.value_selected
        sel_model = radio_model.value_selected
        
        target_models = list(models_data.keys()) if sel_model == 'Both' else [sel_model]
        
        for label in target_models:
            hop, ltc_m, _ = models_data[label]
            ls = MODEL_LS.get(label, '-')
            
            if sel_proc == 'ALL':
                sub = hop[ti, :, :, :].sum(axis=0)
                p_label = 'ALL'
                c = MODEL_COLORS.get(label, 'black') if len(target_models) > 1 else 'purple'
            else:
                pi = PROC_NAMES.index(sel_proc)
                sub = hop[ti, pi, :, :]
                p_label = sel_proc
                c = MODEL_COLORS.get(label, 'black') if len(target_models) > 1 else PROC_COLORS.get(sel_proc, 'black')
                
            if sub.sum() <= 0:
                continue
                
            birth = sub.sum(axis=1)
            land = sub.sum(axis=0)
            
            b_norm = birth / birth.sum() * 100 if birth.sum() > 0 else birth
            l_norm = land / land.sum() * 100 if land.sum() > 0 else land
            
            m_tag = f"[{label}] " if len(target_models) > 1 else ""
            ax.plot(ltc_m, b_norm, ':', color=c, lw=1.8, alpha=0.8,
                    label=f'{m_tag}{p_label} 放出')
            ax.plot(ltc_m, l_norm, ls, color=c, lw=2.4,
                    label=f'{m_tag}{p_label} 着地')
        
        for xv in (6, 12, 18):
            ax.axvline(xv, color='gray', ls=':', alpha=0.7)
        ax.axvspan(0, 6, color='navy', alpha=0.07)
        ax.axvspan(18, 24, color='navy', alpha=0.07)
        
        ax.set_xlim(0, 24)
        ax.set_xlabel('ローカルタイム LT [hour]')
        ax.set_ylabel('割合 [%]')
        ax.set_title(f'TAA = {actual_taa:.1f}° | 過程: {sel_proc} | 表示モデル: {sel_model}', fontsize=12)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.grid(True, ls='--', alpha=0.4)
        ax.legend(fontsize=8, loc='upper right')
        
        fig.canvas.draw_idle()
        
    slider.on_changed(update)
    radio_proc.on_clicked(update)
    radio_model.on_clicked(update)
    
    update()
    
    # ガベージコレクション対策で参照保持
    fig._widgets = [slider, radio_proc, radio_model]
    plt.show()


# ==========================================
# [2] 正味変位の分布と統計情報 (旧図2 復活)
# ==========================================
def plot_displacement_distribution(models_data):
    fig, ax = plt.subplots(figsize=(11, 6))
    print("\n" + "=" * 72)
    print("=== 正味変位の統計 (放出位置から着地位置まで) ===")
    print("=" * 72)
    print(f"{'run':<20}{'過程':>6}{'平均[hour]':>11}{'中央[hour]':>11}"
          f"{'RMS[hour]':>10}{'平均[km]':>11}")
    print("-" * 72)
    
    for label, (hop, ltc, tc) in models_data.items():
        disp = disp_matrix(ltc)
        bw = ltc[1] - ltc[0]
        for p in PROC_TO_PLOT:
            pi = PROC_NAMES.index(p)
            sub = hop[:, pi, :, :].sum(axis=0)
            if sub.sum() <= 0:
                continue
            
            d_bins = np.arange(-12, 13, bw)
            hist = np.zeros(len(d_bins) - 1)
            for k in range(len(d_bins) - 1):
                m = (disp >= d_bins[k]) & (disp < d_bins[k+1])
                hist[k] = sub[m].sum()
            d_c = (d_bins[:-1] + d_bins[1:]) / 2
            if hist.sum() <= 0:
                continue
                
            ls = MODEL_LS.get(label, '-')
            ax.plot(d_c, hist / hist.sum() * 100, ls, color=PROC_COLORS[p], lw=2.2,
                    label=f'{label} {p}' if len(models_data) > 1 else p)

            w = sub / sub.sum()
            mean_d = float((w * disp).sum())
            rms_d = float(np.sqrt((w * disp**2).sum()))
            cum = np.cumsum(hist) / hist.sum()
            med_d = float(np.interp(0.5, cum, d_c))
            km = np.deg2rad(mean_d * 15.0) * R_BODY_KM
            print(f"{label:<20}{p:>6}{mean_d:>11.2f}{med_d:>11.2f}{rms_d:>10.2f}{km:>11.0f}")
            
    print("-" * 72)
    print("  正の変位 = 太陽直下点方向(順行)、 負 = ターミネーター方向(逆行)")
    ax.axvline(0, color='gray', ls='--', alpha=0.7)
    ax.set_xlabel('正味変位 [hour]   (正=SSP方向, 負=ターミネーター方向)')
    ax.set_ylabel('割合 [%]')
    ax.set_title('放出から着地までの正味変位(時間換算)の分布', fontsize=13)
    ax.set_xlim(-4, 4)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.grid(True, ls='--', alpha=0.4)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


# ==========================================
# [3] 夜側への着地率の TAA 依存 (補完図B)
# ==========================================
def plot_night_fraction(models_data):
    fig, ax = plt.subplots(figsize=(11, 6))
    print("\n" + "=" * 74)
    print("=== 夜側(LT<6 または LT>18)へ着地した割合 ===")
    print("=" * 74)
    print(f"{'run':<20}{'過程':>6}{'全TAA平均':>12}{'TAA=0':>10}{'TAA=60':>10}"
          f"{'TAA=120':>10}{'TAA=180':>10}")
    print("-" * 74)
    for label, (hop, ltc, tc) in models_data.items():
        night = is_night(ltc)
        for p in PROC_TO_PLOT:
            pi = PROC_NAMES.index(p)
            frac = np.full(len(tc), np.nan)
            for t in range(len(tc)):
                land = hop[t, pi, :, :].sum(axis=0)
                s = land.sum()
                if s > 0:
                    frac[t] = land[night].sum() / s * 100
            ax.plot(tc, frac, MODEL_LS.get(label, '-'), color=PROC_COLORS[p],
                    lw=2.2, marker='o', ms=3.5, label=f'{label} {p}')
            
            allf = hop[:, pi, :, :].sum(axis=0).sum(axis=0)
            avg = allf[night].sum() / allf.sum() * 100 if allf.sum() > 0 else np.nan
            vals = [frac[taa_index(tc, x)] for x in (0, 60, 120, 180)]
            print(f"{label:<20}{p:>6}{avg:>11.1f}%" + "".join(f"{v:>9.1f}%" for v in vals))
    print("-" * 74)
    ax.axvline(180, color='gray', ls=':', alpha=0.6)
    ax.set_xlabel('TAA [deg]')
    ax.set_ylabel('夜側へ着地した割合 [%]')
    ax.set_title('ターミネーターを越えて夜側に着地した割合の TAA 依存', fontsize=13)
    ax.set_xlim(0, 360)
    ax.xaxis.set_major_locator(MultipleLocator(60))
    ax.grid(True, ls='--', alpha=0.4)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


# ==========================================
# [4] 改良版フラックス (外向きを正に統一) (補完図C)
# ==========================================
def plot_outward_flux(models_data):
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for label, (hop, ltc, tc) in models_data.items():
        n = len(ltc)
        dlt = ltc[1] - ltc[0]
        edges = np.concatenate([[ltc[0] - dlt/2], ltc + dlt/2])
        sgn = np.sign(edges - 12.0)
        sgn[sgn == 0] = 1.0
        
        for p in PROC_TO_PLOT:
            pi = PROC_NAMES.index(p)
            sub = hop[:, pi, :, :].sum(axis=0)
            if sub.sum() <= 0:
                continue
            flux = compute_flux(sub, n)
            ls = MODEL_LS.get(label, '-')
            c = PROC_COLORS[p]
            axes[0].plot(edges, flux * sgn, ls, color=c, lw=2.2, label=f'{label} {p}')
            axes[1].plot(ltc, -np.diff(flux), ls, color=c, lw=2.2, label=f'{label} {p}')

    axes[0].set_title('正味フラックス (太陽直下点から遠ざかる向きを正)\n正 = 夜側へ流れている', fontsize=12)
    axes[0].set_ylabel('外向き正味通過量 [atoms]')
    axes[1].set_title('フラックスの発散 -dF/dLT — 正の場所に原子が溜まる', fontsize=12)
    axes[1].set_ylabel('正味の増減 [atoms]')
    for ax in axes:
        ax.axhline(0, color='gray', ls='--', alpha=0.7)
        for xv in (6, 12, 18):
            ax.axvline(xv, color='gray', ls=':', alpha=0.6)
        ax.axvspan(0, 6, color='navy', alpha=0.07)
        ax.axvspan(18, 24, color='navy', alpha=0.07)
        ax.grid(True, ls='--', alpha=0.4)
        ax.legend(fontsize=9)
    axes[0].annotate('← 明け方側', xy=(3, 0), xytext=(3, 0), fontsize=9, color='navy')
    axes[0].annotate('夕方側 →', xy=(21, 0), xytext=(21, 0), fontsize=9, color='navy')
    axes[-1].set_xlabel('ローカルタイム LT [hour]')
    axes[-1].set_xlim(0, 24)
    axes[-1].xaxis.set_major_locator(MultipleLocator(2))
    plt.tight_layout()
    plt.show()


# ==========================================
# [5] 放出→着地の2次元マップ (旧図4 復活)
# ==========================================
def plot_transfer_map(models_data):
    for label, (hop, ltc, tc) in models_data.items():
        procs = [p for p in PROC_TO_PLOT if hop[:, PROC_NAMES.index(p), :, :].sum() > 0]
        if not procs:
            continue
        fig, axes = plt.subplots(1, len(procs), figsize=(6.5 * len(procs), 5.5), squeeze=False)
        for k, p in enumerate(procs):
            ax = axes[0][k]
            pi = PROC_NAMES.index(p)
            sub = hop[:, pi, :, :].sum(axis=0)
            vmax = sub.max()
            im = ax.pcolormesh(ltc, ltc, np.where(sub > 0, sub, np.nan).T,
                               norm=LogNorm(vmin=max(vmax*1e-5, 1e-30), vmax=vmax),
                               cmap='inferno', shading='auto')
            ax.plot([0, 24], [0, 24], color='cyan', ls='--', lw=1.2, label='変位ゼロ')
            for xv in (6, 12, 18):
                ax.axvline(xv, color='white', ls=':', alpha=0.4)
                ax.axhline(xv, color='white', ls=':', alpha=0.4)
            ax.set_xlabel('放出ローカルタイム [hour]')
            ax.set_ylabel('着地ローカルタイム [hour]')
            ax.set_title(f'{p}')
            ax.set_xlim(0, 24)
            ax.set_ylim(0, 24)
            ax.xaxis.set_major_locator(MultipleLocator(4))
            ax.yaxis.set_major_locator(MultipleLocator(4))
            ax.legend(fontsize=8, loc='upper left')
            plt.colorbar(im, ax=ax, label='atoms')
        fig.suptitle(f'{label} — 放出LT → 着地LT  (対角線より上=SSP方向へ移動)', fontsize=13)
        plt.tight_layout()
        plt.show()


# ==========================================
# [6] 放出LTごとの平均変位 — 放射圧の検証 (補完図D)
# ==========================================
def plot_displacement_vs_birth(models_data):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    print("\n" + "=" * 74)
    print("=== 放出LTごとの平均変位 — 放射圧の検証 ===")
    print("=" * 74)
    print("  放射圧は反太陽方向なので、時角hでの水平成分は b*sin(h) に比例する。")
    print("  → 正午でゼロ、明け方側で負、夕方側で正 の sin 形になるはず。")
    print()
    print(f"{'run':<20}{'過程':>6}{'sin適合 R^2':>13}{'振幅[hour]':>12}{'正午での値':>12}")
    print("-" * 74)

    for label, (hop, ltc, tc) in models_data.items():
        D = disp_matrix(ltc)
        for p in PROC_TO_PLOT:
            pi = PROC_NAMES.index(p)
            sub = hop[:, pi, :, :].sum(axis=0)
            if sub.sum() <= 0:
                continue
            w = sub.sum(axis=1)
            md = np.full(len(ltc), np.nan)
            ok = w > 0
            md[ok] = (sub[ok] * D[ok]).sum(axis=1) / w[ok]

            ls = MODEL_LS.get(label, '-')
            ax.plot(ltc, md, ls, color=PROC_COLORS[p], lw=2.2, marker='o', ms=3.5, label=f'{label} {p}')

            h = np.deg2rad((ltc - 12.0) * 15.0)
            basis = np.sin(h)
            m = ok & np.isfinite(md)
            if m.sum() > 5:
                amp = np.sum(w[m] * basis[m] * md[m]) / np.sum(w[m] * basis[m] ** 2)
                pred = amp * basis
                ss_res = np.sum(w[m] * (md[m] - pred[m]) ** 2)
                ss_tot = np.sum(w[m] * (md[m] - np.average(md[m], weights=w[m])) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                noon = float(np.interp(12.0, ltc, md))
                print(f"{label:<20}{p:>6}{r2:>13.3f}{amp:>12.3f}{noon:>12.3f}")
                if p == 'PSD' and label == list(models_data.keys())[0]:
                    ax.plot(ltc, pred, ':', color='black', lw=1.6, label=f'sin(h)適合 (振幅 {amp:.2f} h)')

    print("-" * 74)
    print("  R^2 が1に近いほど sin 形。正午での値がゼロに近いほど放射圧の描像と整合。")

    ax.axhline(0, color='gray', ls='--', alpha=0.7)
    ax.axvline(12, color='gray', ls=':', alpha=0.8)
    for xv in (6, 18):
        ax.axvline(xv, color='gray', ls=':', alpha=0.6)
    ax.axvspan(0, 6, color='navy', alpha=0.07)
    ax.axvspan(18, 24, color='navy', alpha=0.07)
    ax.set_xlabel('放出ローカルタイム LT [hour]')
    ax.set_ylabel('平均正味変位 [hour]  (正=夕方向き, 負=明け方向き)')
    ax.set_title('放出LTごとの平均変位 — 放射圧なら正午でゼロの sin 形になる', fontsize=13)
    ax.set_xlim(0, 24)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.grid(True, ls='--', alpha=0.4)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


# ==========================================
# [7] 平均正味変位の TAA 依存 (旧図5 復活)
# ==========================================
def plot_taa_dependence(models_data):
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, (hop, ltc, tc) in models_data.items():
        disp = disp_matrix(ltc)
        for p in PROC_TO_PLOT:
            pi = PROC_NAMES.index(p)
            md = np.full(len(tc), np.nan)
            for t in range(len(tc)):
                sub = hop[t, pi, :, :]
                s = sub.sum()
                if s > 0:
                    md[t] = float((sub * disp).sum() / s)
            ls = MODEL_LS.get(label, '-')
            ax.plot(tc, md, ls, color=PROC_COLORS[p], lw=2.2, marker='o', ms=4,
                    label=f'{label} {p}')
    ax.axhline(0, color='gray', ls='--', alpha=0.7)
    ax.axvline(180, color='gray', ls=':', alpha=0.6, label='遠日点')
    ax.set_xlabel('TAA [deg]')
    ax.set_ylabel('平均正味変位 [hour]')
    ax.set_title('正味変位の TAA 依存  (正=SSP方向, 負=ターミネーター方向)', fontsize=13)
    ax.set_xlim(0, 360)
    ax.xaxis.set_major_locator(MultipleLocator(60))
    ax.grid(True, ls='--', alpha=0.4)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


# ==========================================
# メイン
# ==========================================
def main():
    models_data = {}
    for label, subdir in MODELS.items():
        try:
            #models_data[label] = load_hop(os.path.join(BASE_DIR, subdir))
            models_data[label] = load_hop(os.path.join(BASE_DIR, subdir), target_year=1)
            hop = models_data[label][0]
            print(f"[読込] {label}: shape={hop.shape}, 総量={hop.sum():.3e} atoms")
            for pi, pn in enumerate(PROC_NAMES):
                s = hop[:, pi, :, :].sum()
                if s > 0:
                    print(f"        {pn:>4}: {s:.3e} atoms ({s/hop.sum()*100:.1f}%)")
        except Exception as e:
            print(f"[エラー] {label}: {e}")
            
    if not models_data:
        print("読み込めた run がありません。")
        return

    # 全プロットの実行
    plot_interactive_taa_proc(models_data)  # [図1] インタラクティブ (Slider + Buttons)
    
    plot_displacement_distribution(models_data) # [図2] 復活
    plot_night_fraction(models_data)            # [図3]
    plot_outward_flux(models_data)               # [図4]
    plot_transfer_map(models_data)              # [図5] 復活
    plot_displacement_vs_birth(models_data)     # [図6]
    plot_taa_dependence(models_data)           # [図7] 復活


if __name__ == "__main__":
    main()