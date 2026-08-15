# -*- coding: utf-8 -*-
"""
束縛エネルギービン別の表面在庫の時間発展

--- 目的 ---
全体の表面在庫は年々増え続けているのに、大気(外気圏)はほぼ定常になっている。
この一見の矛盾を、束縛エネルギー(U)ビン別に分解して確認する。

想定される答え:
  地下からの拡散供給は DIFFUSION_U_MODE='gaussian' により U=1.4〜2.7 eV に分配される。
  このうち深いビン(U が大きい)は水星の表面温度では熱脱離がほぼ起きないため、
  溜まる一方で放出には寄与しない。
  → 総量は増え続けるが、放出に関わる浅いビンは定常に近い、という描像。

--- 使うデータ ---
surface_density_t*.npy  … shape (N_LON, N_LAT, N_BINS)
  第3軸が束縛エネルギービン。既存の出力にそのまま入っているので再run不要。
  (CSV の "Band" は eff_cos の帯であって U ビンではないので注意)

--- 出力 ---
[1] ビン別の総在庫の時間発展 (対数)
[2] ビン別の年増加率
[3] 全在庫に占める各ビンの割合の変化
"""

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401
from matplotlib.ticker import MultipleLocator
import os
import glob
import re

# ==========================================
# 設定
# ==========================================
BASE_DIR = r"./SimulationResult_202607"
RUN_NAME = "ParabolicHop_72x36_NoEq_DT100_0731_BD0.8_UG1.85_Q0.3_A2.0e+07_LT190k_25yr_FULL"

N_LON, N_LAT = 72, 36
R_BODY_M = 2.440e6
MERCURY_YEAR_SEC = 87.969 * 86400

# 本体の設定に合わせる (SIMULATION_SETTINGS)
U_MIN, U_MAX = 1.4, 2.7
N_U_BINS = 10
U_GAUSSIAN_MU, U_GAUSSIAN_SIGMA = 1.85, 0.25

# 熱脱離が「実質的に効く」とみなす U の上限 [eV]
# (水星の昼側最高温度でも放出時定数が長すぎるビンを不活性とみなす目安)
U_ACTIVE_MAX = 2.0

MAX_FILES = 400   # 読み込む最大ファイル数(多すぎる場合は間引く)


# ==========================================
# ヘルパー
# ==========================================
def cell_areas_m2(n_lon, n_lat, r_m):
    lat_edges = np.linspace(-np.pi/2, np.pi/2, n_lat + 1)
    dlon = 2.0 * np.pi / n_lon
    return (r_m ** 2) * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))


def u_bins_array():
    return np.linspace(U_MIN, U_MAX, N_U_BINS)


def td_timescale(u_ev, temp_k):
    """熱脱離の時定数 [s] (参考表示用)"""
    KB, EV = 1.380649e-23, 1.602e-19
    rate = 1e13 * np.exp(-(u_ev * EV) / (KB * temp_k))
    return np.where(rate > 1e-300, 1.0 / rate, np.inf)


def load_series(target_dir, max_files=MAX_FILES):
    files = glob.glob(os.path.join(target_dir, "surface_density_t*.npy"))
    pairs = []
    for f in files:
        m = re.search(r'surface_density_t(\d+)\.npy', os.path.basename(f))
        if m:
            pairs.append((int(m.group(1)), f))
    if not pairs:
        raise FileNotFoundError(f"surface_density_t*.npy が見つかりません: {target_dir}")
    pairs.sort()
    if len(pairs) > max_files:
        step = int(np.ceil(len(pairs) / max_files))
        pairs = pairs[::step]

    areas = cell_areas_m2(N_LON, N_LAT, R_BODY_M)
    times, stocks = [], []
    skipped = []       # 読めなかったファイル (空・破損・書き込み中)
    bad_shape = []     # 形状が想定と違うファイル

    for th, f in pairs:
        # 0バイトや極端に小さいファイルは書き込み途中/失敗とみなして飛ばす
        try:
            size = os.path.getsize(f)
        except OSError as e:
            skipped.append((os.path.basename(f), f"サイズ取得失敗: {e}"))
            continue
        if size < 128:
            skipped.append((os.path.basename(f), f"サイズ {size} B (空または破損)"))
            continue

        try:
            sd = np.load(f)
        except (EOFError, ValueError, OSError) as e:
            skipped.append((os.path.basename(f), f"{type(e).__name__}: {e}"))
            continue

        if sd.ndim != 3:
            bad_shape.append((os.path.basename(f), str(sd.shape)))
            continue
        if sd.shape[0] != N_LON or sd.shape[1] != N_LAT:
            bad_shape.append((os.path.basename(f), str(sd.shape)))
            continue
        if not np.all(np.isfinite(sd)):
            sd = np.nan_to_num(sd, nan=0.0, posinf=0.0, neginf=0.0)
            skipped.append((os.path.basename(f), "NaN/Inf を0に丸めて使用"))

        # (n_lon, n_lat, n_bins) を面積重みで全球積分 → ビンごとの総原子数
        per_bin = np.einsum('ijb,j->b', sd, areas)
        times.append(th)
        stocks.append(per_bin)

    # --- 読み込み結果の報告 ---
    if skipped:
        print(f"[注意] {len(skipped)} 個のファイルを読み飛ばしました:")
        for name, reason in skipped[:10]:
            print(f"    {name}: {reason}")
        if len(skipped) > 10:
            print(f"    ... 他 {len(skipped)-10} 件")
        print("    (実行中のrunがある、または過去に書き込みが中断された可能性があります)")
    if bad_shape:
        print(f"[注意] {len(bad_shape)} 個は形状が想定外のため除外しました:")
        for name, shp in bad_shape[:5]:
            print(f"    {name}: shape={shp} (期待 ({N_LON}, {N_LAT}, n_bins))")

    if not times:
        raise RuntimeError(
            "有効な surface_density ファイルが1つもありませんでした。\n"
            "  ・run が途中で止まっていないか\n"
            "  ・RUN_NAME のフォルダが正しいか\n"
            "を確認してください。")

    # ビン数が途中で変わっていないか確認
    nb = {len(x) for x in stocks}
    if len(nb) > 1:
        raise ValueError(f"ファイル間でビン数が一致しません: {sorted(nb)}")

    print(f"[読み込み] 有効 {len(times)} ファイル / 全 {len(pairs)} ファイル")
    return np.array(times, dtype=float), np.array(stocks)


# ==========================================
# 本体
# ==========================================
def main():
    target_dir = os.path.join(BASE_DIR, RUN_NAME)
    if not os.path.exists(target_dir):
        print(f"フォルダが見つかりません: {target_dir}")
        return

    times_h, stocks = load_series(target_dir)
    years = times_h * 3600.0 / MERCURY_YEAR_SEC
    ub = u_bins_array()
    n_bins = stocks.shape[1]
    if n_bins != len(ub):
        print(f"[警告] ファイルのビン数 {n_bins} が設定 N_U_BINS={len(ub)} と違います。")
        ub = np.linspace(U_MIN, U_MAX, n_bins)

    total = stocks.sum(axis=1)
    active = stocks[:, ub <= U_ACTIVE_MAX].sum(axis=1)
    inert = stocks[:, ub > U_ACTIVE_MAX].sum(axis=1)

    cmap = plt.get_cmap('viridis')
    colors = [cmap(x) for x in np.linspace(0, 0.9, n_bins)]

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

    # [1] ビン別の総在庫
    for b in range(n_bins):
        axes[0].plot(years, stocks[:, b], color=colors[b], lw=1.8,
                     label=f'U={ub[b]:.2f} eV')
    axes[0].plot(years, total, color='black', lw=2.5, ls='--', label='合計')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('表面在庫 [atoms]')
    axes[0].set_title('束縛エネルギービン別の表面在庫')
    axes[0].grid(True, which='both', ls='--', alpha=0.4)
    axes[0].legend(fontsize=7, ncol=3, loc='lower right')

    # [2] 活性ビン vs 不活性ビン
    axes[1].plot(years, active, color='crimson', lw=2.2,
                 label=f'浅いビン U ≤ {U_ACTIVE_MAX} eV (放出に効く)')
    axes[1].plot(years, inert, color='steelblue', lw=2.2,
                 label=f'深いビン U > {U_ACTIVE_MAX} eV (実質不活性)')
    axes[1].plot(years, total, color='black', lw=2.0, ls='--', label='合計')
    axes[1].set_ylabel('表面在庫 [atoms]')
    axes[1].set_title('放出に効くビンと効かないビンの分解')
    axes[1].grid(True, ls='--', alpha=0.4)
    axes[1].legend(fontsize=9)

    # [3] 全体に占める割合
    frac = stocks / np.maximum(total[:, None], 1e-300)
    for b in range(n_bins):
        axes[2].plot(years, frac[:, b] * 100, color=colors[b], lw=1.8)
    axes[2].set_ylabel('全在庫に占める割合 [%]')
    axes[2].set_xlabel('経過年 [水星年]')
    axes[2].set_title('各ビンの構成比の変化 — 増加がどのビンで起きているか')
    axes[2].grid(True, ls='--', alpha=0.4)
    axes[2].xaxis.set_major_locator(MultipleLocator(5))

    plt.tight_layout()
    plt.show()

    # ==========================================
    # 数値サマリー
    # ==========================================
    print("\n" + "=" * 76)
    print("=== 束縛エネルギービン別の表面在庫 ===")
    print("=" * 76)
    print(f"  読み込み: {len(times_h)} スナップショット, {years[0]:.1f} 〜 {years[-1]:.1f} 年")
    print()
    print(f"{'U [eV]':>8}{'初期 [atoms]':>15}{'最終 [atoms]':>15}{'増加倍率':>11}"
          f"{'最終構成比':>11}{'TD時定数(600K)':>16}")
    print("-" * 76)
    for b in range(n_bins):
        tau = td_timescale(ub[b], 600.0)
        tau_s = f"{tau/86400:.1e}日" if np.isfinite(tau) else "---"
        ratio = stocks[-1, b] / stocks[0, b] if stocks[0, b] > 0 else np.inf
        print(f"{ub[b]:>8.2f}{stocks[0, b]:>15.3e}{stocks[-1, b]:>15.3e}"
              f"{ratio:>11.2f}{frac[-1, b]*100:>10.1f}%{tau_s:>16}")
    print("-" * 76)
    print(f"{'合計':>8}{total[0]:>15.3e}{total[-1]:>15.3e}{total[-1]/total[0]:>11.2f}")
    print()
    print(f"  浅いビン (U ≤ {U_ACTIVE_MAX} eV): "
          f"{active[0]:.3e} → {active[-1]:.3e}  ({active[-1]/active[0]:.2f}倍)")
    print(f"  深いビン (U > {U_ACTIVE_MAX} eV): "
          f"{inert[0]:.3e} → {inert[-1]:.3e}  ({inert[-1]/inert[0]:.2f}倍)")
    print()

    # 最後の数年の年増加率
    if len(years) > 3:
        n_tail = max(2, len(years) // 5)
        dy = years[-1] - years[-n_tail]
        if dy > 0:
            print("  終盤の年増加率:")
            print(f"    合計      : {((total[-1]/total[-n_tail])**(1/dy) - 1)*100:+.2f} %/年")
            print(f"    浅いビン  : {((active[-1]/active[-n_tail])**(1/dy) - 1)*100:+.2f} %/年")
            print(f"    深いビン  : {((inert[-1]/inert[-n_tail])**(1/dy) - 1)*100:+.2f} %/年")
            print()
            print("  [判定] 浅いビンの増加率が合計より十分小さければ、")
            print("         増加は放出に寄与しない深いビンで起きており、")
            print("         大気が定常であることと矛盾しない。")
    print("=" * 76)


if __name__ == "__main__":
    main()