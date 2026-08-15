# -*- coding: utf-8 -*-
"""
=== 課題1(フラックス収支版): 表面在庫の変化率 = 放出 − 吸着 が成り立つか ===

Verify_SubResult.py に追加する関数。

先生の意図:
  「放出と表面の減少、吸着と表面密度の増加が一致してるか」
  → これはストック(総量)でなく フラックス(速度) の収支の話。

定常でなくても各瞬間で必ず成り立つべき保存則:
    d(表面原子数)/dt = (放出で減る) の逆 + (吸着で増える)
  表面から見ると:
    d N_surf / dt = - Gen_Total + Loss_Stuck
  (Gen_Total だけ表面から出ていき、Loss_Stuck だけ表面に戻る。
   Ionized/Escaped は大気側の損失で表面在庫には直接効かない)

これを2通りで出して重ねる:
  [A] 左辺: 表面原子数(surface_density積分)を時間微分した「実測の変化速度」
  [B] 右辺: budget_timeseries の (Loss_Stuck - Gen_Total)/DT_MOVE 「収支から予測した変化速度」
両者が一致すれば、放出・吸着・在庫変化が整合している。

--- 前提 ---
budget_timeseries.csv の Gen_*/Loss_* は「保存時の1ステップ(DT_MOVE秒)分の
weight[atoms]」。/DT_MOVE で [atoms/s] になる。
→ DATA_SOURCE = 'TimeSeries' で使うこと(時間微分するため時系列が必要)。

--- 組み込み ---
関数をコピーし、__main__ に追加:
    plot_flux_balance(RESULT_DIR, dt_move=100.0)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import japanize_matplotlib
import re
from matplotlib.ticker import MultipleLocator


def _surface_atoms_timeseries(target_dir, rm_m=2.440e6):
    """surface_density_t*.npy を全球積分し、(times[h], N_surf[atoms]) を返す。"""
    files = glob.glob(os.path.join(target_dir, "surface_density_t*.npy"))
    if not files:
        return None, None
    pairs = []
    for f in files:
        m = re.search(r'surface_density_t(\d+)\.npy', os.path.basename(f))
        if m:
            pairs.append((int(m.group(1)), f))
    pairs.sort()
    sample = np.load(pairs[0][1])
    n_lon, n_lat, n_bins = sample.shape
    lat_edges = np.linspace(-np.pi/2, np.pi/2, n_lat + 1)
    dlon = 2 * np.pi / n_lon
    cell_areas = (rm_m ** 2) * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))
    times, atoms = [], []
    for th, f in pairs:
        sd = np.load(f)
        times.append(th)
        atoms.append(float(np.sum(sd * cell_areas[np.newaxis, :, np.newaxis])))
    return np.array(times, dtype=float), np.array(atoms, dtype=float)


def plot_flux_balance(target_dir, dt_move=100.0):
    """
    [A] 表面原子数の時間微分(実測の変化速度)
    [B] (Loss_Stuck - Gen_Total)/dt_move (収支から予測した変化速度)
    を重ねて、フラックス収支が閉じているか検証する。
    """
    # --- [A] 表面在庫の時間微分 ---
    times_h, n_surf = _surface_atoms_timeseries(target_dir)
    if times_h is None or len(times_h) < 3:
        print("[スキップ] surface_density スナップショットが足りません(3点以上必要)。")
        return

    # 時間(秒)に直して中心差分で dN/dt [atoms/s]
    t_sec = times_h * 3600.0
    dN_dt = np.gradient(n_surf, t_sec)   # [atoms/s]

    # --- [B] budget_timeseries から収支 ---
    ts_csv = os.path.join(target_dir, "budget_timeseries.csv")
    if not os.path.exists(ts_csv):
        print(f"[スキップ] budget_timeseries.csv がありません: {ts_csv}")
        return
    df = pd.read_csv(ts_csv)

    need = ['Time_hours', 'Gen_Total', 'Loss_Stuck']
    for c in need:
        if c not in df.columns:
            print(f"[スキップ] 列 {c} がありません。")
            return

    # スナップショット時刻に最も近い行の Gen/Loss を対応づけ
    gen = np.interp(times_h, df['Time_hours'], df['Gen_Total'])
    stuck = np.interp(times_h, df['Time_hours'], df['Loss_Stuck'])
    # 1ステップ分の weight を速度[atoms/s]に
    balance = (stuck - gen) / dt_move   # 表面から見た変化速度

    # --- プロット ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    ax1.plot(times_h, dN_dt, 'o-', color='teal', lw=2, ms=4,
             label='[A] 表面在庫の時間微分  dN/dt')
    ax1.plot(times_h, balance, 's--', color='orangered', lw=2, ms=4, alpha=0.8,
             label='[B] 収支 (Loss_Stuck − Gen_Total)/Δt')
    ax1.axhline(0, color='gray', ls=':', alpha=0.6)
    ax1.set_ylabel('表面原子数の変化速度 [atoms/s]')
    ax1.set_title('フラックス収支: 表面在庫の変化率 = 吸着 − 放出')
    ax1.legend(loc='upper right')
    ax1.grid(True, ls='--', alpha=0.5)

    # 残差(A - B): ゼロに近いほど収支が閉じている
    resid = dN_dt - balance
    scale = np.nanmean(np.abs(dN_dt)) if np.nanmean(np.abs(dN_dt)) > 0 else 1.0
    ax2.plot(times_h, resid / scale * 100, '.-', color='purple', lw=1.5,
             label='残差 (A−B) / |dN/dt|平均')
    ax2.axhline(0, color='gray', ls=':', alpha=0.6)
    ax2.fill_between(times_h, -10, 10, color='green', alpha=0.08, label='±10%目安')
    ax2.set_ylabel('相対残差 [%]')
    ax2.set_xlabel('Simulation Time [hours]')
    ax2.legend(loc='upper right')
    ax2.grid(True, ls='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    # --- 数値サマリー ---
    rms_resid = np.sqrt(np.nanmean(resid**2))
    print("\n" + "=" * 66)
    print("=== 課題1: フラックス収支チェック ===")
    print("=" * 66)
    print(f"  比較点数           : {len(times_h)}")
    print(f"  dN/dt の平均振幅    : {np.nanmean(np.abs(dN_dt)):.3e} atoms/s")
    print(f"  収支項の平均振幅    : {np.nanmean(np.abs(balance)):.3e} atoms/s")
    print(f"  残差(A-B) RMS       : {rms_resid:.3e} atoms/s  "
          f"({rms_resid/scale*100:.1f}% of |dN/dt|)")
    if rms_resid / scale < 0.15:
        print("  → 収支はほぼ閉じている。放出・吸着・在庫変化が整合。")
    else:
        print("  → 残差が大きい。以下のいずれかを確認:")
        print("     ・スナップショット間隔が粗く微分が不正確")
        print("     ・Gen/Loss が1ステップ値なので間隔内の平均と乖離")
        print("     ・Supply_Internal(地下供給)など別の在庫源が効いている")
    print("=" * 66)
    print("[注] Gen/Lossは保存時の1ステップ値。スナップショット間隔が広いと")
    print("     その間の平均を代表しないため残差が出やすい。傾向の一致を見ること。")


if __name__ == "__main__":
    RESULT_DIR = r"./SimulationResult_202607/ParabolicHop_72x36_NoEq_DT100_0710_BD0.8_UG1.85_Q0.3_A2.0e+07_LT190k_15yr"

    if os.path.exists(RESULT_DIR):
        plot_flux_balance(RESULT_DIR, dt_move=100.0)
    else:
        print(f"ディレクトリを設定してください: {RESULT_DIR}")