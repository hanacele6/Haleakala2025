# -*- coding: utf-8 -*-
"""
自転による「夜側で蓄積された未消費在庫」が明け方(Dawn)側に掃き込まれる
供給フラックスを、既存の surface_density_t*.npy スナップショットから
事後的に(再シミュレーションせずに)推定し、実際の生成量(Gen_PSD_Dawn + Gen_TD_Dawn)
と重ねて比較するための検証スクリプト（15年目限定・統計CSV対応・堅牢化版）。

--- v2からの修正点 ---
1. is_dawn_side フィルタを削除。
   「夜→昼」への遷移(newly_lit)は定義上すでに必ず明け方(sunrise)側の事象であり、
   追加のフィルタは不要かつ ROTATION_SIGN の符号次第で全消去されるほど壊れやすかった。
2. __main__ 内の存在しない列名参照バグを修正 (Sweep_In_Atoms_per_interval → Sweep_In_Dawn_per_TAA)。
3. 空データでの log スケールクラッシュを防ぐガードを追加。
4. 見かけの自転速度(|d(sub_lon)/dt|)を直接確認する診断プロットを追加
   (ROTATION_SIGN の正しさを、複雑な掃き込み計算に入る前に単体で確認できる)。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import glob
import re
import sys
import time
from tqdm import tqdm


def log(msg):
    print(msg, flush=True)


MAX_SNAPSHOTS_DEBUG = None

# ==========================================
# 1. 設定
# ==========================================
RESULT_DIR = r"./SimulationResult_202607/ParabolicHop_72x36_NoEq_DT100_0710_BD0.8_UG1.85_Q0.3_A2.0e+07_LT190k_15yr"
ROTATION_PERIOD_SEC = 58.6462 * 86400.0
ROTATION_SIGN = 1  # 見かけの自転速度診断プロットで確認してから確定させること
RM_M = 2.440e6

# 診断プロット(見かけの自転速度)だけ先に確認したい場合 True
RUN_ROTATION_DIAGNOSIS_ONLY = False


# ==========================================
# 2. データ読み込みヘルパー
# ==========================================
def load_unwrapped_taa(target_dir):
    """budget_timeseries.csv を読み込み、累積TAAを付与する"""
    csv_path = os.path.join(target_dir, "budget_timeseries.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"budget_timeseries.csv が見つかりません: {csv_path}")

    df = pd.read_csv(csv_path).sort_values('Time_hours').reset_index(drop=True)
    diff = df['TAA'].diff()
    wrap_count = (diff < -180).cumsum().fillna(0)
    df['Unwrapped_TAA'] = df['TAA'] + wrap_count * 360.0
    return df


def load_surface_density_snapshots_final_year(target_dir, final_year_start_hours):
    """15年目（final_year_start_hours以降）のスナップショットのみをロードする"""
    log(f"[STEP] ファイル一覧を取得中(glob)... : {target_dir}")
    t0 = time.time()
    files = glob.glob(os.path.join(target_dir, "surface_density_t*.npy"))
    log(f"[STEP] glob完了 ({time.time()-t0:.2f}s) : {len(files)} 件ヒット")

    if not files:
        raise FileNotFoundError(f"surface_density_t*.npy が見つかりません: {target_dir}")

    pairs = []
    for f in files:
        m = re.search(r'surface_density_t(\d+)\.npy', os.path.basename(f))
        if m:
            t_h = int(m.group(1))
            if t_h >= final_year_start_hours:
                pairs.append((t_h, f))
    pairs.sort()

    if MAX_SNAPSHOTS_DEBUG is not None:
        pairs = pairs[:MAX_SNAPSHOTS_DEBUG]

    if not pairs:
        raise ValueError(
            f"15年目(t>={final_year_start_hours}h)に該当するスナップショットが0件でした。"
            f" 保存間隔やディレクトリを確認してください。"
        )

    log(f"[STEP] 15年目（最終年）の有効なスナップショット数: {len(pairs)} 件 "
        f"(t={pairs[0][0]}h 〜 t={pairs[-1][0]}h)")
    return pairs


def get_grid_geometry(n_lon, n_lat):
    lon_edges = np.linspace(0.0, 2 * np.pi, n_lon + 1)
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, n_lat + 1)
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    dlon = 2 * np.pi / n_lon
    cell_area_per_lat = (RM_M ** 2) * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))
    return lon_centers, lat_centers, cell_area_per_lat


# ==========================================
# 3. 実際の生成量データの取得
# ==========================================
def get_actual_dawn_generation_from_stats(target_dir):
    """budget_statistics_per_taa.csv から TAAごとの生成量を直接取得する"""
    csv_path = os.path.join(target_dir, "budget_statistics_per_taa.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"budget_statistics_per_taa.csv が見つかりません: {csv_path}")

    df = pd.read_csv(csv_path)

    if 'TAA_Bin' in df.columns:
        df['TAA_mod360'] = df['TAA_Bin'] % 360.0
    elif 'TAA' in df.columns:
        df['TAA_mod360'] = df['TAA'] % 360.0
    else:
        raise KeyError(f"TAAを示すカラムが見つかりません。CSVのカラム一覧: {df.columns.tolist()}")

    df['Gen_Dawn_Total'] = df['Gen_PSD_Dawn'] + df['Gen_TD_Dawn']
    df = df.sort_values('TAA_mod360').reset_index(drop=True)

    log(f"[STEP] budget_statistics_per_taa.csv 読み込み完了 ({len(df)}行)")
    return df


# ==========================================
# 4. 診断: 見かけの自転速度の確認 (符号確定用)
# ==========================================
def diagnose_apparent_rotation_rate(target_dir, rotation_period_sec, rotation_sign):
    df_ts = load_unwrapped_taa(target_dir)
    t_h = df_ts['Time_hours'].values
    taa_uw_deg = df_ts['Unwrapped_TAA'].values

    omega_rot = 2.0 * np.pi / rotation_period_sec
    dtaa_dt = np.gradient(np.deg2rad(taa_uw_deg), t_h * 3600.0)  # [rad/s]
    apparent_rate = dtaa_dt - rotation_sign * omega_rot  # d(sub_lon)/dt

    taa_mod = taa_uw_deg % 360.0

    plt.figure(figsize=(9, 5))
    plt.scatter(taa_mod, np.abs(apparent_rate), s=3, alpha=0.5)
    plt.xlabel("TAA [deg]")
    plt.ylabel("|d(sub_lon)/dt| [rad/s]  (見かけの自転速度)")
    plt.title(f"Apparent Rotation Rate Check (ROTATION_SIGN={rotation_sign})\n"
              f"期待: 近日点(TAA=0/360)で最小、遠日点(TAA=180)で最大")
    plt.axvline(180, color='gray', ls=':', label='Aphelion')
    plt.axvline(0, color='gray', ls='-', alpha=0.5, label='Perihelion')
    plt.xlim(0, 360)
    plt.gca().xaxis.set_major_locator(MultipleLocator(60))
    plt.legend()
    plt.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


# ==========================================
# 5. 掃き込み供給フラックスの計算本体
# ==========================================
def compute_sweep_in_flux_final_year(target_dir, rotation_period_sec, rotation_sign):
    log("[STEP] タイムシリーズから15年目の開始時刻を算出中...")
    df_ts = load_unwrapped_taa(target_dir)

    final_year_threshold_taa = df_ts['Unwrapped_TAA'].max() - 360.0
    final_year_start_hours = df_ts.loc[
        df_ts['Unwrapped_TAA'] >= final_year_threshold_taa, 'Time_hours'
    ].min()

    pairs = load_surface_density_snapshots_final_year(target_dir, final_year_start_hours)
    taa_unwrapped_interp = np.interp(
        [t for t, _ in pairs], df_ts['Time_hours'].values, df_ts['Unwrapped_TAA'].values
    )
    omega_rot = 2.0 * np.pi / rotation_period_sec

    sample = None
    for _, fpath in pairs:
        try:
            sample = np.load(fpath)
            break
        except Exception:
            continue

    if sample is None:
        raise ValueError("15年目のすべてのスナップショットが破損しているか、読み込めません。")

    n_lon, n_lat, n_bins = sample.shape
    lon_centers, lat_centers, cell_area_per_lat = get_grid_geometry(n_lon, n_lat)
    log(f"[情報] グリッド: n_lon={n_lon}, n_lat={n_lat}, n_bins={n_bins}")

    records = []
    skipped_files = []
    prev_surf, prev_is_day, prev_t_h, prev_taa_uw = None, None, None, None

    for (t_h, fpath), taa_uw_deg in tqdm(list(zip(pairs, taa_unwrapped_interp)),
                                          desc="15年目スナップショット処理中", file=sys.stdout):
        try:
            surf = np.load(fpath)
            if surf.shape != (n_lon, n_lat, n_bins):
                raise ValueError("shape mismatch")
        except Exception:
            skipped_files.append(fpath)
            prev_surf, prev_is_day, prev_t_h, prev_taa_uw = None, None, None, None
            continue

        taa_rad_uw = np.deg2rad(taa_uw_deg)
        sub_lon = taa_rad_uw - rotation_sign * omega_rot * (t_h * 3600.0)

        lon_sun = (lon_centers[:, None] - sub_lon + np.pi) % (2 * np.pi) - np.pi
        eff_cos = np.cos(lat_centers)[None, :] * np.cos(lon_sun)
        is_day = eff_cos > 0.0
        day_fraction = np.mean(is_day)

        if prev_surf is not None:
            # ★修正: is_dawn_side フィルタを削除。
            # 「夜→昼」への遷移は定義上すでに必ず明け方(sunrise)側の事象であるため、
            # 追加のフィルタは不要かつ、符号次第で結果を全消去してしまう危険がある。
            newly_lit = (~prev_is_day) & (is_day)

            reservoir_areal = np.sum(prev_surf, axis=2)
            cell_area_grid = np.broadcast_to(cell_area_per_lat[None, :], (n_lon, n_lat))

            swept_atoms_dawn = np.sum(reservoir_areal[newly_lit] * cell_area_grid[newly_lit])

            dt_hours = t_h - prev_t_h
            delta_taa = taa_uw_deg - prev_taa_uw
            mid_taa_uw = 0.5 * (prev_taa_uw + taa_uw_deg)

            records.append({
                'Time_hours': 0.5 * (prev_t_h + t_h),
                'Unwrapped_TAA': mid_taa_uw,
                'TAA_mod360': mid_taa_uw % 360.0,
                'Sweep_In_Dawn_Atoms': swept_atoms_dawn,
                'Sweep_In_Dawn_per_TAA': swept_atoms_dawn / delta_taa if delta_taa > 0 else 0.0,
                'Day_Fraction': day_fraction,
                'N_newly_lit_cells': int(np.sum(newly_lit)),
            })

        prev_surf = surf
        prev_is_day = is_day
        prev_t_h = t_h
        prev_taa_uw = taa_uw_deg

    df_sweep = pd.DataFrame(records)

    log("\n" + "=" * 70)
    log("=== 掃き込みフラックス計算 診断情報 ===")
    log(f"  スナップショット数(検出): {len(pairs)}")
    log(f"  読み込み失敗によるスキップ数: {len(skipped_files)}")
    log(f"  有効な区間(データ点)数: {len(df_sweep)}")
    if len(df_sweep) > 0:
        log(f"  平均 照度面積比 (Day_Fraction, 理想値=0.500): {df_sweep['Day_Fraction'].mean():.4f}")
        log(f"  Sweep_In_Dawn_per_TAA: min={df_sweep['Sweep_In_Dawn_per_TAA'].min():.3e}, "
            f"max={df_sweep['Sweep_In_Dawn_per_TAA'].max():.3e}, "
            f"ゼロ件数={int((df_sweep['Sweep_In_Dawn_per_TAA'] <= 0).sum())}/{len(df_sweep)}")
    log("=" * 70 + "\n")

    return df_sweep


# ==========================================
# 6. プロット
# ==========================================
def plot_sweep_in_vs_generation(df_sweep_final, df_gen):
    df_sweep_plot = df_sweep_final[df_sweep_final['Sweep_In_Dawn_per_TAA'] > 0].copy()
    df_sweep_plot = df_sweep_plot.sort_values('TAA_mod360')

    if len(df_sweep_plot) == 0:
        raise ValueError(
            "Sweep_In_Dawn_per_TAA が全て0以下でした。プロットできません。\n"
            "  - ROTATION_SIGN の符号が正しいか diagnose_apparent_rotation_rate() で確認してください\n"
            "  - N_newly_lit_cells が常に0になっていないか df_sweep_final を直接確認してください"
        )

    df_gen_plot = df_gen[df_gen['Gen_Dawn_Total'] > 0].copy()
    if len(df_gen_plot) == 0:
        raise ValueError("Gen_Dawn_Total が全て0以下でした。budget_statistics_per_taa.csv を確認してください。")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    l1, = ax1.plot(df_sweep_plot['TAA_mod360'], df_sweep_plot['Sweep_In_Dawn_per_TAA'],
                    color='darkorange', marker='o', markersize=4, linewidth=1.8,
                    label='Dawn Sweep-In (Estimated per 1° TAA)')

    ax1b = ax1.twinx()
    l2, = ax1b.plot(df_gen_plot['TAA_mod360'], df_gen_plot['Gen_Dawn_Total'],
                     color='blue', linewidth=2.5, label='Actual Gen: PSD+TD (From Stats CSV)')
    l3, = ax1b.plot(df_gen_plot['TAA_mod360'], df_gen_plot['Gen_PSD_Dawn'],
                     color='blue', linestyle=':', alpha=0.6, linewidth=1.5, label='Actual Gen: PSD')
    l4, = ax1b.plot(df_gen_plot['TAA_mod360'], df_gen_plot['Gen_TD_Dawn'],
                     color='blue', linestyle='--', alpha=0.6, linewidth=1.5, label='Actual Gen: TD')

    ax1.set_yscale('log')
    ax1b.set_yscale('log')
    ax1.set_ylabel('Sweep-In Reservoir [atoms / TAA deg]', color='darkorange')
    ax1b.set_ylabel('Actual Dawn Generation [atoms / TAA bin]', color='blue')
    ax1.tick_params(axis='y', labelcolor='darkorange')
    ax1b.tick_params(axis='y', labelcolor='blue')
    ax1.set_title('Rotational Sweep-In Supply (Dawn) vs. Actual Dawn Generation')
    ax1.grid(True, which="both", ls="--", alpha=0.4)

    lines = [l1, l2, l3, l4]
    ax1.legend(lines, [ln.get_label() for ln in lines], loc='upper left', bbox_to_anchor=(1.08, 1))

    sweep_norm = df_sweep_plot['Sweep_In_Dawn_per_TAA'] / df_sweep_plot['Sweep_In_Dawn_per_TAA'].max()
    gen_norm = df_gen_plot['Gen_Dawn_Total'] / df_gen_plot['Gen_Dawn_Total'].max()

    ax2.plot(df_sweep_plot['TAA_mod360'], sweep_norm, color='darkorange', marker='o', markersize=4,
              linewidth=1.8, label='Sweep-In Dawn (normalized)')
    ax2.plot(df_gen_plot['TAA_mod360'], gen_norm, color='blue', linewidth=2.5,
              label='Actual Gen: Dawn Total (normalized)')

    ax2.axvline(180, color='gray', linestyle=':', alpha=0.7, label='Aphelion (TAA=180)')
    ax2.axvline(0, color='gray', linestyle='-', alpha=0.5, label='Perihelion (TAA=0/360)')

    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel('True Anomaly Angle (TAA) [deg]')
    ax2.set_ylabel('Normalized to Peak = 1')
    ax2.set_title('Shape Comparison (Peak-Normalized)')
    ax2.grid(True, ls="--", alpha=0.4)
    ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

    plt.xlim(0, 360)
    ax1.xaxis.set_major_locator(MultipleLocator(60))
    ax2.xaxis.set_major_locator(MultipleLocator(60))

    plt.tight_layout()
    plt.show()

    return df_sweep_plot, df_gen_plot


if __name__ == "__main__":
    print(f"=== 15年目ターゲット解析開始: {RESULT_DIR} ===")

    if not os.path.exists(RESULT_DIR):
        print("エラー: 指定されたディレクトリが存在しません。")
    elif RUN_ROTATION_DIAGNOSIS_ONLY:
        diagnose_apparent_rotation_rate(RESULT_DIR, ROTATION_PERIOD_SEC, ROTATION_SIGN)
    else:
        try:
            df_sweep_final = compute_sweep_in_flux_final_year(RESULT_DIR, ROTATION_PERIOD_SEC, ROTATION_SIGN)
            df_gen = get_actual_dawn_generation_from_stats(RESULT_DIR)

            df_sweep_final = df_sweep_final.sort_values('TAA_mod360')
            df_sweep_plot, df_gen_plot = plot_sweep_in_vs_generation(df_sweep_final, df_gen)

            sweep_peak_taa = df_sweep_plot.loc[
                df_sweep_plot['Sweep_In_Dawn_per_TAA'].idxmax(), 'TAA_mod360'
            ]
            gen_peak_taa = df_gen_plot.loc[df_gen_plot['Gen_Dawn_Total'].idxmax(), 'TAA_mod360']
            print(f"\n[結果] 掃き込み供給フラックスのピーク TAA  : {sweep_peak_taa:.1f} deg")
            print(f"[結果] 実際のDawn生成量(PSD+TD)のピーク TAA: {gen_peak_taa:.1f} deg")

        except Exception as e:
            print(f"エラーが発生しました: {e}")

    print("=== 解析完了 ===")