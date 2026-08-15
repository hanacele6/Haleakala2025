# -*- coding: utf-8 -*-
"""
リージョン別(ターミネーター近傍 / それ以外)の放出量比較 — ハイブリッド版
(差分補間処理 & 日本語プロット対応)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import glob
import re
import sys
from tqdm import tqdm
import platform

# ==========================================
# 日本語フォント設定
# ==========================================
try:
    import japanize_matplotlib
except ImportError:
    # japanize_matplotlibがない場合はOS標準フォントをフォールバックとして設定
    if platform.system() == "Windows":
        plt.rcParams['font.family'] = 'Meiryo'
    elif platform.system() == "Darwin":
        plt.rcParams['font.family'] = 'Hiragino Sans'
    else:
        plt.rcParams['font.family'] = 'Noto Sans CJK JP'

def log(msg):
    print(msg, flush=True)

# ==========================================
# 設定
# ==========================================
BASE_DIR = r"./SimulationResult_202607"

MODELS = {
    "Q2.0 (Standard)": "ParabolicHop_72x36_NoEq_DT100_0713_BD0.8_UG1.85_Q2.0_A2.0e+07_LT190k_15yr",
    "Q0.3 (Weak PSD)": "ParabolicHop_72x36_NoEq_DT100_0710_BD0.8_UG1.85_Q0.3_A2.0e+07_LT190k_15yr",
}

ROTATION_PERIOD_SEC = 58.6462 * 86400.0
ROTATION_SIGN = 1  # 診断プロットで確定させた符号
RM_M = 2.440e6

EFF_COS_TERMINATOR_MAX = 0.15  # これ以下を「ターミネーター近傍」とする
MAX_SNAPSHOTS_DEBUG = None


# ==========================================
# データ読み込みヘルパー
# ==========================================
def load_unwrapped_taa(target_dir):
    csv_path = os.path.join(target_dir, "budget_timeseries.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"budget_timeseries.csv が見つかりません: {csv_path}")
    df = pd.read_csv(csv_path).sort_values('Time_hours').reset_index(drop=True)
    diff = df['TAA'].diff()
    wrap_count = (diff < -180).cumsum().fillna(0)
    df['Unwrapped_TAA'] = df['TAA'] + wrap_count * 360.0
    return df


def load_surface_density_snapshots_final_year(target_dir, final_year_start_hours):
    files = glob.glob(os.path.join(target_dir, "surface_density_t*.npy"))
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
        raise ValueError(f"15年目(t>={final_year_start_hours}h)に該当するスナップショットが0件でした。")
    return pairs


def get_grid_geometry(n_lon, n_lat):
    lon_edges = np.linspace(0.0, 2 * np.pi, n_lon + 1)
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, n_lat + 1)
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    dlon = 2 * np.pi / n_lon
    cell_area_per_lat = (RM_M ** 2) * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))
    return lon_centers, lat_centers, cell_area_per_lat


def get_actual_dawn_generation_from_stats(target_dir):
    csv_path = os.path.join(target_dir, "budget_statistics_per_taa.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"budget_statistics_per_taa.csv が見つかりません: {csv_path}")
    df = pd.read_csv(csv_path)
    taa_col = 'TAA_Bin' if 'TAA_Bin' in df.columns else 'TAA'
    df['TAA_int'] = (df[taa_col] % 360.0).round().astype(int) % 360
    df['Gen_Dawn_Total'] = df['Gen_PSD_Dawn'] + df['Gen_TD_Dawn']
    df = df.groupby('TAA_int', as_index=False)['Gen_Dawn_Total'].sum()
    return df


# ==========================================
# ターミネーター近傍の寄与を測定(差分補間処理を追加)
# ==========================================
def compute_terminator_release(target_dir, rotation_period_sec, rotation_sign, eff_cos_term_max):
    log(f"\n### 解析対象: {target_dir} ###")
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
        raise ValueError("有効なスナップショットが1つもありません。")

    n_lon, n_lat, n_bins = sample.shape
    lon_centers, lat_centers, cell_area_per_lat = get_grid_geometry(n_lon, n_lat)
    cell_area_grid = np.broadcast_to(cell_area_per_lat[None, :], (n_lon, n_lat))
    log(f"[情報] グリッド: n_lon={n_lon}, n_lat={n_lat}, n_bins={n_bins}, スナップショット数={len(pairs)}")

    mid_taas = []
    release_rates = []
    
    prev_total, prev_t_h, prev_taa_uw = None, None, None
    skipped = 0

    for (t_h, fpath), taa_uw_deg in tqdm(list(zip(pairs, taa_unwrapped_interp)),
                                         desc="ターミネーター寄与を測定中", file=sys.stdout):
        try:
            surf = np.load(fpath)
            if surf.shape != (n_lon, n_lat, n_bins):
                raise ValueError("shape mismatch")
        except Exception:
            skipped += 1
            prev_total, prev_t_h, prev_taa_uw = None, None, None
            continue

        curr_total = np.sum(surf, axis=2)

        taa_rad_uw = np.deg2rad(taa_uw_deg)
        sub_lon = taa_rad_uw - rotation_sign * omega_rot * (t_h * 3600.0)
        lon_sun = (lon_centers[:, None] - sub_lon + np.pi) % (2 * np.pi) - np.pi
        eff_cos = np.cos(lat_centers)[None, :] * np.cos(lon_sun)
        is_dawn = lon_sun < 0.0
        region_terminator = is_dawn & (eff_cos > 0.0) & (eff_cos <= eff_cos_term_max)

        if prev_total is not None:
            delta = prev_total - curr_total
            release_est = np.clip(delta, 0.0, None) * cell_area_grid
            terminator_release = np.sum(release_est[region_terminator])

            # スナップショット間のTAA差分を計算し、1度あたりのレート(atoms/deg)に変換
            delta_taa = taa_uw_deg - prev_taa_uw
            if delta_taa > 0:
                rate_per_deg = terminator_release / delta_taa
                mid_taa_uw = 0.5 * (prev_taa_uw + taa_uw_deg)
                
                mid_taas.append(mid_taa_uw)
                release_rates.append(rate_per_deg)

        prev_total, prev_t_h, prev_taa_uw = curr_total, t_h, taa_uw_deg

    log(f"  スキップ数: {skipped}")
    
    if len(mid_taas) == 0:
        return pd.DataFrame(columns=['TAA_int', 'Terminator_Release'])

    # --- 補間処理 ---
    # Unwrapped TAA上で1度刻みの配列を作成し、レートを線形補間する
    min_taa = int(np.ceil(mid_taas[0]))
    max_taa = int(np.floor(mid_taas[-1]))
    target_taas_uw = np.arange(min_taa, max_taa + 1)
    
    interp_rates = np.interp(target_taas_uw, mid_taas, release_rates)
    
    # 1度刻みなので、rate(atoms/deg) * 1(deg) = atoms となりそのまま加算可能
    df_interp = pd.DataFrame({
        'TAA_int': target_taas_uw % 360,
        'Terminator_Release': interp_rates
    })
    
    # 同じ TAA_int (0~359) ごとに合計をとる
    df = df_interp.groupby('TAA_int', as_index=False)['Terminator_Release'].sum()
    return df


# ==========================================
# 統合処理: ターミネーター vs 日常域(逆算)
# ==========================================
def build_hybrid_breakdown(target_dir):
    df_term = compute_terminator_release(target_dir, ROTATION_PERIOD_SEC, ROTATION_SIGN,
                                         EFF_COS_TERMINATOR_MAX)
    df_gen = get_actual_dawn_generation_from_stats(target_dir)

    df = pd.merge(df_gen, df_term, on='TAA_int', how='left')
    df['Terminator_Release'] = df['Terminator_Release'].fillna(0.0)

    # 日常域(それ以外) = 公式のGen_Dawn_Total - ターミネーター近傍(実測)
    df['Routine_Contribution'] = df['Gen_Dawn_Total'] - df['Terminator_Release']

    n_negative = int((df['Routine_Contribution'] < 0).sum())
    if n_negative > 0:
        log(f"[警告] 日常域の逆算値が負になったビンが{n_negative}件あります。"
            f" 単位/binのズレの可能性があるため0にクリップします。")
        df['Routine_Contribution'] = df['Routine_Contribution'].clip(lower=0.0)

    df['Terminator_Fraction'] = np.where(
        df['Gen_Dawn_Total'] > 0, df['Terminator_Release'] / df['Gen_Dawn_Total'], np.nan
    )

    return df.sort_values('TAA_int').reset_index(drop=True)


def plot_hybrid_comparison(dfs_dict):
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    colors = {}
    palette = ['crimson', 'steelblue', 'darkgreen', 'purple']
    for i, label in enumerate(dfs_dict.keys()):
        colors[label] = palette[i % len(palette)]

    ax = axes[0]
    for label, df in dfs_dict.items():
        ax.plot(df['TAA_int'], df['Gen_Dawn_Total'], '-', linewidth=2.2,
                 label=f'{label}: Total (公式統計)', color=colors[label])
        ax.plot(df['TAA_int'], df['Terminator_Release'], ':', linewidth=1.8,
                 label=f'{label}: Terminator(実測/補間)', color=colors[label], alpha=0.7)
    ax.set_yscale('log')
    ax.set_ylabel('Atoms / TAA-bin')
    ax.set_title('Dawn合計 vs ターミネーター近傍(実測)')
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend(fontsize=8)

    ax = axes[1]
    for label, df in dfs_dict.items():
        ax.plot(df['TAA_int'], df['Routine_Contribution'], '-', linewidth=2.0,
                 label=f'{label}: Routine(逆算)', color=colors[label])
    ax.set_yscale('log')
    ax.set_ylabel('Atoms / TAA-bin')
    ax.set_title('日常域(逆算): Gen_Dawn_Total - Terminator_Release')
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend(fontsize=8)

    ax = axes[2]
    for label, df in dfs_dict.items():
        ax.plot(df['TAA_int'], df['Terminator_Fraction'], '-', linewidth=2.0,
                 label=label, color=colors[label])
    ax.set_ylabel('Terminator Fraction')
    ax.set_xlabel('TAA [deg]')
    ax.set_title('ターミネーター寄与率')
    ax.set_ylim(0, 1)
    ax.grid(True, ls='--', alpha=0.4)
    ax.legend(fontsize=8)

    axes[-1].set_xlim(0, 360)
    axes[-1].xaxis.set_major_locator(MultipleLocator(60))
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 70)
    print("=== 全期間合計での比較 ===")
    for label, df in dfs_dict.items():
        total = df['Gen_Dawn_Total'].sum()
        term = df['Terminator_Release'].sum()
        routine = df['Routine_Contribution'].sum()
        print(f"\n[{label}]")
        print(f"  Dawn合計(公式)     : {total:.4e}")
        print(f"  ターミネーター寄与 : {term:.4e}  ({term/total*100:.2f} %)")
        print(f"  日常域寄与(逆算)   : {routine:.4e}  ({routine/total*100:.2f} %)")
    print("=" * 70)


if __name__ == "__main__":
    dfs = {}
    for label, subdir in MODELS.items():
        full_path = os.path.join(BASE_DIR, subdir)
        if not os.path.exists(full_path):
            print(f"エラー: ディレクトリが存在しません: {full_path}")
            continue
        try:
            dfs[label] = build_hybrid_breakdown(full_path)
        except Exception as e:
            print(f"エラー ({label}): {e}")

    if len(dfs) > 0:
        plot_hybrid_comparison(dfs)