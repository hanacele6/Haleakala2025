import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
import sys

# ==============================================================================
# Quality Control Parameters
# ==============================================================================
QC_PARAMS = {
    # 1. 数値チェック
    'max_relative_error': 0.5,

    # 2. スペクトル形状チェック (S/N比や形状を見る)
    'check_window_nm': 0.8,
    'snr_threshold': 2.0,
    'max_fit_vs_raw_ratio': 1.2,

    # 3. 全滅判定 (Drift) & 個別判定 (Outlier) の基準
    'max_drift_threshold': 0.05,  # (nm) これ以上ばらついていたらアウト

    # 4. 生存率チェック
    'min_survival_rate': 0.15
}


# ==============================================================================
# Helper: S/N比と形状だけをチェックする関数
# ==============================================================================
def gaussian_func(x, a, x0, sigma, c):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + c


def check_spectral_quality(file_path, target_wl=589.7558):
    """
    S/N比と形状がまともかどうかだけを判定する（合格/不合格のフラグを返す）
    """
    try:
        data = np.loadtxt(file_path)
        wl = data[:, 0]
        flux = data[:, 1]
    except:
        return False, "File Read Error"

    win = QC_PARAMS['check_window_nm']
    mask = (wl >= target_wl - win) & (wl <= target_wl + win)
    w_cut, f_cut = wl[mask], flux[mask]

    if len(f_cut) < 5: return False, "Data Too Short"

    try:
        # 簡易チェック用にフィット（パラメータは大まかでOK）
        wav_step = np.median(np.diff(w_cut)) if len(w_cut) > 1 else 0.01
        init_base = np.median(f_cut)
        raw_amp = np.max(f_cut) - init_base
        peak_idx = np.argmax(f_cut)
        init_center = w_cut[peak_idx]

        p0 = [raw_amp, init_center, 0.02, init_base]
        bounds = ([0, target_wl - 0.15, 0.005, -np.inf], [np.inf, target_wl + 0.15, 0.1, np.inf])

        popt, pcov = curve_fit(gaussian_func, w_cut, f_cut, p0=p0, bounds=bounds, maxfev=2000)

        h, c, s = popt[0], popt[1], popt[2]

        # S/Nチェック (MAD)
        model = gaussian_func(w_cut, *popt)
        resid = f_cut - model
        mad = np.median(np.abs(resid - np.median(resid)))
        noise = mad * 1.4826 if mad > 1e-9 else 1e-9
        snr = h / noise

        if snr < QC_PARAMS['snr_threshold']: return False, f"Low S/N ({snr:.1f})"
        if h / (np.max(f_cut) - np.min(f_cut)) > QC_PARAMS['max_fit_vs_raw_ratio']:
            return False, "Unrealistic Fit Height"

        return True, "OK"

    except:
        return False, "Fit Failed"


# (Helper Functions: 計算用)
def safe_float(val):
    try:
        return float(val)
    except:
        return np.nan


def calculate_taa(dist, v_rad):
    try:
        d, v = float(dist), float(v_rad)
        val = np.clip((0.387098 * (1 - 0.205630 ** 2) / d - 1) / 0.205630, -1.0, 1.0)
        nu = np.degrees(np.arccos(val))
        return (360.0 - nu) if v < 0 else nu
    except:
        return np.nan


# (Plotting Functions)
def plot_regional_density(taa, n, s, eq, out_dir, d_str):
    try:
        if len(taa) == 0: return
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(taa, n, c='b', label='North', marker='^')
        ax.scatter(taa, s, c='r', label='South', marker='v')
        ax.scatter(taa, eq, c='g', label='Equator', marker='o')
        ax.set_title(f"Regional Density\n{d_str}")
        ax.set_yscale('log')
        ax.legend()
        plt.savefig(out_dir / "Regional_Density_Variation.png");
        plt.close()
    except:
        pass


def plot_equatorial_lt_density(taa, d, ss, dusk, out_dir, d_str):
    try:
        if len(taa) == 0: return
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(taa, d, c='c', label='Dawn', marker='<')
        ax.scatter(taa, ss, c='m', label='SSP', marker='*')
        ax.scatter(taa, dusk, c='k', label='Dusk', marker='>')
        ax.set_title(f"Equatorial Density (LT)\n{d_str}")
        ax.set_yscale('log')
        ax.legend()
        plt.savefig(out_dir / "Equatorial_LT_Density.png");
        plt.close()
    except:
        pass


# ==============================================================================
# Main Logic
# ==============================================================================
def run_qc_analysis(output_dir, csv_file_path=None, date_str="Debug"):
    output_dir = Path(output_dir)
    final_dat = output_dir / 'Na_atoms_final.dat'
    region_dat = output_dir / 'Na_atoms_regions.dat'
    report_out = output_dir / f'Quality_Report_{date_str}.txt'
    summary_out = output_dir / f'Final_Summary_{date_str}.txt'

    print(f"\n--- 最終集計 (Step 13: Holistic Check) ---")

    if not final_dat.exists():
        print("  > Error: Na_atoms_final.dat not found.")
        return

    # 1. データの読み込み
    try:
        data = np.loadtxt(final_dat)
        if data.ndim == 1: data = data.reshape(1, -1)
        if data.shape[1] < 5:
            print("  > Error: Na_atoms_final.dat format invalid (need 5 columns).")
            return
        indices = data[:, 0]
        atoms = data[:, 1]
        errors = data[:, 2]
        center_wls = data[:, 4]  # Step 12で計算された中心波長
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 領域データ・CSV読み込み (省略せず記載)
    reg_map = {}
    if region_dat.exists():
        try:
            rdata = np.loadtxt(region_dat)
            if rdata.ndim == 1: rdata = rdata.reshape(1, -1)
            for r in rdata:
                idx = int(r[0])
                if len(r) >= 8:
                    reg_map[idx] = {'N': r[2], 'S': r[3], 'Eq': r[4], 'D': r[5], 'SS': r[6], 'Dusk': r[7]}
                elif len(r) >= 5:
                    reg_map[idx] = {'N': r[2], 'S': r[3], 'Eq': r[4], 'D': np.nan, 'SS': np.nan, 'Dusk': np.nan}
        except:
            pass

    mercury_df = pd.DataFrame()
    if csv_file_path:
        try:
            if Path(csv_file_path).is_dir():
                cands = list(Path(csv_file_path).glob("*.csv"))
                if cands: mercury_df = pd.read_csv(cands[0])
            else:
                mercury_df = pd.read_csv(csv_file_path)
            if not mercury_df.empty:
                mercury_df = mercury_df.loc[:, ~mercury_df.columns.duplicated()]
                mercury_df = mercury_df[mercury_df['Type'] == 'MERCURY'].reset_index(drop=True)
        except:
            pass

    # --- Phase 1: Holistic Quality Check (S/N判定 + データ収集) ---
    print("\n[Phase 1] Collecting Data & Checking Quality...")

    # 全てのデータをリスト化する（不合格でも一旦入れる）
    all_data_points = []

    spec_dir = output_dir if (output_dir / "2_spectra").exists() == False else output_dir / "2_spectra"

    for i in range(len(indices)):
        idx = int(indices[i])
        val, err, wl = atoms[i], errors[i], center_wls[i]

        # 1. 基本チェック
        qa_status = True
        qa_reason = "OK"

        if val <= 0:
            qa_status = False;
            qa_reason = "Value <= 0"
        elif (err / val) > QC_PARAMS['max_relative_error']:
            qa_status = False;
            qa_reason = "High Error Ratio"

        # 2. スペクトル形状チェック (既存の門番)
        # ★重要: ここで False になっても、ループを continue せず、
        #        「品質NG」というタグをつけてリストには入れる。
        if qa_status:  # 値チェックがOKな場合のみスペクトルを見る
            candidates = list(spec_dir.glob(f"MERCURY{idx}_tr*.exos.dat"))
            if len(candidates) > 0:
                is_good_shape, reason = check_spectral_quality(candidates[0])
                if not is_good_shape:
                    qa_status = False
                    qa_reason = f"Bad Shape ({reason})"
            else:
                qa_status = False
                qa_reason = "File Not Found"

        # TAA取得
        row_idx = idx - 1
        taa = np.nan
        if not mercury_df.empty and 0 <= row_idx < len(mercury_df):
            row = mercury_df.iloc[row_idx]
            if 'true_anomaly_deg' in row:
                taa = safe_float(row['true_anomaly_deg'])
            elif 'mercury_sun_distance_au' in row:
                taa = calculate_taa(row['mercury_sun_distance_au'], row['mercury_sun_radial_velocity_km_s'])

        # ★ 全データをリストに追加 (Phase 2の統計に使うため)
        all_data_points.append({
            'index': i, 'idx': idx,
            'val': val, 'err': err, 'taa': taa, 'center': wl,
            'is_valid': qa_status, 'reason': qa_reason
        })

    # --- Phase 2: Drift Check (using ALL available wavelengths) ---
    # ここでは「S/Nが悪くて不合格になったデータ」の波長も含めて統計を取る
    # これにより、ノイズでフィットが暴れているデータがあれば、Driftが大きくなり、正しく全滅判定できる

    available_centers = [d['center'] for d in all_data_points if d['center'] > 100]  # 0とかNaNは流石に除く

    print(f"\n[Phase 2] Holistic Drift Check (N={len(available_centers)})...")
    reject_all = False
    excluded_log = []

    if len(available_centers) >= 4:
        sorted_c = np.sort(available_centers)
        cut_n = 1 if len(sorted_c) < 10 else 2

        core = sorted_c[cut_n: -cut_n]
        drift = core[-1] - core[0]

        print(f"  > Centers: {sorted_c}")
        print(f"  > Trimmed Drift: {drift:.4f} nm (Limit: {QC_PARAMS['max_drift_threshold']})")

        if drift > QC_PARAMS['max_drift_threshold']:
            print(f"  > [Reject All] Unstable Drift")
            excluded_log.append(f"ALL REJECTED: Large Drift ({drift:.4f} nm)")
            reject_all = True
    elif len(available_centers) > 0:
        print("  > Warning: Not enough data for robust drift check.")

    # 生存率チェック (合格予定者の割合)
    n_total = len(indices)
    if not reject_all:
        valid_count = sum(1 for d in all_data_points if d['is_valid'])
        rate = valid_count / n_total if n_total > 0 else 0
        if rate < QC_PARAMS['min_survival_rate']:
            print(f"  > [Reject All] Low Survival ({rate:.1%})")
            excluded_log.append("ALL REJECTED: Low Survival Rate")
            reject_all = True

    # --- Phase 3: Final Selection & Outlier Check ---
    print("\n[Phase 3] Final Selection...")
    valid_data = {'taa': [], 'atoms': [], 'n': [], 's': [], 'eq': [], 'd': [], 'ss': [], 'dusk': []}

    if not reject_all:
        # 中央値を計算 (S/N悪いデータも含めた全体の中央値を使う)
        med_c = np.median(available_centers) if available_centers else 589.7558
        limit = QC_PARAMS['max_drift_threshold']
        print(f"  > Median Center: {med_c:.4f}")

        for item in all_data_points:
            # 1. そもそもPhase 1 (S/N等) で落ちていたら除外
            if not item['is_valid']:
                excluded_log.append(f"ID {item['idx']}: {item['reason']}")
                continue

            # 2. Phase 3 (Outlier) チェック
            # 日としては合格でも、個別に大きくズレているデータはここで捨てる
            diff = abs(item['center'] - med_c)
            if diff > limit:
                excluded_log.append(f"ID {item['idx']}: Outlier (Diff={diff:.4f})")
                continue

            # --- 合格 ---
            valid_data['atoms'].append(item['val'])
            valid_data['taa'].append(item['taa'])

            idx = item['idx']
            N, S, Eq, D, SS, Du = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            if idx in reg_map:
                r = reg_map[idx]
                N, S, Eq, D, SS, Du = r['N'], r['S'], r['Eq'], r['D'], r['SS'], r['Dusk']

            def flt(v):
                return v if v > 0 else np.nan

            valid_data['n'].append(flt(N));
            valid_data['s'].append(flt(S));
            valid_data['eq'].append(flt(Eq))
            valid_data['d'].append(flt(D));
            valid_data['ss'].append(flt(SS));
            valid_data['dusk'].append(flt(Du))

    # --- Summary ---
    n_valid = len(valid_data['atoms'])
    avg_atoms = np.mean(valid_data['atoms']) if n_valid > 0 else 0
    valid_errs = [d['err'] for d in all_data_points if d['val'] in valid_data['atoms']]
    avg_err = np.mean(valid_errs) if valid_errs else 0

    print(f"\n[Result] Final Valid: {n_valid}/{n_total}")
    print(f"Excluded Count: {len(excluded_log)}")

    # Save
    pa = safe_float(mercury_df.iloc[0]['phase_angle_deg']) if not mercury_df.empty else 0
    taa_avg = np.nanmean(valid_data['taa']) if valid_data['taa'] else 0

    with open(summary_out, 'w') as f:
        f.write(f"{pa} {taa_avg} {avg_atoms} {avg_err}\n")

    grade = "D"
    if n_valid > 0:
        s_rate = (n_valid / n_total * 100)
        grade = "C"
        if s_rate >= 70: grade = "B"
        if s_rate >= 90: grade = "A"

    with open(report_out, 'w') as f:
        f.write(f"Quality Report ({date_str})\nGrade: {grade}\nSurvival: {n_valid}/{n_total}\n")
        f.write("\nExcluded:\n")
        for l in excluded_log: f.write(f" - {l}\n")
    print(f"  > Report Saved: {report_out.name}")

    if len(valid_data['n']) > 0:
        plot_regional_density(valid_data['taa'], valid_data['n'], valid_data['s'], valid_data['eq'], output_dir,
                              date_str)
        plot_equatorial_lt_density(valid_data['taa'], valid_data['d'], valid_data['ss'], valid_data['dusk'], output_dir,
                                   date_str)


if __name__ == '__main__':
    target_dir = r"C:/Users/hanac/University/Senior/Mercury/Haleakala2025/test4_output/20250818"
    csv_path = r"C:/Users/hanac/University/Senior/Mercury/Haleakala2025/2026ver/csvs"

    if Path(target_dir).exists():
        run_qc_analysis(target_dir, csv_path, date_str="DebugRun")
    else:
        print("Error: Path not found.")