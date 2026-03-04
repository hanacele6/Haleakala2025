import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

# ==============================================================================
# Quality Control Parameters (Step 12の結果を尊重 + 形状厳密チェック)
# ==============================================================================
QC_PARAMS = {
    # 1. 基本数値チェック (Na_atoms_final.dat の値を判定)
    'max_relative_error': 1.0,  # Na原子数に対する誤差の許容比率（err/val <= 1.0）
    'max_peak_diff_nm': 0.005,  # [Step12算出] 3枚の画像の中心波長ズレの許容幅 [nm]

    # 2. スペクトル形状チェック (mainファイルをQC用に単独フィットして判定)
    'check_window_nm': 0.3,  # フィット対象範囲 [nm]
    'snr_threshold': 2.0,  # S/N比 (Amplitude / MAD_Noise)
    'residual_quality_threshold': 2.0,  # 残差品質 (Amplitude / Residual_MAD)
    'min_fit_vs_raw_ratio': 1.5,  # 過剰フィット抑制 (Fit_Amp / その他の最大値)
    'max_amp_err_ratio': 1.0,  # フィット不安定性抑制

    'sigma_min': 0.001,  # σ下限 (これより鋭いとノイズの疑い)
    'sigma_max': 0.060,  # σ上限 (これより広いと何かおかしい)

    'min_points_in_fwhm': 1.0,  # FWHMの幅の中に最低限必要なデータ点数

    # 3. クラスター判定
    'max_drift_threshold': 0.05,  # 波長ドリフト許容値

    # 4. 生存率チェック
    'min_survival_count': 3,  # 最低生存数
    'min_survival_rate': 0.33  # 最低生存率
}


# ==============================================================================
# Helper Functions
# ==============================================================================
def safe_float(val):
    try:
        return float(val)
    except:
        return np.nan


def calculate_taa(dist_au, v_rad_km_s):
    try:
        d, v = float(dist_au), float(v_rad_km_s)
        val = np.clip((0.387098 * (1 - 0.205630 ** 2) / d - 1) / 0.205630, -1.0, 1.0)
        return (360.0 - np.degrees(np.arccos(val))) if v < 0 else np.degrees(np.arccos(val))
    except:
        return np.nan


def gaussian_func(x, a, x0, sigma, c):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + c


def check_spectral_quality(file_path, target_wl=589.7558):
    try:
        data = np.loadtxt(file_path)
        if data.ndim < 2 or data.shape[0] < 5: return False, "Data Invalid/Short"
        wl, flux = data[:, 0], data[:, 1]
    except:
        return False, "File Read Error"

    win = QC_PARAMS['check_window_nm']
    mask = (wl >= target_wl - win) & (wl <= target_wl + win)
    w_cut, f_cut = wl[mask], flux[mask]

    if len(f_cut) < 5: return False, "Data Too Short"

    try:
        if len(w_cut) > 1:
            wav_step = np.median(np.diff(w_cut))
        else:
            wav_step = 0.01

        init_base = np.percentile(f_cut, 25)
        raw_amp = np.max(f_cut) - init_base
        peak_idx = np.argmax(f_cut)
        init_center = w_cut[peak_idx]

        p0 = [raw_amp, init_center, 0.02, init_base]
        bounds = ([0, w_cut[0], 0.001, -np.inf], [np.inf, w_cut[-1], 0.15, np.inf])

        popt, pcov = curve_fit(gaussian_func, w_cut, f_cut, p0=p0, bounds=bounds, maxfev=3000)
        h, c, s = popt[0], popt[1], popt[2]

        # --- 1. コントラストチェック ---
        peak_mask = (w_cut > c - 3 * s) & (w_cut < c + 3 * s)
        peak_flux = f_cut[peak_mask]
        continuum_flux = f_cut[~peak_mask]

        if len(continuum_flux) > 0 and len(peak_flux) > 0:
            peak_max_val = np.max(peak_flux)
            bg_max_val = np.max(continuum_flux)
            if bg_max_val <= 0: bg_max_val = 1e-9
            contrast = peak_max_val / bg_max_val
            if contrast < QC_PARAMS['min_fit_vs_raw_ratio']:
                return False, f"Low Contrast (PeakMax/BgMax={contrast:.2f} < {QC_PARAMS['min_fit_vs_raw_ratio']})"
        else:
            return False, "Peak or Continuum missing"

        # --- 2. FWHM内データ点数チェック ---
        fwhm_nm = 2.355 * s
        points_in_peak = fwhm_nm / wav_step
        if points_in_peak < QC_PARAMS['min_points_in_fwhm']:
            return False, f"Too Sharp (Points={points_in_peak:.1f} < {QC_PARAMS['min_points_in_fwhm']})"

        # --- 3. ノイズ推定 ---
        if len(continuum_flux) > 5:
            mad = np.median(np.abs(continuum_flux - np.median(continuum_flux)))
            noise = mad * 1.4826
        else:
            edge_flux = np.concatenate([f_cut[:3], f_cut[-3:]])
            mad = np.median(np.abs(edge_flux - np.median(edge_flux)))
            noise = mad * 1.4826

        if noise < 1e-9: noise = 1e-9
        snr = h / noise

        # --- 4. 残差品質チェック ---
        fitted = gaussian_func(w_cut, *popt)
        residuals = f_cut - fitted
        residual_std = np.median(np.abs(residuals - np.median(residuals))) * 1.4826
        fit_quality = h / (residual_std + 1e-9)

        if fit_quality < QC_PARAMS['residual_quality_threshold']: return False, f"Poor Fit Quality ({fit_quality:.1f})"
        if snr < QC_PARAMS['snr_threshold']: return False, f"Low S/N ({snr:.1f})"
        if s < QC_PARAMS['sigma_min']: return False, f"Too Narrow (Sigma={s:.4f})"
        if s > QC_PARAMS['sigma_max']: return False, f"Too Broad (Sigma={s:.4f})"

        perr = np.sqrt(np.diag(pcov))
        if (perr[0] / h) > QC_PARAMS['max_amp_err_ratio']: return False, "Unstable Fit"

        return True, "OK"
    except Exception as e:
        return False, f"Fit Calculation Failed ({str(e)})"


def find_largest_cluster(items, drift_threshold):
    valid_items = [x for x in items if x['center'] > 100]
    if not valid_items: return [], items
    sorted_items = sorted(valid_items, key=lambda x: x['center'])
    max_cluster = []
    for i in range(len(sorted_items)):
        cluster = []
        start_wl = sorted_items[i]['center']
        for j in range(i, len(sorted_items)):
            if sorted_items[j]['center'] - start_wl <= drift_threshold:
                cluster.append(sorted_items[j])
            else:
                break
        if len(cluster) > len(max_cluster):
            max_cluster = cluster
    cluster_indices = {x['index'] for x in max_cluster}
    rejected_items = [x for x in items if x['index'] not in cluster_indices]
    return max_cluster, rejected_items


# ==============================================================================
# Main Logic
# ==============================================================================
def run(run_info, config):
    output_dir = Path(run_info["output_dir"])
    csv_file_path = run_info["csv_path"]
    date_str = run_info["date"]
    col_conf = config.get("column_density", {})
    target_wl = col_conf.get("target_wavelength", 589.7558)

    print(f"\n--- 1D最終集計QC (Step 13: Clustering Method) ---")

    final_dat = output_dir / 'Na_atoms_final.dat'
    report_out = output_dir / f'Quality_Report_{date_str}.txt'
    summary_out = output_dir / f'Final_Summary_{date_str}.txt'

    if not final_dat.exists():
        print("  > Error: Na_atoms_final.dat not found.")
        return

    try:
        data = np.loadtxt(final_dat)
        if data.ndim == 1: data = data.reshape(1, -1)
        if data.shape[1] < 5:
            print("  > Error: Invalid format in Na_atoms_final.dat")
            return
        indices, atoms, errors, peak_diffs, center_wls = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]
    except Exception as e:
        print(f"Error reading data: {e}")
        return

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

    print("\n[Phase 1] Filtering Step12 Data...")
    all_data_points = []
    spec_dir = output_dir if (output_dir / "2_spectra").exists() == False else output_dir / "2_spectra"

    for i in range(len(indices)):
        idx = int(indices[i])
        val = atoms[i]
        err = errors[i]
        p_diff = peak_diffs[i]
        wl = center_wls[i]

        qa_status = True
        qa_reason = "OK"

        if val <= 0:
            qa_status = False;
            qa_reason = "Value <= 0"
        elif (err / val) > QC_PARAMS['max_relative_error']:
            qa_status = False;
            qa_reason = f"High Error Ratio ({(err / val):.2f})"
        elif p_diff > QC_PARAMS['max_peak_diff_nm']:
            qa_status = False;
            qa_reason = f"Large Peak Diff ({p_diff:.4f} nm)"

        if qa_status:
            # ソートして真ん中のファイルをMainとして扱う（元の仕様）
            cands = sorted(list(spec_dir.glob(f"MERCURY{idx}_tr*.exos.dat")))

            target_file = None
            if len(cands) >= 1:
                target_file = cands[len(cands) // 2]

            if target_file:
                is_good, reason = check_spectral_quality(target_file, target_wl)
                if not is_good:
                    qa_status = False;
                    qa_reason = reason

        row_idx = idx - 1
        taa = np.nan
        if not mercury_df.empty and 0 <= row_idx < len(mercury_df):
            row = mercury_df.iloc[row_idx]
            if 'true_anomaly_deg' in row:
                taa = safe_float(row['true_anomaly_deg'])
            elif 'mercury_sun_distance_au' in row:
                taa = calculate_taa(row['mercury_sun_distance_au'], row['mercury_sun_radial_velocity_km_s'])

        all_data_points.append({
            'index': i, 'idx': idx, 'val': val, 'err': err, 'taa': taa, 'center': wl,
            'is_valid': qa_status, 'reason': qa_reason
        })

    print(f"\n[Phase 2] Clustering Analysis...")
    candidates = [d for d in all_data_points if d['center'] > 100]
    drift_thr = QC_PARAMS['max_drift_threshold']
    best_cluster, rejected_by_cluster = find_largest_cluster(candidates, drift_thr)

    excluded_log = []
    reject_all = False

    for item in rejected_by_cluster:
        reason = item['reason'] if not item['is_valid'] else f"Outlier (Not in main cluster)"
        excluded_log.append(f"ID {item['idx']}: {reason} (WL={item['center']:.4f})")

    final_valid_data = []
    for item in best_cluster:
        if item['is_valid']:
            final_valid_data.append(item)
        else:
            excluded_log.append(f"ID {item['idx']}: {item['reason']} (In cluster)")

    n_total = len(indices)
    n_survived = len(final_valid_data)

    print(f"  > Total Candidates: {len(candidates)}")
    print(f"  > Largest Cluster Size: {len(best_cluster)}")
    print(f"  > Final Valid Data: {n_survived}")

    min_count = QC_PARAMS.get('min_survival_count', 3)
    min_rate = QC_PARAMS.get('min_survival_rate', 0.15)

    if n_survived < min_count or (n_survived / n_total) < min_rate:
        print(f"  > [Reject All] Too few valid data points ({n_survived})")
        excluded_log.insert(0, f"ALL REJECTED: Only {n_survived} valid points found (Cluster size={len(best_cluster)})")
        reject_all = True
        final_valid_data = []

        # --- Phase 3: Prepare Data (1D Only) ---
    valid_data = {'taa': [], 'atoms': []}
    if not reject_all:
        for item in final_valid_data:
            valid_data['atoms'].append(item['val'])
            valid_data['taa'].append(item['taa'])

    # --- Summary & Save ---
    n_valid = len(valid_data['atoms'])
    avg_atoms = np.mean(valid_data['atoms']) if n_valid > 0 else 0
    valid_errs = [d['err'] for d in final_valid_data]

    # 誤差の計算方法:
    # (単純平均): 各観測の典型的な誤差を表す
    # avg_err = np.mean(valid_errs) if valid_errs else 0

    # 旧方式 (二乗和平方根/N): 独立な誤差の合成として計算 (過去データとの一貫性のため採用)
    avg_err = np.sqrt(np.sum(np.array(valid_errs) ** 2)) / len(valid_errs) if valid_errs else 0

    print(f"  > Final Valid: {n_valid}/{n_total}")

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
        f.write("\nExclusion Details:\n")
        seen = set()
        for l in sorted(list(set(excluded_log)), key=lambda x: int(x.split()[1].rstrip(':')) if 'ID' in x else -1):
            if l not in seen:
                f.write(f" - {l}\n")
                seen.add(l)

    print(f"  > Report Saved: {report_out.name}")


if __name__ == '__main__':
    print("Use as module.")