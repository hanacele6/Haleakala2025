import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==============================================================================
# 設定 & 物理定数
# ==============================================================================
PI = np.pi

def gaussian_linear_baseline(x, height, center, sigma, const, linear):
    return height * np.exp(-(x - center) ** 2 / (2 * abs(sigma) ** 2)) + const + linear * x

# ==============================================================================
# 関数: 基本フィット (1D専用)
# ==============================================================================
def fit_single_spec(wl, flux, fit_config, target_wl):
    c_idx = np.argmin(np.abs(wl - target_wl))
    half_w = fit_config.get('fit_half_width_pix', 35)

    s = max(0, c_idx - half_w)
    e = min(len(wl), c_idx + half_w)

    x_data = wl[s:e]
    y_data = flux[s:e]

    if len(y_data) < 5: return 0.0, None, 0

    peak_idx_rel = np.argmax(y_data)
    peak_val = y_data[peak_idx_rel]
    min_val = np.min(y_data)

    p0 = [peak_val - min_val, x_data[peak_idx_rel], 0.005, min_val, 0]

    try:
        popt, _ = curve_fit(gaussian_linear_baseline, x_data, y_data, p0=p0, maxfev=10000)
    except RuntimeError:
        return 0.0, None, 0

    height, center_wl, sigma_wl = popt[0], popt[1], abs(popt[2])
    start_wl = center_wl - 3 * sigma_wl
    end_wl = center_wl + 3 * sigma_wl

    mask = (x_data >= start_wl) & (x_data <= end_wl)
    #integration_width_pix = np.sum(mask)
    wav_step = np.median(np.diff(x_data)) if len(x_data) > 1 else 0.01
    sigma_pix = sigma_wl / wav_step
    integration_width_pix = int(round(6 * sigma_pix)) + 1

    print(
        f"DEBUG width: {integration_width_pix}, sigma_wl={sigma_wl:.5f}, sigma_pix={sigma_pix:.2f}, wav_step={wav_step:.6f}")

    if integration_width_pix == 0: return 0.0, None, 0

    if fit_config.get('subtract_baseline', False):
        baseline_func = popt[3] + popt[4] * x_data
        counts = np.sum(y_data[mask] - baseline_func[mask])
    else:
        counts = np.sum(y_data[mask])

    return counts, popt, integration_width_pix

def process_total_atoms(paths, base_name, output_dir, gfac1, fit_config, target_wl):
    data_map = {}
    for k, p in paths.items():
        try:
            data_map[k] = np.loadtxt(p)
        except:
            pass

    if 'main' not in data_map: return None

    create_plots = fit_config.get('plot_config', {}).get('create_plots', False)
    if create_plots: plt.figure(figsize=(10, 6))

    colors = {'main': 'blue', 'plus': 'green', 'minus': 'red'}
    fit_results = {}
    centers = []
    main_center_wl = np.nan
    width_main = 1

    for k in ['minus', 'main', 'plus']:
        if k not in data_map:
            fit_results[k] = 0.0
            continue

        wl, flux = data_map[k][:, 0], data_map[k][:, 1]
        cnt, popt, width = fit_single_spec(wl, flux, fit_config, target_wl)
        fit_results[k] = cnt

        if popt is not None:
            centers.append(popt[1])
            if k == 'main':
                main_center_wl = popt[1]
                width_main = width

        if create_plots:
            c_idx = np.argmin(np.abs(wl - target_wl))
            hw = fit_config.get('fit_half_width_pix', 35)
            s, e = max(0, c_idx - hw), min(len(wl), c_idx + hw)
            plt.scatter(wl[s:e], flux[s:e], s=10, c=colors.get(k, 'k'), alpha=0.5, label=f"{k}")
            if popt is not None:
                x_sm = np.linspace(wl[s], wl[e - 1], 200)
                plt.plot(x_sm, gaussian_linear_baseline(x_sm, *popt), c=colors.get(k, 'k'), lw=1.5)

    if create_plots:
        plt.title(f"Fit: {base_name} (1D Total)")
        plt.legend()
        plt.grid(True, ls=':')
        plt.savefig(output_dir / f"plot_{base_name}_fit_1d.png")
        plt.close()

    if fit_results.get('main', 0) == 0: return None

    wing_width = fit_config.get('noise_wing_width_pix', 60)
    cts_main = data_map['main'][:, 1]

    if len(cts_main) > 2 * wing_width:
        wings_combined = np.concatenate((cts_main[0:wing_width], cts_main[-wing_width:]))
        sigma_noise = np.std(wings_combined, ddof=1)
    else:
        sigma_noise = 0.0

    err_stat = sigma_noise * np.sqrt(width_main)
    cts2 = fit_results['main']
    err_sys = max(abs(fit_results.get('plus', 0) - cts2), abs(fit_results.get('minus', 0) - cts2))

    N_total = cts2 / gfac1
    N_err = (err_stat + err_sys) / gfac1
    peak_diff = (max(centers) - min(centers)) if len(centers) >= 2 else 0.0


    print(f"DEBUG [{base_name}] cts2={cts2:.4e}, err_stat={err_stat:.4e}, err_sys={err_sys:.4e}")
    print(f"DEBUG [{base_name}] N_total={N_total:.4e}, N_err={N_err:.4e}")

    return N_total, N_err, peak_diff, main_center_wl

# ==============================================================================
# Main
# ==============================================================================
def run(run_info, config):
    output_dir = Path(run_info["output_dir"])
    csv_path = run_info["csv_path"]

    col_conf = config.get("column_density", {})
    target_wl = col_conf.get("target_wavelength", 589.7558)
    fit_config = col_conf.get("fit_config", {})
    fit_config['plot_config'] = col_conf.get('plot_config', {'create_plots': True})

    F_lambda_cgs = 5.18e14 * 1e7
    lambda_cm = target_wl * 1e-7
    JL_nu = F_lambda_cgs * (lambda_cm ** 2 / (299792.458 * 1e5))
    sigma_D1_nu = PI * (4.8032e-10) ** 2 / 9.109e-28 / (299792.458 * 1e5) * 0.327
    GFAC_CONST = sigma_D1_nu * JL_nu



    print(f"\n--- 1D原子数密度計算 (Step 12: Total Abundance Only) ---")

    try:
        df = pd.read_csv(csv_path)
    except:
        return

    res_final = []
    target_df = df[df['Type'] == 'MERCURY'].reset_index(drop=True)

    for i, row in target_df.iterrows():
        file_num = i + 1
        base = f"MERCURY{file_num}_tr"

        cands = sorted(list(output_dir.glob(f"{base}*.exos.dat")))
        if not cands: cands = sorted(list((output_dir / "2_spectra").glob(f"{base}*.exos.dat")))
        if not cands: continue

        paths = {'main': cands[len(cands) // 2]} if len(cands) < 3 else {'minus': cands[0], 'main': cands[1], 'plus': cands[2]}

        print(f"  > Processing: {paths['main'].name}")

        gamma = row.get('g_factor', np.nan)
        dist = row.get('mercury_sun_distance_au', np.nan)

        if pd.isna(gamma) or pd.isna(dist): continue

        gfac1 = (GFAC_CONST * gamma) / (dist ** 2)

        print(f"DEBUG [run] GFAC_CONST={GFAC_CONST:.4e}, gamma={gamma:.4e}, dist={dist:.4f}, gfac1={gfac1:.4e}")

        # 1D全体計算のみを実行
        ret = process_total_atoms(paths, base, output_dir, gfac1, fit_config, target_wl)
        if ret:
            N, Err, PDiff, MainCenter = ret
            res_final.append([file_num, N, Err, PDiff, MainCenter])

    if res_final:
        np.savetxt(output_dir / 'Na_atoms_final.dat', np.array(res_final),
                   fmt="%d %.4e %.4e %.6f %.6f", header="Index Total_Atoms Error Peak_Diff_nm Center_WL_nm")
        print("  > Saved: Na_atoms_final.dat")