import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.ndimage import center_of_mass
import csv

# ==============================================================================
# 設定 & 物理定数
# ==============================================================================
PI = np.pi

def gaussian_linear_baseline(x, height, center, sigma, const, linear):
    return height * np.exp(-(x - center) ** 2 / (2 * abs(sigma) ** 2)) + const + linear * x

# ==============================================================================
# 関数: 基本フィット
# ==============================================================================
def fit_single_spec(wl, flux, fit_config, target_wl):
    """
    1Dスペクトルに対してガウスフィットを行い面積を返す。
    【修正点】
    - QCモード削除
    - リトライ/救済措置(Fallback)削除 -> 失敗時は 0 を返す
    - ベースライン処理は config に完全従属
    """
    # --- 1. データの切り出し ---
    c_idx = np.argmin(np.abs(wl - target_wl))
    half_w = fit_config.get('fit_half_width_pix', 35)

    # 配列外参照を防ぐためのクリップ
    s = max(0, c_idx - half_w)
    e = min(len(wl), c_idx + half_w)

    x_data = wl[s:e]
    y_data = flux[s:e]

    # データ点が少なすぎる場合は解析不可とする
    if len(y_data) < 5:
        return 0.0, None, 0

    # --- 2. 初期値推定 ---
    # x_data は波長そのものなので、indexではなく値で扱う必要がありますが、
    # curve_fitの安定性のため、旧式同様 index (0, 1, 2...) でフィットしてから波長に戻すか、
    # あるいは旧式のように pixel index ベースで計算するのが安全ですが、
    # ここでは新型の構造（波長配列を渡す）に合わせて x_data (波長) でフィットします。

    peak_idx_rel = np.argmax(y_data)
    peak_val = y_data[peak_idx_rel]
    min_val = np.min(y_data)

    p0 = [
        peak_val - min_val,  # height
        x_data[peak_idx_rel],  # center (wavelength)
        0.005,  # sigma (approx 0.005 nm or similar)
        min_val,  # const (baseline)
        0  # linear (baseline slope)
    ]

    # --- 3. フィッティング (旧式: 一発勝負) ---
    try:
        # 旧式同様、boundsなし、失敗したら例外へ
        popt, _ = curve_fit(gaussian_linear_baseline, x_data, y_data, p0=p0, maxfev=10000)
    except RuntimeError:
        # フィット失敗時は 0 を返す (旧式の挙動)
        return 0.0, None, 0

    # --- 4. 積分範囲の決定 (3シグマ) ---
    height, center_wl, sigma_wl = popt[0], popt[1], abs(popt[2])

    # 積分範囲（波長）
    start_wl = center_wl - 3 * sigma_wl
    end_wl = center_wl + 3 * sigma_wl

    # 波長範囲をマスクに変換
    mask = (x_data >= start_wl) & (x_data <= end_wl)
    integration_width_pix = np.sum(mask)  # 統計誤差計算用にピクセル数をカウント

    if integration_width_pix == 0:
        return 0.0, None, 0

    # --- 5. 原子数(カウント)計算 ---
    # configの指示に厳密に従う
    do_sub = fit_config.get('subtract_baseline', False)

    if do_sub:
        # ベースラインを引く
        baseline_func = popt[3] + popt[4] * x_data
        counts = np.sum(y_data[mask] - baseline_func[mask])
    else:
        # ベースラインを引かない (生データの積分)
        counts = np.sum(y_data[mask])


    return counts, popt, integration_width_pix


def process_total_atoms(paths, base_name, output_dir, gfac1, fit_config, target_wl):
    """
    全体の原子数計算。
    【修正点】
    - ノイズレベル(統計誤差)の計算を復活
    - 誤差 = 統計誤差 + 系統誤差(シフト間の差)
    """
    # データの読み込み
    data_map = {}
    for k, p in paths.items():
        try:
            # unpack=Trueしないと (N, 2) になるので注意。ここでは新型に合わせてそのまま読む
            data_map[k] = np.loadtxt(p)
        except:
            pass

    if 'main' not in data_map: return None

    # --- プロット準備 ---
    create_plots = fit_config.get('plot_config', {}).get('create_plots', False)
    if create_plots:
        plt.figure(figsize=(10, 6))

    colors = {'main': 'blue', 'plus': 'green', 'minus': 'red'}
    fit_results = {}  # 結果格納用
    centers = []  # 中心波長格納用

    # メイン画像の中心波長・積分幅
    main_center_wl = np.nan
    width_main = 1  # ゼロ除算防止の初期値

    # --- メインループ (Main, Plus, Minus) ---
    for k in ['minus', 'main', 'plus']:
        if k not in data_map:
            fit_results[k] = 0.0
            continue

        wl, flux = data_map[k][:, 0], data_map[k][:, 1]

        # フィット実行 (QCなし、失敗時は0)
        cnt, popt, width = fit_single_spec(wl, flux, fit_config, target_wl)
        fit_results[k] = cnt

        if popt is not None:
            centers.append(popt[1])
            if k == 'main':
                main_center_wl = popt[1]
                width_main = width

        # プロット作成
        if create_plots:
            # 表示範囲の設定
            c_idx = np.argmin(np.abs(wl - target_wl))
            hw = fit_config.get('fit_half_width_pix', 35)
            s, e = max(0, c_idx - hw), min(len(wl), c_idx + hw)

            plt.scatter(wl[s:e], flux[s:e], s=10, c=colors.get(k, 'k'), alpha=0.5, label=f"{k}")
            if popt is not None:
                x_sm = np.linspace(wl[s], wl[e - 1], 200)
                plt.plot(x_sm, gaussian_linear_baseline(x_sm, *popt), c=colors.get(k, 'k'), lw=1.5)

    # プロット保存
    if create_plots:
        plt.title(f"Fit: {base_name}")
        plt.legend()
        plt.grid(True, ls=':')
        plot_path = output_dir / f"plot_{base_name}_fit.png"
        plt.savefig(plot_path)
        plt.close()

    # --- 誤差計算 (旧式ロジックの復活) ---
    if fit_results.get('main', 0) == 0:
        return None  # メインのフィットに失敗したら結果なしとする

    # 1. 統計誤差 (Statistical Error) の計算
    # ウイング部分のノイズレベルを計算
    wing_width = fit_config.get('noise_wing_width_pix', 60)
    cts_main = data_map['main'][:, 1]  # Fluxのみ抽出

    if len(cts_main) > 2 * wing_width:
        wing1 = cts_main[0:wing_width]
        wing2 = cts_main[-wing_width:]
        wings_combined = np.concatenate((wing1, wing2))
        sigma_noise = np.std(wings_combined, ddof=1)
    else:
        sigma_noise = 0.0  # データが短すぎる場合

    # 統計誤差 = ノイズ * sqrt(積分幅)
    # width_main は fit_single_spec から返ってきたピクセル数
    err_stat = sigma_noise * np.sqrt(width_main)

    # 2. 系統誤差 (Systematic Error) の計算
    cts2 = fit_results['main']
    ctsp = fit_results.get('plus', 0)
    ctsm = fit_results.get('minus', 0)

    # シフトデータとの差分の大きい方
    errp = abs(ctsp - cts2)
    errm = abs(ctsm - cts2)
    err_sys = max(errp, errm)

    # 3. 最終計算
    N_total = cts2 / gfac1
    # 誤差は (統計 + 系統) / gfac
    N_err = (err_stat + err_sys) / gfac1

    # 3枚の画像のピーク位置のズレ (最大 - 最小)
    peak_diff = (max(centers) - min(centers)) if len(centers) >= 2 else 0.0

    sigma_noise = np.std(wings_combined, ddof=1)
    print(f"DEBUG paths: minus={paths['minus'].name}, main={paths['main'].name}, plus={paths['plus'].name}")
    print(f"DEBUG err_sys: {err_sys:.4e}, errp: {errp:.4e}, errm: {errm:.4e}")
    print(f"DEBUG sigma_noise: {sigma_noise:.4e}, wing_width: {wing_width}, wings_len: {len(wings_combined)}")

    # 戻り値: 原子数, 誤差, ピークズレ, メイン中心波長
    return N_total, N_err, peak_diff, main_center_wl

# ==============================================================================
# 関数: 領域別計算
# ==============================================================================
def get_geometry_masks(data_sub, geom_params):
    nx = geom_params['nx']
    ny = geom_params['ny']
    plate_scale = geom_params['plate_scale']
    diameter_arcsec = geom_params['diameter_arcsec']
    pa = geom_params.get('phase_angle', 0.0)

    n_fibers = data_sub.shape[0]
    flux_map_1d = np.sum(data_sub, axis=1)

    if len(flux_map_1d) != nx * ny:
        return None, None, None, None, None, None, None

    img_2d = flux_map_1d.reshape(ny, nx)
    threshold = np.max(img_2d) * 0.2
    img_masked = np.where(img_2d > threshold, img_2d, 0)
    cy, cx = center_of_mass(img_masked)

    if np.isnan(cy) or np.isnan(cx):
        return None, None, None, None, None, None, None

    r_pix = (diameter_arcsec / 2.0) / plate_scale
    boundary_pix = 0.3 * r_pix

    mask_n, mask_s, mask_eq = [], [], []
    mask_dawn, mask_ss, mask_dusk = [], [], []

    for i in range(n_fibers):
        y = i // nx
        x = i % nx
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        if dist > r_pix * 1.1: continue
        dy = y - cy
        if dy > boundary_pix:
            mask_n.append(i)
        elif dy < -boundary_pix:
            mask_s.append(i)
        else:
            mask_eq.append(i)
            r_local = np.sqrt(max(0.0001, r_pix ** 2 - dy ** 2))
            sin_theta = np.clip((x - cx) / r_local, -1.0, 1.0)
            theta_pix = np.arcsin(sin_theta) * (180.0 / PI)
            delta_theta = theta_pix - pa
            if delta_theta < -20.0:
                mask_dawn.append(i)
            elif delta_theta > 20.0:
                mask_dusk.append(i)
            else:
                mask_ss.append(i)

    debug_info = f"Center=({cx:.1f}, {cy:.1f}), Rd={r_pix:.1f}pix, PA={pa:.1f}deg"
    return mask_n, mask_s, mask_eq, mask_dawn, mask_ss, mask_dusk, debug_info


def filter_bad_fibers(idx_list, data_sub, wl, target_wl, fit_config):
    valid_idx = []
    c_idx = np.argmin(np.abs(wl - target_wl))
    hw = fit_config.get('fit_half_width_pix', 35)
    s, e = max(0, c_idx - hw), min(len(wl), c_idx + hw)
    fluxes = []
    for i in idx_list:
        y_d = data_sub[i, s:e]
        bg = np.nanmean(np.concatenate((y_d[:5], y_d[-5:])))
        peak_flux = np.sum(y_d - bg)
        fluxes.append(peak_flux)
    fluxes = np.array(fluxes)
    positive_mask = fluxes > 0
    if np.sum(positive_mask) == 0: return []
    med_pos = np.median(fluxes[positive_mask])
    upper_limit = med_pos * 10
    for j, i in enumerate(idx_list):
        if 0 < fluxes[j] <= upper_limit: valid_idx.append(i)
    return valid_idx


def process_regions_from_sub_fits(exos_path, target_wl, gfac1, geom_params, cts2MR, fit_config):
    try:
        sub_name = exos_path.name.replace(".exos.dat", ".sub.fits")
        sub_fits_path = exos_path.parent / sub_name
        if not sub_fits_path.exists():
            candidates = list(exos_path.parent.parent.glob(f"**/{sub_name}"))
            if candidates: sub_fits_path = candidates[0]
        if not sub_fits_path.exists():
            return None, f"Sub fits not found: {sub_name}"

        with fits.open(sub_fits_path) as hdul:
            data_sub = hdul[0].data
            header = hdul[0].header
            nx = header['NAXIS1']
            crval1 = header['CRVAL1']
            cdelt1 = header['CDELT1']
            wl = crval1 + np.arange(nx) * cdelt1

        if geom_params['enabled']:
            idx_n, idx_s, idx_eq, idx_dawn, idx_ss, idx_dusk, debug_msg = get_geometry_masks(data_sub, geom_params)
        else:
            return None, "Geometry params disabled"

        if idx_n is None: return None, "Geometry mask failed"

        idx_n = filter_bad_fibers(idx_n, data_sub, wl, target_wl, fit_config)
        idx_s = filter_bad_fibers(idx_s, data_sub, wl, target_wl, fit_config)
        idx_eq = filter_bad_fibers(idx_eq, data_sub, wl, target_wl, fit_config)
        idx_dawn = filter_bad_fibers(idx_dawn, data_sub, wl, target_wl, fit_config)
        idx_ss = filter_bad_fibers(idx_ss, data_sub, wl, target_wl, fit_config)
        idx_dusk = filter_bad_fibers(idx_dusk, data_sub, wl, target_wl, fit_config)

        spec_n = np.sum(data_sub[idx_n, :], axis=0) if idx_n else np.zeros_like(wl)
        spec_s = np.sum(data_sub[idx_s, :], axis=0) if idx_s else np.zeros_like(wl)
        spec_eq = np.sum(data_sub[idx_eq, :], axis=0) if idx_eq else np.zeros_like(wl)
        spec_dawn = np.sum(data_sub[idx_dawn, :], axis=0) if idx_dawn else np.zeros_like(wl)
        spec_ss = np.sum(data_sub[idx_ss, :], axis=0) if idx_ss else np.zeros_like(wl)
        spec_dusk = np.sum(data_sub[idx_dusk, :], axis=0) if idx_dusk else np.zeros_like(wl)

        if idx_n: spec_n /= len(idx_n)
        if idx_s: spec_s /= len(idx_s)
        if idx_eq: spec_eq /= len(idx_eq)
        if idx_dawn: spec_dawn /= len(idx_dawn)
        if idx_ss: spec_ss /= len(idx_ss)
        if idx_dusk: spec_dusk /= len(idx_dusk)

        cnt_n, _, _ = fit_single_spec(wl, spec_n, fit_config, target_wl)
        cnt_s, _, _ = fit_single_spec(wl, spec_s, fit_config, target_wl)
        cnt_eq, _, _ = fit_single_spec(wl, spec_eq, fit_config, target_wl)
        cnt_dawn, _, _ = fit_single_spec(wl, spec_dawn, fit_config, target_wl)
        cnt_ss, _, _ = fit_single_spec(wl, spec_ss, fit_config, target_wl)
        cnt_dusk, _, _ = fit_single_spec(wl, spec_dusk, fit_config, target_wl)

        r_mercury_cm = 2.4397e8
        a_cm2 = PI * (r_mercury_cm ** 2)
        r_pix = (geom_params['diameter_arcsec'] / 2.0) / geom_params['plate_scale']
        a_pix = PI * (r_pix ** 2)
        factor_cd = (a_pix * cts2MR) / (gfac1 * a_cm2)

        val_n, val_s, val_eq = cnt_n * factor_cd, cnt_s * factor_cd, cnt_eq * factor_cd
        val_dawn, val_ss, val_dusk = cnt_dawn * factor_cd, cnt_ss * factor_cd, cnt_dusk * factor_cd

        debug_msg += f" | Survived Fibers: N({len(idx_n)}), S({len(idx_s)}), Eq({len(idx_eq)})"
        return [val_n, val_s, val_eq, val_dawn, val_ss, val_dusk], debug_msg

    except Exception as e:
        print(f"    [Region Exception] {e}")
        return None, str(e)


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

    geom_conf = {
        'enabled': True,
        'plate_scale': col_conf.get('plate_scale', 1.0),
        'nx': config.get('resample', {}).get('params', {}).get('n_fib_x', 10),
        'ny': config.get('resample', {}).get('params', {}).get('n_fib_y', 12),
        'diameter_arcsec': 0.0,
        'phase_angle': 0.0
    }

    calib_map = {}
    calib_file = output_dir / "calibration_factors.csv"
    if calib_file.exists():
        try:
            with open(calib_file, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        calib_map[row[0]] = float(row[1])
        except:
            pass

    F_lambda_cgs = 5.18e14 * 1e7
    lambda_cm = target_wl * 1e-7
    JL_nu = F_lambda_cgs * (lambda_cm ** 2 / (299792.458 * 1e5))
    sigma_D1_nu = PI * (4.8032e-10) ** 2 / 9.109e-28 / (299792.458 * 1e5) * 0.327
    GFAC_CONST = sigma_D1_nu * JL_nu

    print(f"\n--- 原子数密度計算 (Step 12: Add Center WL) ---")

    try:
        df = pd.read_csv(csv_path)
    except:
        return

    res_final = []
    res_region = []
    target_df = df[df['Type'] == 'MERCURY'].reset_index(drop=True)

    for i, row in target_df.iterrows():
        file_num = i + 1
        base = f"MERCURY{file_num}_tr"

        cands = sorted(list(output_dir.glob(f"{base}*.exos.dat")))
        if not cands: cands = sorted(list((output_dir / "2_spectra").glob(f"{base}*.exos.dat")))
        if not cands: continue

        if len(cands) < 3:
            paths = {'main': cands[len(cands) // 2]}
        else:
            paths = {'minus': cands[0], 'main': cands[1], 'plus': cands[2]}

        print(f"  > Processing: {paths['main'].name}")

        gamma = row.get('g_factor', np.nan)
        dist = row.get('mercury_sun_distance_au', np.nan)
        dia = row.get('apparent_diameter_arcsec', np.nan)
        pa = row.get('phase_angle_deg', 0.0)

        if pd.isna(gamma) or pd.isna(dist): continue

        geom_conf['diameter_arcsec'] = dia if not pd.isna(dia) else 6.0
        geom_conf['phase_angle'] = pa
        gfac1 = (GFAC_CONST * gamma) / (dist ** 2)

        base_key = paths['main'].name.replace('.exos.dat', '')
        cts2MR = calib_map.get(base_key, 1.0e-12)

        # ★ 全体計算
        ret = process_total_atoms(paths, base, output_dir, gfac1, fit_config, target_wl)
        if ret:
            # 戻り値: N, Err, Peak_Diff, Main_Center
            N, Err, PDiff, MainCenter = ret
            res_final.append([file_num, N, Err, PDiff, MainCenter])

        # ★ 領域計算
        regs, debug_info = process_regions_from_sub_fits(paths['main'], target_wl, gfac1, geom_conf, cts2MR, fit_config)
        if regs:
            avg_r = np.nanmean(regs[:3])
            res_region.append([file_num, avg_r, regs[0], regs[1], regs[2], regs[3], regs[4], regs[5]])
            print(f"    -> Regions: N={regs[0]:.2e}, S={regs[1]:.2e}, Eq={regs[2]:.2e}")

    if res_final:
        # ★ ヘッダーに Center_WL_nm を追加し、Peak_Diff_nm も残す
        np.savetxt(output_dir / 'Na_atoms_final.dat', np.array(res_final),
                   fmt="%d %.4e %.4e %.6f %.6f", header="Index Total_Atoms Error Peak_Diff_nm Center_WL_nm")

    if res_region:
        np.savetxt(output_dir / 'Na_atoms_regions.dat', np.array(res_region),
                   fmt="%d %.4e %.4e %.4e %.4e %.4e %.4e %.4e",
                   header="Index Disk_Avg North South Equator Dawn SS Dusk")
        print(f"  > Saved: Na_atoms_regions.dat")


if __name__ == "__main__":
    print("Use as module.")