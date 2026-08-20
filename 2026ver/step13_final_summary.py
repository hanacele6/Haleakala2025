import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

# ==============================================================================
# 物理定数
# ==============================================================================
K_B = 1.380649e-23  # ボルツマン定数 [J/K]
M_NA = 22.98976928 * 1.66053906660e-27  # Na原子の質量 [kg]
C_M_S = 2.99792458e8  # 光速 [m/s]
C_KM_S = 2.99792458e5  # 光速 [km/s]

# ==============================================================================
# 物理QCの既定パラメータ
#   「この値でなければ物理的におかしい」と言える量だけをここに置く。
#   経験的なフィット品質の閾値は LEGACY_QC_PARAMS 側に隔離してある。
# ==============================================================================
PHYS_PARAMS = {
    # --- 検出の有意性 ---
    'min_detection_sigma': 3.0,  # N_total / N_err がこれ未満なら「非検出」

    # --- 輝線中心波長 (ドップラー予測との一致) ---
    'center_reference': 'doppler',  # doppler | fixed | median
    'center_tolerance_pix': 3.0,  # 予測波長からのずれの許容値 [波長ステップ単位]
    #  その日の波長ゼロ点のずれ (較正由来の系統誤差) をどう扱うか。
    #    median : 全観測に共通のオフセットを中央値で見積もって差し引き、
    #             「その日の中でのばらつき」を判定する (既定)
    #    none   : 予測波長からの絶対的なずれで判定する
    #  ゼロ点ずれは較正の問題であって個々の観測の良否ではないので、既定は median。
    #  ただし見積もったオフセットが大きい場合は警告を出す。
    'center_offset_correction': 'median',
    'center_offset_warn_pm': 5.0,  # 日ごとのゼロ点ずれがこれを超えたら警告 [pm]
    'center_offset_min_points': 2,  # ゼロ点を見積もるのに最低限必要な有意検出数
    'center_tolerance_floor_nm': 0.003,  # 上記が小さすぎる場合の下限 [nm]
    'apply_sft_offset': True,  # step10 の sft (太陽光モデルのずらし量) を予測波長に反映する

    # --- 輝線幅 (装置プロファイル + 熱ドップラー幅) ---
    #
    #  この装置では sigma_instrument ~ 13 pm に対し sigma_thermal(1200K) ~ 1.3 pm しかない。
    #  つまり輝線幅はほぼ完全に装置プロファイルで決まっており、線幅から温度は測れない。
    #  したがってこの項目は「温度の妥当性チェック」ではなく
    #  「装置プロファイルと整合しているか (= ブレンド・フィット失敗・ノイズの尖りでないか)」
    #  の整合性チェックとして機能する。
    #
    'check_line_width': True,
    'width_mode': 'auto',  # auto = その日の実測値から装置幅を推定 / fixed = 下の値を使う
    'instrumental_fwhm_nm': None,  # fixed のときの装置FWHM [nm]
    'instrumental_fwhm_pix': None, # 同上をピクセルで指定する場合 (step10 の FWHM 出力と同じ単位)
    'width_tolerance_factor': 1.5,  # 広い側の許容倍率 (装置幅の何倍まで許すか)
    #  装置プロファイル幅は分光器固有の性質であり、日によって変わるものではない。
    #  auto (その日の中央値) がこの範囲を外れた場合、それは装置幅ではなく
    #  「その日の全データが一様に劣化している」ことを意味するので、
    #  範囲の端に丸めた上で警告する。既定値は step10 の PSF 探索範囲に合わせてある。
    'instrumental_fwhm_pix_range': [3.0, 10.0],
    'width_tolerance_factor_low': 3.0,  # 狭い側の許容倍率 (装置幅の何分の1まで許すか)
    #   狭い側を広い側より緩くしてあるのは、連続光の引き方やブレンドの影響で
    #   フィットの sigma は小さく出やすい一方、「装置幅より細い輝線」は
    #   物理的にはナイキスト限界より下でしか起こり得ないため。
    #   null / 0 にすると、狭い側はナイキスト限界だけで判定する。
    'temperature_min_k': 500.0,  # 想定する外圏Na温度の下限 [K]
    'temperature_max_k': 5000.0,  # 同上限 (放射圧で加速された高速成分を含む) [K]

    # --- フィットの窓 ---
    'check_window_nm': 0.3,

    # --- 日単位の採否 ---
    'min_survival_count': 3,
    'min_survival_rate': 0.33,
}

# ==============================================================================
# 旧来の経験的QCパラメータ (qc_mode: legacy 用。再現性確認のために残してある)
# ==============================================================================
LEGACY_QC_PARAMS = {
    'max_relative_error': 1.0,
    'max_peak_diff_nm': 0.005,
    'check_window_nm': 0.3,
    'snr_threshold': 2.0,
    'residual_quality_threshold': 2.0,
    'min_fit_vs_raw_ratio': 1.5,
    'max_amp_err_ratio': 1.0,
    'sigma_min': 0.001,
    'sigma_max': 0.060,
    'min_points_in_fwhm': 1.0,
    'max_drift_threshold': 0.05,
    'min_survival_count': 3,
    'min_survival_rate': 0.33,
}

# 後方互換 (外部から QC_PARAMS を参照しているコード用)
QC_PARAMS = LEGACY_QC_PARAMS


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


def thermal_sigma_nm(temperature_k, lambda_nm):
    """指定温度におけるNa輝線の熱ドップラー幅 sigma [nm]"""
    v_thermal = np.sqrt(K_B * temperature_k / M_NA)  # [m/s]
    return lambda_nm * v_thermal / C_M_S


def doppler_shifted_wavelength(lambda_rest_nm, v_km_s):
    """視線速度によるドップラーシフト後の波長 [nm] (v>0 = 遠ざかる = 赤方偏移)"""
    return lambda_rest_nm * (1.0 + v_km_s / C_KM_S)


def parse_sft_from_name(path):
    """
    ファイル名 (..._sftNNN.exos.dat) から step10 で使われた sft [nm] を復元する。

    step10 は太陽光モデルを  direct_solar_wl = solar_wl - sft  として観測に合わせている。
    つまり観測側の波長軸は「真の波長 - sft」を読む状態になっているので、
    輝線の期待中心波長も同じだけずらして比較しないと sft の分だけ系統的にずれる。
    """
    import re
    m = re.search(r'_sft(-?\d+)', Path(path).name)
    if not m:
        return np.nan
    return int(m.group(1)) / 10000.0


# ==============================================================================
# 測定 (判定はしない)
#   スペクトルからガウシアンフィットで観測量を取り出すだけの関数。
#   「測る」と「弾く」を分離しておくと、判定基準を変えても測定側を触らずに済む。
# ==============================================================================
def measure_line(file_path, target_wl=589.7558, window_nm=0.3):
    result = {
        'ok': False, 'msg': '', 'amp': np.nan, 'center': np.nan, 'sigma': np.nan,
        'amp_err': np.nan, 'noise': np.nan, 'snr': np.nan,
        'residual_rms': np.nan, 'wav_step': np.nan,
    }

    try:
        data = np.loadtxt(file_path)
        if data.ndim < 2 or data.shape[0] < 5:
            result['msg'] = "Data Invalid/Short"
            return result
        wl, flux = data[:, 0], data[:, 1]
    except Exception:
        result['msg'] = "File Read Error"
        return result

    mask = (wl >= target_wl - window_nm) & (wl <= target_wl + window_nm)
    w_cut, f_cut = wl[mask], flux[mask]

    if len(f_cut) < 5:
        result['msg'] = "Data Too Short"
        return result

    wav_step = np.median(np.diff(w_cut)) if len(w_cut) > 1 else np.nan
    result['wav_step'] = wav_step

    try:
        init_base = np.percentile(f_cut, 25)
        raw_amp = np.max(f_cut) - init_base
        init_center = w_cut[np.argmax(f_cut)]

        p0 = [raw_amp, init_center, 0.02, init_base]
        bounds = ([0, w_cut[0], 1e-4, -np.inf], [np.inf, w_cut[-1], 0.15, np.inf])
        popt, pcov = curve_fit(gaussian_func, w_cut, f_cut, p0=p0, bounds=bounds, maxfev=3000)

        amp, center, sigma = popt[0], popt[1], abs(popt[2])
        perr = np.sqrt(np.diag(pcov))

        # フィット窓の端に張り付いた解は、輝線ではなくノイズや窓外の構造を掴んでいる
        edge_margin = 3.0 * wav_step if np.isfinite(wav_step) else 0.0
        if (center - w_cut[0]) < edge_margin or (w_cut[-1] - center) < edge_margin:
            result['msg'] = "Fit railed at window edge"
            result['center'], result['sigma'] = center, sigma
            result['wav_step'] = wav_step
            return result

        # 連続光部分からノイズを推定 (MAD -> sigma 換算)
        peak_mask = (w_cut > center - 3 * sigma) & (w_cut < center + 3 * sigma)
        cont = f_cut[~peak_mask]
        if len(cont) > 5:
            noise = np.median(np.abs(cont - np.median(cont))) * 1.4826
        else:
            edge = np.concatenate([f_cut[:3], f_cut[-3:]])
            noise = np.median(np.abs(edge - np.median(edge))) * 1.4826
        noise = max(noise, 1e-30)

        residuals = f_cut - gaussian_func(w_cut, *popt)
        residual_rms = np.median(np.abs(residuals - np.median(residuals))) * 1.4826

        result.update({
            'ok': True, 'msg': 'OK', 'amp': amp, 'center': center, 'sigma': sigma,
            'amp_err': perr[0], 'noise': noise, 'snr': amp / noise,
            'residual_rms': residual_rms,
        })
        return result

    except Exception as e:
        result['msg'] = f"Fit Failed ({str(e)[:40]})"
        return result


# ==============================================================================
# 判定 (物理基準)
# ==============================================================================
def estimate_instrumental_sigma(pp, points, wav_step):
    """
    装置プロファイルの sigma [nm] を決める。

    優先順位:
      1. instrumental_fwhm_nm  (fixed 指定)
      2. instrumental_fwhm_pix (fixed 指定, ピクセル単位)
      3. auto: その日の実測 sigma の中央値
         輝線は装置幅で決まっているので、実測値の中央値がそのまま装置プロファイル幅になる。
         ただし「その日の全データが同じように壊れている」ケースは検出できない。
    """
    mode = str(pp.get('width_mode', 'auto')).lower()

    sigmas = [p['meas']['sigma'] for p in points
              if p['meas']['ok'] and np.isfinite(p['meas']['sigma'])]
    measured = float(np.median(sigmas)) if sigmas else np.nan

    fixed, fixed_src = np.nan, ''
    if pp.get('instrumental_fwhm_nm'):
        fixed, fixed_src = float(pp['instrumental_fwhm_nm']) / 2.3548, 'fixed(nm)'
    elif pp.get('instrumental_fwhm_pix') and np.isfinite(wav_step):
        fixed, fixed_src = float(pp['instrumental_fwhm_pix']) * wav_step / 2.3548, 'fixed(pix)'

    if mode == 'fixed' and np.isfinite(fixed):
        # 指定値が実測と大きく食い違う場合、そのまま使うと全データを弾いてしまう。
        # 指定値の方が疑わしいので警告して実測値に切り替える (strict_width=True なら従う)。
        if np.isfinite(measured) and not (fixed / 2.0 <= measured <= fixed * 2.0):
            print(f"  > 警告: 指定された装置幅 (sigma={fixed * 1000:.2f} pm) は "
                  f"実測の中央値 (sigma={measured * 1000:.2f} pm) と大きく異なります。")
            if pp.get('strict_width', False):
                print("         strict_width=True のため指定値をそのまま使用します。")
                return fixed, fixed_src + '!'
            print("         指定値を無視し、実測値を装置幅として使用します。")
            return measured, 'auto(median, fixed ignored)'
        return fixed, fixed_src

    if not np.isfinite(measured):
        return np.nan, 'unavailable'

    # auto の場合、実測中央値が装置幅としてありえない値なら丸める
    rng = pp.get('instrumental_fwhm_pix_range', None)
    if rng and np.isfinite(wav_step) and wav_step > 0:
        lo = float(rng[0]) * wav_step / 2.3548
        hi = float(rng[1]) * wav_step / 2.3548
        if measured > hi:
            print(f"  > [!] 実測の輝線幅 (sigma={measured * 1000:.2f} pm = "
                  f"FWHM {2.3548 * measured / wav_step:.1f} pix) が "
                  f"装置幅としてありえない太さです。")
            print(f"        step10 の太陽光減算が破綻している可能性があります "
                  f"(PSF探索範囲は {rng[0]:.0f}-{rng[1]:.0f} pix)。")
            print(f"        装置幅は上限 {rng[1]:.0f} pix 相当に丸めて判定します。")
            return hi, 'auto(clamped to max)'
        if measured < lo:
            print(f"  > [!] 実測の輝線幅 (sigma={measured * 1000:.2f} pm) が "
                  f"装置幅としてありえない細さです。")
            return lo, 'auto(clamped to min)'

    return measured, 'auto(median)'


def build_width_window(pp, target_wl, wav_step, sigma_inst):
    """許容される輝線幅 sigma の範囲 [nm] を組み立てる"""
    if not np.isfinite(sigma_inst):
        return np.nan, np.nan, np.nan, False

    sig_lo = np.sqrt(sigma_inst ** 2 + thermal_sigma_nm(pp['temperature_min_k'], target_wl) ** 2)
    sig_hi = np.sqrt(sigma_inst ** 2 + thermal_sigma_nm(pp['temperature_max_k'], target_wl) ** 2)

    tol_hi = pp.get('width_tolerance_factor', pp.get('sigma_tolerance_factor', 2.0))
    tol_lo = pp.get('width_tolerance_factor_low', None)
    if tol_lo is None:
        tol_lo = tol_hi

    sigma_max = sig_hi * tol_hi
    # tol_lo が 0 / null なら下限はナイキスト限界のみに任せる
    sigma_min = (sig_lo / tol_lo) if tol_lo else 0.0

    # ナイキスト条件: FWHMが2ピクセル未満の輝線は「分解できていない = ノイズの尖り」
    nyquist_sigma = np.nan
    if np.isfinite(wav_step) and wav_step > 0:
        nyquist_sigma = 2.0 * wav_step / 2.3548
        sigma_min = max(sigma_min, nyquist_sigma)

    resolvable = (not np.isfinite(nyquist_sigma)) or (sigma_min < sigma_max)
    return sigma_min, sigma_max, nyquist_sigma, resolvable


def judge_physical(meas, val, err, p_diff, expected_center, pp, target_wl, width_window=None,
                   center_offset=0.0):
    """物理的に意味のある4項目だけで判定する"""
    notes = []

    # --- 1. 検出の有意性 ---
    if not np.isfinite(val) or val <= 0:
        return False, "Non-detection (N <= 0)"
    if not np.isfinite(err) or err <= 0:
        notes.append("error undefined")
    else:
        det_sigma = val / err
        if det_sigma < pp['min_detection_sigma']:
            return False, f"Non-detection ({det_sigma:.1f}sigma < {pp['min_detection_sigma']}sigma)"

    if not meas['ok']:
        return False, meas['msg']

    # --- 2. 輝線中心波長: ドップラー予測との一致 ---
    wav_step = meas['wav_step']
    tol = pp['center_tolerance_pix'] * wav_step if np.isfinite(wav_step) else np.nan
    if not np.isfinite(tol):
        tol = pp['center_tolerance_floor_nm']
    tol = max(tol, pp['center_tolerance_floor_nm'])

    if np.isfinite(expected_center):
        # その日に共通のゼロ点ずれを差し引いた残差で判定する
        d_lambda = meas['center'] - expected_center - center_offset
        if abs(d_lambda) > tol:
            v_off = d_lambda / target_wl * C_KM_S
            label = "Center scatter" if center_offset else "Center off by"
            return False, (f"{label} {d_lambda * 1000:+.1f} pm "
                           f"({v_off:+.1f} km/s, tol={tol * 1000:.1f} pm)")

    # --- 3. 太陽光減算の系統誤差 (sft違いによる中心波長のばらつき) ---
    if np.isfinite(p_diff) and p_diff > tol:
        return False, f"Solar-subtraction systematics ({p_diff * 1000:.1f} pm > {tol * 1000:.1f} pm)"

    # --- 4. 輝線幅: 装置幅 + 熱ドップラー幅と整合するか ---
    if pp['check_line_width'] and width_window is not None:
        s_min, s_max, sigma_inst, resolvable = width_window
        if not resolvable:
            notes.append("width test skipped")
        else:
            s = meas['sigma']
            if s < s_min:
                return False, (f"Too narrow (sigma={s * 1000:.2f} pm < {s_min * 1000:.2f} pm; "
                               f"unresolved / noise spike)")
            if s > s_max:
                ratio = s / sigma_inst if (np.isfinite(sigma_inst) and sigma_inst > 0) else np.nan
                extra = f", {ratio:.1f}x instrumental" if np.isfinite(ratio) else ""
                return False, f"Too broad (sigma={s * 1000:.2f} pm > {s_max * 1000:.2f} pm{extra})"

    return True, ("OK" if not notes else "OK (" + "; ".join(notes) + ")")


# ==============================================================================
# 判定 (旧来の経験基準) — 再現性確認用
# ==============================================================================
def judge_legacy(meas, val, err, p_diff, qc):
    if val <= 0:
        return False, "Value <= 0"
    if (err / val) > qc['max_relative_error']:
        return False, f"High Error Ratio ({(err / val):.2f})"
    if p_diff > qc['max_peak_diff_nm']:
        return False, f"Large Peak Diff ({p_diff:.4f} nm)"
    if not meas['ok']:
        return False, meas['msg']

    fit_quality = meas['amp'] / (meas['residual_rms'] + 1e-30)
    if fit_quality < qc['residual_quality_threshold']:
        return False, f"Poor Fit Quality ({fit_quality:.1f})"
    if meas['snr'] < qc['snr_threshold']:
        return False, f"Low S/N ({meas['snr']:.1f})"
    if meas['sigma'] < qc['sigma_min']:
        return False, f"Too Narrow (Sigma={meas['sigma']:.4f})"
    if meas['sigma'] > qc['sigma_max']:
        return False, f"Too Broad (Sigma={meas['sigma']:.4f})"
    if np.isfinite(meas['amp_err']) and meas['amp'] > 0 \
            and (meas['amp_err'] / meas['amp']) > qc['max_amp_err_ratio']:
        return False, "Unstable Fit"
    if np.isfinite(meas['wav_step']) and meas['wav_step'] > 0:
        if (2.3548 * meas['sigma'] / meas['wav_step']) < qc['min_points_in_fwhm']:
            return False, "Too Sharp"
    return True, "OK"


def find_largest_cluster(items, drift_threshold):
    """legacy モード専用: 中心波長が最も密集している群を主クラスタとみなす"""
    valid_items = [x for x in items if np.isfinite(x['center_fit']) and x['center_fit'] > 100]
    if not valid_items:
        return [], items
    sorted_items = sorted(valid_items, key=lambda x: x['center_fit'])
    max_cluster = []
    for i in range(len(sorted_items)):
        cluster = []
        start_wl = sorted_items[i]['center_fit']
        for j in range(i, len(sorted_items)):
            if sorted_items[j]['center_fit'] - start_wl <= drift_threshold:
                cluster.append(sorted_items[j])
            else:
                break
        if len(cluster) > len(max_cluster):
            max_cluster = cluster
    cluster_ids = {x['idx'] for x in max_cluster}
    return max_cluster, [x for x in items if x['idx'] not in cluster_ids]


# ==============================================================================
# 手動選別
# ==============================================================================
def read_manual_selection(path):
    """
    手動選別ファイルを読む。書式:
        mode refine          # refine(既定)=自動QCに加減算 / override=手動指定のみ採用
        include 1 2 5
        exclude 3
        include 7            # 複数行に分けてもよい
    '#' 以降はコメント。
    """
    if not path.exists():
        return None
    sel = {'mode': 'refine', 'include': set(), 'exclude': set()}
    try:
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.split('#')[0].strip()
            if not line:
                continue
            tokens = line.split()
            key = tokens[0].lower()
            if key == 'mode' and len(tokens) > 1:
                sel['mode'] = tokens[1].lower()
            elif key in ('include', 'exclude'):
                for t in tokens[1:]:
                    try:
                        sel[key].add(int(t))
                    except ValueError:
                        pass
    except Exception as e:
        print(f"  > 警告: 手動選別ファイルの読み込みに失敗しました ({e})")
        return None
    return sel


def format_row(p, use_flag=None):
    return "  {:>4} {:>11.4e} {:>7} {:>12} {:>12} {:>10} {:>5}  {}".format(
        p['idx'],
        p['val'],
        f"{p['val'] / p['err']:.1f}" if (np.isfinite(p['err']) and p['err'] > 0) else "-",
        f"{p['center_fit']:.5f}" if np.isfinite(p['center_fit']) else "-",
        f"{p['d_resid'] * 1000:+.1f}" if np.isfinite(p.get('d_resid', np.nan)) else "-",
        f"{p['sigma'] * 1000:.2f}" if np.isfinite(p['sigma']) else "-",
        use_flag if use_flag is not None else ("OK" if p['auto_valid'] else "NG"),
        p.get('reason', p.get('auto_reason', '')),
    )


def write_manual_template(path, points):
    """観測ごとの診断値をコメントとして並べた、編集用テンプレートを書き出す"""
    lines = [
        "# ---- 手動選別ファイル ----",
        "# このファイル名から '.sample' を外すと、次回実行時に適用されます。",
        "#",
        "#   mode refine   : 自動QCの結果を出発点に、include/exclude で加減算する",
        "#   mode override : 自動QCを無視し、include に書いたIDだけを採用する",
        "#",
        "# 以下は今回の自動判定の結果です。参考にして include/exclude を編集してください。",
        "#",
        "#   ID    N_total       S/N   Center[nm]  dResid[pm]   sigma[pm]  Auto  Reason",
    ]
    for p in sorted(points, key=lambda x: x['idx']):
        lines.append("#" + format_row(p))
    lines += ["", "mode refine", "include", "exclude", ""]
    path.write_text("\n".join(lines), encoding='utf-8')


# ==============================================================================
# Main Logic
# ==============================================================================
def run(run_info, config):
    output_dir = Path(run_info["output_dir"])
    csv_file_path = run_info["csv_path"]
    date_str = run_info["date"]
    col_conf = config.get("column_density", {})
    target_wl = col_conf.get("target_wavelength", 589.7558)

    # --- 設定の読み込み ---
    sum_conf = config.get("summary", {}) or {}
    qc_mode = str(sum_conf.get("qc_mode", "physical")).lower()  # physical | legacy | off

    user_phys = sum_conf.get("physical_params", {}) or {}
    unknown_keys = [k for k in user_phys if k not in PHYS_PARAMS and k != 'strict_width']
    if unknown_keys:
        print(f"  > [警告] physical_params に未知のキーがあります: {unknown_keys}")
        print(f"           (タイポの可能性。有効なキー: {sorted(PHYS_PARAMS.keys())})")
    pp = dict(PHYS_PARAMS)
    pp.update(user_phys)
    # 注意: gfactor.fwhm は step11 が太陽スペクトルを畳み込むカーネル幅であって
    #       分光器の装置プロファイル幅ではない。ここでは絶対に流用しないこと。

    qc = dict(LEGACY_QC_PARAMS)
    qc.update(sum_conf.get("qc_params", {}) or {})

    man_conf = sum_conf.get("manual_selection", {}) or {}
    manual_enabled = bool(man_conf.get("enabled", True))

    print(f"\n--- 1D最終集計QC (Step 13) ---")
    print(f"  > QC mode: {qc_mode}")

    final_dat = output_dir / 'Na_atoms_final.dat'
    report_out = output_dir / f'Quality_Report_{date_str}.txt'
    summary_out = output_dir / f'Final_Summary_{date_str}.txt'
    manual_file = output_dir / f'manual_selection_{date_str}.txt'
    manual_sample = output_dir / f'manual_selection_{date_str}.txt.sample'

    if not final_dat.exists():
        print("  > Error: Na_atoms_final.dat not found.")
        return

    try:
        data = np.loadtxt(final_dat)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 5:
            print("  > Error: Invalid format in Na_atoms_final.dat")
            return
        indices, atoms, errors = data[:, 0], data[:, 1], data[:, 2]
        peak_diffs, center_wls = data[:, 3], data[:, 4]
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    # --- 観測メタデータ ---
    mercury_df = pd.DataFrame()
    if csv_file_path:
        try:
            if Path(csv_file_path).is_dir():
                cands = list(Path(csv_file_path).glob("*.csv"))
                if cands:
                    mercury_df = pd.read_csv(cands[0])
            else:
                mercury_df = pd.read_csv(csv_file_path)
            if not mercury_df.empty:
                mercury_df = mercury_df.loc[:, ~mercury_df.columns.duplicated()]
                mercury_df = mercury_df[mercury_df['Type'] == 'MERCURY'].reset_index(drop=True)
        except Exception:
            pass

    spec_dir = output_dir if not (output_dir / "2_spectra").exists() else output_dir / "2_spectra"

    # ==========================================================================
    # Phase 1: 測定
    # ==========================================================================
    print("\n[Phase 1] Measuring lines...")
    points = []
    for i in range(len(indices)):
        idx = int(indices[i])
        row_idx = idx - 1

        v_geo, taa = np.nan, np.nan
        if not mercury_df.empty and 0 <= row_idx < len(mercury_df):
            row = mercury_df.iloc[row_idx]
            if 'mercury_earth_radial_velocity_km_s' in row:
                v_geo = safe_float(row['mercury_earth_radial_velocity_km_s'])
            if 'true_anomaly_deg' in row:
                taa = safe_float(row['true_anomaly_deg'])
            elif 'mercury_sun_distance_au' in row:
                taa = calculate_taa(row['mercury_sun_distance_au'],
                                    row['mercury_sun_radial_velocity_km_s'])

        cands = sorted(list(spec_dir.glob(f"MERCURY{idx}_tr*.exos.dat")))
        sft = np.nan
        if cands:
            used_file = cands[len(cands) // 2]
            sft = parse_sft_from_name(used_file)
            meas = measure_line(used_file, target_wl, pp['check_window_nm'])
        else:
            meas = measure_line(Path("__missing__"), target_wl, pp['check_window_nm'])
            meas['msg'] = "Spectrum file not found"

        points.append({
            'idx': idx, 'val': atoms[i], 'err': errors[i], 'p_diff': peak_diffs[i],
            'center_step12': center_wls[i], 'taa': taa, 'v_geo': v_geo, 'meas': meas,
            'center_fit': meas['center'], 'sigma': meas['sigma'], 'd_lambda': np.nan,
            'sft': sft,
        })

    # --- 期待される輝線中心波長 ---
    ref_mode = pp['center_reference']
    median_center = np.nan
    if ref_mode == 'median':
        good = [p['center_fit'] for p in points if np.isfinite(p['center_fit'])]
        median_center = np.median(good) if good else np.nan

    for p in points:
        # step10 が太陽光モデルを -sft ずらして合わせているぶん、観測波長軸も同じだけずれている
        sft_off = p['sft'] if (pp['apply_sft_offset'] and np.isfinite(p['sft'])) else 0.0

        if ref_mode == 'doppler' and np.isfinite(p['v_geo']):
            p['expected_center'] = doppler_shifted_wavelength(target_wl, p['v_geo']) - sft_off
        elif ref_mode == 'median':
            p['expected_center'] = median_center
        else:
            p['expected_center'] = target_wl - sft_off
        if np.isfinite(p['center_fit']) and np.isfinite(p['expected_center']):
            p['d_lambda'] = p['center_fit'] - p['expected_center']

    # --- その日に共通の波長ゼロ点ずれ ---
    center_offset = 0.0
    if str(pp.get('center_offset_correction', 'median')).lower() == 'median':
        # 非検出やフィット失敗の中心波長は無意味なので、オフセットの見積もりから除く。
        # (これらを混ぜると中央値が引きずられ、正常なデータが弾かれてしまう)
        usable = [p for p in points
                  if np.isfinite(p['d_lambda']) and p['meas']['ok']
                  and np.isfinite(p['err']) and p['err'] > 0
                  and (p['val'] / p['err']) >= pp['min_detection_sigma']]
        dls = [p['d_lambda'] for p in usable]
        n_min = int(pp.get('center_offset_min_points', 2))

        if len(dls) < n_min:
            print(f"  > 有意な検出が {len(dls)} 点しかないため、ゼロ点補正は行いません。")
        else:
            center_offset = float(np.median(dls))
            scatter = float(np.median(np.abs(np.array(dls) - center_offset))) * 1.4826
            print(f"  > Zero-point from {len(dls)}/{len(points)} detections: "
                  f"{center_offset * 1000:+.1f} pm (scatter {scatter * 1000:.1f} pm)")
            if scatter * 1000 > pp.get('center_offset_warn_pm', 5.0) * 2:
                print("  > [!] 有意検出の中でも中心波長がばらついています。"
                      "波長較正か太陽光減算を確認してください。")
            if abs(center_offset) * 1000 > pp.get('center_offset_warn_pm', 5.0):
                print(f"  > [!] この日の波長ゼロ点が {center_offset * 1000:+.1f} pm "
                      f"({center_offset / target_wl * C_KM_S:+.1f} km/s) ずれています。")
                print("        step06 の波長較正か resample.params.wavshift を確認してください。")
                print("        (判定はこのオフセットを差し引いた残差で行います)")
            else:
                print(f"  > Day zero-point offset: {center_offset * 1000:+.1f} pm (removed)")

    if ref_mode == 'doppler':
        vs = [p['v_geo'] for p in points if np.isfinite(p['v_geo'])]
        if vs:
            print(f"  > Doppler reference: v_geo = {np.min(vs):+.2f} .. {np.max(vs):+.2f} km/s"
                  f" -> {doppler_shifted_wavelength(target_wl, float(np.mean(vs))):.5f} nm (mean)")
            sfts = sorted({p['sft'] for p in points if np.isfinite(p['sft'])})
            if pp['apply_sft_offset'] and sfts:
                print(f"  > sft offset applied: {[f'{v:+.4f}' for v in sfts]} nm")
            elif not pp['apply_sft_offset']:
                print("  > sft offset: 適用しません (apply_sft_offset=False)")
        else:
            print("  > 警告: 視線速度が取得できないため、静止波長を基準にします。")

    # ==========================================================================
    # Phase 2: 自動判定
    # ==========================================================================
    print(f"\n[Phase 2] Judging ({qc_mode})...")
    median_step = np.nanmedian([p['meas']['wav_step'] for p in points])
    width_window, sigma_inst, width_src = None, np.nan, 'n/a'
    if qc_mode == 'physical':
        sigma_inst, width_src = estimate_instrumental_sigma(pp, points, median_step)
        s_min, s_max, nyq, resolvable = build_width_window(pp, target_wl, median_step, sigma_inst)
        width_window = (s_min, s_max, sigma_inst, resolvable)
        if np.isfinite(sigma_inst):
            fwhm_pix = 2.3548 * sigma_inst / median_step if median_step else np.nan
            print(f"  > Instrumental sigma: {sigma_inst * 1000:.2f} pm "
                  f"(FWHM {2.3548 * sigma_inst * 1000:.1f} pm = {fwhm_pix:.1f} pix) [{width_src}]")
            tl = pp.get('width_tolerance_factor_low', None)
            print(f"  > Allowed sigma: {s_min * 1000:.2f} - {s_max * 1000:.2f} pm "
                  f"(/{tl if tl else 'nyquist'}, x{pp.get('width_tolerance_factor', 2.0)})")
        else:
            print("  > 警告: 装置幅を決められないため、幅による判定は無効化されます。")
        if not resolvable:
            print("  > 警告: この分散では輝線が分解できていません。幅による判定は無効化されます。")

    for p in points:
        p['d_resid'] = p['d_lambda'] - center_offset if np.isfinite(p['d_lambda']) else np.nan

    for p in points:
        if qc_mode == 'off':
            p['auto_valid'], p['auto_reason'] = True, "QC disabled"
        elif qc_mode == 'legacy':
            p['auto_valid'], p['auto_reason'] = judge_legacy(
                p['meas'], p['val'], p['err'], p['p_diff'], qc)
        else:
            p['auto_valid'], p['auto_reason'] = judge_physical(
                p['meas'], p['val'], p['err'], p['p_diff'], p['expected_center'], pp, target_wl,
                width_window, center_offset)

    # legacy モードのみ、従来のクラスタリングによる外れ値除去を追加で行う
    if qc_mode == 'legacy':
        best_cluster, _ = find_largest_cluster(points, qc['max_drift_threshold'])
        cluster_ids = {x['idx'] for x in best_cluster}
        for p in points:
            if p['idx'] not in cluster_ids and p['auto_valid']:
                p['auto_valid'], p['auto_reason'] = False, "Outlier (Not in main cluster)"

    n_auto = sum(1 for p in points if p['auto_valid'])
    print(f"  > Auto-passed: {n_auto}/{len(points)}")

    # --- 判定基準そのものが疑わしい場合の警告 ---
    #   「同じ理由で大半が落ちた」ときは、データではなく基準の較正が疑わしい。
    if qc_mode == 'physical' and len(points) >= 3 and n_auto <= len(points) * 0.34:
        from collections import Counter
        kinds = Counter(p['auto_reason'].split('(')[0].strip()
                        for p in points if not p['auto_valid'])
        top_kind, top_n = kinds.most_common(1)[0]
        if top_n >= len(points) * 0.66:
            print(f"  > [!] {len(points)} 点中 {top_n} 点が同じ理由 '{top_kind}' で落ちています。")
            print("        データではなく判定基準の設定を疑ってください。")
            if top_kind.startswith('Too broad') or top_kind.startswith('Too narrow'):
                print("        -> summary.physical_params.check_line_width: False で幅判定を切るか、")
                print("           width_tolerance_factor を上げてください。")
            elif top_kind.startswith('Center'):
                print("        -> center_reference / apply_sft_offset の設定、")
                print("           もしくは center_tolerance_pix を確認してください。")
            print("        -> 個別に採用したい場合は manual_selection ファイルを使ってください。")

    # ==========================================================================
    # Phase 3: 手動選別
    # ==========================================================================
    for p in points:
        p['valid'] = p['auto_valid']
        p['reason'] = p['auto_reason']
        p['manual'] = ""

    selection = read_manual_selection(manual_file) if manual_enabled else None

    if selection is not None:
        mode = selection['mode']
        inc, exc = selection['include'], selection['exclude']
        print(f"\n[Phase 3] Manual selection: {manual_file.name} (mode={mode}, "
              f"include={sorted(inc)}, exclude={sorted(exc)})")
        for p in points:
            if mode == 'override':
                p['valid'] = (p['idx'] in inc)
                p['manual'] = "MANUAL-INCLUDE" if p['valid'] else "MANUAL-DROP"
                p['reason'] = f"[{p['manual']}] auto={p['auto_reason']}"
            else:
                if p['idx'] in exc:
                    p['valid'], p['manual'] = False, "MANUAL-EXCLUDE"
                    p['reason'] = f"[MANUAL-EXCLUDE] auto={p['auto_reason']}"
                elif p['idx'] in inc:
                    p['valid'], p['manual'] = True, "MANUAL-INCLUDE"
                    p['reason'] = f"[MANUAL-INCLUDE] auto={p['auto_reason']}"
        unknown = (inc | exc) - {p['idx'] for p in points}
        if unknown:
            print(f"  > 警告: データに存在しないID {sorted(unknown)} が指定されています。")
    elif manual_enabled:
        write_manual_template(manual_sample, points)
        print(f"\n[Phase 3] Manual selection: なし "
              f"(編集用テンプレート {manual_sample.name} を出力しました)")

    final_valid = [p for p in points if p['valid']]
    n_total, n_survived = len(points), len(final_valid)

    # --- 日単位の採否 (手動指定がある場合は人の判断を優先し、発動させない) ---
    reject_all = False
    if qc_mode != 'off' and selection is None:
        min_count = pp['min_survival_count'] if qc_mode == 'physical' else qc['min_survival_count']
        min_rate = pp['min_survival_rate'] if qc_mode == 'physical' else qc['min_survival_rate']
        if n_survived < min_count or (n_total > 0 and (n_survived / n_total) < min_rate):
            print(f"  > [Reject All] Too few valid data points ({n_survived}/{n_total})")
            reject_all = True
            final_valid = []

    # ==========================================================================
    # Phase 4: 集計と出力
    # ==========================================================================
    n_valid = len(final_valid)
    avg_atoms = np.mean([p['val'] for p in final_valid]) if n_valid > 0 else 0
    valid_errs = [p['err'] for p in final_valid]
    avg_err = np.sqrt(np.sum(np.array(valid_errs) ** 2)) / len(valid_errs) if valid_errs else 0
    taa_list = [p['taa'] for p in final_valid if np.isfinite(p['taa'])]
    taa_avg = np.mean(taa_list) if taa_list else 0

    pa = safe_float(mercury_df.iloc[0]['phase_angle_deg']) if not mercury_df.empty else 0

    print(f"  > Final Valid: {n_valid}/{n_total}")

    with open(summary_out, 'w') as f:
        f.write(f"{pa} {taa_avg} {avg_atoms} {avg_err}\n")

    grade = "D"
    if n_valid > 0:
        s_rate = n_valid / n_total * 100
        grade = "C"
        if s_rate >= 70:
            grade = "B"
        if s_rate >= 90:
            grade = "A"

    with open(report_out, 'w', encoding='utf-8') as f:
        f.write(f"Quality Report ({date_str})\n")
        f.write(f"QC mode: {qc_mode}\n")
        if selection is not None:
            f.write(f"Manual selection: {manual_file.name} (mode={selection['mode']})\n")
        if reject_all:
            f.write("ALL REJECTED: too few valid points\n")
        f.write(f"Grade: {grade}\nSurvival: {n_valid}/{n_total}\n")

        if qc_mode == 'physical':
            sg_min, sg_max, nyq, resolvable = build_width_window(
                pp, target_wl, median_step, sigma_inst)
            f.write("\n[Physical criteria]\n")
            f.write(f"  Detection  : N / N_err >= {pp['min_detection_sigma']} sigma\n")
            f.write(f"  Day zero-point offset removed: {center_offset * 1000:+.1f} pm\n")
            f.write(f"  Center     : {ref_mode} reference, tolerance = "
                    f"{pp['center_tolerance_pix']} pix "
                    f"(floor {pp['center_tolerance_floor_nm'] * 1000:.1f} pm)\n")
            sfts = sorted({p['sft'] for p in points if np.isfinite(p['sft'])})
            f.write(f"  sft offset : {'applied ' + str([round(v, 4) for v in sfts]) if (pp['apply_sft_offset'] and sfts) else 'not applied'}\n")
            if np.isfinite(sigma_inst):
                tl = pp.get('width_tolerance_factor_low', None)
                f.write(f"  Line width : sigma = {sg_min * 1000:.2f} - {sg_max * 1000:.2f} pm "
                        f"(instrumental sigma {sigma_inst * 1000:.2f} pm [{width_src}], "
                        f"low /{tl if tl else 'nyquist'}, high x"
                        f"{pp.get('width_tolerance_factor', 2.0)})\n")
                f.write(f"               thermal sigma at 1200 K is only "
                        f"{thermal_sigma_nm(1200.0, target_wl) * 1000:.2f} pm -> "
                        f"line is instrument-dominated; this is a consistency check, "
                        f"not a temperature measurement\n")
            else:
                f.write("  Line width : not evaluated (instrumental width unknown)\n")
            if not resolvable:
                f.write("  (line unresolved at this dispersion -> width test disabled)\n")

        f.write("\n[Per-observation table]\n")
        f.write("   ID    N_total       S/N   Center[nm]  dResid[pm]   sigma[pm]    Use  Reason\n")
        for p in sorted(points, key=lambda x: x['idx']):
            f.write(format_row(p, "YES" if p['valid'] else "no") + "\n")

        f.write("\n[Excluded]\n")
        for p in sorted(points, key=lambda x: x['idx']):
            if not p['valid']:
                f.write(f" - ID {p['idx']}: {p['reason']}\n")

    print(f"  > Report Saved: {report_out.name}")


if __name__ == '__main__':
    print("Use as module.")