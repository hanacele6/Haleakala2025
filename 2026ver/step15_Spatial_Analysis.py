import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from astropy.time import Time
from astroquery.jplhorizons import Horizons
import scipy.ndimage as ndimage
from matplotlib.colors import LogNorm

# ==============================================================================
# 設定パラメータ
# ==============================================================================
SPATIAL_CONFIG = {
    'nx': 10,  # X方向のファイバー数
    'ny': 12,  # Y方向のファイバー数
    'plate_scale': 1.0,  # 1ピクセル(ファイバー)あたりの秒角
    'seeing_arcsec': 2.0,  # 大気の揺らぎ(シーイング)の仮定値(秒角)
    'hapke_scale_factor': 100.0  # Hapke画像のスケール (1秒角 = 100ピクセル)
}


# ==============================================================================
# 根幹3: JPL Horizonsから太陽のポジションアングルを取得
# ==============================================================================
def get_sun_position_angle(date_obs_str):
    """
    観測日時(DATE-OBS)からJPL Horizonsにアクセスし、
    水星に対する太陽のPosition Angle(天の北から東回り)を取得する。
    """
    print(f"    -> Querying JPL Horizons for Sun Position Angle at {date_obs_str}...")
    try:
        jd_time = Time(date_obs_str).jd
        obj = Horizons(id='199', location='F65', epochs=jd_time)
        eph = obj.ephemerides()
        sun_pa = eph['sunTargetPA'][0]
        print(f"    -> Sun Position Angle retrieved: {sun_pa:.2f} deg")
        return sun_pa
    except Exception as e:
        print(f"    -> [Warning] Failed to get Sun PA from JPL Horizons: {e}")
        print("    -> Using default Sun PA = 0.0")
        return 0.0


# ==============================================================================
# 根幹1,2,4: 中心推定関数 (Hapke Template Matching)
# ==============================================================================
def find_mercury_center_with_hapke(obs_img_2d, hapke_img_highres, sun_pa_deg):
    ny_obs, nx_obs = obs_img_2d.shape
    plate_scale = SPATIAL_CONFIG['plate_scale']
    hapke_scale = SPATIAL_CONFIG['hapke_scale_factor']

    # --- 根幹1 & 2: 観測データの重心計算 (最大輝度の50%以上) ---
    thresh_50 = np.max(obs_img_2d) * 0.5
    mask = (obs_img_2d >= thresh_50)
    if np.sum(mask) == 0:
        mask = (obs_img_2d >= 0)
    cy_obs, cx_obs = ndimage.center_of_mass(obs_img_2d * mask)

    if np.isnan(cy_obs) or np.isnan(cx_obs):
        cy_obs, cx_obs = ny_obs / 2.0, nx_obs / 2.0

    # シーイングの候補と、微調整の探索範囲
    seeing_candidates = np.linspace(1.0, 3.5, 11)
    fine_search_arcsec = 1.0
    fine_steps = 11

    best_error = np.inf
    best_results = {
        'cx': cx_obs, 'cy': cy_obs,
        'seeing': 2.0, 'model': np.zeros_like(obs_img_2d),
        'rot_angle': 0.0
    }

    # 観測データを最大値1.0に正規化
    obs_norm = obs_img_2d / np.max(obs_img_2d) if np.max(obs_img_2d) > 0 else obs_img_2d

    # Hapkeモデルの回転角計算 (PAを使用)
    HAPKE_LIGHT_ANGLE = 0.0
    NORTH_ANGLE_ON_IMAGE = 180.0
    telescope_flip_angle = 180.0 if sun_pa_deg > 180.0 else 0.0
    target_light_angle = NORTH_ANGLE_ON_IMAGE + sun_pa_deg + telescope_flip_angle
    rotation_angle = target_light_angle - HAPKE_LIGHT_ANGLE

    # --- 根幹4: 最小二乗法でフィット位置とシーイングを探す ---
    for s_arcsec in seeing_candidates:
        # A. モデルの準備 (ぼかし -> 回転 -> 縮小)
        sigma_pix = (s_arcsec / 2.355) * hapke_scale
        h_blurred = ndimage.gaussian_filter(hapke_img_highres, sigma=sigma_pix)

        hapke_flipped = np.fliplr(h_blurred)
        h_rotated = ndimage.rotate(hapke_flipped, -rotation_angle, reshape=False, mode='constant', cval=0.0)

        zoom_factor = 1.0 / (hapke_scale * plate_scale)
        h_lowres = ndimage.zoom(h_rotated, zoom_factor, order=1)

        # 12x10キャンバスへの配置
        h_canvas = np.zeros((ny_obs, nx_obs))
        h_h, h_w = h_lowres.shape
        y_c, x_c = (ny_obs - h_h) // 2, (nx_obs - h_w) // 2
        y_s, y_e = max(0, y_c), min(ny_obs, y_c + h_h)
        x_s, x_e = max(0, x_c), min(nx_obs, x_c + h_w)

        # サイズの安全な切り出し
        canvas_h, canvas_w = y_e - y_s, x_e - x_s
        h_canvas[y_s:y_e, x_s:x_e] = h_lowres[0:canvas_h, 0:canvas_w]

        true_cy_canvas = y_s + (canvas_h / 2.0)
        true_cx_canvas = x_s + (canvas_w / 2.0)

        # B. モデルの重心
        h_thresh_50 = np.max(h_canvas) * 0.5
        h_mask = (h_canvas >= h_thresh_50)
        if np.sum(h_mask) == 0:
            h_mask = (h_canvas >= 0)
        cy_mod, cx_mod = ndimage.center_of_mass(h_canvas * h_mask)

        if np.isnan(cy_mod) or np.isnan(cx_mod):
            continue

        # C. 重心周辺での微調整 (Least Squares)
        fine_range = fine_search_arcsec / plate_scale
        for dy in np.linspace(-fine_range, fine_range, fine_steps):
            for dx in np.linspace(-fine_range, fine_range, fine_steps):
                shift_y = (cy_obs - cy_mod) + dy
                shift_x = (cx_obs - cx_mod) + dx

                shifted_model = ndimage.shift(h_canvas, (shift_y, shift_x), order=1)

                max_val = np.max(shifted_model)
                model_norm = shifted_model / max_val if max_val > 0 else shifted_model

                # 誤差の二乗和 (Sum of Squared Errors)
                error = np.sum((obs_norm - model_norm) ** 2)

                if error < best_error:
                    best_error = error
                    best_results.update({
                        'cx': true_cx_canvas + shift_x,
                        'cy': true_cy_canvas + shift_y,
                        'seeing': s_arcsec,
                        'model': shifted_model,
                        'rot_angle': rotation_angle
                    })

    print(
        f"    -> Best Fit: Seeing={best_results['seeing']:.2f}\", Center=({best_results['cx']:.2f}, {best_results['cy']:.2f})")
    return best_results['cx'], best_results['cy'], best_results['model'], best_results['rot_angle']


# ==============================================================================
# プロット関数
# ==============================================================================
def plot_hapke_alignment(obs_img_2d, hapke_model_matched, cx, cy, r_pix, sun_pa, rotation_angle, out_path, title):
    fig, ax = plt.subplots(figsize=(10, 8))

    norm = LogNorm(vmin=np.max(obs_img_2d) * 0.01, vmax=np.max(obs_img_2d)) if np.max(obs_img_2d) > 0 else None
    im = ax.imshow(obs_img_2d, origin='lower', cmap='viridis', norm=norm,
                   extent=[-0.5, SPATIAL_CONFIG['nx'] - 0.5, -0.5, SPATIAL_CONFIG['ny'] - 0.5])
    plt.colorbar(im, label='Observed Continuum Flux (Log)')

    if np.max(hapke_model_matched) > 0:
        ax.contour(hapke_model_matched, levels=5, colors='white', alpha=0.6,
                   extent=[-0.5, SPATIAL_CONFIG['nx'] - 0.5, -0.5, SPATIAL_CONFIG['ny'] - 0.5])

    circle = plt.Circle((cx, cy), r_pix, color='red', fill=False, linewidth=2, linestyle='-')
    ax.add_patch(circle)
    ax.plot(cx, cy, 'rx', markersize=12, markeredgewidth=2, label='True Center')

    # 太陽方向の矢印
    theta_rad = np.radians(rotation_angle)
    sun_x = cx + (r_pix * 1.8) * np.cos(theta_rad)
    sun_y = cy + (r_pix * 1.8) * np.sin(theta_rad)
    target_x = cx + (r_pix * 0.8) * np.cos(theta_rad)
    target_y = cy + (r_pix * 0.8) * np.sin(theta_rad)

    ax.annotate('', xy=(target_x, target_y), xytext=(sun_x, sun_y),
                arrowprops=dict(facecolor='yellow', edgecolor='orange', shrink=0.0, width=3, headwidth=10), zorder=5)

    ax.text(sun_x, sun_y, '☀ SUN', color='yellow', fontsize=14, ha='center', va='center', fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))

    ax.set_title(f"{title}\nCenter: ({cx:.2f}, {cy:.2f}) | Sun PA: {sun_pa:.1f} deg")
    ax.set_xlabel("Fiber X (pixels)")
    ax.set_ylabel("Fiber Y (pixels)")
    ax.set_xticks(np.arange(SPATIAL_CONFIG['nx']) - 0.5, minor=True)
    ax.set_yticks(np.arange(SPATIAL_CONFIG['ny']) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', alpha=0.2)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.legend(loc='upper right')

    plt.savefig(out_path, dpi=100)
    plt.close()


# ==============================================================================
# パイプライン呼び出し関数 (Main Logic)
# ==============================================================================
def run(run_info, config):
    output_dir = Path(run_info["output_dir"])
    csv_file_path = run_info["csv_path"]
    date_str = run_info.get("date", "20250819")

    print(f"\n--- 2D Spatial Analysis (Hapke Matching with JPL Horizons) ---")

    df = pd.DataFrame()
    if csv_file_path:
        try:
            if Path(csv_file_path).is_dir():
                cands = list(Path(csv_file_path).glob("*.csv"))
                if cands: df = pd.read_csv(cands[0])
            else:
                df = pd.read_csv(csv_file_path)
        except Exception as e:
            print(f"  > CSV Read Error: {e}")

    # Hapkeモデルの読み込み
    hapke_path = output_dir / "1_fits" / f"Hapke{date_str}.fits"
    if not hapke_path.exists():
        hapke_path = output_dir / f"Hapke{date_str}.fits"
    if not hapke_path.exists():
        print(f"  > Error: Hapke model not found at {hapke_path}")
        return

    with fits.open(hapke_path) as hdul_h:
        hapke_img_highres = hdul_h[0].data

    spatial_dir = output_dir / "4_spatial_analysis"
    spatial_dir.mkdir(exist_ok=True)

    fits_dir = output_dir / "1_fits" if (output_dir / "1_fits").exists() else output_dir
    img_fits_files = list(fits_dir.glob("*_tr.img.fits"))

    if not img_fits_files:
        print(f"  > No *_tr.img.fits files found.")
        return

    for fits_path in img_fits_files:
        try:
            with fits.open(fits_path) as hdul:
                img_data = hdul[0].data

            img_2d = img_data.reshape((SPATIAL_CONFIG['ny'], SPATIAL_CONFIG['nx']))
            dia_arcsec = SPATIAL_CONFIG.get('default_diameter_arcsec', 6.0)
            date_obs = None

            if not df.empty:
                base_name = fits_path.stem.replace(".img", "").replace("_tr", "")
                target_df = df[df['Type'] == 'MERCURY']
                if not target_df.empty:
                    target_row = target_df.iloc[0]
                    dia_arcsec = target_row.get('apparent_diameter_arcsec', dia_arcsec)
                    date_obs = target_row.get('DATE-OBS', None)

            if date_obs:
                sun_pa_deg = get_sun_position_angle(date_obs)
            else:
                print("  > Warning: 'DATE-OBS' not found in CSV. Using Sun PA = 0.0")
                sun_pa_deg = 0.0

            r_pix = (dia_arcsec / 2.0) / SPATIAL_CONFIG['plate_scale']

            cx, cy, matched_hapke, rot_angle = find_mercury_center_with_hapke(img_2d, hapke_img_highres, sun_pa_deg)

            plot_name = spatial_dir / f"{fits_path.stem}_hapke_aligned.png"
            plot_hapke_alignment(img_2d, matched_hapke, cx, cy, r_pix, sun_pa_deg, rot_angle, plot_name, fits_path.name)

            print(f"  > Processed: {fits_path.name} | PA: {sun_pa_deg:.1f} | Center: ({cx:.2f}, {cy:.2f})")

        except Exception as e:
            print(f"  > Error processing {fits_path.name}: {e}")


# ==============================================================================
# テスト用スタンドアロン実行ブロック
# ==============================================================================
if __name__ == '__main__':
    # ==========================================================================
    test_date = "20250819"  # テストする日付
    # ==========================================================================

    project_base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025")
    csv_dir_name = "2026ver/csvs"
    output_dir_name = "test4_output"

    base_out = project_base_dir / output_dir_name / test_date
    csv_path = project_base_dir / csv_dir_name / f"mcparams{test_date}.csv"

    print(f"=== 試運転モード起動: 対象日 {test_date} ===")
    mock_run_info = {
        "output_dir": str(base_out),
        "csv_path": str(csv_path),
        "date": test_date
    }

    run(mock_run_info, {})