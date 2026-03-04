import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from astropy.time import Time
from astroquery.jplhorizons import Horizons
import scipy.ndimage as ndimage
from matplotlib.colors import LogNorm
import warnings

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
# ヘルパー: JPL Horizonsから太陽のポジションアングルを取得
# ==============================================================================
def get_sun_position_angle(date_obs_str):
    """
    観測日時(DATE-OBS)からJPL Horizonsにアクセスし、
    水星に対する太陽のPosition Angle(天の北から東回り)を取得する。
    """
    print(f"    -> Querying JPL Horizons for Sun Position Angle at {date_obs_str}...")
    try:
        # 時刻文字列をユリウス日に変換 (例: '2024-10-16 14:12:38' -> JD)
        jd_time = Time(date_obs_str).jd

        # Target: 199 (水星), Location: F65 (ハレアカラ T60)
        obj = Horizons(id='199', location='F65', epochs=jd_time)
        eph = obj.ephemerides()

        # sunTargetPA: ターゲット(水星)の中心から見た、太陽方向のPosition Angle
        sun_pa = eph['sunTargetPA'][0]
        print(f"    -> Sun Position Angle retrieved: {sun_pa:.2f} deg")
        return sun_pa
    except Exception as e:
        print(f"    -> [Warning] Failed to get Sun PA from JPL Horizons: {e}")
        print("    -> Using default Sun PA = 0.0")
        return 0.0


# ==============================================================================
# 中心推定関数 (Hapke Template Matching) - サイズ不一致修正版
# ==============================================================================
def find_mercury_center_with_hapke(obs_img_2d, hapke_img_highres, sun_pa_deg):
    """
    Hapkeモデルをシーイングでぼかし、太陽の角度に合わせて回転させ、
    観測画像の解像度(12x10)のキャンバスに配置してマッチングを行う。
    """
    ny_obs, nx_obs = obs_img_2d.shape

    # 1. シーイング(2Dぼかし)の適用
    sigma_arcsec = SPATIAL_CONFIG['seeing_arcsec'] / 2.355
    sigma_pix = sigma_arcsec * SPATIAL_CONFIG['hapke_scale_factor']
    hapke_blurred = ndimage.gaussian_filter(hapke_img_highres, sigma=sigma_pix)

    # ---------------------------------------------------------
    # 2. 太陽のPosition Angleに合わせてHapke画像を回転 (パイプライン固定版)
    # ---------------------------------------------------------

    # 鏡面反転の補正 (カセグレン焦点のため東西が反転している場合の固定処理)
    # ※もし反転が不要と分かれば、この1行は消してください
    hapke_flipped = np.fliplr(hapke_blurred)

    # [定数1] Hapkeモデルの初期の光の方向 (fliplrをした場合、光は「右(0度)」になります)
    # ※ fliplrをしない場合は 180.0 (左) にしてください。
    HAPKE_LIGHT_ANGLE = 0.0

    # [定数2] カメラの固定向き (画像上の「北」は何度方向か？)
    # 10の側が南北なので、北は「右(0度)」か「左(180度)」のどちらかです。
    # ※ 一度決めたら全日程で固定します。
    NORTH_ANGLE_ON_IMAGE = 0.0

    # [定数3] カメラの東西の向き (画像上の「東」は何度方向か？)
    # 通常、北から反時計回りに90度回した方向が東です。
    EAST_ANGLE_ON_IMAGE = NORTH_ANGLE_ON_IMAGE + 90.0

    # ドイツ式赤道儀の子午線反転の自動検知 (夕方観測なら望遠鏡が180度逆立ちしている)
    telescope_flip_angle = 180.0 if sun_pa_deg > 180.0 else 0.0

    # ---------------------------------------------------------
    # 天球上の太陽PA(sun_pa_deg)を、画像上の光の入射角に変換する
    # ---------------------------------------------------------
    # 太陽のPA(北から東回り)を、Python画像座標(北から東方向への角度)にマッピング
    # 太陽の方向 = 北の角度 + (PA * 東の方向への符号) + 望遠鏡反転

    # 東が +90度の方向にあるので、PAをそのまま足す
    target_light_angle = NORTH_ANGLE_ON_IMAGE + sun_pa_deg + telescope_flip_angle

    # ---------------------------------------------------------
    # 最終的な回転処理
    # ---------------------------------------------------------
    # 目的の光の角度(target_light_angle)から、Hapkeの初期角度(HAPKE_LIGHT_ANGLE)を引き算する
    rotation_angle = target_light_angle - HAPKE_LIGHT_ANGLE

    # ndimage.rotate は時計回りがデフォルトなので、マイナスをつけて反時計回りにする
    hapke_rotated = ndimage.rotate(hapke_flipped, -rotation_angle, reshape=False, mode='constant', cval=0.0)


    # 3. ダウンスケール(縮小)
    zoom_factor = 1.0 / (SPATIAL_CONFIG['hapke_scale_factor'] * SPATIAL_CONFIG['plate_scale'])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        hapke_lowres = ndimage.zoom(hapke_rotated, zoom_factor, order=1)

    # ---------------------------------------------------------
    # サイズの不一致を防ぐため、12x10の黒いキャンバスに配置する
    # ---------------------------------------------------------
    h_low, w_low = hapke_lowres.shape
    hapke_canvas = np.zeros((ny_obs, nx_obs), dtype=float)

    # キャンバスの中央付近に配置するためのオフセット計算
    y0 = (ny_obs - h_low) // 2
    x0 = (nx_obs - w_low) // 2

    # はみ出し防止（水星が大きすぎる場合の安全対策）
    y_start, y_end = max(0, y0), min(ny_obs, y0 + h_low)
    x_start, x_end = max(0, x0), min(nx_obs, x0 + w_low)
    h_start, h_end = max(0, -y0), max(0, -y0) + (y_end - y_start)
    w_start, w_end = max(0, -x0), max(0, -x0) + (x_end - x_start)

    # ペースト実行
    hapke_canvas[y_start:y_end, x_start:x_end] = hapke_lowres[h_start:h_end, w_start:w_end]

    # キャンバス上での幾何学的な「真の中心」座標を再計算
    true_cy_hapke = y_start + (h_low / 2.0) - h_start
    true_cx_hapke = x_start + (w_low / 2.0) - w_start
    # ---------------------------------------------------------

    # 4. 観測画像の光度重心で初期位置のアタリをつける
    threshold = np.max(obs_img_2d) * 0.2
    img_masked = np.where(obs_img_2d > threshold, obs_img_2d, 0)
    cy_init, cx_init = ndimage.center_of_mass(img_masked)

    if np.isnan(cy_init) or np.isnan(cx_init):
        cy_init, cx_init = ny_obs / 2.0, nx_obs / 2.0

    # 5. グリッドサーチによるテンプレートマッチング
    search_range = 2.0  # ±2ピクセル探索
    steps = 40
    best_score = -np.inf
    best_cx, best_cy = cx_init, cy_init
    best_shifted_model = np.zeros_like(obs_img_2d)

    for dy in np.linspace(-search_range, search_range, steps):
        for dx in np.linspace(-search_range, search_range, steps):
            cx_test = cx_init + dx
            cy_test = cy_init + dy

            # Hapkeキャンバスの中心を cx_test, cy_test にシフト
            shift_y = cy_test - true_cy_hapke
            shift_x = cx_test - true_cx_hapke

            shifted_hapke = ndimage.shift(hapke_canvas, (shift_y, shift_x), order=1, mode='constant', cval=0.0)

            # 内積でスコア化（12x10同士なのでエラーにならない）
            score = np.sum(obs_img_2d * shifted_hapke)

            if score > best_score:
                best_score = score
                best_cx = cx_test
                best_cy = cy_test
                best_shifted_model = shifted_hapke

    return best_cx, best_cy, best_shifted_model, rotation_angle


# ==============================================================================
# プロット関数 (Hapke重ね合わせ確認用)
# ==============================================================================
def plot_hapke_alignment(obs_img_2d, hapke_model_matched, cx, cy, r_pix, sun_pa, rotation_angle, out_path, title):
    fig, ax = plt.subplots(figsize=(10, 8))

    # 1. 元画像とHapkeモデルの等高線
    norm = LogNorm(vmin=np.max(obs_img_2d) * 0.01, vmax=np.max(obs_img_2d)) if np.max(obs_img_2d) > 0 else None
    im = ax.imshow(obs_img_2d, origin='lower', cmap='viridis', norm=norm,
                   extent=[-0.5, SPATIAL_CONFIG['nx'] - 0.5, -0.5, SPATIAL_CONFIG['ny'] - 0.5])
    plt.colorbar(im, label='Observed Continuum Flux (Log)')

    ax.contour(hapke_model_matched, levels=5, colors='white', alpha=0.6,
               extent=[-0.5, SPATIAL_CONFIG['nx'] - 0.5, -0.5, SPATIAL_CONFIG['ny'] - 0.5])

    # 2. 真の輪郭（赤い円）と中心
    circle = plt.Circle((cx, cy), r_pix, color='red', fill=False, linewidth=2, linestyle='-')
    ax.add_patch(circle)
    ax.plot(cx, cy, 'rx', markersize=12, markeredgewidth=2, label='True Center')

    # ---------------------------------------------------------
    # 太陽光の入射方向を示す矢印の描画
    # ---------------------------------------------------------
    # Hapkeモデルは fliplr で右側(0度)から光が当たる状態になり、
    # そこから rotation_angle 分だけ回転させているため、太陽の位置角度は rotation_angle となる。
    theta_rad = np.radians(rotation_angle)

    # 矢印の始点 (水星の少し外側)
    sun_x = cx + (r_pix * 1.8) * np.cos(theta_rad)
    sun_y = cy + (r_pix * 1.8) * np.sin(theta_rad)

    # 矢印の終点 (水星のリム付近)
    target_x = cx + (r_pix * 0.8) * np.cos(theta_rad)
    target_y = cy + (r_pix * 0.8) * np.sin(theta_rad)

    # 矢印を描画 (黄色で分かりやすく)
    ax.annotate('', xy=(target_x, target_y), xytext=(sun_x, sun_y),
                arrowprops=dict(facecolor='yellow', edgecolor='orange',
                                shrink=0.0, width=3, headwidth=10),
                zorder=5)

    # 太陽マーク（テキスト）を始点付近に配置
    ax.text(sun_x, sun_y, '☀ SUN', color='yellow', fontsize=14,
            ha='center', va='center', fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))
    # ---------------------------------------------------------

    ax.set_title(f"{title}\nCenter: ({cx:.2f}, {cy:.2f}) | Sun PA: {sun_pa:.1f} deg")
    ax.set_xlabel("Fiber X (East-West axis)")
    ax.set_ylabel("Fiber Y (North-South axis)")

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

    # CSVの読み込み
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

    # Hapkeのマスターファイルを検索
    hapke_path = output_dir / "1_fits" / f"Hapke{date_str}.fits"
    if not hapke_path.exists():
        hapke_path = output_dir / f"Hapke{date_str}.fits"

    if not hapke_path.exists():
        print(f"  > Error: Hapke model not found at {hapke_path}")
        return

    # 高解像度Hapkeの読み込み
    with fits.open(hapke_path) as hdul_h:
        hapke_img_highres = hdul_h[0].data

    spatial_dir = output_dir / "4_spatial_analysis"
    spatial_dir.mkdir(exist_ok=True)

    # 観測データ(連続光)の取得
    fits_dir = output_dir / "1_fits" if (output_dir / "1_fits").exists() else output_dir
    img_fits_files = list(fits_dir.glob("*_tr.img.fits"))

    if not img_fits_files:
        print(f"  > No *_tr.img.fits files found.")
        return

    # 各画像の処理
    for fits_path in img_fits_files:
        try:
            # 観測画像の読み込み
            with fits.open(fits_path) as hdul:
                img_data = hdul[0].data
                header = hdul[0].header

            img_2d = img_data.reshape((SPATIAL_CONFIG['ny'], SPATIAL_CONFIG['nx']))

            # 視直径の取得 (CSVから。なければデフォルト)
            dia_arcsec = SPATIAL_CONFIG.get('default_diameter_arcsec', 6.0)
            date_obs = None

            if not df.empty:
                # このファイル名に合致する行を探すロジック（適宜調整してください）
                base_name = fits_path.stem.replace(".img", "")

                # MERCURYの最初の行を仮に使用
                target_row = df[df['Type'] == 'MERCURY'].iloc[0]
                dia_arcsec = target_row.get('apparent_diameter_arcsec', dia_arcsec)
                date_obs = target_row.get('DATE-OBS', None)

            # JPL Horizons から太陽の角度を取得
            if date_obs:
                sun_pa_deg = get_sun_position_angle(date_obs)
            else:
                print("  > Warning: 'DATE-OBS' not found in CSV. Using Sun PA = 0.0")
                sun_pa_deg = 0.0

            r_pix = (dia_arcsec / 2.0) / SPATIAL_CONFIG['plate_scale']

            # マッチング実行
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
    # ★ ここを書き換えてテストしたい日付を指定してください ★
    test_date = "20250819"
    # ==========================================================================

    project_base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025")
    csv_dir_name = "2026ver/csvs"
    output_dir_name = "test4_output"

    # パスの構築
    base_out = project_base_dir / output_dir_name / test_date
    csv_path = project_base_dir / csv_dir_name / f"mcparams{test_date}.csv"

    print(f"=== 試運転モード起動 (エッジ可視化版): 対象日 {test_date} ===")
    print(f"Output Dir : {base_out}")
    print(f"CSV Path   : {csv_path}")

    # モックアップの run_info を作成
    mock_run_info = {
        "output_dir": str(base_out),
        "csv_path": str(csv_path),
        "date": test_date
    }

    # 実行
    run(mock_run_info, {})