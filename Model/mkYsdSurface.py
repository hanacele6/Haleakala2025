import numpy as np
import matplotlib.pyplot as plt


def calculate_yasuda_dawn_model(taa_deg, lat_grid, lon_grid, night_accumulation=0.15):
    """
    安田モデル（Dawn側非対称性を考慮）

    Parameters:
    - night_accumulation: Dawn側の夜面に適用する蓄積成分の割合 (0.0 ~ 1.0)
    """

    # 水星の軌道パラメータ
    e = 0.2056
    a = 0.387  # [AU]

    # 太陽距離 R の計算
    nu = np.radians(taa_deg)
    dist_au = (a * (1 - e ** 2)) / (1 + e * np.cos(nu))

    # 太陽天頂角 (SZA) の計算 (SSP = Lat0, Lon0)
    lat_rad = np.radians(lat_grid)
    lon_rad = np.radians(lon_grid)
    cos_sza = np.cos(lat_rad) * np.cos(lon_rad)

    # --- 密度の構成要素 ---

    # 1. 昼側の直接放出 (Source Flux ∝ cos(SZA))
    # 昼側 (cos_sza > 0) のみ値を持ち、夜側は0
    direct_component = np.maximum(0, cos_sza)

    # 2. Dawn側夜面の蓄積 (Dawn-side Night Accumulation)
    # 条件A: 夜側であること (cos_sza <= 0)
    # 条件B: Dawn側であること (Longitude < 0) ※西経側をDawnと仮定
    background_map = np.zeros_like(cos_sza)

    # 条件マスクを作成
    is_night = (cos_sza <= 0)
    is_dawn = (lon_grid < 0) & (lon_grid > -180)  # -180~0度をDawn半球とする

    # Dawn側の夜面にのみ値を代入
    background_map[is_night & is_dawn] = night_accumulation

    # 合計密度 (全体強度 ∝ 1/R^2)
    density_map = (1.0 / dist_au ** 2) * (direct_component + background_map)

    return density_map, dist_au


# --- メイン処理 ---

# 1. グリッド作成
lat = np.linspace(-90, 90, 180)
lon = np.linspace(-180, 180, 360)
lon_grid, lat_grid = np.meshgrid(lon, lat)

# 2. パラメータ設定
taa_list = [0, 90, 180, 270]
night_ratio = 0.20  # Dawn側の夜面に残る割合（少し強調して20%に設定）

# 3. プロット準備 (constrained_layout使用)
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
fig.suptitle(f"Predicted Surface Density (Yasuda Model: Dawn-Asymmetry Only)", fontsize=16)

# 共通カラースケールの設定
_, max_dist_au = calculate_yasuda_dawn_model(0, np.array([0]), np.array([0]), night_ratio)
vmax_val = (1.0 / max_dist_au ** 2) * (1.0 + night_ratio)

im = None

# 4. 計算と描画
for ax, taa in zip(axes.flat, taa_list):
    density, current_dist = calculate_yasuda_dawn_model(taa, lat_grid, lon_grid, night_accumulation=night_ratio)

    # コンタープロット
    im = ax.contourf(lon_grid, lat_grid, density, levels=30, cmap='inferno', vmin=0, vmax=vmax_val)

    # ガイドライン (Terminator)
    ax.axvline(x=-90, color='cyan', linestyle='--', linewidth=1.0, alpha=0.6, label='Dawn')
    ax.axvline(x=90, color='white', linestyle=':', linewidth=0.8, alpha=0.4, label='Dusk')

    # Dawn/Duskのラベルを表示
    if taa == 0:  # 最初のプロットにだけテキストを入れる
        ax.text(-135, 60, "Dawn Night", color='cyan', fontsize=10, ha='center', weight='bold')
        ax.text(135, 60, "Dusk Night", color='white', fontsize=10, ha='center', alpha=0.7)

    # 装飾
    ax.set_title(f"TAA = {taa}° (R = {current_dist:.3f} AU)")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.grid(True, linestyle=':', alpha=0.4)

# 5. カラーバー
cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05, aspect=40)
cbar.set_label('Surface Density', rotation=270, labelpad=15)

plt.show()