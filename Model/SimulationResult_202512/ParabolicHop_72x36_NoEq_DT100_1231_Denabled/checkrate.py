# -*- coding: utf-8 -*-
import numpy as np
import os

# --- 物理定数と設定 (正木さんのコードより抜粋) ---
AU_METER = 1.496e11
RM = 2.440e6  # 水星半径 [m]
KB = 1.380649e-23
EV_TO_J = 1.602e-19
NA_MASS = 3.8175e-26

# シミュレーション設定値 (コードから引用)
DIFF_REF_FLUX = 1.0e7 * (100.0 ** 2)  # atoms/m^2/s
DIFF_REF_TEMP = 700.0
DIFF_E_A_EV = 0.80
KB_EV_CONST = 8.617e-5
DIFF_PRE_FACTOR = DIFF_REF_FLUX / np.exp(-DIFF_E_A_EV / (KB_EV_CONST * DIFF_REF_TEMP))

SWS_FLUX_1AU = 10.0 * 100 ** 3 * 400e3 * 4
SWS_YIELD = 0.06
SWS_REF_DENS = 7.5e14 * 100 ** 2

Q_PSD = 1.0e-20 / (100 ** 2)
F_UV_1AU = 1.5e14 * (100 ** 2)


# 温度モデル (Leblanc)
def get_temp(cos_theta, au):
    T_BASE, T_AMP, T_NIGHT = 100.0, 600.0, 100.0
    scaling = np.sqrt(0.306 / au)
    if cos_theta <= 0: return T_NIGHT
    return T_BASE + T_AMP * (cos_theta ** 0.25) * scaling


# 熱的脱離率 (TD Rate) - 正木さんのコードの修正版ロジック
def get_td_rate(temp_K):
    if temp_K < 10: return 0.0
    VIB_FREQ = 1e13
    U_EFF_EV = 1.85
    exponent = -(U_EFF_EV * EV_TO_J) / (KB * temp_K)
    if exponent < -700: return 0.0
    return VIB_FREQ * np.exp(exponent)


def analyze_flux(npy_path, current_au=0.313, subsolar_lon_idx=36):  # 0.313は近日点付近を想定
    """
    npyファイルを読み込み、その瞬間の全放出フラックスを計算してTable 1と比較する
    """
    if not os.path.exists(npy_path):
        print(f"File not found: {npy_path}")
        return

    surface_density = np.load(npy_path)  # shape (72, 36)
    N_LON, N_LAT = surface_density.shape

    # グリッド定義
    lon_edges = np.linspace(-np.pi, np.pi, N_LON + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)

    total_flux_td = 0.0
    total_flux_psd = 0.0
    total_flux_sws = 0.0
    total_supply_diff = 0.0

    max_flux_td = 0.0
    max_flux_psd = 0.0

    # 表面積計算用
    area_sum = 0.0

    print(f"--- Analyzing: {os.path.basename(npy_path)} (AU={current_au}) ---")

    for i in range(N_LON):
        for j in range(N_LAT):
            # 座標と面積
            dlon = lon_edges[1] - lon_edges[0]
            lat_center = (lat_edges[j] + lat_edges[j + 1]) / 2
            cell_area = (RM ** 2) * dlon * (np.sin(lat_edges[j + 1]) - np.sin(lat_edges[j]))
            area_sum += cell_area

            # 太陽位置 (簡易的に中央をサブソーラーとする場合)
            lon_center = (lon_edges[i] + lon_edges[i + 1]) / 2
            # subsolar_lon_idx を中心(0)とする
            rel_lon = lon_center  # 本来はsubsolar_lonとの差分

            # cos theta
            cos_theta = np.cos(lat_center) * np.cos(rel_lon)
            # ※注: 実際のsimulationではsub_lonが動くので、適宜調整してください。
            # ここでは「サブソーラー点(i=N_LON/2, j=N_LAT/2)付近で最大」となるよう簡易計算します。

            # 温度
            temp = get_temp(cos_theta, current_au)

            # 1. Diffusion Supply Flux (供給フラックス)
            flux_diff = 0.0
            if temp > 100:
                flux_diff = DIFF_PRE_FACTOR * np.exp(-DIFF_E_A_EV / (KB_EV_CONST * temp))
            total_supply_diff += flux_diff * cell_area

            # 2. TD Flux (放出フラックス = 表面密度 * レート)
            n_surf = surface_density[i, j]
            rate_td = get_td_rate(temp)
            flux_td = n_surf * rate_td
            total_flux_td += flux_td * cell_area
            max_flux_td = max(max_flux_td, flux_td)

            # 3. PSD Flux
            f_uv = F_UV_1AU / (current_au ** 2)
            illum = 1.0 if cos_theta > 0 else 0.0
            rate_psd = f_uv * Q_PSD * max(0, cos_theta) * illum
            flux_psd = n_surf * rate_psd  # ※ここが重要（後述）
            # Killen論文では PSDは表面被覆率に依存するが、ここでは密度依存にしているか確認
            total_flux_psd += flux_psd * cell_area
            max_flux_psd = max(max_flux_psd, flux_psd)

            # 4. SWS Flux
            # 簡易的に全球平均に近い形で計算（実際は磁気圏モデルによる）
            sw_flux = SWS_FLUX_1AU / (current_au ** 2)
            # コード内の実装: rate = (Flux * Yield) / Ref_Dens
            rate_sws = (sw_flux * SWS_YIELD) / SWS_REF_DENS
            flux_sws = n_surf * rate_sws  # 密度依存

            # 昼側のみ、あるいは特定領域のみの判定が必要だが、ここでは簡易積算
            if cos_theta > 0:
                total_flux_sws += flux_sws * cell_area

    # 単位変換: m^-2 s^-1 -> cm^-2 s^-1
    to_cm2 = 1e-4

    print(f"\n[Calculated Max Local Fluxes (cm^-2 s^-1)]")
    print(f"  Diffusion Supply (Max): {DIFF_REF_FLUX * to_cm2:.2e} (Setting)")
    print(f"  TD Emission (Max):      {max_flux_td * to_cm2:.2e}")
    print(f"  PSD Emission (Max):     {max_flux_psd * to_cm2:.2e}")

    print(f"\n[Total Source Rates (atoms s^-1)]")
    print(f"  Diffusion Supply Total: {total_supply_diff:.2e}")
    print(f"  TD Emission Total:      {total_flux_td:.2e}")
    print(f"  PSD Emission Total:     {total_flux_psd:.2e}")
    print(f"  SWS Emission Total:     {total_flux_sws:.2e}")

    # Killen 2004 Table 1 (Perihelion) との比較
    # PSD Max Rate: 6.0e7 cm^-2 s^-1
    # Impact Max: 6.0e5 cm^-2 s^-1
    # Sputter Max: 3.5e7 cm^-2 s^-1 (Upper Limit)

    print(f"\n[Comparison with Killen (2004) Table 1 (Perihelion)]")
    print(f"  Killen PSD Max Flux:    6.00e+07")
    print(f"  Your PSD Max Flux:      {max_flux_psd * to_cm2:.2e}")
    print(f"  --------------------------------")
    print(f"  Killen Sputter Max:     3.50e+07")
    print(f"  Your SWS Max Flux:      {(max_flux_td if max_flux_td > 0 else 0) * 0 + (flux_sws * to_cm2):.2e} (Approx)")
    # SWSは場所によるので目安


# ==========================================================
# 自動実行用コード (ここを checkrate.py の末尾に貼り付けてください)
# ==========================================================
if __name__ == "__main__":
    import glob

    # 1. カレントディレクトリにある surface_density ファイルを検索
    #    (ファイル名パターン: surface_density_t*.npy)
    search_pattern = "surface_density_t*.npy"
    files = sorted(glob.glob(search_pattern))

    if len(files) > 0:
        # 見つかったファイルの中で一番新しいもの（最後のもの）を使用
        target_file = files[-1]
        print(f"Target file found: {target_file}")

        # 2. 解析を実行
        #    近日点付近(0.313AU)を想定して計算します
        analyze_flux(target_file, current_au=0.313)

    else:
        print(f"Error: No files matching '{search_pattern}' were found in this directory.")
        print("Please make sure you place this script in the same folder as the .npy files.")