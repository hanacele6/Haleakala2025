import numpy as np
import pandas as pd
from pathlib import Path
import sys

# --- 定数 ---
TARGET_WL = 589.7571833  # nm (Na D1)
FWHM_CONST = 0.005  # nm
C_KMS = 299792.458  # km/s


def calculate_gamma_convolution(solar_data, v_rad, target_wl, fwhm):
    """
    太陽スペクトルとガウス関数の畳み込み積分を行う。
    物理定数は掛けず、純粋な畳み込み係数(gamma0)のみを返す。
    """
    wl0 = solar_data[:, 0]
    sol = solar_data[:, 1]

    # ドップラーシフト
    wl_shifted = wl0 * (1.0 + v_rad / C_KMS)

    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    idx = np.argmin(np.abs(wl_shifted - target_wl))

    # 範囲切り出し
    hw = int(5 * sigma / np.mean(np.diff(wl0))) + 5
    s_idx = max(0, idx - hw)
    e_idx = min(len(wl0), idx + hw + 1)

    wl_crop = wl_shifted[s_idx:e_idx]
    sol_crop = sol[s_idx:e_idx]

    # ガウス関数
    phi = np.exp(-((wl_crop - target_wl) ** 2) / (2.0 * sigma ** 2))

    # 正規化
    if np.sum(phi) == 0: return 0.0
    phi_norm = phi / np.sum(phi)

    # 畳み込み
    gamma0 = np.sum(sol_crop * phi_norm)

    return gamma0


def run(run_info, config):
    output_dir = run_info["output_dir"]
    csv_path = run_info["csv_path"]

    solar_conf = config.get("solar_subtraction", {})
    solar_path = Path(solar_conf.get("solar_spec_path", ""))

    if not solar_path.exists():
        solar_path = output_dir.parent / "SolarSpectrum.txt"
    if not solar_path.exists():
        print(f"エラー: 太陽スペクトルが見つかりません: {solar_path}")
        return

    print(f"\n--- g-factor 係数計算 (Gamma Convolution Only) ---")

    try:
        solar_data = np.loadtxt(solar_path)
        df = pd.read_csv(csv_path)

        g_factors = []
        print(f"  > {len(df)} 件のデータを計算中...")

        for idx, row in df.iterrows():
            v_rad = row['mercury_sun_radial_velocity_km_s']

            if pd.isna(v_rad):
                g_factors.append(np.nan)
                continue

            # 物理定数を掛けない、純粋なgamma値を計算 (~0.24付近になるはず)
            gamma_val = calculate_gamma_convolution(solar_data, v_rad, TARGET_WL, FWHM_CONST)
            g_factors.append(gamma_val)

        # CSVに保存 (列名は g_factor のままだが、中身は gamma係数)
        df['g_factor'] = g_factors
        df.to_csv(csv_path, index=False)

        valid_g = [x for x in g_factors if not pd.isna(x)]
        if valid_g:
            print(f"  > 計算完了: 平均 gamma = {np.mean(valid_g):.8f}")
        else:
            print("  > 警告: 有効な値がありません。")

    except Exception as e:
        print(f"  > エラー: {e}")


if __name__ == "__main__":
    print("Use as module.")