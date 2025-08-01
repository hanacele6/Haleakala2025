import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ==============================================================================
# 設定と実行
# ==============================================================================
if __name__ == "__main__":

    # ▼▼▼ 設定項目 ▼▼▼
    # --------------------------------------------------------------------------
    # 1. パスの設定
    base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
    data_dir = base_dir / "output/20250701/"
    solar_spec_path = base_dir / "SolarSpectrum.txt"

    # 2. 比較したい観測データファイル名
    #    (3番目のコードで背景光減算後に作られた .dat ファイル)
    observation_dat_filename = "MERCURY1_tr.totfib.dat"

    # 3. シフト量や物理定数の設定
    #    4番目のコードで使っている値や、試したい値を入力します。
    #    CSVから読み込む値も、ここでは手で入力します。
    constants = {
        # 水星-地球間の視線速度 (km/s)
        'Vme': -27.98578,
        # 水星-太陽間の視線速度 (km/s)
        'Vms': 6.1923,
        # 手動で加える波長シフト量 (nm)
        'sft': 0.0005,
        # 光速 (km/s)
        'c': 299792.458
    }
    # --------------------------------------------------------------------------
    # ▲▲▲ 設定はここまで ▲▲▲


    # --- 1. ファイルの読み込み ---
    observation_dat_path = data_dir / observation_dat_filename
    try:
        print(f"観測データを読み込んでいます: {observation_dat_path.name}")
        obs_data = np.loadtxt(observation_dat_path, skiprows=1)
        wl, Nat = obs_data[:, 0], obs_data[:, 1]

        print(f"太陽光モデルを読み込んでいます: {solar_spec_path.name}")
        sol_data = np.loadtxt(solar_spec_path)
    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません。パスを確認してください。")
        print(e)
        sys.exit()

    # --- 2. 太陽光モデルの波長をシフト ---
    sft = constants['sft']
    Vms = constants['Vms']
    Vme = constants['Vme']
    c = constants['c']

    # 太陽光の波長を真空→大気に変換（必要に応じて）
    # wavair_factor = 1.000276
    wavair_factor = 1.000 # 元のコードに合わせて1.0に設定
    sol_wl_orig = sol_data[:, 0] / wavair_factor
    sol_flux1 = sol_data[:, 2] # 直接光成分
    sol_flux2 = sol_data[:, 1] # 反射光成分

    # シフト後の波長軸を計算
    direct_solar_wl = sol_wl_orig - sft
    reflected_solar_wl = direct_solar_wl * (1 + Vms / c) * (1 + Vme / c)

    # --- 3. 4番目のコードが行う「有効範囲」を計算 ---
    valid_wl_min = max(np.min(direct_solar_wl), np.min(reflected_solar_wl))
    valid_wl_max = min(np.max(direct_solar_wl), np.max(reflected_solar_wl))

    # --- 4. グラフの描画 ---
    print("グラフを生成しています...")
    fig, ax = plt.subplots(figsize=(15, 8))

    # 観測データをプロット (強度を正規化)
    ax.plot(wl, Nat / np.median(Nat), label=f"Observation: {observation_dat_filename}", color='black', linewidth=2, zorder=10)

    # シフトさせた太陽光モデルをプロット (強度を正規化)
    ax.plot(direct_solar_wl, sol_flux1 / np.median(sol_flux1),
            label=f"Solar Model (Direct, sft={sft})", color='dodgerblue', linestyle='--')
    ax.plot(reflected_solar_wl, sol_flux2 / np.median(sol_flux2),
            label=f"Solar Model (Reflected, Vms={Vms}, Vme={Vme})", color='red', linestyle='--')

    # 4番目のコードがデータを切り取る範囲を縦線で表示
    ax.axvline(x=valid_wl_min, color='limegreen', linestyle=':', label=f"Crop Min: {valid_wl_min:.4f} nm")
    ax.axvline(x=valid_wl_max, color='limegreen', linestyle=':', label=f"Crop Max: {valid_wl_max:.4f} nm")

    # グラフの装飾
    ax.set_title("Diagnostic Plot: Observation vs. Shifted Solar Models")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized Intensity")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

    print("\n処理が完了しました。")