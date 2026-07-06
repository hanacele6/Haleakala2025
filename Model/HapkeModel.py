import numpy as np
import matplotlib.pyplot as plt
import warnings

#def calc_hapke_disk(g_deg, R_pix=150, theta_deg=16.0, w=0.2, h=0.065, B0=2.4, b=0.20, c=0.18):
def calc_hapke_disk(g_deg, R_pix=150, theta_deg=9.0, w=0.195, h=0.075, B0=2.3, b=0.21, c=0.08):
    """
    元のコードの計算ロジックを独立させた関数。
    指定した位相角(g_deg)における2Dの反射率マップ(numpy配列)を返します。
    """
    pi = np.pi
    theta = np.deg2rad(theta_deg)
    dim = int(R_pix * 2)
    
    x_coords = np.arange(dim, dtype=np.float64) - R_pix + 0.5
    y_coords = np.arange(dim, dtype=np.float64) - R_pix + 0.5
    x, y = np.meshgrid(x_coords, y_coords)
    disk_mask = (x ** 2 + y ** 2) < R_pix ** 2

    with np.errstate(divide='ignore', invalid='ignore'):
        g = np.deg2rad(g_deg)
        LL = np.arcsin(y / R_pix)
        LambdaL = np.arcsin(x / (R_pix * np.cos(LL)))
        terminator_mask = LambdaL < (pi / 2.0 - g)
        valid_mask = disk_mask & terminator_mask

        LL_v = LL[valid_mask]
        LambdaL_v = LambdaL[valid_mask]

        cosi = np.clip(np.cos(LambdaL_v + g) * np.cos(LL_v), 0, 1)
        cose = np.clip(np.cos(LambdaL_v) * np.cos(LL_v), 0, 1)
        i = np.arccos(cosi)
        e = np.arccos(cose)
        sini = np.sin(i)
        sine = np.sin(e)
        
        # ゼロ除算回避のため分母に微小値(1e-10)を追加しています
        cospsi = np.clip((np.cos(g) - cosi * cose) / (np.sin(i) * np.sin(e) + 1e-10), -1, 1)
        psi = np.arccos(cospsi)

        # --- Hapke式の計算 ---
        tantheta = np.tan(theta)
        # ゼロ除算回避
        tani = np.clip(np.tan(i), 1e-10, None)
        tane = np.clip(np.tan(e), 1e-10, None)
        sinpsihalf = np.sin(psi / 2.0)

        kai = 1.0 / np.sqrt(1.0 + pi * (tantheta) ** 2)
        E1i = np.exp(-2.0 / pi / tantheta / tani)
        E2i = np.exp(-1.0 / pi / tantheta ** 2 / tani ** 2)
        E1e = np.exp(-2.0 / pi / tantheta / tane)
        E2e = np.exp(-1.0 / pi / tantheta ** 2 / tane ** 2)

        mu0e0 = kai * (cosi + sini * tantheta * E2i / (2.0 - E1i))
        cond = (i <= e)

        mu0e_true = kai * (cosi + sini * tantheta * (cospsi * E2e + sinpsihalf ** 2 * E2i) / (2 - E1e - (psi / pi) * E1i))
        mue_true = kai * (cose + sine * tantheta * (E2e - sinpsihalf ** 2 * E2i) / (2 - E1e - (psi / pi) * E1i))
        mue0_true = kai * (cose + sine * tantheta * E2e / (2 - E1e))
        fpsi_true = np.exp(-2.0 * np.tan(psi / 2.0))
        Siepsi_true = (mue_true / mue0_true) * (cosi / mu0e0) * kai / (1 - fpsi_true + fpsi_true * kai * cosi / mu0e0)

        mu0e_false = kai * (cosi + sini * tantheta * (E2i - sinpsihalf ** 2 * E2e) / (2 - E1i - (psi / pi) * E1e))
        mue_false = kai * (cose + sine * tantheta * (cospsi * E2i + sinpsihalf ** 2 * E2e) / (2 - E1i - (psi / pi) * E1e))
        mue0_false = kai * (cose + sine * tantheta * E2e / (2 - E1e))
        fpsi_false = np.exp(-2.0 * np.tan(psi / 2.0))
        Siepsi_false = (mue_false / mue0_false) * (cosi / mu0e0) * kai / (1 - fpsi_false + fpsi_false * kai * cose / mue0_false)

        mu0e = np.where(cond, mu0e_true, mu0e_false)
        mue = np.where(cond, mue_true, mue_false)
        Siepsi = np.where(cond, Siepsi_true, Siepsi_false)

        Bg = B0 / (1.0 + np.tan(g / 2.0) / h)
        pg = 1.0 + b * np.cos(g) + c * (3.0 * (np.cos(g)) ** 2 - 1.0) / 2.0
        gamma = np.sqrt(1.0 - w)
        
        Hmu0e = (1.0 + 2.0 * mu0e) / (1.0 + 2.0 * gamma * mu0e)
        Hmue = (1.0 + 2.0 * mue) / (1.0 + 2.0 * gamma * mue)
        
        # 反射率(rRieg)の算出
        rRieg = (w / (4.0 * pi)) * (mu0e / (mu0e + mue)) * ((1.0 + Bg) * pg + Hmu0e * Hmue - 1.0) * Siepsi

    # 背景をNaNにして円盤のみを描画できるようにする
    RR = np.full((dim, dim), np.nan, dtype=np.float64)
    RR[valid_mask] = rRieg.astype(np.float64)

    return np.fliplr(RR)

def plot_hapke_results():
    """図とグラフを生成して表示するメイン関数"""
    print("シミュレーションを実行中...")

    # --- 1. 位相角ごとの2D画像を描画 ---
    phase_angles = [0, 45, 90, 135]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('Hapke Model Simulated Disks (Bidirectional Reflectance)', fontsize=16)

    for ax, g in zip(axes, phase_angles):
        # 警告を非表示にして計算
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            img = calc_hapke_disk(g)
            
        # vmaxを少し絞ることでコントラストを調整（白飛び防止）
        im = ax.imshow(img, cmap='gray', vmin=0, vmax=0.03)
        ax.set_title(f'Phase Angle: {g}°')
        ax.axis('off')

    plt.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
    plt.show()

    # --- 2. 位相角に対する平均反射率のグラフを描画 ---
    print("位相曲線(Phase Curve)を計算中...")
    g_list = np.arange(0, 160, 5)
    mean_reflectances = []

    for g in g_list:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            img = calc_hapke_disk(g)
            # 有効なピクセル(NaN以外)の平均値を計算
            mean_ref = np.nanmean(img)
            mean_reflectances.append(mean_ref)

    plt.figure(figsize=(8, 5))
    plt.plot(g_list, mean_reflectances, marker='o', linestyle='-', color='b')
    plt.title('Average Disk Reflectance vs Phase Angle (Phase Curve)')
    plt.xlabel('Phase Angle g (deg)')
    plt.ylabel('Mean Bidirectional Reflectance')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 衝効果（g=0付近の跳ね上がり）を強調
    plt.axvspan(0, 15, color='orange', alpha=0.2, label='Opposition Effect region')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plot_hapke_results()