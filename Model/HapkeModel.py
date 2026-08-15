import numpy as np
import matplotlib.pyplot as plt
import warnings

def calc_hapke_disk(g_deg, R_pix=150, theta_deg=16.0, w=0.2, h=0.065, B0=2.4, b=0.20, c=0.18):
    """
    指定した位相角(g_deg)における2Dの反射率マップ(numpy配列)を計算する関数。
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
        
        # 方位角の差 psi の計算 (不定形 0/0 の回避)
        num = np.cos(g) - cosi * cose
        den = sini * sine
        cospsi = np.ones_like(cosi) # i=0, e=0 など den=0 の場合は 1.0 (psi=0) とする
        safe_mask = den > 1e-10
        cospsi[safe_mask] = np.clip(num[safe_mask] / den[safe_mask], -1.0, 1.0)
        psi = np.arccos(cospsi)

        # --- Hapke式の計算 ---
        tantheta = np.tan(theta)
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

    RR = np.full((dim, dim), np.nan, dtype=np.float64)
    RR[valid_mask] = rRieg.astype(np.float64)

    return np.fliplr(RR)


def compare_hapke_models():
    # 比較する2つのパラメータセットを定義
    # 数値は元のDefaultを使用し、表記を2002年に見立てる
    params_2002 = {
        'theta_deg': 16.0, 'w': 0.2, 'h': 0.065, 'B0': 2.4, 'b': 0.20, 'c': 0.18
    }
    # 数値は元のNa 589nmを使用し、表記を2010年に見立てる
    params_2010 = {
        'theta_deg': 9.0, 'w': 0.1949, 'h': 0.075, 'B0': 2.3, 'b': 0.2095, 'c': 0.0789
    }

    print("位相曲線(Phase Curve)を計算中...")
    g_list = np.arange(0, 160, 5)
    mean_ref_2002 = []
    mean_ref_2010 = []

    for g in g_list:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            img_2002 = calc_hapke_disk(g, **params_2002)
            img_2010 = calc_hapke_disk(g, **params_2010)
            
            mean_ref_2002.append(np.nanmean(img_2002))
            mean_ref_2010.append(np.nanmean(img_2010))

    # --- 1. 位相曲線と比率のグラフ ---
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 絶対値の比較
    ax1.plot(g_list, mean_ref_2002, marker='o', linestyle='-', color='blue', label='2002 (Mallama et al.)')
    ax1.plot(g_list, mean_ref_2010, marker='s', linestyle='--', color='red', label='2010 (Domingue et al.)')
    ax1.set_title('Average Disk Reflectance vs Phase Angle')
    ax1.set_xlabel('Phase Angle g (deg)')
    ax1.set_ylabel('Mean Bidirectional Reflectance')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # 比率 (2010 / 2002) の比較
    ratio = np.array(mean_ref_2010) / np.array(mean_ref_2002)
    ax2.plot(g_list, ratio, marker='^', linestyle='-', color='green')
    ax2.axhline(1.0, color='black', linestyle='--')
    ax2.set_title('Ratio (2010 / 2002)')
    ax2.set_xlabel('Phase Angle g (deg)')
    ax2.set_ylabel('Reflectance Ratio')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

    # --- 2. 2D画像の視覚的比較 (差分マップ付き) ---
    print("2D画像シミュレーションを計算中...")
    phase_angles = [10, 45, 90, 135]
    fig2, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig2.suptitle('2D Reflectance Comparison: 2002 vs 2010 Models', fontsize=18)

    for i, g in enumerate(phase_angles):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            img_2002 = calc_hapke_disk(g, **params_2002)
            img_2010 = calc_hapke_disk(g, **params_2010)
        
        # 差分の割合(%)を計算 : (2010 - 2002) / 2002 * 100
        diff_pct = np.zeros_like(img_2002)
        valid = (img_2002 > 1e-6) # ゼロ除算回避
        diff_pct[valid] = (img_2010[valid] - img_2002[valid]) / img_2002[valid] * 100.0
        diff_pct[~valid] = np.nan
        
        # 背景(宇宙空間)を黒く描画するためのカラーマップ設定
        cmap_gray = plt.get_cmap('gray').copy()
        cmap_gray.set_bad(color='black')
        cmap_diff = plt.get_cmap('RdBu_r').copy()
        cmap_diff.set_bad(color='black')
        
        # 上段: 2002年
        im0 = axes[0, i].imshow(img_2002, cmap=cmap_gray, vmin=0, vmax=0.03)
        axes[0, i].set_title(f'2002 Model (g={g}°)')
        axes[0, i].axis('off')
        
        # 中段: 2010年
        im1 = axes[1, i].imshow(img_2010, cmap=cmap_gray, vmin=0, vmax=0.03)
        axes[1, i].set_title(f'2010 Model (g={g}°)')
        axes[1, i].axis('off')

        # 下段: 差分(%)マップ
        vmax_diff = 15 # 差分のカラースケールを適正値に戻す
        im2 = axes[2, i].imshow(diff_pct, cmap=cmap_diff, vmin=-vmax_diff, vmax=vmax_diff)
        axes[2, i].set_title(f'Difference % (g={g}°)')
        axes[2, i].axis('off')

        # 一番右の列にだけカラーバーを追加
        if i == 3:
            fig2.colorbar(im0, ax=axes[0, i], fraction=0.046, pad=0.04)
            fig2.colorbar(im1, ax=axes[1, i], fraction=0.046, pad=0.04)
            cbar = fig2.colorbar(im2, ax=axes[2, i], fraction=0.046, pad=0.04)
            cbar.set_label('Difference (%)')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

if __name__ == '__main__':
    compare_hapke_models()