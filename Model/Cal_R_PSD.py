import numpy as np

# ==============================================================================
# 1. 物理定数 (CGS単位系: cm, g, s)
# ==============================================================================
F_UV_1AU = 1.5e14      # 1AUでの紫外線光子束 [photons/cm^2/s]
Q_PSD = 2.0e-20          # 光刺激脱離の断面積 [cm^2]
# Na表面密度を [atoms/m^2] から [atoms/cm^2] に変換
SIGMA_NA_CM = 1.5e17 / (100**2) # [atoms/cm^2]

def calculate_rpsd_cm(distance_au, cos_z):
    """
    指定された太陽からの距離とcos(Z)でR_PSDをcm単位で計算する関数

    Args:
        distance_au (float): 水星と太陽の距離 [AU]
        cos_z (float): 天頂角のコサイン (0から1の値)

    Returns:
        float: 粒子放出率 R_PSD [atoms/cm^2/s]
    """
    if not (0 <= cos_z <= 1):
        print("警告: cos_z は0から1の範囲で指定してください。")
        return 0

    # 1. 指定された距離での紫外線光子束 F_UV を計算 [photons/cm^2/s]
    #    メートルへの変換は行わない
    f_uv_at_r0 = F_UV_1AU * (1 / distance_au)**2

    # 2. R_PSD を計算 [atoms/cm^2/s]
    #    全ての値がcm単位なので、そのまま計算する
    r_psd = f_uv_at_r0 * Q_PSD * cos_z * SIGMA_NA_CM

    return r_psd

# ==============================================================================
# 2. 使い方 (計算例)
# ==============================================================================

# 例1：以前に手計算した条件 (距離0.4 AU, cos(Z)=1)
distance1 = 0.4
cos_z1 = 1.0
rpsd1 = calculate_rpsd_cm(distance1, cos_z1)
print(f"距離が {distance1} AU, cos(Z)={cos_z1} の場合:")
print(f"  R_PSD = {rpsd1:,.3e} atoms/cm^2/s")
print("-" * 30)


# 例2：水星が近日点(最も太陽に近い位置)にいる場合 (距離 ~0.31 AU, cos(Z)=1)
distance2 = 0.31
cos_z2 = 1.0
rpsd2 = calculate_rpsd_cm(distance2, cos_z2)
print(f"水星の近日点 (距離 {distance2} AU, cos(Z)={cos_z2}) の場合:")
print(f"  R_PSD = {rpsd2:,.3e} atoms/cm^2/s")
print("-" * 30)


# 例3：平均距離で、太陽が斜めから差す場合 (距離 0.387 AU, cos(Z)=0.5)
distance3 = 0.387
cos_z3 = 0.5 # 太陽高度30度 (天頂角60度)
rpsd3 = calculate_rpsd_cm(distance3, cos_z3)
print(f"平均距離 (距離 {distance3} AU) で太陽が斜め (cos(Z)={cos_z3}) の場合:")
print(f"  R_PSD = {rpsd3:,.3e} atoms/cm^2/s")
print("-" * 30)