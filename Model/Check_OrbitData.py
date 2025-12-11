import numpy as np


# 1. ユーザー定義関数の再現 (現状のコード)
def calculate_surface_temperature_current(lon_rad, lat_rad, AU, subsolar_lon_rad):
    T0 = 100.0
    T1 = 600.0
    # 現在の実装: ((0.306 / AU) ** 2) をかけている
    cos_theta = np.cos(lat_rad) * np.cos(lon_rad - subsolar_lon_rad)
    if cos_theta <= 0:
        return T0
    return T0 + T1 * (cos_theta ** 0.25) * ((0.306 / AU) ** 2)


# 2. 論文/物理法則に基づいた修正案 (1/√R 則)
def calculate_surface_temperature_corrected(lon_rad, lat_rad, AU, subsolar_lon_rad):
    T0 = 100.0
    T1 = 600.0
    # 修正: 温度は距離の平方根に反比例 (ステファン・ボルツマン則: T ∝ Flux^0.25 ∝ (1/R^2)^0.25 = 1/√R)
    # あるいは論文 [Hale and Hapke, 2002] の記述に従いパラメータ化
    scaling = np.sqrt(0.306 / AU)

    cos_theta = np.cos(lat_rad) * np.cos(lon_rad - subsolar_lon_rad)
    if cos_theta <= 0:
        return T0
    return T0 + T1 * (cos_theta ** 0.25) * scaling


def check_temperature_extremes():
    # 近日点 (Perihelion) と 遠日点 (Aphelion) の距離
    au_peri = 0.3075
    au_aphe = 0.4667

    print(
        f"{'Condition':<15} | {'AU':<8} | {'Current Temp(K)':<20} | {'Corrected Temp(K)':<20} | {'Paper Reference(K)'}")
    print("-" * 90)

    # 1. 近日点 (TAA=0, SubSolarLon=0)
    t_curr_peri = calculate_surface_temperature_current(0, 0, au_peri, 0)
    t_corr_peri = calculate_surface_temperature_corrected(0, 0, au_peri, 0)
    print(f"{'Perihelion':<15} | {au_peri:<8.4f} | {t_curr_peri:<20.1f} | {t_corr_peri:<20.1f} | ~700 K")

    # 2. 遠日点 (TAA=180, SubSolarLon=pi/2 or -pi/2)
    # ここでは計算簡略化のため、subsolar_lon=0として、直下点の温度を計算
    t_curr_aphe = calculate_surface_temperature_current(0, 0, au_aphe, 0)
    t_corr_aphe = calculate_surface_temperature_corrected(0, 0, au_aphe, 0)
    print(f"{'Aphelion':<15} | {au_aphe:<8.4f} | {t_curr_aphe:<20.1f} | {t_corr_aphe:<20.1f} | ~575 K")


if __name__ == "__main__":
    check_temperature_extremes()