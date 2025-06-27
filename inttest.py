import numpy as np
from scipy.interpolate import interp1d

# --- ステップ1でIDLが出力した値をここにコピー＆ペーストしてください ---
coord_x = 396.156374
val_at_396 = 1926#<IDLが出力したval_at_396の値>
val_at_397 = 1917#<IDLが出力したval_at_397の値>
idl_result = 1924.0
# --------------------------------------------------------------------

# 1. Pythonで線形補間を手計算してみる
fraction = coord_x - 396  # 小数部分 (0.156374)
manual_result = val_at_396 + (val_at_397 - val_at_396) * fraction

# 2. Pythonのinterp1dを使って計算してみる
#    IDLの interpolate(Array, X) は、配列のインデックスをx座標と見なすため、
#    x_dataは [396, 397] となります。
x_data = np.array([396, 397])
y_data = np.array([val_at_396, val_at_397])

interp_func = interp1d(x_data, y_data)
scipy_result = interp_func(coord_x)

# --- 結果の比較 ---
print("--- Python Manual Check ---")
print(f"IDL's Result: {idl_result}")
print(f"Manual Calculation in Python: {manual_result}")
print(f"Scipy's interp1d Result: {scipy_result}")

# 3つの値は理論上、ほぼ一致するはずです。