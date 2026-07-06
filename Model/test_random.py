import numpy as np
from numba import njit

@njit()
def draw_rand_numba(n, seed):
    np.random.seed(seed)
    r = np.empty(n)
    for i in range(n):
        r[i] = np.random.random()
    return r

@njit()
def draw_gamma_numba(seed):
    np.random.seed(seed)
    return np.random.gamma(2.0, 1.0)

# ウォームアップ（コンパイル）
_ = draw_rand_numba(1, 42)
_ = draw_gamma_numba(42)

# --- random() の比較 ---
np.random.seed(42)
py = np.array([np.random.random() for _ in range(5)])

nb = draw_rand_numba(5, 42)

print("=== random() の比較 ===")
print("Python:", py)
print("Numba: ", nb)
print("一致?  ", np.allclose(py, nb))

# --- gamma() の比較 ---
np.random.seed(42)
py_gamma = np.random.gamma(2.0, 1.0)

nb_gamma = draw_gamma_numba(42)

print("\n=== gamma(2.0, 1.0) の比較 ===")
print("Python:", py_gamma)
print("Numba: ", nb_gamma)
print("一致?  ", np.isclose(py_gamma, nb_gamma))