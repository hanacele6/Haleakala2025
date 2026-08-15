import numpy as np
import os

# スキップされたファイルの一つを指定
test_file = r"./SimulationResult_202607/ParabolicHop_72x36_NoEq_DT100_0707_BD0.8_UG1.85_Q2.0_A2.0e+07_LT190k_15yr/surface_density_t16412.npy"

print(f"ファイルサイズ: {os.path.getsize(test_file) / (1024*1024):.2f} MB")

try:
    data = np.load(test_file)
    print(f"読み込み成功。Shape: {data.shape}")
except Exception as e:
    print(f"エラー発生: {type(e).__name__} - {e}")