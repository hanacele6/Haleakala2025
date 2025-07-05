import numpy as np

# 128個の要素を持つ、値がすべて1.0の配列を作成
A = np.ones(128)

# 順方向にFFTを実行
FA = np.fft.fft(A)

# 結果の最初の要素（直流成分）を表示
print('Python Forward FFT[0]:', FA[0])

# テスト1で計算したFAを使って逆変換を実行
B = np.fft.ifft(FA)

# 結果の最初の要素を表示
print('Python Inverse FFT[0]:', np.real(B[0]))