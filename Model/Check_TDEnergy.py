import numpy as np
import matplotlib.pyplot as plt


def plot_flux_distribution(temp_k=700):
    """
    熱脱離フラックス分布（表面から放出される粒子のエネルギー分布）のみをプロットする
    temp_k: 表面温度 (Kelvin)
    """
    # 物理定数
    k_B_ev = 8.617e-5  # ボルツマン定数 [eV/K]
    kT = k_B_ev * temp_k

    # エネルギー範囲 (0 eV 〜 0.5 eV)
    E = np.linspace(0, 0.5, 1000)

    # === フラックス分布 (Flux Distribution) ===
    # シミュレーションで使用されている分布
    # f(E) ∝ E * exp(-E/kT)
    # これは np.random.gamma(shape=2.0, scale=kT) の理論曲線です
    flux_pdf = (E / (kT ** 2)) * np.exp(-E / kT)

    # プロット
    plt.figure(figsize=(10, 6))

    # フラックス分布 (赤線)
    plt.plot(E, flux_pdf, color='red', lw=2,
             label=r'Flux Dist. (Desorbing): $f(E) \propto E \exp(-E/k_B T)$')

    # 平均エネルギーの表示 (Flux分布の平均は 2kT)
    mean_E_flux = 2 * kT
    plt.axvline(mean_E_flux, color='black', alpha=0.5, linestyle=':',
                label=f'Mean Energy = ({mean_E_flux:.3f} eV)')

    plt.title(f'Thermal Desorption Energy Distribution (T = {temp_k} K)\n(Simulation Input Model)')
    plt.xlabel('Energy (eV)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # グラフを表示
    plt.show()


# 実行
if __name__ == "__main__":
    plot_flux_distribution(temp_k=700)