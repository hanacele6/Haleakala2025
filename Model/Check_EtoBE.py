import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 定数と設定
# ==========================================
K_BOLTZMANN = 1.380649e-23
EV_TO_JOULE = 1.602e-19

settings = {
    'USE_DYNAMIC_U_MODEL': True,
    'N_U_BINS': 5,   
    'U_MIN': 1.4,     
    'U_MAX': 2.7,     
}

# ==========================================
# 2. コア関数群 
# ==========================================
def setup_binding_energy_bins(settings: dict) -> tuple:
    U_bins = np.linspace(settings['U_MIN'], settings['U_MAX'], settings['N_U_BINS'])
    # 表面の穴の分布 (1.85 eV が中心)
    V_weights = np.exp(-0.5 * ((U_bins - 1.85) / 0.35)**2)
    V_weights /= np.max(V_weights)
    return U_bins, V_weights

def calculate_probability_distribution(E_in_eV: float, temp_impact_K: float, U_bins: np.ndarray, V_weights: np.ndarray, settings: dict) -> np.ndarray:
    kBT_eV = K_BOLTZMANN * temp_impact_K / EV_TO_JOULE
    alpha = (settings['U_MAX'] - settings['U_MIN']) / 0.5
    
    # 上限クリップなし
    U_target = settings['U_MIN'] + alpha * E_in_eV

    # 二乗距離で計算 (滑らかなS字カーブ)
    gamma = 0.5
    energy_factor = np.exp(- ((U_bins - U_target)**2) / (kBT_eV * gamma))
    probabilities = V_weights * energy_factor

    P_total = np.sum(probabilities)
    
    # 計算限界を超えた場合は「一番近いビン」に100%割り振る
    if P_total > 1e-100:
        return probabilities / P_total
    else:
        # 目標値 U_target に最も近いビンのインデックスを取得
        closest_idx = np.argmin(np.abs(U_bins - U_target))
        fallback = np.zeros(len(U_bins))
        fallback[closest_idx] = 1.0
        return fallback

def assign_sticking_bin(E_in_eV: float, temp_impact_K: float, U_bins: np.ndarray, V_weights: np.ndarray, settings: dict) -> int:
    probabilities = calculate_probability_distribution(E_in_eV, temp_impact_K, U_bins, V_weights, settings)
    return np.random.choice(len(U_bins), p=probabilities)

# ==========================================
# 3. 検証とプロットの実行
# ==========================================
def run_verification():
    U_bins, V_weights = setup_binding_energy_bins(settings)
    T_surface = 400.0  
    N_particles = 10000 
    
    # モンテカルロ試行
    E_incident = np.random.uniform(0.0, 4.0, N_particles)
    U_assigned = np.zeros(N_particles)
    
    for i in range(N_particles):
        idx = assign_sticking_bin(E_incident[i], T_surface, U_bins, V_weights, settings)
        jitter = np.random.uniform(-0.02, 0.02)
        U_assigned[i] = U_bins[idx] + jitter

    # === ここからグラフ描画 (4画面) ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 【左上図】 状態密度 V(U)
    axes[0, 0].bar(U_bins, V_weights, width=(settings['U_MAX']-settings['U_MIN'])/settings['N_U_BINS']*0.8, color='lightblue', edgecolor='black')
    axes[0, 0].set_title("1. Surface State Density V(U)")
    axes[0, 0].set_xlabel("Binding Energy U [eV]")
    axes[0, 0].set_ylabel("Relative Density")
    
    # 【右上図】 入射エネルギー E vs 割り当てられた U (散布図)
    axes[0, 1].scatter(E_incident, U_assigned, alpha=0.05, s=10, color='blue')
    alpha_coeff = (settings['U_MAX'] - settings['U_MIN']) / 0.5
    E_line = np.linspace(0, 4, 100)
    U_target_line = settings['U_MIN'] + alpha_coeff * E_line
    axes[0, 1].plot(E_line, U_target_line, color='red', linestyle='--', linewidth=2, label='U_target (Ideal)')
    axes[0, 1].set_title(f"2. Assigned U vs Incident E (T={T_surface}K)")
    axes[0, 1].set_xlabel("Incident Energy E [eV]")
    axes[0, 1].set_ylabel("Assigned Binding Energy U [eV]")
    axes[0, 1].set_xlim(0, 0.75)
    axes[0, 1].set_ylim(1.3, 2.8)
    axes[0, 1].legend()

    # 【左下図】 特定のエネルギーにおける確率分布 P(U) の比較
    E_tests = [0.2, 0.5, 1.5]
    colors_line = ['green', 'orange', 'purple']
    for E_test, col in zip(E_tests, colors_line):
        P_dist = calculate_probability_distribution(E_test, T_surface, U_bins, V_weights, settings)
        axes[1, 0].plot(U_bins, P_dist, marker='o', label=f'E = {E_test} eV', color=col)
    axes[1, 0].set_title(f"3. Probability Distribution P(U) at T={T_surface}K")
    axes[1, 0].set_xlabel("Binding Energy U [eV]")
    axes[1, 0].set_ylabel("Probability P(U)")
    axes[1, 0].legend()

    # 🌟【右下図: 100% 積み上げ面グラフ (Stacked Area)】
    E_array = np.linspace(0.0, 4.0, 200)
    P_matrix = np.zeros((len(U_bins), len(E_array)))
    
    for i, E_val in enumerate(E_array):
        P_matrix[:, i] = calculate_probability_distribution(E_val, T_surface, U_bins, V_weights, settings)
    
    cmap = plt.get_cmap('viridis')
    colors_stack = [cmap(i / (len(U_bins)-1)) for i in range(len(U_bins))]
    labels = [f'U={u:.2f} eV' for u in U_bins]
    
    axes[1, 1].stackplot(E_array, P_matrix, labels=labels, colors=colors_stack, alpha=0.85)
        
    axes[1, 1].set_title(f"4. Stacked Probability of U bins vs Incident E (T={T_surface}K)")
    axes[1, 1].set_xlabel("Incident Energy E [eV]")
    axes[1, 1].set_ylabel("Probability P (Stacked)")
    axes[1, 1].set_xlim(0, 0.75)
    axes[1, 1].set_ylim(0, 1.0)
    
    axes[1, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_verification()