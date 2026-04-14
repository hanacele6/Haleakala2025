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
    'V_DIST_TYPE': 'gaussian',  # 'gaussian' または 'uniform'
    'MAX_BARRIER': 2.0          # 最も深いサイトに入るための最大ハードル [eV]
}

# ==========================================
# 2. コア関数群 
# ==========================================
def setup_binding_energy_bins(settings: dict) -> tuple:
    U_bins = np.linspace(settings['U_MIN'], settings['U_MAX'], settings['N_U_BINS'])
    
    if settings.get('V_DIST_TYPE') == 'uniform':
        V_weights = np.ones(len(U_bins))
    else:
        V_weights = np.exp(-0.5 * ((U_bins - 1.85) / 0.35)**2)
        
    V_weights /= np.max(V_weights)
    return U_bins, V_weights

def calculate_E_req(U_bins: np.ndarray, settings: dict) -> np.ndarray:
    """
    【新設】サイトの深さ (U) に応じて、必要な入射エネルギーのハードル (E_req) を計算する関数
    """
    # 現在はシンプルな線形補間モデル (U_MIN なら 0 eV, U_MAX なら MAX_BARRIER eV)
    E_req = (U_bins - settings['U_MIN']) / (settings['U_MAX'] - settings['U_MIN']) * settings['MAX_BARRIER']
    return E_req

def calculate_probability_distribution(E_in_eV: float, temp_impact_K: float, U_bins: np.ndarray, V_weights: np.ndarray, settings: dict) -> np.ndarray:
    # 1. 独立させた関数から要求バリア (E_req) を取得
    E_req = calculate_E_req(U_bins, settings)
    
    # 2. 到達確率 (ゴンペルツ関数)
    W_spread = 0.4
    activation_factor = np.exp(-np.exp(-(E_in_eV - E_req) / W_spread))
    
    # 3. 状態密度 V(U) と 掛け合わせる
    probabilities = V_weights * activation_factor
    P_total = np.sum(probabilities)
    
    if P_total > 1e-10:
        return probabilities / P_total
    else:
        fallback = np.zeros(len(U_bins))
        fallback[0] = 1.0
        return fallback

def assign_sticking_bin(E_in_eV: float, temp_impact_K: float, U_bins: np.ndarray, V_weights: np.ndarray, settings: dict) -> int:
    probabilities = calculate_probability_distribution(E_in_eV, temp_impact_K, U_bins, V_weights, settings)
    return np.random.choice(len(U_bins), p=probabilities)

# ==========================================
# 3. 検証とプロットの実行
# ==========================================
def run_verification():
    U_bins, V_weights = setup_binding_energy_bins(settings)
    E_req_bins = calculate_E_req(U_bins, settings) # ハードルの一覧を取得
    
    # 🌟【追加】コンソールに現在のハードル設定を出力
    print("=" * 45)
    print(f" 現在の設定 (V_DIST_TYPE: {settings['V_DIST_TYPE']})")
    print("-" * 45)
    print(" サイト深さ (U)  |  状態密度 (V)  |  要求ハードル (E_req)")
    print("-" * 45)
    for u, v, req in zip(U_bins, V_weights, E_req_bins):
        print(f" {u:5.2f} eV       |      {v:4.2f}      |      {req:4.2f} eV")
    print("=" * 45)

    T_surface = 400.0  
    N_particles = 10000 
    
    E_incident = np.random.uniform(0.0, 4.0, N_particles)
    U_assigned = np.zeros(N_particles)
    
    for i in range(N_particles):
        idx = assign_sticking_bin(E_incident[i], T_surface, U_bins, V_weights, settings)
        jitter = np.random.uniform(-0.02, 0.02)
        U_assigned[i] = U_bins[idx] + jitter

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 【左上図】 状態密度 ＋ 🌟要求ハードルの重ね合わせ
    ax_V = axes[0, 0]
    ax_V.bar(U_bins, V_weights, width=(settings['U_MAX']-settings['U_MIN'])/settings['N_U_BINS']*0.8, color='lightblue', edgecolor='black', label='State Density V(U)')
    ax_V.set_title(f"1. Surface State Density & Required Barrier")
    ax_V.set_xlabel("Binding Energy U [eV]")
    ax_V.set_ylabel("Relative Density V(U)")
    ax_V.set_ylim(0, 1.1)
    
    # 右側のY軸（第2軸）を作成してハードルを描画
    ax_req = ax_V.twinx()
    ax_req.plot(U_bins, E_req_bins, color='red', marker='x', linestyle='-', linewidth=2, markersize=8, label='Req. Barrier (E_req)')
    ax_req.set_ylabel("Required Barrier E_req [eV]", color='red')
    ax_req.tick_params(axis='y', labelcolor='red')
    ax_req.set_ylim(-0.2, settings['MAX_BARRIER'] + 0.5)
    
    # 凡例をまとめる
    lines_1, labels_1 = ax_V.get_legend_handles_labels()
    lines_2, labels_2 = ax_req.get_legend_handles_labels()
    ax_V.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    # 【右上図】 散布図と期待値
    axes[0, 1].scatter(E_incident, U_assigned, alpha=0.05, s=10, color='blue')
    E_line = np.linspace(0, 4, 100)
    U_mean_line = np.zeros_like(E_line)
    for idx, e_val in enumerate(E_line):
        p_dist = calculate_probability_distribution(e_val, T_surface, U_bins, V_weights, settings)
        U_mean_line[idx] = np.sum(U_bins * p_dist)
        
    axes[0, 1].plot(E_line, U_mean_line, color='red', linestyle='--', linewidth=2, label='Expected U (Mean)')
    axes[0, 1].set_title(f"2. Assigned U vs Incident E (T={T_surface}K)")
    axes[0, 1].set_xlabel("Incident Energy E [eV]")
    axes[0, 1].set_ylabel("Assigned Binding Energy U [eV]")
    axes[0, 1].set_xlim(0, 3.0)
    axes[0, 1].set_ylim(1.3, 2.8)
    axes[0, 1].legend()

    # 【左下図】 確率分布 P(U) の比較
    E_tests = [0.2, 0.8, 2.0]
    colors_line = ['green', 'orange', 'purple']
    for E_test, col in zip(E_tests, colors_line):
        P_dist = calculate_probability_distribution(E_test, T_surface, U_bins, V_weights, settings)
        axes[1, 0].plot(U_bins, P_dist, marker='o', label=f'E = {E_test} eV', color=col)
    axes[1, 0].set_title(f"3. Probability Distribution P(U) at T={T_surface}K")
    axes[1, 0].set_xlabel("Binding Energy U [eV]")
    axes[1, 0].set_ylabel("Probability P(U)")
    axes[1, 0].legend()

    # 【右下図: 100% 積み上げ面グラフ】
    E_array = np.linspace(0.0, 4.0, 200)
    P_matrix = np.zeros((len(U_bins), len(E_array)))
    
    for i, E_val in enumerate(E_array):
        P_matrix[:, i] = calculate_probability_distribution(E_val, T_surface, U_bins, V_weights, settings)
    
    cmap = plt.get_cmap('viridis')
    colors_stack = [cmap(i / (len(U_bins)-1)) for i in range(len(U_bins))]
    labels = [f'U={u:.2f} eV' for u in U_bins]
    
    axes[1, 1].stackplot(E_array, P_matrix, labels=labels, colors=colors_stack, alpha=0.85)
    axes[1, 1].set_title(f"4. Stacked Probability vs Incident E (T={T_surface}K)")
    axes[1, 1].set_xlabel("Incident Energy E [eV]")
    axes[1, 1].set_ylabel("Probability P (Stacked)")
    axes[1, 1].set_xlim(0, 3.0)
    axes[1, 1].set_ylim(0, 1.0)
    axes[1, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_verification()