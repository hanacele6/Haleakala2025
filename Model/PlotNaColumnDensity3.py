import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import pandas as pd
import matplotlib.font_manager as fm

# --- フォント設定 ---
try:
    font_path = 'C:/Windows/Fonts/meiryo.ttc'
    jp_font = fm.FontProperties(fname=font_path)
    # plt.rcParams['font.family'] = jp_font.get_name()
except FileNotFoundError:
    pass

# ==========================================
# 1. 物理定数・パラメータ設定
# ==========================================

RM_m = 2.440e6
CM_PER_M = 100.0
NORMALIZATION_AREA_CM2 = 3.7408e17
NORMALIZATION_AREA_HALF_CM2 = NORMALIZATION_AREA_CM2 / 2.0
NORMALIZATION_AREA_QUARTER_CM2 = NORMALIZATION_AREA_CM2 / 4.0

# --- 物理モデル用 ---
G = 6.67384e-5
MS = 1.9884e30
MM = 3.3e23
AU = 1.495978707e13
RM_cm = 2440e5
E = 0.2056
A_AU = 0.3871
L_AU = 0.37078

# モデルパラメータ
TAU0 = 169200.0
PHI0 = 4.6e7
D_PARAM = 4.6e7
ROT_ANGLE_ACCUM = 90

# ==========================================
# 2. ユーザー設定
# ==========================================

# ★★★ グリッド設定
GRID_RESOLUTION = 101
GRID_MAX_RM = 5.0
output_dir = r"./SimulationResult_202512/ParabolicHop_72x36_NoEq_DT100_0116_0.4Denabled_2.7_LowestQ"

# ★★★ プロットモード
PLOT_MODE = "DUSK"  # "ALL"にするとSimのDawn/Dusk両方が見れます
COMMON_Y_LABEL = "Column Density [atoms/cm²]"
SHOW_LEGEND = True

CSV_PLOT_SELECTION = "DUSK"  #DAWN, DUSK, BOTH

# ★★★ 比較用CSVファイル
SHOW_CSV_OVERLAY = True
CSV_USE_SHARED_Y_AXIS = True
CSV_SETTINGS = [
    {"path": r"./dawn.csv", "label": "Obs: Dawn", "color": "green", "marker": "x", "type": "DAWN"},
    {"path": r"./dusk.csv", "label": "Obs: Dusk", "color": "magenta", "marker": "+", "type": "DUSK"}
]

# ★★★ 物理モデル(Analytical)の設定
SHOW_ANALYTICAL_MODEL_DAWN = False
LABEL_ANALYTICAL_DAWN = "Analytical Model: Dawn"
COLOR_ANALYTICAL_DAWN = "red"

SHOW_ANALYTICAL_MODEL_DUSK = True
LABEL_ANALYTICAL_DUSK = "Analytical Model: Dusk"
COLOR_ANALYTICAL_DUSK = "red"


# ==========================================
# 3. 物理モデル計算関数群
# ==========================================

def get_sun_distance(taa_deg):
    return L_AU / (1.0 + E * np.cos(np.deg2rad(taa_deg)))


def get_orbital_angular_velocity(r_au):
    return np.sqrt((G / AU ** 3) * MS * L_AU) / r_au ** 2 * np.rad2deg(1)


def get_relative_angular_velocity(taa_deg):
    t_orbit = 2.0 * np.pi * np.sqrt(A_AU ** 3 / ((G / AU ** 3) * (MS + MM)))
    t_rot = t_orbit * (2.0 / 3.0)
    rot_dot = 360.0 / t_rot
    r = get_sun_distance(taa_deg)
    taa_dot = get_orbital_angular_velocity(r)
    return rot_dot - taa_dot


def calculate_sun_rotation(taa_range):
    rot = 0.0
    step = 10.0
    rot_history = []
    for taa in taa_range:
        for i in range(int(step)):
            taa_current = taa + i / step
            r = get_sun_distance(taa_current)
            taa_dot = get_orbital_angular_velocity(r)
            omega = get_relative_angular_velocity(taa_current)
            drot = (omega / taa_dot) / step
            rot += drot
        rot_history.append(rot)
    return np.array(rot_history)


def calculate_analytical_models():
    """DawnとDusk両方の物理モデルを計算して返す"""

    # SRPデータの取得
    try:
        data = np.loadtxt('TAA_SRP.txt')
        taa_obs = data[:, 0]
        srp_obs = data[:, 1]
    except OSError:
        taa_obs = np.arange(360)
        srp_obs = 1.0 + 0.5 * np.sin(np.deg2rad(taa_obs) * 2)

    taa_range = np.arange(360)
    total_rotation = calculate_sun_rotation(taa_range)
    omega_values = get_relative_angular_velocity(taa_range)

    # 蓄積計算 (Dawn用)
    s_accumulated = np.zeros(360)
    for taa in taa_range:
        current_rot = total_rotation[taa]
        for i in range(ROT_ANGLE_ACCUM):
            rot_to_check = current_rot - ROT_ANGLE_ACCUM + i
            if rot_to_check < 0: rot_to_check += 180.0

            srp_at_rot = np.interp(rot_to_check, total_rotation, srp_obs)
            omega_at_rot = np.interp(rot_to_check, total_rotation, omega_values)
            s_rate = np.sqrt(srp_at_rot) / (np.abs(omega_at_rot) + 1e-9)
            s_accumulated[taa] += s_rate

    # モデル計算
    dawn_model = np.zeros(360)
    dusk_model = np.zeros(360)

    for taa in taa_range:
        r_au = get_sun_distance(taa)

        # 基本パラメータ
        ph = PHI0 * (0.306 / r_au) ** 2
        tau_val = TAU0 * r_au ** 2

        srp_val = srp_obs[int(taa) % len(srp_obs)]
        tm = np.sqrt(2.0 * RM_cm / srp_val)

        # === 修正箇所: Duskモデルの計算式 ===
        # Duskも移動時間(tm)の影響を受けるため、Dawnの第1項と同じ形にする
        # これが「蓄積スパイクがない状態」のベースライン密度
        n_dusk_term = ph * tau_val * (1.0 - np.exp(-tm / tau_val))

        # Dawnモデル (Base + Accumulation)
        n_add_term = D_PARAM * s_accumulated[taa] * omega_values[taa]
        dawn_model[taa] = n_dusk_term + n_add_term

        # Duskモデル (Base only)
        dusk_model[taa] = n_dusk_term

    return taa_range, dawn_model, dusk_model


# ==========================================
# 4. グリッドデータ処理
# ==========================================
grid_total_width_m = 2 * GRID_MAX_RM * RM_m
cell_size_m = grid_total_width_m / GRID_RESOLUTION
cell_volume_m3 = cell_size_m ** 3
mid_index_x = (GRID_RESOLUTION - 1) // 2
mid_index_y = (GRID_RESOLUTION - 1) // 2
idx_dawn_outer_limit = mid_index_y // 2
idx_dusk_outer_start = (GRID_RESOLUTION - 1) - idx_dawn_outer_limit


def process_simulation_data(target_dir, mode):
    try:
        all_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.npy') and f.startswith('density_grid_')])
        if not all_files: return None, None
    except FileNotFoundError:
        return None, None

    sim_results_taa = []
    results_dict = {"DAWN": [], "DUSK": []}
    single_result_density = []

    for filename in tqdm(all_files, desc="Processing Grid"):
        try:
            taa = int(filename.split('_taa')[-1].split('.')[0])
        except:
            continue

        filepath = os.path.join(target_dir, filename)
        density_grid_m3 = np.load(filepath)
        dayside_grid = density_grid_m3[mid_index_x:, :, :]
        atoms_grid = dayside_grid * cell_volume_m3
        atoms_grid[0, :, :] *= 0.5

        sum_mid = np.sum(atoms_grid[:, mid_index_y, :])

        if mode == "ALL":
            sum_dawn = np.sum(atoms_grid[:, :mid_index_y, :])
            results_dict["DAWN"].append((sum_dawn + 0.5 * sum_mid) / NORMALIZATION_AREA_HALF_CM2)
            sum_dusk = np.sum(atoms_grid[:, mid_index_y + 1:, :])
            results_dict["DUSK"].append((sum_dusk + 0.5 * sum_mid) / NORMALIZATION_AREA_HALF_CM2)
        else:
            total = 0
            area = 1
            if mode == "DAWN":
                total = np.sum(atoms_grid[:, :mid_index_y, :]) + 0.5 * sum_mid
                area = NORMALIZATION_AREA_HALF_CM2
            elif mode == "DUSK":
                total = np.sum(atoms_grid[:, mid_index_y + 1:, :]) + 0.5 * sum_mid
                area = NORMALIZATION_AREA_HALF_CM2
            elif mode == "DAYSIDE_TOTAL":
                total = np.sum(atoms_grid)
                area = NORMALIZATION_AREA_CM2
            single_result_density.append(total / area)

        sim_results_taa.append(taa)

    sim_results_taa = np.array(sim_results_taa)
    sorted_idx = np.argsort(sim_results_taa)
    sim_results_taa = sim_results_taa[sorted_idx]

    if mode == "ALL":
        for k in results_dict: results_dict[k] = np.array(results_dict[k])[sorted_idx]
        return sim_results_taa, results_dict
    else:
        return sim_results_taa, np.array(single_result_density)[sorted_idx]


# ==========================================
# 5. メイン処理とプロット
# ==========================================

sim_taa, sim_data = process_simulation_data(output_dir, PLOT_MODE)
model_taa, model_dawn, model_dusk = calculate_analytical_models()

if sim_taa is not None:
    fig, ax1 = plt.subplots(figsize=(10, 7))
    y1_max_data = 0

    # 1. Simulation Plot
    if PLOT_MODE == "ALL":
        for key, val_array in sim_data.items():
            c = "blue" if key == "DAWN" else "darkblue"
            m = "^" if key == "DAWN" else "v"
            ax1.plot(sim_taa, val_array, color=c, label=f"Sim: {key}", marker=m, linestyle='', alpha=0.6)
            if len(val_array) > 0: y1_max_data = max(y1_max_data, np.max(val_array))
    else:
        ax1.scatter(sim_taa, sim_data, label=f'Sim: {PLOT_MODE}', color='blue', alpha=0.6, s=50)
        if len(sim_data) > 0: y1_max_data = np.max(sim_data)

    # 2. Analytical Model Plot
    if SHOW_ANALYTICAL_MODEL_DAWN:
        ax1.plot(model_taa, model_dawn, color=COLOR_ANALYTICAL_DAWN, label=LABEL_ANALYTICAL_DAWN, linewidth=2.5,
                 alpha=0.9)
        y1_max_data = max(y1_max_data, np.max(model_dawn))

    if SHOW_ANALYTICAL_MODEL_DUSK:
        ax1.plot(model_taa, model_dusk, color=COLOR_ANALYTICAL_DUSK, label=LABEL_ANALYTICAL_DUSK, linewidth=2.5,
                  alpha=0.9)
        # Duskは通常低いのでMax更新には寄与しないことが多いが念のため
        y1_max_data = max(y1_max_data, np.max(model_dusk))

    ax1.set_xlabel('True Anomaly Angle (deg)', fontsize=18)
    ax1.set_ylabel(COMMON_Y_LABEL, fontsize=18)
    ax1.tick_params(labelsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlim(0, 360)
    ax1.set_xticks(np.arange(0, 361, 60))

    # 3. CSV Observation
    target_ax = ax1
    has_csv = False
    if SHOW_CSV_OVERLAY and not CSV_USE_SHARED_Y_AXIS:
        target_ax = ax1.twinx()
        target_ax.set_ylabel(COMMON_Y_LABEL + " (Obs)", fontsize=18)
        target_ax.grid(False)

    if SHOW_CSV_OVERLAY:
        for setting in CSV_SETTINGS:
            if CSV_PLOT_SELECTION != "BOTH" and setting["type"] != CSV_PLOT_SELECTION: continue

            if os.path.exists(setting["path"]):
                try:
                    df = pd.read_csv(setting["path"], encoding='shift_jis')
                    if df.shape[1] >= 4:
                        target_ax.scatter(df.iloc[:, 3], df.iloc[:, 4], label=setting["label"],
                                          color=setting["color"], marker=setting["marker"], s=80, zorder=3,
                                          linewidths=1.5)
                        has_csv = True
                except:
                    pass

    # 4. Axis Scaling
    if not CSV_USE_SHARED_Y_AXIS and has_csv:
        def get_align_val(t, v):
            if len(t) == 0: return 0, 1
            diff = np.abs(np.mod(t, 360))
            idx = np.argmin(np.minimum(diff, 360 - diff))
            return v[idx], np.max(v)


        # Sim or Model基準
        sim_p, sim_m = 0, y1_max_data
        if len(sim_taa) > 0:
            if isinstance(sim_data, dict):
                sim_p, _ = get_align_val(sim_taa, sim_data["DAWN"])
            else:
                sim_p, _ = get_align_val(sim_taa, sim_data)
        elif SHOW_ANALYTICAL_MODEL_DAWN:
            sim_p, _ = get_align_val(model_taa, model_dawn)

        # Obs基準
        obs_vals, obs_taas = [], []
        for c in target_ax.collections:
            o = c.get_offsets()
            if len(o) > 0:
                obs_taas.append(o[:, 0])
                obs_vals.append(o[:, 1])
        if obs_vals:
            obs_p, obs_m = get_align_val(np.concatenate(obs_taas), np.concatenate(obs_vals))
        else:
            obs_p, obs_m = 0, 1

        ratio = obs_p / sim_p if sim_p > 1e-9 else 1.0
        print(f"Align Ratio: {ratio:.4f}")

        final_sim_top = max(sim_m, obs_m / ratio) * 1.1
        ax1.set_ylim(0, final_sim_top)
        target_ax.set_ylim(0, final_sim_top * ratio)

    if SHOW_LEGEND:
        l1, lb1 = ax1.get_legend_handles_labels()
        l2, lb2 = target_ax.get_legend_handles_labels() if target_ax != ax1 else ([], [])
        ax1.legend(l1 + l2, lb1 + lb2, loc='upper left', fontsize=10, ncol=2)

    plt.tight_layout()
    plt.show()
else:
    print("No data processed.")