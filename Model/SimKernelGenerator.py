# -*- coding: utf-8 -*-
"""
simulation_kernel_gen_full_verbose.py
==============================================================================
逆解析用カーネル生成シミュレーション (全プロセス対応・進捗表示強化版)

概要:
    Leblanc (2003) の物理モデルに基づき、全表面から粒子を放出して
    輸送行列（カーネル）を生成します。
    ターミナルへの進捗表示を充実させ、実行状況をモニタリングしやすくしています。
==============================================================================
"""

import numpy as np
import sys
import os
from multiprocessing import Pool, cpu_count
import time
import datetime

# ==============================================================================
# 0. 物理定数・設定
# ==============================================================================
PHYSICAL_CONSTANTS = {
    'PI': np.pi,
    'AU': 1.496e11,
    'MASS_NA': 3.8175e-26,
    'K_BOLTZMANN': 1.380649e-23,
    'GM_MERCURY': 2.2032e13,
    'RM': 2.440e6,
    'C': 299792458.0,
    'EV_TO_JOULE': 1.602e-19,
    'H': 6.62607015e-34,
    'ME': 9.109e-31,
    'E_CHARGE': 1.602e-19,
    'EPSILON_0': 8.854e-12,
}

# グリッド設定
N_SRC_LON = 18  # 20度刻み
N_SRC_LAT = 9  # 20度刻み

OUTPUT_DIR = "./Inverse_Kernels_Full_2"

# シミュレーション設定
SIMULATION_SETTINGS = {
    'DT_MOVE': 500.0,
    'GRID_RADIUS_RM': 6.0,
    'BETA': 1.0,
    'T1AU': 54500.0,
    'USE_CORIOLIS_FORCES': True,
    'TEMP_BASE': 100.0,
    'TEMP_AMP': 600.0,
    'TEMP_NIGHT': 100.0,
}

# Leblancパラメータ
PARAMS_LEBLANC = {
    'Q_PSD': 1.0e-20,  # PSD断面積 [cm2] (論文値)
    'F_UV_1AU': 1.5e14,  # UV Flux [ph/cm2/s]
    'SWS_FLUX_1AU': 4.0e8,  # Solar Wind Flux [cm-2 s-1] (概算)
    'SWS_YIELD': 0.06,  # Sputtering Yield
    'MMV_FLUX_TOTAL': 3.5e23,  # MMV Total Rate [atoms/s] (近日点)
    'TEMP_PSD': 1500.0,
    'TEMP_MMV': 3000.0,
    'SWS_U_EV': 0.27,  # Binding Energy for Thompson
}

# ==============================================================================
# 1. ヘルパー関数 (物理計算)
# ==============================================================================
src_lon_edges = np.linspace(-np.pi, np.pi, N_SRC_LON + 1)
src_lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_SRC_LAT + 1)


def get_source_indices(pos_surf, sub_lon):
    lon_inertial = np.arctan2(pos_surf[1], pos_surf[0])
    lat = np.arcsin(np.clip(pos_surf[2] / np.linalg.norm(pos_surf), -1, 1))
    lon_fixed = (lon_inertial - sub_lon + np.pi) % (2 * np.pi) - np.pi
    ix = np.searchsorted(src_lon_edges, lon_fixed) - 1
    iy = np.searchsorted(src_lat_edges, lat) - 1
    return max(0, min(N_SRC_LON - 1, ix)), max(0, min(N_SRC_LAT - 1, iy))


def calculate_surface_temperature(lon_rad, lat_rad, AU, subsolar_lon):
    scaling = np.sqrt(0.306 / AU)
    cos_theta = np.cos(lat_rad) * np.cos(lon_rad - subsolar_lon)
    if cos_theta <= 0:
        return SIMULATION_SETTINGS['TEMP_NIGHT']
    return SIMULATION_SETTINGS['TEMP_BASE'] + SIMULATION_SETTINGS['TEMP_AMP'] * (cos_theta ** 0.25) * scaling


def calc_local_rates(lon_inertial, lat, AU, sub_lon):
    """
    ある地点における各プロセスの放出フラックス(相対値)を計算する
    """
    cos_sza = np.cos(lat) * np.cos(lon_inertial - sub_lon)
    illum_frac = 1.0 if cos_sza > 0 else 0.0
    eff_cos = max(0.0, cos_sza)

    T_surf = calculate_surface_temperature(lon_inertial, lat, AU, sub_lon)

    # 1. PSD
    rate_psd = (PARAMS_LEBLANC['F_UV_1AU'] / AU ** 2) * PARAMS_LEBLANC['Q_PSD'] * eff_cos

    # 2. TD
    rate_td = 0.0
    if T_surf > 100:
        U_J = 1.85 * PHYSICAL_CONSTANTS['EV_TO_JOULE']
        rate_td = 1e13 * np.exp(-U_J / (PHYSICAL_CONSTANTS['K_BOLTZMANN'] * T_surf))

    # 3. SWS
    rate_sws = (PARAMS_LEBLANC['SWS_FLUX_1AU'] / AU ** 2) * PARAMS_LEBLANC['SWS_YIELD'] * illum_frac

    # 4. MMV
    area_total = 4 * np.pi * (PHYSICAL_CONSTANTS['RM'] * 100) ** 2  # cm2
    mmv_total_at_au = PARAMS_LEBLANC['MMV_FLUX_TOTAL'] * (0.306 / AU) ** 1.9
    rate_mmv = mmv_total_at_au / area_total

    return rate_td, rate_psd, rate_sws, rate_mmv, T_surf


def sample_thompson_sigmund(U_eV):
    while True:
        E = np.random.uniform(0, 100.0)
        prob = E / (E + U_eV) ** 3
        if np.random.random() < prob * 10.0:
            return E


def lonlat_to_xyz(lon, lat, r):
    return np.array([r * np.cos(lat) * np.cos(lon), r * np.cos(lat) * np.sin(lon), r * np.sin(lat)])


def sample_lambertian_local():
    u1, u2 = np.random.random(2)
    phi = 2 * np.pi * u1
    st = np.sqrt(u2)
    ct = np.sqrt(1 - u2)
    return np.array([st * np.cos(phi), st * np.sin(phi), ct])


def transform_local_to_world(local, norm):
    z = norm
    if abs(z[2]) > 0.99:
        ref = np.array([1., 0., 0.])
    else:
        ref = np.array([0., 0., 1.])
    x = np.cross(ref, z);
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return local[0] * x + local[1] * y + local[2] * z


def get_orbital_params_linear(t, orbit_data, t0):
    times = orbit_data[:, 2]
    taa = np.interp(t, times, orbit_data[:, 0])
    au = np.interp(t, times, orbit_data[:, 1])
    vr = np.interp(t, times, orbit_data[:, 3])
    vt = np.interp(t, times, orbit_data[:, 4])
    sl = np.deg2rad(np.interp(t, times, orbit_data[:, 5]))
    return taa, au, vr, vt, sl


# ==============================================================================
# 2. 粒子移動 (Kernel Step)
# ==============================================================================
def simulate_particle_kernel_step(args):
    settings = args['settings']
    p = args['particle_state']
    TAA, AU, V_rad, V_tan, sl = args['orbit']
    dt = args['duration']

    pos = p['pos']
    vel = p['vel']
    RM = PHYSICAL_CONSTANTS['RM']

    r = np.linalg.norm(pos)
    ag = -PHYSICAL_CONSTANTS['GM_MERCURY'] * pos / r ** 3

    acc_srp = np.array([-2.0 / (AU ** 2), 0, 0]) if pos[0] > 0 else np.zeros(3)
    acc = ag + acc_srp

    pos_new = pos + vel * dt + 0.5 * acc * dt ** 2
    vel_new = vel + acc * dt

    r_new = np.linalg.norm(pos_new)

    if r_new <= RM: return {'status': 'stuck', 'state': None}
    if r_new > RM * settings['GRID_RADIUS_RM']: return {'status': 'escaped', 'state': None}

    tau = settings['T1AU'] * AU ** 2
    if pos[0] > 0 and np.random.random() < (1 - np.exp(-dt / tau)):
        return {'status': 'ionized', 'state': None}

    p['pos'] = pos_new
    p['vel'] = vel_new
    return {'status': 'alive', 'state': p}


# ==============================================================================
# 3. メイン処理
# ==============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("========================================================")
    print("   Inverse Kernel Generation Simulation (Full Process)")
    print(f"   Output Directory: {OUTPUT_DIR}")
    print("========================================================")

    # 軌道データ読み込み
    try:
        orbit_data = np.loadtxt('orbit2025_spice_unwrapped.txt')
        orbit_data[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(orbit_data[:, 0])))
        orbit_data[:, 5] = np.rad2deg(np.unwrap(np.deg2rad(orbit_data[:, 5])))
        print("[Info] Orbit data loaded successfully.")
    except Exception as e:
        print(f"[Error] Failed to load orbit data: {e}")
        return

    MERCURY_YEAR = 87.97 * 86400
    t_start = orbit_data[0, 2]
    t_end = t_start + MERCURY_YEAR * 1.5  # 1.5年分
    t_spinup = t_start + MERCURY_YEAR * 0.5

    t_curr = t_start
    dt_move = SIMULATION_SETTINGS['DT_MOVE']
    step = 0
    total_steps = int((t_end - t_start) / dt_move)

    active_particles = []
    PARTICLES_PER_STEP = 5000

    print(f"[Info] Total Simulation Steps: {total_steps}")
    print(f"[Info] Spin-up until step: {int((t_spinup - t_start) / dt_move)}")
    print("--------------------------------------------------------")

    start_time = time.time()
    spinup_completed = False

    while t_curr < t_end:
        step += 1
        TAA, AU, vr, vt, sub_lon = get_orbital_params_linear(t_curr, orbit_data, t_start)

        # --- A. 粒子生成 ---
        new_particles = []
        rnd_lon = np.random.uniform(-np.pi, np.pi, PARTICLES_PER_STEP)
        rnd_lat = np.arcsin(np.random.uniform(-1, 1, PARTICLES_PER_STEP))

        for k in range(PARTICLES_PER_STEP):
            lon_in = rnd_lon[k] + sub_lon - np.pi
            lat = rnd_lat[k]

            r_td, r_psd, r_sws, r_mmv, T_surf = calc_local_rates(lon_in, lat, AU, sub_lon)
            total_rate = r_td + r_psd + r_sws + r_mmv

            if total_rate <= 0: continue

            probs = np.array([r_td, r_psd, r_sws, r_mmv]) / total_rate
            proc_idx = np.random.choice([0, 1, 2, 3], p=probs)

            mass = PHYSICAL_CONSTANTS['MASS_NA']
            kb = PHYSICAL_CONSTANTS['K_BOLTZMANN']

            spd = 0.0
            if proc_idx == 0:  # TD
                spd = np.random.gamma(2.0, kb * T_surf)
                spd = np.sqrt(2 * spd / mass)
            elif proc_idx == 1:  # PSD
                spd = np.sqrt(2 * np.random.gamma(2.0, kb * PARAMS_LEBLANC['TEMP_PSD']) / mass)
            elif proc_idx == 2:  # SWS
                E_eV = sample_thompson_sigmund(PARAMS_LEBLANC['SWS_U_EV'])
                spd = np.sqrt(2 * E_eV * PHYSICAL_CONSTANTS['EV_TO_JOULE'] / mass)
            else:  # MMV
                spd = np.sqrt(2 * np.random.gamma(2.0, kb * PARAMS_LEBLANC['TEMP_MMV']) / mass)

            pos_surf = lonlat_to_xyz(lon_in, lat, PHYSICAL_CONSTANTS['RM'])
            norm = pos_surf / PHYSICAL_CONSTANTS['RM']
            vel = spd * transform_local_to_world(sample_lambertian_local(), norm)
            ix, iy = get_source_indices(pos_surf, sub_lon)

            new_particles.append({
                'pos': pos_surf, 'vel': vel,
                'src_ix': ix, 'src_iy': iy,
                'weight': total_rate
            })

        active_particles.extend(new_particles)

        # --- B. 移動計算 ---
        tasks = [{'settings': SIMULATION_SETTINGS, 'spec': None, 'orbit': (TAA, AU, vr, vt, sub_lon),
                  'particle_state': p, 'duration': dt_move} for p in active_particles]

        if len(tasks) > 0:
            with Pool(cpu_count() - 1) as pool:
                results = pool.map(simulate_particle_kernel_step, tasks)

            next_p = []
            for res in results:
                if res['status'] == 'alive':
                    next_p.append(res['state'])
            active_particles = next_p

        # --- C. カーネル保存 & スピンアップ判定 ---
        deg_taa = TAA % 360

        # スピンアップ完了通知
        if not spinup_completed and t_curr > t_spinup:
            spinup_completed = True
            print("\n" + "=" * 56)
            print(f" [Notification] Spin-up Period Completed at Step {step}.")
            print(" Starting Data Collection Phase...")
            print("=" * 56 + "\n")

        # カーネル保存
        if t_curr > t_spinup and (step % 50 == 0):
            kernel_mat = np.zeros((N_SRC_LON, N_SRC_LAT, 2), dtype=np.float32)
            for p in active_particles:
                if p['pos'][0] > 0:
                    reg = 0 if p['pos'][1] < 0 else 1
                    kernel_mat[p['src_ix'], p['src_iy'], reg] += p['weight']

            # ファイル名をTAAの整数値で保存
            fname = f"kernel_taa{int(deg_taa):03d}.npy"
            np.save(os.path.join(OUTPUT_DIR, fname), kernel_mat)
            print(f" [Save] Kernel saved: {fname} (Particles: {len(active_particles)})")

        # --- D. 進捗表示 (Enhanced Print) ---
        if step % 10 == 0 or step == 1:
            elapsed = time.time() - start_time
            progress = step / total_steps
            remaining = elapsed / progress * (1.0 - progress) if progress > 0 else 0

            # 時間フォーマット (HH:MM:SS)
            elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
            remaining_str = str(datetime.timedelta(seconds=int(remaining)))

            status_str = "SPIN-UP" if not spinup_completed else "RECORDING"

            # 1行で更新表示 (キャリッジリターン \r は環境によってはログが消えるので、今回は普通にprint)
            # ログが見やすいように適度にフォーマット
            print(f"Step {step:5d}/{total_steps} [{progress * 100:5.1f}%] | "
                  f"Mode: {status_str:9s} | TAA: {deg_taa:5.1f} | "
                  f"Particles: {len(active_particles):6d} | "
                  f"Elapsed: {elapsed_str} (Left: {remaining_str})")

        t_curr += dt_move

    print("\n========================================================")
    print("   Simulation Completed Successfully.")
    print("========================================================")


if __name__ == '__main__':
    main()