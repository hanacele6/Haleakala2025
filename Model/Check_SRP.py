# -*- coding: utf-8 -*-
"""
SRP Doppler Shift Test Code

This script extracts the physics calculations from the original 3D simulation
to plot the behavior of radiation pressure on a single particle
at specific TAAs (60 deg and 300 deg).

Required files:
- orbit2025_v5.txt
- SolarSpectrum_Na0.txt
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# ==============================================================================
# Physical Constants (Copied from original code)
# ==============================================================================
PHYSICAL_CONSTANTS = {
    'PI': np.pi,
    'AU': 1.496e11,
    'MASS_NA': 22.98976928 * 1.66054e-27,
    'K_BOLTZMANN': 1.380649e-23,
    'GM_MERCURY': 2.2032e13,
    'RM': 2.440e6,
    'C': 299792458.0,
    'H': 6.62607015e-34,
    'E_CHARGE': 1.602176634e-19,
    'ME': 9.1093897e-31,
    'EPSILON_0': 8.854187817e-12,
    'G': 6.6743e-11,
    'MASS_SUN': 1.989e30,
}


# ==============================================================================
# Physics Model (Copied from original code)
# ==============================================================================

def _calculate_acceleration(pos, vel, V_radial_ms, V_tangential_ms, AU, spec_data, settings):
    """
    【シミュレーションの核】(元のコードからコピー)
    粒子にかかる総加速度（重力＋放射圧＋見かけの力）を計算します。

    ★★★ 変更点: 放射圧(SRP)以外の力をすべて無効化 ★★★
    """
    x, y, z = pos
    r0 = AU * PHYSICAL_CONSTANTS['AU']

    # --- 1. 太陽放射圧 (Solar Radiation Pressure, SRP) ---
    # (この部分は変更しない)
    velocity_for_doppler = vel[0] + V_radial_ms
    w_na_d2 = 589.1582e-9 * (1.0 - velocity_for_doppler / PHYSICAL_CONSTANTS['C'])
    w_na_d1 = 589.7558e-9 * (1.0 - velocity_for_doppler / PHYSICAL_CONSTANTS['C'])

    b = 0.0
    wl, gamma, sigma0_perdnu2, sigma0_perdnu1, JL = spec_data.values()

    if (wl[0] * 1e-9 <= w_na_d2 < wl[-1] * 1e-9) and \
            (wl[0] * 1e-9 <= w_na_d1 < wl[-1] * 1e-9):
        gamma2 = np.interp(w_na_d2 * 1e9, wl, gamma)
        gamma1 = np.interp(w_na_d1 * 1e9, wl, gamma)
        F_lambda_1AU_m = JL * 1e4 * 1e9
        F_lambda_d1_at_Mercury = F_lambda_1AU_m / (AU ** 2) * gamma1
        F_nu_d1 = F_lambda_d1_at_Mercury * (w_na_d1 ** 2) / PHYSICAL_CONSTANTS['C']
        J1 = sigma0_perdnu1 * F_nu_d1
        F_lambda_d2_at_Mercury = F_lambda_1AU_m / (AU ** 2) * gamma2
        F_nu_d2 = F_lambda_d2_at_Mercury * (w_na_d2 ** 2) / PHYSICAL_CONSTANTS['C']
        J2 = sigma0_perdnu2 * F_nu_d2
        b = 1 / PHYSICAL_CONSTANTS['MASS_NA'] * (
                (PHYSICAL_CONSTANTS['H'] / w_na_d1) * J1 + (PHYSICAL_CONSTANTS['H'] / w_na_d2) * J2)

    # 影の計算は単純化のためコメントアウトしてもよい
    # if x < 0 and np.sqrt(y ** 2 + z ** 2) < PHYSICAL_CONSTANTS['RM']:
    #     b = 0.0
    accel_srp = np.array([-b, 0.0, 0.0])

    # --- 2. 水星の重力 (無効化) ---
    # r_sq = np.sum(pos ** 2)
    # accel_g = -PHYSICAL_CONSTANTS['GM_MERCURY'] * pos / (r_sq ** 1.5) if r_sq > 0 else np.array([0., 0., 0.])
    accel_g = np.array([0.0, 0.0, 0.0])

    # --- 3. 太陽の重力 (無効化) ---
    accel_sun = np.array([0.0, 0.0, 0.0])
    # if settings.get('USE_SOLAR_GRAVITY', False):
    #     ...

    # --- 4. 見かけの力 (コリオリ力・遠心力) (無効化) ---
    accel_coriolis = np.array([0.0, 0.0, 0.0])
    accel_centrifugal = np.array([0.0, 0.0, 0.0])
    # if settings.get('USE_CORIOLIS_FORCES', False):
    #     ...

    # --- 総加速度 ---
    # 放射圧のみを返す
    return accel_srp + accel_g + accel_sun + accel_centrifugal + accel_coriolis

def calculate_srp_acceleration(vel, V_radial_ms, AU, spec_data, settings):
    """
    Calculates and returns only the SRP acceleration vector.
    (For plotting purposes)
    """
    x = 0.0  # Simplified shadow check (assume sunlit)

    velocity_for_doppler = vel[0] + V_radial_ms
    w_na_d2 = 589.1582e-9 * (1.0 - velocity_for_doppler / PHYSICAL_CONSTANTS['C'])
    w_na_d1 = 589.7558e-9 * (1.0 - velocity_for_doppler / PHYSICAL_CONSTANTS['C'])

    b = 0.0
    wl, gamma, sigma0_perdnu2, sigma0_perdnu1, JL = spec_data.values()

    if (wl[0] * 1e-9 <= w_na_d2 < wl[-1] * 1e-9) and \
            (wl[0] * 1e-9 <= w_na_d1 < wl[-1] * 1e-9):
        gamma2 = np.interp(w_na_d2 * 1e9, wl, gamma)
        gamma1 = np.interp(w_na_d1 * 1e9, wl, gamma)
        F_lambda_1AU_m = JL * 1e4 * 1e9
        F_lambda_d1_at_Mercury = F_lambda_1AU_m / (AU ** 2) * gamma1
        F_nu_d1 = F_lambda_d1_at_Mercury * (w_na_d1 ** 2) / PHYSICAL_CONSTANTS['C']
        J1 = sigma0_perdnu1 * F_nu_d1
        F_lambda_d2_at_Mercury = F_lambda_1AU_m / (AU ** 2) * gamma2
        F_nu_d2 = F_lambda_d2_at_Mercury * (w_na_d2 ** 2) / PHYSICAL_CONSTANTS['C']
        J2 = sigma0_perdnu2 * F_nu_d2
        b = 1 / PHYSICAL_CONSTANTS['MASS_NA'] * (
                (PHYSICAL_CONSTANTS['H'] / w_na_d1) * J1 + (PHYSICAL_CONSTANTS['H'] / w_na_d2) * J2)

    # SRP acts in the -X direction
    return np.array([-b, 0.0, 0.0])


# ==============================================================================
# Helper Functions (New)
# ==============================================================================

def get_params_by_taa(taa_target, orbit_data):
    """
    Gets the orbital parameters from the closest TAA in the orbit data.
    """
    taa_col = orbit_data[:, 0]
    idx = np.argmin(np.abs(taa_col - taa_target))

    taa = orbit_data[idx, 0]
    au = orbit_data[idx, 1]
    v_radial = orbit_data[idx, 3]
    v_tangential = orbit_data[idx, 4]

    return taa, au, v_radial, v_tangential


def run_particle_test(taa_target, orbit_data, spec_data, settings, pos0, vel0, duration_sec, dt):
    """
    Simulation function to trace a single particle under fixed TAA conditions.
    """
    # 1. Get orbital parameters for the target TAA
    taa, au, v_rad, v_tan = get_params_by_taa(taa_target, orbit_data)

    v_rad = -v_rad

    print(f"--- Running test for TAA {taa_target} deg (Actual: {taa:.1f} deg) ---")
    print(f"    AU = {au:.3f}")
    print(f"    V_radial = {v_rad / 1000:.2f} km/s (Sunward is positive)")
    print(f"    V_tangential = {v_tan / 1000:.2f} km/s")

    # 2. Arrays to store results
    n_steps = int(duration_sec / dt)
    time_array = np.zeros(n_steps)
    srp_accel_x = np.zeros(n_steps)  # SRP X-component [m/s^2]
    velocity_x = np.zeros(n_steps)  # Particle X-velocity [m/s]
    doppler_vel = np.zeros(n_steps)  # Velocity for Doppler shift [m/s]

    # 3. Initial state
    pos = pos0.copy()
    vel = vel0.copy()

    # 4. Time integration loop (RK4)
    for i in range(n_steps):
        # 4a. Record current state
        time_array[i] = i * dt
        srp_vec = calculate_srp_acceleration(vel, v_rad, au, spec_data, settings)
        srp_accel_x[i] = srp_vec[0]  # Negative for -X direction
        velocity_x[i] = vel[0]
        doppler_vel[i] = vel[0] + v_rad

        # 4b. RK4 integration (Copied from original code)
        # k1
        k1_vel = dt * _calculate_acceleration(pos, vel, v_rad, v_tan, au, spec_data, settings)
        k1_pos = dt * vel
        # k2
        k2_vel = dt * _calculate_acceleration(pos + 0.5 * k1_pos, vel + 0.5 * k1_vel, v_rad, v_tan, au, spec_data,
                                              settings)
        k2_pos = dt * (vel + 0.5 * k1_vel)
        # k3
        k3_vel = dt * _calculate_acceleration(pos + 0.5 * k2_pos, vel + 0.5 * k2_vel, v_rad, v_tan, au, spec_data,
                                              settings)
        k3_pos = dt * (vel + 0.5 * k2_vel)
        # k4
        k4_vel = dt * _calculate_acceleration(pos + k3_pos, vel + k3_vel, v_rad, v_tan, au, spec_data, settings)
        k4_pos = dt * (vel + k3_vel)

        # Update position and velocity
        pos += (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos) / 6.0
        vel += (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel) / 6.0

    print(f"    -> Final X-Velocity: {vel[0] / 1000:.2f} km/s")
    return time_array, srp_accel_x, velocity_x, doppler_vel


def plot_results(results_60, results_300):
    """
    Plots the simulation results in separate windows.
    """
    time_60, srp_60, vel_60, dop_60 = results_60
    time_300, srp_300, vel_300, dop_300 = results_300

    # Convert time to hours
    time_h_60 = time_60 / 3600.0
    time_h_300 = time_300 / 3600.0

    # Convert velocity to km/s
    vel_km_60 = np.array(vel_60) / 1000.0
    vel_km_300 = np.array(vel_300) / 1000.0
    dop_km_60 = np.array(dop_60) / 1000.0
    dop_km_300 = np.array(dop_300) / 1000.0

    # --- 1. SRP Acceleration Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(time_h_60, srp_60, 'b-', label='TAA 60 deg (Receding)')
    plt.plot(time_h_300, srp_300, 'r-', label='TAA 300 deg (Approaching)')
    plt.ylabel("SRP Acceleration [m/s^2]")
    plt.xlabel("Simulation Time [Hours]")
    plt.legend()
    plt.grid(True)
    plt.title("1. SRP Acceleration vs. Time")
    plt.tight_layout()
    plt.savefig("srp_doppler_test_en_1_accel.png")
    print("\nGraph 1 saved as 'srp_doppler_test_en_1_accel.png'.")
    plt.show()

    # --- 2. Particle X-Velocity Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(time_h_60, vel_km_60, 'b-', label='TAA 60 deg')
    plt.plot(time_h_300, vel_km_300, 'r-', label='TAA 300 deg')
    plt.ylabel("Particle X-Velocity [km/s]\n(Anti-Sunward is Negative)")
    plt.xlabel("Simulation Time [Hours]")
    plt.legend()
    plt.grid(True)
    plt.title("2. Particle Velocity vs. Time (Mercury Frame)")
    plt.tight_layout()
    plt.savefig("srp_doppler_test_en_2_velocity.png")
    print("Graph 2 saved as 'srp_doppler_test_en_2_velocity.png'.")
    plt.show()

    # --- 3. Doppler Velocity Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(time_h_60, dop_km_60, 'b-', label='TAA 60 deg (V_particle + V_orbit)')
    plt.plot(time_h_300, dop_km_300, 'r-', label='TAA 300 deg (V_particle + V_orbit)')
    plt.xlabel("Simulation Time [Hours]")
    plt.ylabel("Doppler Velocity [km/s]\n(Sunward is Positive)")
    plt.legend()
    plt.grid(True)
    plt.title("3. Total Velocity for Doppler Shift")
    plt.tight_layout()
    plt.savefig("srp_doppler_test_en_3_doppler.png")
    print("Graph 3 saved as 'srp_doppler_test_en_3_doppler.png'.")
    plt.show()

# ==============================================================================
# Main execution
# ==============================================================================

def main_test():
    """
    Main function to run the test simulation.
    """
    print("Starting SRP test simulation.")

    # --- 1. Load external files ---
    try:
        spec_data_np = np.loadtxt('SolarSpectrum_Na0.txt', usecols=(0, 3))
        orbit_data = np.loadtxt('orbit2025_v5.txt')
    except FileNotFoundError as e:
        print(f"Error: Data file '{e.filename}' not found.");
        sys.exit()

    # --- 2. Pre-process spectral data (Copied from original) ---
    wl, gamma = spec_data_np[:, 0], spec_data_np[:, 1]
    if not np.all(np.diff(wl) > 0):
        sort_indices = np.argsort(wl)
        wl, gamma = wl[sort_indices], gamma[sort_indices]
    sigma_const = PHYSICAL_CONSTANTS['E_CHARGE'] ** 2 / (
            4 * PHYSICAL_CONSTANTS['ME'] * PHYSICAL_CONSTANTS['C'] * PHYSICAL_CONSTANTS['EPSILON_0'])
    spec_data_dict = {'wl': wl, 'gamma': gamma,
                      'sigma0_perdnu2': sigma_const * 0.641,
                      'sigma0_perdnu1': sigma_const * 0.320,
                      'JL': 5.18e14}

    # --- 3. Simulation settings ---
    settings = {
        'USE_SOLAR_GRAVITY': True,
        'USE_CORIOLIS_FORCES': True
    }

    # --- 4. Particle initial conditions ---
    # Place at the sub-solar point surface
    pos0 = np.array([PHYSICAL_CONSTANTS['RM'], 0.0, 0.0])
    # Initial velocity zero relative to Mercury
    vel0 = np.array([1000.0, 0.0, 0.0])

    # --- 5. Simulation time settings ---
    SIMULATION_DURATION_SEC = 10000  # 10 hours
    DT_SEC = 10.0  # 10 second steps

    # --- 6. Run for TAA 60° ---
    results_60 = run_particle_test(
        taa_target=60,
        orbit_data=orbit_data,
        spec_data=spec_data_dict,
        settings=settings,
        pos0=pos0,
        vel0=vel0,
        duration_sec=SIMULATION_DURATION_SEC,
        dt=DT_SEC
    )

    # --- 7. Run for TAA 300° ---
    results_300 = run_particle_test(
        taa_target=300,
        orbit_data=orbit_data,
        spec_data=spec_data_dict,
        settings=settings,
        pos0=pos0,
        vel0=vel0,
        duration_sec=SIMULATION_DURATION_SEC,
        dt=DT_SEC
    )

    # --- 8. Plot results ---
    plot_results(results_60, results_300)


if __name__ == '__main__':
    # Check for required files before starting
    print("Checking required files...")
    for f in ['orbit2025_v5.txt', 'SolarSpectrum_Na0.txt']:
        if not os.path.exists(f):
            print(f"Error: Required file '{f}' not found. Place it in the same directory as the script.")
            sys.exit()
    print("Files OK.")

    main_test()