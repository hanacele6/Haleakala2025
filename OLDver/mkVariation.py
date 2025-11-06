import numpy as np
import os


def pro_daily_variation3_python():
    """
    IDL 'pro dailyvariation3' procedure translated to Python.
    This script calculates daily variations based on observational data,
    applies a polynomial correction, and writes the results to a file.
    """
    # --- Initialization ---
    Month = 'November2024'
    # Use os.path.join for cross-platform compatibility
    filef0 =  r'C:\Users\hanac\University\Senior\Mercury\Haleakala2025'
    fileg = os.path.join(filef0, 'output', 'gamma_factor')

    day = ['20241107', '20241111', '20241115', '20241116', '20241119', 'test']

    #filed = os.path.join(filef0, 'output', f'{day[i]}')
    # Use NumPy arrays for easier mathematical operations
    Rms = np.array([0.4377959174,  0.4204883305, 0.400042978, 0.3945494659, 0.377476745, 0.3658675989],
                   dtype=np.float64)
    beta = np.array([-6.6218, -6.9658, -6.9294, -6.8466, -6.3849, -5.8759], dtype=np.float64)
    elon = np.array([299.2087, 312.3199, 326.682, 330.5111, 342.6696, 351.3963], dtype=np.float64)
    elat = beta  # ecliptic latitude

    # Constants
    pi = np.pi
    d2r = pi / 180.0
    dayN = len(day)

    # Pre-allocate arrays with numpy.zeros
    atoms = np.zeros(dayN, dtype=np.float64)
    err = np.zeros(dayN, dtype=np.float64)
    cf4 = np.zeros(dayN, dtype=np.float64)
    PA = np.zeros(dayN, dtype=np.float64)
    sf = np.zeros(dayN, dtype=np.float64)
    df = np.zeros(dayN, dtype=np.float64)
    h = np.zeros(dayN, dtype=np.float64)
    taa = np.zeros(dayN, dtype=np.float64)

    # --- Read PA correction file and perform polynomial fit ---
    correction_file = os.path.join(filef0, 'PA_correction2.txt')
    try:
        # Read the data from the file
        cf3 = np.loadtxt(correction_file)
        pa_data = cf3[:, 0]
        cf_data = cf3[:, 1]

        # Perform a 2nd-degree polynomial fit.
        # np.polyfit returns coefficients [p2, p1, p0] for p2*x^2 + p1*x + p0
        # This order is used by np.polyval, which simplifies calculation later.
        p2_coeffs = np.polyfit(pa_data, cf_data, 2)

    except FileNotFoundError:
        print(f"Error: Correction file not found at {correction_file}")
        return
    except Exception as e:
        print(f"An error occurred while reading or fitting the correction data: {e}")
        return

    # ;;;;;;Parameter for g-factor;;;;;;;;; (Commented out in original IDL)
    # NaD1 = 589.7558  # [nm]
    # NaD2 = 589.1583  # [nm]
    # c = 299792.458   # Light speed [km/s]
    # me = 9.1093897e-28 # Electron mass [g]
    # e = 4.8032068e-10  # Elementary charge [esu=g^1/2*cm^3/2*s^-1]
    # JL = 5.18e+14     # solar flux@1AU [phs/cm^2/nm/s]
    # f1 = 0.327
    # f2 = 0.654
    # Planck = 6.626e-27 # [cm^2*g/s]
    #
    # JLnu1 = JL * (1e9) * ((NaD1 * 1e-9)**2 / (c * 1e3))
    # JLnu2 = JL * (1e9) * ((NaD2 * 1e-9)**2 / (c * 1e3))
    # sigmaD1dnu = np.pi * e**2 / (me * c * 1e5) * f1 # * Planck * (c * 1e5) / (NaD1 * 1e-7) - Original has error, this is simplified form
    # sigmaD2dnu = np.pi * e**2 / (me * c * 1e5) * f2 # * Planck * (c * 1e5) / (NaD2 * 1e-7) - Original has error, this is simplified form
    # ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

    # --- Main Loop ---
    output_filename = os.path.join(filef0, f'DailyVariation3_Y1_{Month}_python.dat')

    try:
        with open(output_filename, 'w') as lunw4:
            for i in range(dayN):
                # --- Read daily data files ---
                try:
                    # Read 'num2.txt' file
                    #num_file = os.path.join(filef0, f'{day[i]}num2.txt')
                    num_file = os.path.join(filef0, 'output', day[i], f'{day[i]}num2_python.txt')
                    print(f"Reading: {num_file}")
                    pa0, taa0, a0, a1 = np.loadtxt(num_file, max_rows=1)
                    PA[i] = pa0
                    taa[i] = taa0
                    atoms[i] = a0
                    err[i] = a1

                    # Read 'gamma.txt' file
                    gamma_file = os.path.join(fileg, f'gamma_{day[i]}.txt')
                    g11, g12 = np.loadtxt(gamma_file, max_rows=1)
                    gamma1 = g12

                except FileNotFoundError as e:
                    print(f"Warning: Could not find a data file for day {day[i]}: {e}")
                    continue  # Skip to the next day
                except Exception as e:
                    print(f"Warning: Error reading data for day {day[i]}: {e}")
                    continue

                # --- Calculations ---
                # Solar radiation acceleration (commented out in original IDL)
                # gfac1 = sigmaD1dnu * JLnu1 / Rms[i]**2 * gamma1
                # gfac2 = sigmaD2dnu * JLnu2 / Rms[i]**2 * gamma2 # gamma2 is not defined
                # m_na = 3.82e-23  # [g]
                # acc = 1 / (m_na * c * 1e5) * (gfac1 + gfac2)

                sf[i] = pi / (pi - PA[i] * d2r)

                # Evaluate the polynomial fit at the current PA value
                cf4[i] = np.polyval(p2_coeffs, PA[i])

                h[i] = Rms[i] * np.sin(beta[i] * d2r)

                # --- Write to output file ---
                # Format: 2 floating-point numbers, followed by 5 in scientific notation
                # to match the 7 variables provided in the IDL code.
                output_line = (
                    f"{taa[i]:15.5f}"
                    f"{h[i]:15.5f}"
                    f"{Rms[i]:15.6e}"
                    f"{elon[i]:15.6e}"
                    f"{elat[i]:15.6e}"
                    f"{atoms[i] / cf4[i]:15.6e}"
                    f"{err[i] / cf4[i]:15.6e}\n"
                )
                lunw4.write(output_line)

    except IOError as e:
        print(f"Error: Could not write to output file {output_filename}: {e}")

    print('end')


# --- Execute the main function ---
if __name__ == '__main__':
    pro_daily_variation3_python()