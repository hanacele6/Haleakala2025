import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import sys
import warnings

# Suppress RankWarning from polyfit
warnings.filterwarnings('ignore', category=np.RankWarning)


def plot_exosphere_results(exosphere_data_list, output_base_path, date_str):
    """
    Plots the final exosphere signals. This function is from your original script.
    """
    if not exosphere_data_list:
        print("No exosphere data to plot.")
        return

    plot_subdir_name = "py_ExospherePlots"
    plot_output_dir = output_base_path / "fits" / date_str / plot_subdir_name
    plot_output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 10))
    colors = plt.cm.get_cmap('tab10', max(10, len(exosphere_data_list)))

    for i, data in enumerate(exosphere_data_list):
        file_id = data['file_identifier']
        wl = data['wavelength']
        exos = data['exos_signal']
        airm = data['airm']
        fwhm = data['FWHM']
        color = colors(i % 10)
        label = f"File ID {file_id} (Airmass={airm:.1f}, FWHM={fwhm:.1f} pix)"
        plt.plot(wl, exos, color=color, linewidth=1.5, label=label, alpha=0.8)

    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Intensity (MR-scaled)', fontsize=12)
    plt.title(f'Mercury Exosphere Spectra - Extracted Signals ({date_str})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    combined_plot_path = plot_output_dir / f"all_exos_combined.png"
    plt.savefig(combined_plot_path)
    print(f"Saved combined plot: {combined_plot_path}")
    plt.show()


def process_single_file(file_id, base_name, totfib_filepath, solar_params, physical_params):
    """
    Processes a single observation file using the logic from your original pro_subsurfnew2_refactored script.
    This function contains the core processing loop for one file.
    """
    print(f"\nProcessing File: {base_name} ({totfib_filepath.name})")

    # Unpack parameters
    valid_interp_min_wl = solar_params['valid_interp_min_wl']
    valid_interp_max_wl = solar_params['valid_interp_max_wl']
    solar_wl_air_shifted = solar_params['solar_wl_air_shifted']
    solar_flux_comp0 = solar_params['solar_flux_comp0']
    solar_wl_reflected = solar_params['solar_wl_reflected']
    solar_flux_comp1 = solar_params['solar_flux_comp1']

    vme_obs = physical_params['vme_obs']
    vms_obs = physical_params['vms_obs']
    rmn_mercury_arcsec = physical_params['rmn_mercury_arcsec']
    na_d2_line_wavelength_nm = physical_params['na_d2_line_wavelength_nm']
    na_line_exclusion_half_width_pixels = physical_params['na_line_exclusion_half_width_pixels']

    # --- Read and Crop Observation Data ---
    try:
        # Assume one header row
        totfib_data_full = np.loadtxt(totfib_filepath, skiprows=1)
        wl_obs_full, Nat_obs_full = totfib_data_full[:, 0], totfib_data_full[:, 1]
    except Exception as e:
        print(f"  -> ERROR reading {totfib_filepath.name}: {e}. Skipping.")
        return None

    processing_mask = (wl_obs_full >= valid_interp_min_wl) & (wl_obs_full <= valid_interp_max_wl)
    wl_obs, Nat_obs = wl_obs_full[processing_mask], Nat_obs_full[processing_mask]
    ixm = len(wl_obs)

    if ixm < 20:
        print(f"  -> WARNING: Only {ixm} data points remain after cropping. Skipping.")
        return None
    print(f"  -> Processing {ixm} data points within range {wl_obs.min():.2f}-{wl_obs.max():.2f} nm.")

    # --- Interpolate Solar Model ---
    solar_model_comp0_interp = np.interp(wl_obs, solar_wl_air_shifted, solar_flux_comp0)
    solar_model_comp1_interp = np.interp(wl_obs, solar_wl_reflected, solar_flux_comp1)
    surf_model_components = np.column_stack([solar_model_comp0_interp, solar_model_comp1_interp])

    # --- D2 Line Exclusion (Your Original, Correct Logic) ---
    na_d2_center_idx = np.argmin(np.abs(wl_obs - na_d2_line_wavelength_nm))
    idx_exclude_start = max(0, na_d2_center_idx - na_line_exclusion_half_width_pixels)
    idx_exclude_end = min(ixm - 1, na_d2_center_idx + na_line_exclusion_half_width_pixels)
    fitting_mask = np.ones(ixm, dtype=bool)
    fitting_mask[idx_exclude_start: idx_exclude_end + 1] = False
    Nat_for_fitting = Nat_obs[fitting_mask]
    pixel_indices_for_fitting = np.arange(ixm)[fitting_mask]

    if len(Nat_for_fitting) < 3:
        print(f"  -> WARNING: Not enough points ({len(Nat_for_fitting)}) for fitting after D2 exclusion. Skipping.")
        return None

    # --- Best Fit Search ---
    zansamin = 1e32
    best_params_fit = {}
    for iFWHM_val in range(30, 101):
        FWHM = iFWHM_val * 0.1
        sigma = FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        x_psf = np.arange(ixm) - (ixm / 2.0)
        psf = np.exp(-(x_psf / sigma) ** 2 / 2.0)
        psf_normalized = psf / np.sum(psf)
        fft_psf_rolled = np.fft.fft(np.roll(psf_normalized, -(ixm // 2)))

        surf1_convolved = np.zeros_like(surf_model_components)
        for i in range(2):
            fft_surf_comp = np.fft.fft(surf_model_components[:, i])
            convolved_fft = np.fft.ifft(fft_surf_comp * fft_psf_rolled)
            surf1_convolved[:, i] = np.real(convolved_fft)

        for iairm_val in range(51):
            airm = 0.1 * iairm_val
            term1_positive = np.maximum(surf1_convolved[:, 0], 1e-9)
            model_shape = (term1_positive ** airm) * surf1_convolved[:, 1]
            model_shape_masked = model_shape[fitting_mask]

            try:
                # Use a safe division to avoid errors with very small model values
                ratio = np.divide(Nat_for_fitting, model_shape_masked, out=np.zeros_like(Nat_for_fitting),
                                  where=model_shape_masked != 0)
                poly_coeffs = np.polyfit(pixel_indices_for_fitting, ratio, 2)

                fitted_ratio = np.polyval(poly_coeffs, pixel_indices_for_fitting)
                residuals_sq = (Nat_for_fitting - model_shape_masked * fitted_ratio) ** 2
                zansa = np.sum(residuals_sq)

                if zansa < zansamin:
                    zansamin = zansa
                    best_params_fit = {
                        'poly_coeffs': poly_coeffs,
                        'airm': airm,
                        'FWHM': FWHM,
                        'final_model': model_shape * np.polyval(poly_coeffs, np.arange(ixm)),
                        'model_shape_for_ratio2': model_shape
                    }
            except (np.linalg.LinAlgError, ValueError):
                continue

    if not best_params_fit:
        print(f"  -> WARNING: No valid fit found for {base_name}. Skipping.")
        return None

    print(f"  -> Best fit: Airmass={best_params_fit['airm']:.2f}, FWHM={best_params_fit['FWHM']:.2f} pix")

    # --- Final Signal Extraction and Saving ---
    Natb_exosphere_counts = Nat_obs - best_params_fit['final_model']

    # cts2MR Factor Calculation
    dwl_obs_mean = np.mean(np.diff(wl_obs))
    Rmc_cm = 4.879e8
    Shap_cm2_per_pix = (Rmc_cm / rmn_mercury_arcsec) ** 2 / 1e4 if rmn_mercury_arcsec > 1e-9 else 0.0

    model_for_ratio2 = best_params_fit['model_shape_for_ratio2'][fitting_mask]
    denom = np.sum(model_for_ratio2 ** 2)
    ratio2_equivalent = np.sum(Nat_for_fitting * model_for_ratio2) / denom if denom > 1e-30 else 0.0

    cts2MR_factor = 1.0
    hapke_filepath = totfib_filepath.parent / f"Hapke{physical_params['date_str']}.fits"
    if hapke_filepath.exists() and Shap_cm2_per_pix > 0 and abs(ratio2_equivalent) > 0:
        try:
            hap_data = fits.getdata(hapke_filepath)
            tothap = np.sum(hap_data) * Shap_cm2_per_pix * dwl_obs_mean * 1e12
            cts2MR_factor = tothap / ratio2_equivalent
        except Exception as e:
            print(f"  -> WARNING: Could not process Hapke file: {e}")

    exos_signal_final_MR = Natb_exosphere_counts * cts2MR_factor

    # Save Files using the base_name
    output_dir = totfib_filepath.parent
    exos_output_dat = output_dir / f"{base_name}.exos.dat"
    np.savetxt(exos_output_dat, np.column_stack([wl_obs, exos_signal_final_MR]), fmt="%.6e",
               header="Wavelength(nm) Flux(MR)")
    print(f"  -> Saved exosphere signal: {exos_output_dat.name}")

    return {
        'file_identifier': file_id,  # Keep original file_id for plot labels
        'wavelength': wl_obs,
        'exos_signal': exos_signal_final_MR,
        'airm': best_params_fit['airm'],
        'FWHM': best_params_fit['FWHM']
    }


def main_processor(date_str, base_dir, type_col_name='Type'):
    """
    Main function to drive the processing. It reads the solar spectrum and the CSV file,
    then loops through each observation, calling the processing function for each.
    """
    base_dir = Path(base_dir)
    # The directory structure from your second script
    data_dir = base_dir / "output" / date_str

    # --- Read Solar Spectrum (once for all files) ---
    solar_file_path = base_dir / "SolarSpectrum.txt"
    try:
        solar_raw = np.loadtxt(solar_file_path)
    except Exception as e:
        print(f"FATAL Error: Could not read solar spectrum file {solar_file_path}: {e}")
        return

    VACUUM_TO_AIR_FACTOR = 1.000276
    solar_wl_air = solar_raw[:, 0] / VACUUM_TO_AIR_FACTOR
    solar_params = {
        'solar_flux_comp0': solar_raw[:, 2],
        'solar_flux_comp1': solar_raw[:, 1],
    }

    # --- Read CSV to get file list and parameters ---
    csv_file_path = base_dir / "2025ver" / f"mcparams{date_str[:6]}.csv"
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"FATAL ERROR: Parameter CSV file not found: {csv_file_path}")
        return

    print(f"\n--- Starting Mercury Exosphere Extraction for {date_str} ---")

    all_results_for_plotting = []

    # --- Loop through specified types, e.g., 'MERCURY' ---
    types_to_process = ['MERCURY']
    for process_type in types_to_process:
        print("\n" + "=" * 25 + f" Processing Type: {process_type} " + "=" * 25)
        target_df = df[df[type_col_name] == process_type].copy()

        if target_df.empty:
            print(f"-> No data found for type '{process_type}' in CSV.")
            continue

        # --- Loop through each entry for this type ---
        for idx, (row_index, row) in enumerate(target_df.iterrows(), start=1):
            # Generate filename like 'MERCURY1_tr'
            base_name = f"{process_type}{idx}_tr"
            input_file = data_dir / f"{base_name}.totfib.dat"

            # Fallback to check for _orig.dat file
            if not input_file.exists():
                input_file_orig = data_dir / f"{base_name}.totfib_orig.dat"
                if input_file_orig.exists():
                    input_file = input_file_orig
                else:
                    print(f"\nSkipping {base_name}: Input file not found.")
                    continue

            try:
                # Get parameters from the current row of the CSV
                file_id = int(row.get('file_id', idx))  # Use file_id from CSV if it exists, otherwise use index
                vme_obs = float(row['mercury_earth_radial_velocity_km_s'])
                vms_obs = float(row['mercury_sun_radial_velocity_km_s'])
                rmn_arcsec = float(row['apparent_diameter_arcsec'])
                sft_solar_nm = float(row.get('sft_solar_nm', 0.002))
            except (KeyError, ValueError) as e:
                print(f"\nERROR reading row {row_index} from CSV: {e}. Skipping this entry.")
                continue

            # Update solar parameters with per-file Doppler shifts
            c_light_km_s = 299792.458
            solar_params['solar_wl_air_shifted'] = solar_wl_air - sft_solar_nm
            doppler_factor = (1 + vms_obs / c_light_km_s) * (1 + vme_obs / c_light_km_s)
            solar_params['solar_wl_reflected'] = solar_params['solar_wl_air_shifted'] * doppler_factor
            solar_params['valid_interp_min_wl'] = max(np.min(solar_params['solar_wl_air_shifted']),
                                                      np.min(solar_params['solar_wl_reflected']))
            solar_params['valid_interp_max_wl'] = min(np.max(solar_params['solar_wl_air_shifted']),
                                                      np.max(solar_params['solar_wl_reflected']))

            physical_params = {
                'vme_obs': vme_obs,
                'vms_obs': vms_obs,
                'rmn_mercury_arcsec': rmn_arcsec,
                'na_d2_line_wavelength_nm': 589.594,
                'na_line_exclusion_half_width_pixels': 4,#もともとは7
                'date_str': date_str
            }

            # Call the processing function for this single file
            result = process_single_file(file_id, base_name, input_file, solar_params, physical_params)

            if result:
                all_results_for_plotting.append(result)

    # --- Plotting ---
    if all_results_for_plotting:
        print("\n--- Plotting Results ---")
        plot_exosphere_results(all_results_for_plotting, base_dir, date_str)

    print("\n--- All Processing Complete ---")


if __name__ == "__main__":
    # --- 1. Main Settings (Please change these to match your setup) ---
    obs_date = "20250501"
    # Use r-string for Windows paths or Path object for cross-platform compatibility
    project_base_directory = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")

    # --- 2. Run the Processor ---
    main_processor(date_str=obs_date, base_dir=project_base_directory)