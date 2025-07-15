import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Circle
import numpy as np
from datetime import datetime, timedelta, timezone
import os

from skyfield.api import Topos, load, wgs84
from skyfield.framelib import ecliptic_frame
import spiceypy as spice

# Matplotlib font settings (adjust to your environment if needed for other languages)
# try:
#     plt.rcParams['font.family'] = 'your_font_name_here'
# except RuntimeError:
#     print("Font not found, default font will be used.")

# JST timezone (still used for JST display in titles if desired)
JST = timezone(timedelta(hours=9))

# Absolute path to SPICE kernels, as per user's environment
SPICE_KERNEL_BASE_PATH = r'C:\Users\hanac\University\Senior\PythonProject\merc2025a\kernels\generic_kernels'
# SCRIPT_DIR definition (can be kept for reference, but not directly used for SPICE_KERNEL_BASE_PATH here)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_spice_kernels(planet_spk_name='de430.bsp'):
    """Load SPICE kernels"""
    try:
        spice.furnsh(os.path.join(SPICE_KERNEL_BASE_PATH, 'lsk', 'naif0012.tls'))
        spice.furnsh(os.path.join(SPICE_KERNEL_BASE_PATH, 'pck', 'pck00011.tpc'))
        spice.furnsh(os.path.join(SPICE_KERNEL_BASE_PATH, 'spk', 'planets', planet_spk_name))
        return True
    except Exception as e:
        print(f"Failed to load SPICE kernels: {e}")
        print("Please ensure kernel files are in the correct path and subdirectory structure.")
        print(f"Expected base path: {SPICE_KERNEL_BASE_PATH}")
        print("Example subdirectories: lsk/naif0012.tls, pck/pck00011.tpc, spk/planets/your_spk.bsp")
        return False


def unload_spice_kernels(planet_spk_name='de430.bsp'):
    """Unload SPICE kernels (cleanup)"""
    try:
        spice.unload(os.path.join(SPICE_KERNEL_BASE_PATH, 'lsk', 'naif0012.tls'))
        spice.unload(os.path.join(SPICE_KERNEL_BASE_PATH, 'pck', 'pck00011.tpc'))
        spice.unload(os.path.join(SPICE_KERNEL_BASE_PATH, 'spk', 'planets', planet_spk_name))
    except Exception:
        pass


def get_true_anomaly_spice(skyfield_time, skyfield_target, skyfield_sun):
    """Calculate True Anomaly of the target using Skyfield objects and SPICE"""
    utc_iso_string = skyfield_time.utc_iso()
    et = spice.str2et(utc_iso_string)

    target_pos_ssb_km = skyfield_target.at(skyfield_time).position.km
    target_vel_ssb_km_s = skyfield_target.at(skyfield_time).velocity.km_per_s
    sun_pos_ssb_km = skyfield_sun.at(skyfield_time).position.km
    sun_vel_ssb_km_s = skyfield_sun.at(skyfield_time).velocity.km_per_s

    target_state_wrt_sun_km = np.concatenate([
        target_pos_ssb_km - sun_pos_ssb_km,
        target_vel_ssb_km_s - sun_vel_ssb_km_s
    ])

    try:
        mu_sun_km3_s2 = spice.bodvrd('SUN', 'GM', 1)[1][0]
    except Exception as e:
        print(f"Could not retrieve Sun's GM from SPICE kernel: {e}")
        mu_sun_km3_s2 = 1.32712440018e11
        print(f"Using hardcoded Sun's GM value {mu_sun_km3_s2} km^3/s^2.")

    try:
        orbital_elements = spice.oscelt(target_state_wrt_sun_km, et, mu_sun_km3_s2)
        true_anomaly_degrees = np.degrees(orbital_elements[5])
        return true_anomaly_degrees
    except Exception as e:
        print(f"Error during True Anomaly calculation: {e}")
        return None


def plot_planetary_positions_and_visibility_with_taa(date_str, lat, lon, skyfield_eph_file='de442.bsp',
                                                     spice_planet_spk='de430.bsp'):
    """
    Calculates and displays TAA (True Anomaly) value using SPICE.
    """
    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        print("Incorrect date format. Please use YYYY-MM-DD.")
        return

    if not load_spice_kernels(planet_spk_name=spice_planet_spk):
        return

    ts = load.timescale()
    try:
        sky_eph = load(skyfield_eph_file)
    except Exception as e:
        print(f"Failed to load Skyfield ephemeris file '{skyfield_eph_file}': {e}")
        unload_spice_kernels(planet_spk_name=spice_planet_spk)
        return

    sun_sky = sky_eph['sun']
    earth_sky = sky_eph['earth']
    mercury_sky = sky_eph['mercury']
    observer_location = wgs84.latlon(lat, lon)

    dt_noon_jst = datetime(target_date.year, target_date.month, target_date.day, 12, 0, 0, tzinfo=JST)
    t_noon_utc_sky = ts.utc(dt_noon_jst)

    times_for_plot_utc_sky = []
    start_day_utc = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=JST).astimezone(
        timezone.utc)
    for i in range(24 * 6 + 1):  # 10-minute intervals
        dt_utc = start_day_utc + timedelta(minutes=i * 10)
        times_for_plot_utc_sky.append(ts.utc(dt_utc))

    # --- 1. Calculate Solar System body positions (at Noon JST) ---
    sun_pos_ssb_ecl_au = sun_sky.at(t_noon_utc_sky).frame_xyz(ecliptic_frame).au
    earth_pos_ssb_ecl_au = earth_sky.at(t_noon_utc_sky).frame_xyz(ecliptic_frame).au
    mercury_pos_ssb_ecl_au = mercury_sky.at(t_noon_utc_sky).frame_xyz(ecliptic_frame).au

    earth_x = earth_pos_ssb_ecl_au[0] - sun_pos_ssb_ecl_au[0]
    earth_y = earth_pos_ssb_ecl_au[1] - sun_pos_ssb_ecl_au[1]
    mercury_x = mercury_pos_ssb_ecl_au[0] - sun_pos_ssb_ecl_au[0]
    mercury_y = mercury_pos_ssb_ecl_au[1] - sun_pos_ssb_ecl_au[1]

    obs_point_sky = earth_sky + observer_location
    sot_degrees = obs_point_sky.at(t_noon_utc_sky).observe(mercury_sky).apparent().separation_from(
        obs_point_sky.at(t_noon_utc_sky).observe(sun_sky).apparent()
    ).degrees
    phase_angle_degrees = obs_point_sky.at(t_noon_utc_sky).observe(mercury_sky).phase_angle(sun_sky).degrees

    taa_degrees = get_true_anomaly_spice(t_noon_utc_sky, mercury_sky, sun_sky)

    # --- 2. Calculate Alt/Az for visibility plot (for the whole day) ---
    observer_sky = earth_sky + observer_location
    sun_altitudes = []
    mercury_altitudes = []
    plot_times_jst = []

    for t_sky_utc in times_for_plot_utc_sky:
        sun_app = observer_sky.at(t_sky_utc).observe(sun_sky).apparent()
        mercury_app = observer_sky.at(t_sky_utc).observe(mercury_sky).apparent()
        sun_alt, _, _ = sun_app.altaz()
        mercury_alt, _, _ = mercury_app.altaz()
        sun_altitudes.append(sun_alt.degrees)
        mercury_altitudes.append(mercury_alt.degrees)
        plot_times_jst.append(t_sky_utc.astimezone(JST).replace(tzinfo=None))  # Naive datetime for matplotlib

    # --- Plotting ---
    fig = plt.figure(figsize=(14, 7))  # Adjusted figure size for two plots
    fig.suptitle(f"{date_str} (JST) / Obs. Location: Lat {lat:.2f}°, Lon {lon:.2f}° E", fontsize=14)

    # Subplot 1: Solar System Positions
    ax1 = fig.add_subplot(1, 2, 1, aspect='equal')
    ax1.set_title(f"Sun-Earth-Mercury Positions ({dt_noon_jst.strftime('%H:%M JST')})", fontsize=10)
    ax1.plot(0, 0, 'o', color='gold', markersize=15, label='Sun (Origin)')
    earth_orbit = Circle((0, 0), 1.0, edgecolor='dodgerblue', facecolor='none', linestyle=':')
    ax1.add_patch(earth_orbit)
    ax1.plot(earth_x, earth_y, 'o', color='dodgerblue', markersize=8, label='Earth')
    ax1.plot([0, earth_x], [0, earth_y], '-', color='dodgerblue', linewidth=0.5)

    mercury_orbit_radius_actual = np.sqrt(mercury_x ** 2 + mercury_y ** 2)
    mercury_orbit = Circle((0, 0), mercury_orbit_radius_actual, edgecolor='darkgrey', facecolor='none', linestyle=':')
    ax1.add_patch(mercury_orbit)
    ax1.plot(mercury_x, mercury_y, 'o', color='grey', markersize=6, label='Mercury')
    ax1.plot([0, mercury_x], [0, mercury_y], '-', color='grey', linewidth=0.5)
    ax1.plot([earth_x, mercury_x], [earth_y, mercury_y], '--', color='red', linewidth=0.7)

    ax1.set_xlabel("Ecliptic X (AU, Sun-centered)", fontsize=9)
    ax1.set_ylabel("Ecliptic Y (AU, Sun-centered)", fontsize=9)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.6)
    max_range = 1.5
    ax1.set_xlim(-max_range, max_range)
    ax1.set_ylim(-max_range, max_range)
    ax1.tick_params(axis='both', which='major', labelsize=8)

    taa_text_en = f"{taa_degrees:.1f}°" if taa_degrees is not None else "Calculation Failed"
    info_text = (f"SOT (Elongation): {sot_degrees:.1f}°\n"
                 f"Phase Angle: {phase_angle_degrees:.1f}°\n"
                 f"TAA (True Anomaly): {taa_text_en}")
    ax1.text(0.03, 0.03, info_text, transform=ax1.transAxes, fontsize=8,
             verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.7))

    # Subplot 2: Altitude Profile
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("Sun and Mercury Altitude Profile (JST)", fontsize=10)
    ax2.plot(plot_times_jst, sun_altitudes, label='Sun Altitude', color='orange')
    ax2.plot(plot_times_jst, mercury_altitudes, label='Mercury Altitude', color='dimgray')

    ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, label='Horizon')
    ax2.axhline(10, color='gray', linestyle='--', linewidth=0.5, label='Mercury Alt 10°')
    ax2.axhline(-1.5, color='blue', linestyle='--', linewidth=0.5, label='Sun Alt -1.5° (Twilight)')

    ax2.set_xlabel("Time (JST)", fontsize=9)
    ax2.set_ylabel("Altitude (°)", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.6)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.set_xticks([t for i, t in enumerate(plot_times_jst) if i % (6 * 2) == 0])  # Approx every 2 hours
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right", fontsize=8)  # Adjusted for readability
    ax2.tick_params(axis='y', which='major', labelsize=8)

    ax2.set_ylim(-30, 90)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust rect to prevent suptitle overlap and make space for x-labels
    plt.show()

    unload_spice_kernels(planet_spk_name=spice_planet_spk)


if __name__ == '__main__':
    print("Sun-Earth-Mercury Positions and Visibility (calculating TAA value)")

    print("\n--- Regarding SPICE Kernel Preparation ---")
    print(f"This script expects SPICE kernels to be arranged in the following directory structure:")
    print(f"  {SPICE_KERNEL_BASE_PATH}/")
    print(f"  ├── lsk/naif0012.tls")
    print(f"  ├── pck/pck00011.tpc")
    print(f"  └── spk/planets/(Planetary SPK file, e.g., de430.bsp or de442.bsp)")
    print("Please download the necessary kernels from the NASA NAIF site and place them in the path above.")
    print("---------------------------------\n")

    while True:
        date_input_str = input("Enter date to display (YYYY-MM-DD, e.g., 2025-07-01): ")
        if not date_input_str: date_input_str = "2025-07-01"  # Default date
        try:
            datetime.strptime(date_input_str, "%Y-%m-%d")
            break
        except ValueError:
            print("Incorrect date format. Please re-enter.")

    default_lat, default_lon = 20.7083119, 203.742  # Haleakala Observatory (Longitude East positive)
    try:
        lat_str = input(f"Observer's Latitude (default Haleakala {default_lat}): ")
        obs_lat = float(lat_str) if lat_str else default_lat
    except ValueError:
        obs_lat = default_lat
        print(f"Invalid or empty latitude. Using default value {default_lat}.")

    try:
        lon_str = input(f"Observer's Longitude (East positive, default Haleakala {default_lon}): ")
        obs_lon = float(lon_str) if lon_str else default_lon
    except ValueError:
        obs_lon = default_lon
        print(f"Invalid or empty longitude. Using default value {default_lon}.")

    skyfield_eph_name = 'de442.bsp'
    spice_planet_spk_name = 'de430.bsp'

    custom_spice_spk = input(
        f"Planetary SPK file for SPICE calculations (e.g., de430.bsp, default: {spice_planet_spk_name}): ")
    if custom_spice_spk:
        spice_planet_spk_name = custom_spice_spk

    print(f"\nSkyfield ephemeris: {skyfield_eph_name} (loaded from script directory or Skyfield path)")
    print(
        f"SPICE planetary ephemeris: {spice_planet_spk_name} (loaded from spk/planets/ within {SPICE_KERNEL_BASE_PATH})")
    print(f"Displaying for Date: {date_input_str}, Observer: Lat {obs_lat:.2f}°, Lon {obs_lon:.2f}° E\n")

    plot_planetary_positions_and_visibility_with_taa(date_input_str, obs_lat, obs_lon,
                                                     skyfield_eph_file=skyfield_eph_name,
                                                     spice_planet_spk=spice_planet_spk_name)