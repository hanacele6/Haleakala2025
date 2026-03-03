import numpy as np
import matplotlib.pyplot as plt


def miv_longitude_sampling_histogram():
    """
    Performs MIV longitude sampling and plots the resulting
    probability density as a filled bar histogram.
    """

    # --- MIV Sampling Logic ---
    PI = np.pi
    M_rejection = 4.0 / 3.0
    TOTAL_SAMPLES = 1_000_000  # Number of samples for statistics

    print(f"Sampling {TOTAL_SAMPLES} particles for longitude...")

    # Use numpy for faster sampling
    sampled_longitudes_rad = []
    num_generated = 0

    batch_size = int(TOTAL_SAMPLES * M_rejection * 1.2)

    while num_generated < TOTAL_SAMPLES:
        random_lon_rad = np.random.uniform(-PI, PI, size=batch_size)
        prob_accept = (1.0 - (1.0 / 3.0) * np.sin(random_lon_rad)) / M_rejection
        accepted = random_lon_rad[np.random.random(size=batch_size) < prob_accept]
        sampled_longitudes_rad.extend(accepted)
        num_generated += len(accepted)

    sampled_longitudes_rad = sampled_longitudes_rad[:TOTAL_SAMPLES]

    print("Sampling complete. Creating histogram...")

    # --- Plotting ---
    plt.figure(figsize=(12, 7))

    # Calculate histogram data
    bins = 100
    hist, bin_edges = np.histogram(
        sampled_longitudes_rad,
        bins=bins,
        range=(-PI, PI),
        density=True
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # --- ★★★ MODIFIED ★★★ ---
    # Draw the histogram as filled bars
    plt.bar(
        bin_centers,
        hist,
        width=bin_width,
        label='MIV Sampling Result (Histogram)',
        color='dodgerblue',
        edgecolor='darkblue',  # Add edge color for clarity
        linewidth=0.5
    )
    # --- ★★★ END MODIFIED ★★★ ---

    # --- Graph Styling ---
    plt.title(f'MIV Longitude Sampling Density (N={TOTAL_SAMPLES})', fontsize=16)
    plt.xlabel('Longitude [radians]', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xlim(-PI, PI)
    plt.ylim(bottom=0)  # Y-axis starts at 0

    # Set X-axis ticks corresponding to physical locations
    plt.xticks(
        ticks=[-PI, -PI / 2, 0, PI / 2, PI],
        labels=[r'$-\pi$ (-180°)', r'$-\pi/2$ (-90°, Dawn)', '0 (0°, Subsolar)', r'$+\pi/2$ (+90°, Dusk)',
                r'$+\pi$ (180°)']
    )

    plt.tight_layout()

    # Save the figure to a file
    output_filename = 'miv_longitude_histogram_filled.png'
    plt.savefig(output_filename)
    print(f"Histogram image saved as '{output_filename}'")

    # Display the plot on screen
    plt.show()


if __name__ == '__main__':
    miv_longitude_sampling_histogram()