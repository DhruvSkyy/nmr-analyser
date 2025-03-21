import os
import argparse
from pathlib import Path

from cluster import (
    hierarchical_clustering,
    assign_functional_groups,
    load_csv_coordinates,
    determine_noise_threshold,
    detect_and_plot_peaks,
    calculate_cluster_count,
)

from j_val import (
    factorize_multiplicity,
    match_peaks_to_multiplicity,
)

def nmr_peak_analysis(peaks, k, frequency, uncertainty=1.0, extra=0):
    """
    Clusters NMR peaks, assigns functional groups, classifies multiplets, and
    prints ordered peaks with multiplicity and J values in the standard NMR format.
    Example output: 7.26 (d, J = 7.5 Hz)
    """
    output_dir = "nmrdata_output"
    os.makedirs(output_dir, exist_ok=True)
    detailed_path = os.path.join(output_dir, "detailed_calculations.txt")
    simple_path = os.path.join(output_dir, "simple_calculations.txt")

    for path in [detailed_path, simple_path]:
        if os.path.exists(path):
            os.remove(path)
    
    if k > len(peaks): # Potential error when k is input again by user. 
        raise ValueError("K is higher than the number of peaks available.")

    max_k = min(k + extra, len(peaks)) # In the case extra leads to more k than peaks 
    min_multiplets, best_k = float('inf'), k

    for current_k in range(k, max_k + 1):
        multiplet_count, clusters, cluster_data = evaluate_clusters(peaks, current_k, uncertainty, frequency)
        if multiplet_count < min_multiplets:
            min_multiplets, best_k = multiplet_count, current_k
            best_cluster_data = cluster_data
            best_clusters = clusters

    # Order all data by peak values
    best_cluster_data.sort(key=lambda x: min(x[0]))

    with open(detailed_path, "a") as f_detailed, open(simple_path, "a") as f_simple:
        for cluster_peaks, best_match, avg_j_values, assigned_groups, all_multiplicities, all_j_vals in best_cluster_data:
            write_detailed_output(f_detailed, cluster_peaks, all_j_vals, assigned_groups, all_multiplicities)
            write_simple_output(f_simple, cluster_peaks, best_match, avg_j_values, assigned_groups)

            if avg_j_values:
                j_values_str = ', '.join(f'{j:.2f}' for j in avg_j_values)
                peak_str = f"{min(cluster_peaks):.2f} ({best_match}, J = {j_values_str} Hz)"
            else:
                peak_str = f"{min(cluster_peaks):.2f} ({best_match})"
            print(peak_str)

    print(f"Analysis complete. Files written to: {detailed_path}, {simple_path}")
    return best_k, best_clusters


def evaluate_clusters(peaks, k, uncertainty, frequency):
    labels, centroids = hierarchical_clustering(peaks.reshape(-1, 1), k)
    clusters = {i: [] for i in range(k)}
    cluster_data = []
    multiplet_count = 0
    for peak, label in zip(peaks, labels):
        clusters[label].append(peak)
    for cluster_peaks in clusters.values():
        cluster_peaks.sort()
        best_match, avg_j_values, assigned_groups, all_multiplicities, all_j_vals = process_cluster(cluster_peaks, uncertainty, frequency)
        cluster_data.append((cluster_peaks, best_match, avg_j_values, assigned_groups, all_multiplicities, all_j_vals))
        if best_match == 'multiplet':
            multiplet_count += 1
    return multiplet_count, clusters, cluster_data


def process_cluster(cluster_peaks, uncertainty, frequency):
    if len(cluster_peaks) == 1:
        return "s", [], [], [], []

    all_mults = factorize_multiplicity(len(cluster_peaks))
    best_match, matched_j_vals, raw_j_vals = match_peaks_to_multiplicity(
        peaks=cluster_peaks, 
        multiplicities=all_mults, 
        frequency=frequency, 
        uncertainty=uncertainty
    )

    assigned_groups = assign_functional_groups(cluster_peaks[0])

    return best_match, matched_j_vals, assigned_groups, all_mults, raw_j_vals

def write_detailed_output(f, peaks, j_vals, groups, all_multiplicities):
    peaks_str = ', '.join(f"{p:.5f}" for p in peaks)
    j_vals_str = ', '.join(f"{j:.5f}" for j in j_vals) if len(j_vals) > 0 else "None"
    groups_str = " , ".join(groups)
    mult_str = ', '.join(str(m) for m in all_multiplicities) if all_multiplicities else "None"
    f.write(f"Peaks: {peaks_str}\nJ-values: {j_vals_str}\nGroups: {groups_str}\nAll Multiplicities: {mult_str}\n\n")

def write_simple_output(f, peaks, best_match, j_vals, groups):
    range_str = f"{min(peaks):.2f} - {max(peaks):.2f} ppm" if best_match == "multiplet" else f"{peaks[0]:.2f} ppm"
    j_vals_str = ', '.join(f"{j:.1f}" for j in j_vals) if j_vals else "None"
    groups_str = " , ".join(groups)
    f.write(f"Peak: {range_str}\nMultiplicity: {best_match}\nJ-values: {j_vals_str}\nGroups: {groups_str}\n\n")


def main(path, frequency, uncertainty=1.0, extra=10):
    """Main function to perform NMR peak analysis with adjustable parameters."""
    x, y = load_csv_coordinates(path)
    noise_threshold = determine_noise_threshold(y)
    os.makedirs('nmrdata_output', exist_ok=True)

    # Peak Detection and Plotting
    while True:
        peak_x, peak_y, y_axis_scale = detect_and_plot_peaks(x, y, threshold_baseline=noise_threshold)
        scaled_threshold = noise_threshold / (10 ** y_axis_scale)
        
        print(f"Current threshold: {scaled_threshold:.2f} ×10^{y_axis_scale}")

        if input("Is this threshold acceptable? (y/n): ").strip().lower() == 'y':
            break

        user_input_str = input(f"Enter new threshold (as seen on y-axis, e.g. 4.5 if axis shows 4.5 ×10^{y_axis_scale}): ")
        try:
            user_input_value = float(user_input_str)
            noise_threshold = user_input_value * (10 ** y_axis_scale)
        except ValueError:
            print("Invalid input; please enter a numeric value.")
            continue
    
    
    # Clustering and Analysis
    k = calculate_cluster_count(peak_x)
    while True:
        k, clusters  = nmr_peak_analysis(peak_x, k, frequency=frequency, uncertainty=uncertainty, extra=extra)
        detect_and_plot_peaks(x, y, threshold_baseline=noise_threshold, clusters=clusters)
        print(f"Number of clusters: {k}")
        if input("Is this number of clusters acceptable? (y/n): ").strip().lower() == 'y':
            break
        k = int(input("Enter preferred number of clusters: "))
        extra = 0


def validate_file(arg):
    if Path(arg).is_file():
        return arg
    raise FileNotFoundError(arg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NMR Peak Analysis Tool")
    parser.add_argument("--file", type=validate_file, required=True, help="Input file path")
    parser.add_argument("--extra", type=int, default=0, help="Extra clusters to consider during analysis")
    parser.add_argument("--uncertainty", type=float, default=1.0, help="Uncertainty tolerance for multiplicity matching")
    parser.add_argument("--frequency", type=float, required=True, help="NMR frequency (optional, default is 400.0 MHz)")
    args = parser.parse_args()
    main(args.file, uncertainty=args.uncertainty, extra=args.extra, frequency=args.frequency)

