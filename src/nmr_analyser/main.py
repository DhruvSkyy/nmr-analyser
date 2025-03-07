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
    calculate_j_values,
    match_j_values_to_multiplicity,
)
import os

def nmr_peak_analysis(peaks, k, uncertainty=1.0, extra=10):
    """Clusters NMR peaks, assigns functional groups, and classifies multiplets."""
    output_dir = "nmrdata_output"
    os.makedirs(output_dir, exist_ok=True)
    detailed_path = os.path.join(output_dir, "detailed_calculations.txt")
    simple_path = os.path.join(output_dir, "simple_calculations.txt")
    for path in [detailed_path, simple_path]:
        if os.path.exists(path):
            os.remove(path)

    max_k = min(k + extra, len(peaks))
    min_multiplets, best_k = float('inf'), k
    for current_k in range(k, max_k + 1):
        multiplet_count, clusters, cluster_data = evaluate_clusters(peaks, current_k, uncertainty)
        if multiplet_count < min_multiplets:
            min_multiplets, best_k = multiplet_count, current_k
            best_cluster_data = cluster_data
            best_clusters = clusters

    with open(detailed_path, "a") as f_detailed, open(simple_path, "a") as f_simple:
        for cluster_peaks, best_match, avg_j_values, assigned_groups, all_multiplicities, all_j_vals in best_cluster_data:
            write_detailed_output(f_detailed, cluster_peaks, all_j_vals, assigned_groups, all_multiplicities)
            write_simple_output(f_simple, cluster_peaks, best_match, avg_j_values, assigned_groups)

    print(f"Analysis complete. Files written to: {detailed_path}, {simple_path}")
    return best_k, best_clusters


def evaluate_clusters(peaks, k, uncertainty):
    labels, centroids = hierarchical_clustering(peaks.reshape(-1, 1), k)
    clusters = {i: [] for i in range(k)}
    cluster_data = []
    multiplet_count = 0
    for peak, label in zip(peaks, labels):
        clusters[label].append(peak)
    for cluster_peaks in clusters.values():
        cluster_peaks.sort()
        best_match, avg_j_values, assigned_groups, all_multiplicities, all_j_vals = process_cluster(cluster_peaks, uncertainty)
        cluster_data.append((cluster_peaks, best_match, avg_j_values, assigned_groups, all_multiplicities, all_j_vals))
        if best_match == 'multiplet':
            multiplet_count += 1
    return multiplet_count, clusters, cluster_data


def process_cluster(cluster_peaks, uncertainty, frequency=400.0):
    if len(cluster_peaks) == 1:
        return "singlet", [], [], [], []
    j_vals = calculate_j_values(cluster_peaks, frequency=frequency)
    all_mults = factorize_multiplicity(len(cluster_peaks))
    best_match, avg_j_values = match_j_values_to_multiplicity(j_vals, all_mults, uncertainty=uncertainty)
    assigned_groups = assign_functional_groups(cluster_peaks[0])
    return best_match, avg_j_values, assigned_groups, all_mults, j_vals


def write_detailed_output(f, peaks, j_vals, groups, all_multiplicities):
    peaks_str = ', '.join(f"{p:.2f}" for p in peaks)
    j_vals_str = ', '.join(f"{j:.5f}" for j in j_vals) if len(j_vals) > 0 else "None"
    groups_str = " , ".join(groups)
    mult_str = ', '.join(str(m) for m in all_multiplicities) if all_multiplicities else "None"
    f.write(f"Peaks: {peaks_str}\nJ-values: {j_vals_str}\nGroups: {groups_str}\nAll Multiplicities: {mult_str}\n\n")

def write_simple_output(f, peaks, best_match, j_vals, groups):
    """Writes simplified peak, multiplicity, and group info to file."""
    range_str = f"{min(peaks):.2f} - {max(peaks):.2f} ppm" if best_match == "multiplet" else f"{peaks[0]:.2f} ppm"
    j_vals_str = ', '.join(f"{j:.1f}" for j in j_vals) if j_vals else "None"
    groups_str = " , ".join(groups)
    f.write(f"Peaks: {range_str}\nMultiplicity: {best_match}\nJ-values: {j_vals_str}\nGroups: {groups_str}\n\n")


def main(path, uncertainty=1.0, extra=10, frequency=400.0):
    """Main function to perform NMR peak analysis with adjustable parameters."""
    x, y = load_csv_coordinates(path)
    noise_threshold = determine_noise_threshold(y)
    os.makedirs('nmrdata_output', exist_ok=True)

    # Peak Detection and Plotting
    while True:
        peak_x, peak_y = detect_and_plot_peaks(x, y, threshold_baseline=noise_threshold)
        print(f"Threshold: {noise_threshold}")
        if input("Is this threshold acceptable? (y/n): ").strip().lower() == 'y':
            break
        noise_threshold = float(input("Enter new noise threshold: "))

    # Clustering and Analysis
    k = calculate_cluster_count(peak_x)
    while True:
        k, clusters  = nmr_peak_analysis(peak_x, k, uncertainty=uncertainty, extra=extra)
        detect_and_plot_peaks(x, y, threshold_baseline=noise_threshold, clusters=clusters)
        print(f"Number of clusters: {k}")
        if input("Is this number of clusters acceptable? (y/n): ").strip().lower() == 'y':
            break
        k = int(input("Enter preferred number of clusters: "))
        extra = 1


def validate_file(arg):
    if Path(arg).is_file():
        return arg
    raise FileNotFoundError(arg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NMR Peak Analysis Tool")
    parser.add_argument("--file", type=validate_file, required=True, help="Input file path")
    parser.add_argument("--extra", type=int, default=10, help="Extra clusters to consider during analysis")
    parser.add_argument("--uncertainty", type=float, default=1.0, help="Uncertainty tolerance for multiplicity matching")
    parser.add_argument("--frequency", type=float, default=400.0, help="NMR frequency (optional, default is 400.0 MHz)")
    args = parser.parse_args()
    main(args.file, uncertainty=args.uncertainty, extra=args.extra, frequency=args.frequency)

