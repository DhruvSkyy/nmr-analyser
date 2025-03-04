import os
import numpy as np
import argparse
from pathlib import Path

from helper import (
    hierarchical_clustering,
    factorize_multiplicity,
    compute_j_values,
    match_j_values_to_multiplicity,
    assign_functional_groups,
    getcsvcoords,
    find_noise_threshold,
    detect_and_plot_peaks,
    get_kmeans_distance_threshold,
)

def nmr_peak_analysis(peaks, k, uncertainty=1.0):
    """
    Clusters NMR peaks (using 'hierarchical_clustering'),
    assigns functional groups, and classifies multiplets
    by determining the best matching multiplicity using J-value ranking.
    If the multiplicity is ambiguous, 'multiple' will be shown.

    Generates TWO files:
      1) nmrdata_output/detailed_calculations.txt
         - Shifts with 2 decimals
         - All J-values with 5 decimals
         - All possible multiplicities
      2) nmrdata_output/simple_calculations.txt
         - Average shift with 2 decimals
         - J-values as integers if a single match
         - Range of ppm values if "multiple"
         - Best matched or "multiple" multiplicity
    """

    labels, centroids = hierarchical_clustering(peaks.reshape(-1, 1), k)

    # Prepare dict for each cluster
    cluster_dict = {i: [] for i in range(k)}
    for peak, label in zip(peaks, labels):
        cluster_dict[label].append(peak)

    # Sort clusters by their minimum peak value (lowest ppm first)
    sorted_clusters = sorted(cluster_dict.items(),
                             key=lambda item: min(item[1]) if item[1] else float('inf'))

    # Ensure output folder
    os.makedirs("nmrdata_output", exist_ok=True)

    # File paths
    detailed_path = "nmrdata_output/detailed_calculations.txt"
    simple_path = "nmrdata_output/simple_calculations.txt"

    # Remove files if they exist
    for path in [detailed_path, simple_path]:
        if os.path.exists(path):
            os.remove(path)

    # Open both files in append mode
    with open(detailed_path, "a") as f_detailed, open(simple_path, "a") as f_simple:
        print("NMR Peak Analysis:\n")
        
        for cluster, cluster_peaks in sorted_clusters:
            if not cluster_peaks:
                continue
            cluster_peaks.sort()

            n_peaks = len(cluster_peaks)
            if n_peaks == 1:
                best_match = "singlet"
                avg_j_values = []
            else:
                all_mults = factorize_multiplicity(n_peaks)
                j_vals = compute_j_values(cluster_peaks, freq=400.0)
                best_match, avg_j_values = match_j_values_to_multiplicity(j_vals, all_mults, uncertainty=uncertainty)

            avg_shift = np.mean(cluster_peaks) if n_peaks > 0 else 0.0
            representative_shift = centroids[cluster]
            assigned_groups = assign_functional_groups(representative_shift)
            assigned_groups_str = " , ".join(assigned_groups)

            # DETAILED file output
            cluster_peaks_str = ", ".join(f"{p:.2f}" for p in cluster_peaks)
            j_vals_str = ", ".join(f"{j:.5f}" for j in avg_j_values) if avg_j_values else "None"

            f_detailed.write(f"Cluster {cluster}\n")
            f_detailed.write(f"  Peaks (ppm, 2dp): {cluster_peaks_str}\n")
            f_detailed.write(f"  J-values (Hz, 5dp): {j_vals_str}\n")
            f_detailed.write(f"  multiplicity: {all_mults}\n")
            f_detailed.write(f"  Assigned groups: {assigned_groups_str}\n\n")

            # SIMPLE file output
            if best_match == "multiple":
                range_str = f"{min(cluster_peaks):.2f} - {max(cluster_peaks):.2f} ppm"
                j_vals_str = "None"
            else:
                range_str = f"{avg_shift:.2f} ppm"
                j_vals_str = ", ".join(f"{j:.1f}" for j in avg_j_values) if avg_j_values else "None"

            f_simple.write(f"Cluster {cluster}\n")
            f_simple.write(f"  Average shift: {range_str}\n")
            f_simple.write(f"  J-values (Hz, int): {j_vals_str}\n")
            f_simple.write(f"  Best multiplicity: {best_match}\n")
            f_simple.write(f"  Assigned groups: {assigned_groups_str}\n\n")

            print(f"{cluster_peaks_str} ppm: {best_match}, {assigned_groups_str}")

        print("\nAnalysis complete. Files written to:")
        print(f"  {detailed_path}")
        print(f"  {simple_path}")


def main(path):
    """
    Allows user to iteratively adjust the noise threshold and cluster count for analysis.
    Save data to a nmrdata_output folder. 
    """
    x, y = getcsvcoords(path)
    noise_threshold = find_noise_threshold(y)
    
    folder_path = 'nmrdata_output'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    # Peak Detection and Plotting
    while True:
        peak_x, peak_y = detect_and_plot_peaks(x, y, threshold_baseline=noise_threshold)
        print(f"Threshold: {noise_threshold}", noise_threshold)
        user_input = input("Is this threshold acceptable? (y/n): ").strip().lower()
        if user_input == 'y':
            break
        noise_threshold = float(input("Enter new noise threshold: "))
    
    threshold, count = get_kmeans_distance_threshold(peak_x)
    # Clustering Peak Analysis
    while True:
        nmr_peak_analysis(peak_x, count)
        print(f"Number of clusters: {count}")
        user_input = input("Is this number of clusters acceptable? (y/n): ").strip().lower()
        if user_input == 'y':
            break
        count = int(input("Enter preferred number of clusters: "))

def validate_file(arg):
    if (Path(arg)).is_file():
        return arg
    else:
        raise FileNotFoundError(arg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=validate_file, help="Input file path", required=True)
    args = parser.parse_args()
    main(args.file)
