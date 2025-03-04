import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

# Define empirical 1H chemical shift ranges
# https://www.ucl.ac.uk/nmr/sites/nmr/files/L2_3_web.pdf
CHEMICAL_SHIFT_RANGES = {
    "Aldehyde": (9.5, 10.5),
    "Aromatic": (6.5, 8.2),
    "Alkene": (4.5, 6.1),
    "Alkyne": (2.0, 3.2),
    "Acetal": (4.5, 6.0),
    "Alkoxy": (3.4, 4.8),
    "N-Methyl": (3.0, 3.5),
    "Methoxy": (3.3, 3.8),
    "Methyl": (0.9, 1.0),
    "Methyl (double bond/aromatic)": (1.8, 2.5),
    "Methyl (CO-CH3)": (1.8, 2.7),
    "Methylene (CH2-O)": (3.6, 4.7),
    "Methylene (CH2-R1R2)": (1.3, 1.4),
    "Methine": (1.5, 1.6),
    "Cyclopropane": (0.22, 0.25),
    "TMS (Reference)": (0.0, 0.0),
    "Metal Hydride": (-5, -20)
}

# Adapted from https://stanford.edu/~cpiech/cs221/handouts/kmeans.html
# tolerance from https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
def kmeans(data, k, max_iter=300, tol=1e-4, random_state=42):
    """
    Implements K-Means clustering. 
    
    Parameters:
        data (numpy array): Data points to be clustered.
        k (int): Number of clusters.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.
        random_state (int): Seed for random initialisation.
    
    Returns:
        tuple: Cluster labels and centroids as numpy arrays.
    """ 
    np.random.seed(random_state)
    centroids = data[np.random.choice(data.shape[0], k, replace=False)].astype(float)
    for _ in range(max_iter):
        distances = np.linalg.norm(data[:, None] - centroids, axis=2)
        labels = distances.argmin(axis=1)
        old_centroids = centroids.copy()
        for i in range(k):
            pts = data[labels == i]
            centroids[i] = pts.mean(axis=0) if len(pts) else data[np.random.randint(0, data.shape[0])]
        if np.linalg.norm(centroids - old_centroids) < tol:
            break
    return labels, centroids

def hierarchical_clustering(peaks, k, linkage='ward'):
    """
    Clusters NMR peaks using hierarchical (agglomerative) clustering,
    assigns functional groups, and classifies multiplets.
    
    Parameters:
        peaks (numpy array): Array of peak positions (1D).
        k (int): Number of clusters to find.

    Returns:
        None: Prints the analysis results.
    """
    data = peaks.reshape(-1, 1)
    
    hc = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = hc.fit_predict(data)
    
    cluster_dict = {i: [] for i in range(k)}
    for peak, lbl in zip(peaks, labels):
        cluster_dict[lbl].append(peak)
    
    clusters = [np.mean(val) for key, val in cluster_dict.items()]
    
    return labels, clusters

def getcsvcoords(path):
    """
    Reads a CSV file and extracts x and y coordinates.
    
    Parameters:
        path (str): Path to the CSV file.
    
    Returns:
        Two numpy arrays representing x and y values.
    """
    x, y = [], []
    with open(path) as file:
        next(file)  # Skip header
        for line in file: 
            parts = line.split()
            x.append(float(parts[0]))
            y.append(float(parts[1]))
    return np.array(x), np.array(y)

def find_noise_threshold(y_values, k=2):
    """
    Uses K-Means clustering to find a threshold separating noise from signal in peak heights.

    Parameters:
        y_values (array-like): 1D array of y-values (peak heights).
        k (int): Number of clusters (default is 2: noise vs signal).

    Returns:
        float: Threshold value separating noise from peaks.
    """
    if len(y_values) < 2:
        print("Need at least two data points to apply clustering.")
        return None

    y_values = np.array(y_values).reshape(-1, 1)
    labels, centroids = kmeans(y_values, k)
    sorted_centroids = np.sort(centroids.ravel())
    threshold = sorted_centroids[0]

    return threshold

def find_peaks(y, height=None, distance=1, window_size=10):
    """
    Identifies peaks in a 1D data array using local maxima detection.
    
    Parameters:
        y (list or numpy array): Input data.
        height (float, optional): Minimum peak height.
        distance (int, optional): Minimum separation between peaks.
        window_size (int, optional): Smoothing window size.
    
    Returns:
        tuple: List of peak indices and smoothed data.
    """
    y = np.array(y, dtype=float)

    if height is None:
        height = np.mean(y)

    peaks = [i for i in range(1, len(y) - 1)
             if y[i] > height and y[i] > y[i - 1] and y[i] > y[i + 1]]
    
    return peaks

def detect_and_plot_peaks(x, y, threshold_baseline=5000000):
    """
    Detects and plots peaks in the given data.
    
    Parameters:
        x (array-like): X-axis data.
        y (array-like): Y-axis data.
        threshold_baseline (float): Minimum height for peak detection.
    
    Returns:
        tuple: Arrays of peak x and y values.
    """
    x = np.array(x)
    y = np.array(y)
    xhigh = max(x)
    xlow = min(x)
    peaks = find_peaks(y, threshold_baseline)
    peak_x = x[peaks]
    peak_y = y[peaks]
    
    # Plot the data
    plt.figure(figsize=(15, 5))
    plt.plot(x, y, marker="o", linestyle="-", markersize=1, label="Data")
    plt.scatter(peak_x, peak_y, color="red", marker="x", label="Peaks")  # Highlight peaks
    plt.axhline(y=threshold_baseline, color='gray', linestyle="--", label="Threshold")  # Threshold line
    
    # Adjust x limits
    plt.xlim(xhigh, xlow)

    # Labels and title
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Plot of Data with Peak Detection")
    plt.legend()
    
    plt.show(block=False)  # Non-blocking plot
    plt.savefig('nmrdata_output/output.jpg')
    
    return peak_x, peak_y

def get_kmeans_distance_threshold(peaks):
    """
    Computes a distance threshold for peak clustering using K-Means.
    
    Parameters:
        peaks (numpy array): Sorted array of peak positions.
    
    Returns:
        tuple: Computed threshold and count of distances above the threshold.
    """
    peaks = np.sort(np.asarray(peaks, dtype=float))
    if len(peaks) < 2:
        print("Need at least two peaks to measure distances.")
        return None, 0

    distances = np.diff(peaks)
    if len(distances) == 0:
        print("Not enough distances to compute a threshold.")
        return None, 0

    labels, centroids = kmeans(distances.reshape(-1, 1), k=2)
    threshold = np.min(centroids)

    count = np.sum(distances > threshold)
    return threshold, count

def assign_functional_groups(peak):
    """
    Determines possible functional groups based on a given chemical shift.
    
    Parameters:
        peak (float): Chemical shift value.
    
    Returns:
        list: Possible functional groups.
    """
    possible_groups = [group for group, (low, high) in CHEMICAL_SHIFT_RANGES.items() if low <= peak <= high]
    return possible_groups if possible_groups else ["Unassigned"]

def factorize_multiplicity(num_peaks):
    """
    Returns all possible multiplicities for 'num_peaks':

      - If num_peaks > 36, returns ["multiple"].
      - If num_peaks == 1, returns ["s"] (singlet).

      - If num_peaks <= 7:
           * Uses possible factors [2..7].
           * Includes single-factor labels for 2..7:
               2 -> "d"
               3 -> "t"
               4 -> "q"
               5 -> "quintet"
               6 -> "hex"
               7 -> "hept"
           * Generates permutations of any factorisations.

      - If num_peaks > 7:
           * Only uses factors [2, 3, 4] (mapped to "d", "t", "q").
           * If no factorisation is possible using these factors, returns ["multiple"].
           * No single-factor labels for numbers > 7 (e.g. "8" doesn't map to anything).

    Examples:
      factorize_multiplicity(4)  -> ["dd", "q"]
      factorize_multiplicity(6)  -> ["dt", "td", "hex"]  (since 6 <= 7)
      factorize_multiplicity(8)  -> ["ddd", "dq", "qd"]  (only 2,3,4 factors)
      factorize_multiplicity(1)  -> ["s"]
      factorize_multiplicity(40) -> ["multiple"] (because >36)
      factorize_multiplicity(11) -> ["multiple"] (cannot be built from 2,3,4)
    """
    if num_peaks > 36:
        return ["multiple"]
    if num_peaks == 1:
        return ["s"]

    # Factor label maps
    FACTORS_UP_TO_7 = {
        2: "d",
        3: "t",
        4: "q",
        5: "quintet",
        6: "hex",
        7: "hept"
    }
    FACTORS_ABOVE_7 = {
        2: "d",
        3: "t",
        4: "q"
    }

    # Decide which factor dictionary and factor range to use
    if num_peaks <= 7:
        factor_dict = FACTORS_UP_TO_7
        factor_range = range(2, 8)  # 2..7 inclusive
    else:
        factor_dict = FACTORS_ABOVE_7
        factor_range = range(2, 5)  # 2..4 inclusive

    # Collect all factor sequences
    results = []

    def backtrack(current, path):
        if current == 1:
            results.append(path[:])
            return
        for f in factor_range:
            if current % f == 0:
                path.append(f)
                backtrack(current // f, path)
                path.pop()

    # Perform backtracking
    backtrack(num_peaks, [])

    # Build label strings from the factor sequences
    labels = set()
    for seq in results:
        label_str = "".join(factor_dict[f] for f in seq)
        labels.add(label_str)

    # For numbers <= 7, also include the single-factor label if it exists
    # (e.g. 6 -> "hex") – but only if num_peaks <= 7
    if num_peaks <= 7 and num_peaks in factor_dict:
        labels.add(factor_dict[num_peaks])

    # If we got no labels at all, return ["multiple"]
    final_labels = sorted(labels)
    return final_labels if final_labels else ["multiplet"]


def compute_j_values(peaks, freq=400.0):
    """
    Calculate consecutive J values for the sorted 'peaks'.
    J = difference_in_ppm * freq (in Hz).
    Returns a list of J values in Hz (floats).
    """
    if len(peaks) < 2:
        return []
    sorted_peaks = sorted(peaks)
    diffs_ppm = np.diff(sorted_peaks)
    j_vals = diffs_ppm * freq
    return j_vals

def letter_to_lines(item):
    """Map 'd' -> 2, 't' -> 3, 'q' -> 4, etc., including full terms like 'quintet' or 'hept'."""
    lookup = {'d': 2, 't': 3, 'q': 4, 'quintet': 5, 'hex': 6, 'hept': 7}
    # If the item is a full word (e.g., 'quintet'), return the value directly
    if item.lower() in lookup:
        return [lookup[item.lower()]]
    # Otherwise, handle as individual letters
    return [lookup.get(char.lower()) for char in item]


def build_splitting_path(sizes):
    """
    Recursively build a 'splitting tree' path:
      - If there's 1 dimension, just yield (0), (1), (2), ...
      - If multiple dimensions, 'snake' over dimension 0
        and recursively generate the sub-dimensions.
    This ensures we only change one index at a time.
    """
    if not sizes:
        return  # no dimensions at all
    if len(sizes) == 1:
        for i in range(sizes[0]):
            yield (i,)
    else:
        first = sizes[0]
        rest = sizes[1:]
        subpath = list(build_splitting_path(rest))
        rev_subpath = subpath[::-1]
        for i in range(first):
            if i % 2 == 0:
                for combo in subpath:
                    yield (i,) + combo
            else:
                for combo in rev_subpath:
                    yield (i,) + combo

def generate_j_pattern(multiplicity):
    """
    Given a multiplicity like 'td', 'hept', etc.,
    return the list of ranks (largest coupling = k, smallest = 1)
    that describe the order we flip each coupling.
    """
    sizes = letter_to_lines(multiplicity)
    k = len(sizes)
    path = list(build_splitting_path(sizes))
    ranks = []
    for p1, p2 in zip(path, path[1:]):
        for dim in range(k):
            if p1[dim] != p2[dim]:
                ranks.append(k - dim)
                break
    return ranks

def rank_with_uncertainty_dbscan(j_values, uncertainty=1):
    """
    Use DBSCAN to rank j_values based on clustering within a given uncertainty.
    """
    
    j_values_reshaped = np.array(j_values).reshape(-1, 1)
    clustering = DBSCAN(eps=uncertainty, min_samples=1).fit(j_values_reshaped)
    labels = clustering.labels_
    unique_labels = np.unique(labels)
    sorted_clusters = sorted(unique_labels, key=lambda lbl: np.mean(j_values_reshaped[labels == lbl]))
    rank_map = {label: rank + 1 for rank, label in enumerate(sorted_clusters)}
    ranks = [rank_map[label] for label in labels]
    return ranks

def match_j_values_to_multiplicity(j_values, multiplicities, uncertainty=1):
    """
    Given a list of j_values and multiplicities, match the j_values
    to the expected multiplicity patterns or output 'multiple' if ambiguous.
    Additionally, return averaged J values for specific patterns.
    """
    if multiplicities[0] == 'multiplet':
        return 'multiplet', None
    
    # Generate ranks from j_values using the DBSCAN method
    j_ranks = rank_with_uncertainty_dbscan(j_values, uncertainty)
    
    matches = []
    for multiplicity in multiplicities:
        pattern = generate_j_pattern(multiplicity)
        if j_ranks == pattern:
            matches.append(multiplicity)
    
    if len(matches) == 1:
        matched_multiplicity = matches[0]
        if matched_multiplicity != 'multiplet':
            # Calculate averaged J values based on the pattern
            unique_ranks = np.unique(j_ranks)
            avg_j_values = [np.mean([j for j, rank in zip(j_values, j_ranks) if rank == ur]) for ur in unique_ranks]
            return matched_multiplicity, avg_j_values
        else:
            return matched_multiplicity, None
    else:
        return 'multiplet', None
