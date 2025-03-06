import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

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
# tolerance/ maxiter from https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
def kmeans_clustering(data, k, max_iter=300, tol=1e-4, random_state=42):
    """
    Perform K-Means clustering on the given dataset.
    
    Parameters
    ----------
    data : numpy.ndarray
        Data points to be clustered, shape (n_samples, n_features).
    k : int
        Number of clusters to form.
    max_iter : int, optional
        Maximum number of iterations of the k-means algorithm, by default 300.
    tol : float, optional
        Tolerance to declare convergence, by default 1e-4.
    random_state : int, optional
        Random seed for reproducibility, by default 42.
    
    Returns
    -------
    labels : numpy.ndarray
        Cluster labels for each data point.
    centroids : numpy.ndarray
        Coordinates of cluster centroids.
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

def hierarchical_clustering(peaks, n_clusters, linkage='ward'):
    """
    Perform hierarchical (agglomerative) clustering on NMR peaks.
    
    Parameters
    ----------
    peaks : numpy.ndarray
        Array of peak positions (1D).
    n_clusters : int
        Number of clusters to find.
    linkage : str, optional
        Linkage criterion ('ward', 'complete', 'average', 'single'), by default 'ward'.

    Returns
    -------
    labels : numpy.ndarray
        Cluster labels for each peak.
    centroids : numpy.ndarray
        Mean position of each cluster.
    """
    data = peaks.reshape(-1, 1)
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = hc.fit_predict(data)
    cluster_dict = {i: [] for i in range(n_clusters)}
    for peak, lbl in zip(peaks, labels):
        cluster_dict[lbl].append(peak)
    centroids = np.array([np.mean(val) for val in cluster_dict.values()])
    return labels, centroids


def load_csv_coordinates(file_path):
    """
    Load x and y coordinates from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    x : numpy.ndarray
        X coordinates.
    y : numpy.ndarray
        Y coordinates.
    """
    x, y = [], []
    with open(file_path) as file:
        next(file)  # Skip header
        for line in file: 
            parts = line.split()
            x.append(float(parts[0]))
            y.append(float(parts[1]))
    return np.array(x), np.array(y)

def determine_noise_threshold(y_values, n_clusters=2):
    """
    Calculate a threshold to differentiate noise from signals using K-Means clustering.

    Parameters
    ----------
    y_values : array-like
        1D array of y-values (peak heights).
    n_clusters : int, optional
        Number of clusters to form, by default 2 (noise vs signal).

    Returns
    -------
    float
        Threshold value separating noise from peaks.
    """
    if len(y_values) < 2:
        print("Need at least two data points to apply clustering.")
        return None
    y_values = np.array(y_values).reshape(-1, 1)
    labels, centroids = kmeans_clustering(y_values, n_clusters)
    threshold = np.sort(centroids.ravel())[0]
    return threshold

def find_peaks(y, height):
    """
    Identify peaks in a 1D data array using local maxima detection.
    
    Parameters
    ----------
    y : numpy.ndarray
        Input data array.
    min_height : float
        Minimum height for peak detection.

    Returns
    -------
    list
        Indices of detected peaks.
    """

    peaks = [i for i in range(1, len(y) - 1)
             if y[i] > height and y[i] > y[i - 1] and y[i] > y[i + 1]]
    
    return peaks

def detect_and_plot_peaks(x, y, threshold_baseline, xlim=None, clusters=None):
    """
    Detects and plots peaks in the given data with a wide, detailed NMR spectrum style.
    Optionally highlights peaks in different colours based on clusters.
    
    Parameters
    ----------
    x : numpy.ndarray
        X-axis data (Chemical shift in ppm).
    y : numpy.ndarray
        Y-axis data (Intensity).
    threshold_baseline : float
        Minimum height for peak detection.
    xlim : tuple of (float, float), optional
        X-axis limits as (min, max). Defaults to the full range of x data.
    clusters : dict, optional
        A dictionary where keys are cluster labels and values are lists of peak values.
        Peaks in different clusters are plotted in different colours.

    Returns
    -------
    peak_x : numpy.ndarray
        X coordinates of detected peaks.
    peak_y : numpy.ndarray
        Y coordinates of detected peaks.
    """
    xhigh, xlow = max(x), min(x)
    peaks = find_peaks(y, height=threshold_baseline)
    peak_x, peak_y = x[peaks], y[peaks]

    plt.figure(figsize=(12, 8))
    plt.plot(x, y, linestyle="-", linewidth=0.8, color="black", label="NMR Spectrum")
    plt.axhline(y=threshold_baseline, color='gray', linestyle="--", linewidth=1, label="Threshold")
    
    # Default colour map for clusters
    colours = plt.cm.get_cmap('tab10', len(clusters) if clusters else 1)
    
    if clusters:
        for i, (label, peak_values) in enumerate(clusters.items()):
            mask = np.isin(peak_x, peak_values)
            cluster_x = peak_x[mask]
            cluster_y = peak_y[mask]
            plt.scatter(cluster_x, cluster_y, color=colours(i), marker="x", s=50)
    else:
        plt.scatter(peak_x, peak_y, color="red", marker="x", s=50, label="Detected Peaks")

    if xlim:
        plt.xlim(xlim[1], xlim[0]) 
    else:
        plt.xlim(xhigh, xlow)

    plt.xlabel("Chemical Shift (ppm)", fontsize=12)
    plt.ylabel("Intensity (a.u.)", fontsize=12)
    plt.title("1H NMR Spectrum", fontsize=14)
    if not clusters:
        plt.legend(loc="upper right", fontsize=10)

    plt.savefig('nmrdata_output/compact_nmr_output.jpg', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    
    return peak_x, peak_y

def calculate_cluster_count(peaks):
    """
    Estimate the number of clusters in peak data using K-Means on peak distances.
    
    Parameters
    ----------
    peaks : numpy.ndarray
        Sorted array of peak positions.

    Returns
    -------
    count: int
        Estimated count of clusters based on distance thresholding.
    """
    peaks = np.sort(np.asarray(peaks, dtype=float))
    if len(peaks) < 2:
        print("Need at least two peaks to measure distances.")
        return 0
    distances = np.diff(peaks)
    labels, centroids = kmeans_clustering(distances.reshape(-1, 1), k=2)
    threshold = np.min(centroids)
    count = np.sum(distances > threshold)
    
    return int(count)

def assign_functional_groups(peak):
    """
    Determine possible functional groups based on a chemical shift value.

    Parameters
    ----------
    peak : float
        Chemical shift value to classify.

    Returns
    -------
    list
        List of possible functional groups corresponding to the chemical shift.
        Returns ["Unassigned"] if no match is found.
    """
    possible_groups = [group for group, (low, high) in CHEMICAL_SHIFT_RANGES.items() if low <= peak <= high]
    return possible_groups if possible_groups else ["Unassigned"]

