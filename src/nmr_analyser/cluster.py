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
    # maybe try use ward clustering over k means, i tried k means works same or better. 
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
