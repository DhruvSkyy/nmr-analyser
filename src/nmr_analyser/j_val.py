import numpy as np
from sklearn.cluster import DBSCAN

def factorize_multiplicity(num_peaks: int) -> list[str]:
    """
    Algorithm to generate all possible multiplicity labels for a given number of peaks using backtracking.
    
    Parameters
    ----------
    num_peaks : int
        The number of peaks to factorize.

    Returns
    -------
    list of str
        A sorted list of possible multiplicity labels or ["multiple"] if no valid factorisation exists.
        
    Algorithm
    ---------
    This function uses a backtracking algorithm to explore all possible factorizations of `num_peaks` 
    using specific factor ranges and maps these factorizations to their respective labels. 

    Rules
    -----
    - If num_peaks > 36, returns ["multiple"].
    - If num_peaks == 1, returns ["s"] (singlet).
    - If num_peaks <= 7:
        - Uses possible factors [2..7] with specific labels.
        - Generates all permutations of valid factorizations.
    - If num_peaks > 7:
        - Restricts factors to [2, 3, 4] with labels ("d", "t", "q").
        - Returns ["multiple"] if no valid factorization is found.

    Examples
    --------
    >>> factorize_multiplicity(4)
    ['dd', 'q']
    >>> factorize_multiplicity(6)
    ['dt', 'hex', 'td']
    >>> factorize_multiplicity(8)
    ['ddd', 'dq', 'qd']
    >>> factorize_multiplicity(1)
    ['s']
    >>> factorize_multiplicity(40)
    ['multiple']
    >>> factorize_multiplicity(11)
    ['multiple']
    """
    if num_peaks > 36:
        return ["multiple"]
    if num_peaks == 1:
        return ["s"]

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

    if num_peaks <= 7:
        factor_dict = FACTORS_UP_TO_7
        factor_range = range(2, 8)
    else:
        factor_dict = FACTORS_ABOVE_7
        factor_range = range(2, 5)

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

    backtrack(num_peaks, [])

    labels = set()
    for seq in results:
        label_str = "".join(factor_dict[f] for f in seq)
        labels.add(label_str)

    if num_peaks <= 7 and num_peaks in factor_dict:
        labels.add(factor_dict[num_peaks])

    final_labels = sorted(labels)
    
    return final_labels if final_labels else ["multiplet"]

# Allow frequency for calculating j values to be changed asap - from argparse as well 
def calculate_j_values(peaks, frequency=400.0):
    """
    Calculate J coupling values in Hertz (Hz) from a list of chemical shift peaks.

    Parameters
    ----------
    peaks : list of float
        Chemical shift peaks in parts per million (ppm).
    frequency : float, optional
        Spectrometer frequency in MHz, default is 400.0 MHz.

    Returns
    -------
    list of float
        J values in Hz, calculated as the difference between consecutive peaks multiplied by the frequency.

    Examples
    --------
    >>> calculate_j_values([1.0, 2.0, 2.5], frequency=400.0)
    [400.0, 200.0]

    """
    if len(peaks) < 2:
        return []
    sorted_peaks = sorted(peaks)
    diffs_ppm = np.diff(sorted_peaks)
    j_values = diffs_ppm * frequency
    return j_values


def multiplicity_to_line_count(multiplicity):
    """
    Convert multiplicity notation ('d', 't', 'q', 'quintet', 'hept') to the corresponding line counts.

    Parameters
    ----------
    multiplicity : str
        Multiplicity indicator (e.g., 'd', 't', 'quintet', 'hept').

    Returns
    -------
    list of int
        Line counts for each component of the multiplicity.

    Examples
    --------
    >>> multiplicity_to_line_count('td')
    [3, 2]

    """
    lookup = {'d': 2, 't': 3, 'q': 4, 'quintet': 5, 'hex': 6, 'hept': 7}
    if multiplicity.lower() in lookup:
        return [lookup[multiplicity.lower()]]
    return [lookup.get(char.lower()) for char in multiplicity]


def generate_splitting_path(dimensions):
    """
    Recursively generate the splitting path for given multiplicity dimensions.

    Parameters
    ----------
    dimensions : list of int
        Number of lines in each splitting dimension (e.g., [2, 3] for a doublet-triplet).

    Yields
    ------
    tuple of int
        Indexes representing the splitting pattern path.

    Algorithm:
    ----------
    1. Generate the path for the innermost dimension as sequential indices.
    2. For each additional dimension:
       - Combine each index of the current dimension with all indices of the existing path.
       - Alternate the direction of the index combination for every row.
       - Append the resulting combinations to the path.
    3. The final path maintains a logical order where each step only changes a single index.

    Example:
    --------
    For dimensions = [2, 3] (representing a doublet-triplet):

    list(generate_splitting_path([2, 3]))

    Output:
    [(0, 0), (0, 1), (0, 2), 
     (1, 2), (1, 1), (1, 0)]
    """
    if not dimensions:
        return
    if len(dimensions) == 1:
        for i in range(dimensions[0]):
            yield (i,)
    else:
        first_dim = dimensions[0]
        remaining_dims = dimensions[1:]
        subpath = list(generate_splitting_path(remaining_dims))
        reversed_subpath = subpath[::-1]
        for i in range(first_dim):
            if i % 2 == 0:
                for combo in subpath:
                    yield (i,) + combo
            else:
                for combo in reversed_subpath:
                    yield (i,) + combo


def generate_j_pattern(multiplicity):
    """
    Generate the expected rank order of J values for a given multiplicity pattern.

    This function translates a multiplicity string (e.g., 'dt' for doublet-triplet) 
    into line counts using `multiplicity_to_line_count` and generates the splitting 
    path using `generate_splitting_path`. It then evaluates which dimension changes 
    at each step to assign ranks, with higher ranks assigned to higher dimension changes.

    Parameters
    ----------
    multiplicity : str
        Multiplicity pattern (e.g., 'td', 'hept', etc.).

    Returns
    -------
    list of int
        Rank order of J values from largest to smallest.

    Examples
    --------
    >>> generate_j_pattern('dt')
    [1, 1, 2, 1, 1]

    Notes
    -----
    The function generates the splitting path as:
    
    [(0, 0), (0, 1), (0, 2), 
     (1, 2), (1, 1), (1, 0)]

    Rank Assignment:
    - (0, 0) to (0, 1): Second dimension changes → Rank 1
    - (0, 1) to (0, 2): Second dimension changes → Rank 1
    - (0, 2) to (1, 2): First dimension changes → Rank 2
    - (1, 2) to (1, 1): Second dimension changes → Rank 1
    - (1, 1) to (1, 0): Second dimension changes → Rank 1

    Resulting in the rank order:
    [1, 1, 2, 1, 1]
    """

    sizes = multiplicity_to_line_count(multiplicity)
    dimension_count = len(sizes)
    path = list(generate_splitting_path(sizes))
    ranks = []
    for p1, p2 in zip(path, path[1:]):
        for dim in range(dimension_count):
            if p1[dim] != p2[dim]:
                ranks.append(dimension_count - dim)
                break
    return ranks

def cluster_and_rank_j_values(j_values, uncertainty=1.0):
    """
    Cluster J values based on uncertainty and output a ranked list of clusters.

    Parameters
    ----------
    j_values : list of float
        J coupling values in Hz.
    uncertainty : float, optional
        Clustering threshold in Hz, default is 1.0 Hz.

    Returns
    -------
    list of int
        Ranked J values based on cluster ordering.

    Examples
    --------
    >>> cluster_and_rank_j_values([7.1, 7.1, 9], uncertainty=1)
    [1, 1, 2]

    Notes
    -----
    Clusters J values using DBSCAN with the specified uncertainty as the clustering radius.
    Outputs a ranked list of clusters based on ascending mean J values.
    """
    j_values_reshaped = np.array(j_values).reshape(-1, 1)
    clustering = DBSCAN(eps=uncertainty, min_samples=1).fit(j_values_reshaped)
    labels = clustering.labels_
    unique_labels = np.unique(labels)
    sorted_clusters = sorted(unique_labels, key=lambda lbl: np.mean(j_values_reshaped[labels == lbl]))
    rank_map = {label: rank + 1 for rank, label in enumerate(sorted_clusters)}
    return [rank_map[label] for label in labels]


def match_j_values_to_multiplicity(j_values, multiplicities, uncertainty=1.0):
    """
    Match observed J values to expected multiplicity patterns.

    Parameters
    ----------
    j_values : list of float
        Observed J values in Hz.
    multiplicities : list of str
        Expected multiplicity patterns (e.g., ['d', 't', 'q']).
    uncertainty : float, optional
        Uncertainty threshold in Hz for J value clustering, default is 1.0 Hz.

    Returns
    -------
    tuple (str, list of float or None)
        Matched multiplicity and averaged J values, or 'multiplet' if ambiguous.

    Notes
    -----
    Uses `cluster_and_rank_j_values` and `generate_j_pattern` to compare observed and expected patterns.
    If multiple patterns match, 'multiplet' is returned. Averaged J values are provided if a match is unique.
    """
    if multiplicities[0] == 'multiplet':
        return 'multiplet', None
    j_ranks = cluster_and_rank_j_values(j_values, uncertainty)
    matches = []
    for multiplicity in multiplicities:
        pattern = generate_j_pattern(multiplicity)
        if j_ranks == pattern:
            matches.append(multiplicity)
    if len(matches) == 1:
        matched_multiplicity = matches[0]
        unique_ranks = np.unique(j_ranks)
        avg_j_values = [np.mean([j for j, rank in zip(j_values, j_ranks) if rank == ur]) for ur in unique_ranks]
        return matched_multiplicity, avg_j_values
    return 'multiplet', None

