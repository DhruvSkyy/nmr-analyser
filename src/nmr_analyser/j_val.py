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
    - If num_peaks > 36, returns ["multiplet"].
    - If num_peaks == 1, returns ["s"] (singlet).
    - If num_peaks <= 7:
        - Uses possible factors [2..7] with specific labels.
        - Generates all permutations of valid factorizations.
    - If num_peaks > 7:
        - Restricts factors to [2, 3, 4] with labels ("d", "t", "q").
        - Returns ["multiplet"] if no valid factorization is found.

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
    ['multiplet']
    >>> factorize_multiplicity(11)
    ['multiplet']
    """
    if num_peaks > 36:
        return ["multiplet"]
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

def match_peaks_to_multiplicity(peaks, multiplicities, frequency, uncertainty=1.0):
    """
    Match observed peak positions to one of several expected multiplicity patterns.

    The function calculates J values for each candidate multiplicity by calling 
    `calculate_j_vals(peaks, multiplicity, frequency)`, then compares the 
    resulting rank-pattern against `generate_j_pattern(multiplicity)` using 
    `cluster_and_rank_j_values`.

    Parameters
    ----------
    peaks : list of float
        Observed peak positions (in ppm).
    multiplicities : list of str
        List of candidate multiplicity patterns (e.g. ['d', 't', 'td', 'dt']).
    frequency : float
        Spectrometer frequency in MHz (for ppm->Hz conversion).
    uncertainty : float, optional
        Uncertainty threshold in Hz used by `cluster_and_rank_j_values` to
        decide if two J-values are 'the same'.

    Returns
    -------
    (matched_multiplicity, j_values_or_adj)
        matched_multiplicity : str
            - A single matched pattern (e.g., 'd', 'dt') if exactly one match is found.
            - 'multiplet' if none or multiple patterns match, or if the user specifically
              wants to label it as 'multiplet'.
        j_values_or_adj : list of float or None
            - If exactly one pattern matches, returns the J values (possibly averaged
              by rank) that correspond to that pattern.
            - If zero or multiple patterns match, returns the list of adjacent J values
              derived directly from consecutive peak differences in Hz, so you can 
              see the raw spacing. (Or returns None, if you prefer the old behaviour.)

    Notes
    -----
    - If the first item in `multiplicities` is 'multiplet', we immediately return
      ('multiplet', <adjacent_Js>).
    - Otherwise, we test each multiplicity pattern by:
        1. Calculating J values from the peaks.
        2. Clustering and ranking them with `cluster_and_rank_j_values`.
        3. Comparing that rank-pattern to `generate_j_pattern(multiplicity)`.
      If exactly one matches, we return it. If 0 or >1 match, we label 'multiplet'.
    """

    # -- 1) If the user has an explicit 'multiplet' or no pattern at all, short-circuit:
    if not multiplicities or multiplicities[0].lower() == 'multiplet':
        # Return 'multiplet' plus raw adjacent J-values (consecutive peak diffs)
        adjacent_js = _adjacent_j_values(peaks, frequency)
        return 'multiplet', '', adjacent_js

    # -- 2) For each candidate multiplicity, compute J-values and check rank pattern
    matched_patterns = []
    stored_j_vals_and_ranks = []  # store (multiplicity, j_vals, j_ranks) for debugging

    for m in multiplicities:
        # Calculate J-values for these peaks under pattern m
        j_vals = calculate_j_vals(peaks, m, frequency)

        # Cluster and rank them -> e.g. j_ranks = [1,2,2,1] for dd, ...
        j_ranks = cluster_and_rank_j_values(j_vals, uncertainty=uncertainty)

        # Compare to the "expected" pattern from generate_j_pattern(m)
        expected_pattern = generate_j_pattern(m)  # e.g. [1,2,2,1]
        
        if m == 'dt':
            print(j_vals)
            print(j_ranks)
            print(expected_pattern)
        if j_ranks == expected_pattern:
            matched_patterns.append(m)
            stored_j_vals_and_ranks.append((m, j_vals, j_ranks))
    
    # -- 3) Decide how to return
    if len(matched_patterns) == 1:
        # Exactly one match; return that multiplicity & the matched J-values
        # Optionally, you could average the J's by rank or just return them directly.
        mpat, j_vals, j_ranks = stored_j_vals_and_ranks[0]

        # If you want to average by rank (like the old code):
        unique_ranks = np.unique(j_ranks)
        # e.g. for a dd pattern [1,2,2,1], unique_ranks is [1,2].
        # We'll average all J's that have rank 1, then rank 2, etc.
        # That yields a smaller set: [J1, J2] or so.
        avg_j_by_rank = []
        for ur in unique_ranks:
            matching_js = [j for j, rk in zip(j_vals, j_ranks) if rk == ur]
            avg_j_by_rank.append(np.mean(matching_js))
        
        return mpat, avg_j_by_rank, j_vals

    # -- 4) If zero or multiple matches, label 'multiplet'
    # Return the adjacent J-values from the raw peaks so user can see raw spacing.
    return 'multiplet', '', _adjacent_j_values(peaks, frequency)


def _adjacent_j_values(peaks, frequency):
    """
    Helper function to convert consecutive peak differences (in ppm) into Hz.
    """
    sorted_peaks = np.sort(peaks)
    diffs_ppm = np.diff(sorted_peaks)
    diffs_hz = [d * frequency for d in diffs_ppm]
    return diffs_hz


def calculate_j_vals(peaks, multiplicity, frequency):
    peaks = np.sort(peaks)
    line_counts = multiplicity_to_line_count(multiplicity)

    if len(line_counts) == 1:
        adj_diffs_hz = np.diff(peaks) * frequency
        return adj_diffs_hz.tolist()

    line_counts_reversed = line_counts[::-1]
    current_peaks = peaks.copy()
    rank_pattern = generate_j_pattern(multiplicity)
    final_j_vals = [0] * len(rank_pattern)

    for iteration, n in enumerate(line_counts_reversed):
        rank_number = iteration + 1
        n_groups = len(current_peaks) // n
        groups = [current_peaks[i * n:(i + 1) * n] for i in range(n_groups)]

        j_vals_iteration = []
        for g in groups:
            diffs = np.diff(g) * frequency
            j_vals_iteration.extend(diffs.tolist())

        # Fill in the final_j_vals array at positions matching current rank number
        indices_to_fill = [i for i, rank in enumerate(rank_pattern) if rank == rank_number]
        for idx, val in zip(indices_to_fill, j_vals_iteration):
            final_j_vals[idx] = val

        current_peaks = np.array([np.mean(g) for g in groups])

    return final_j_vals

