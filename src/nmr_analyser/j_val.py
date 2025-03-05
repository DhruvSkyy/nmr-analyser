import numpy as np
from sklearn.cluster import DBSCAN

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
      factorize_multiplicity(6)  -> ["dt", "td", "hex"]
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
