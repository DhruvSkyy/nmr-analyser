import numpy as np

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

import numpy as np

def calculate_j_vals(peaks, multiplicity, frequency):
    """
    Calculate J values from a list of peak positions and a given multiplicity.

    Parameters
    ----------
    peaks : array-like
        Observed peak positions (e.g. in ppm), assumed to be the lines
        belonging to the multiplet.
    multiplicity : str
        Multiplicity notation (e.g. 'd', 't', 'dt', 'td', 'ddd', etc.).
    frequency : float
        Spectrometer frequency in MHz. Used to convert ppm differences to Hz.

    Returns
    -------
    list of float
        A list of J values in an order that matches the rank pattern from
        `generate_j_pattern(multiplicity)`. For example, if the rank pattern
        is [1, 1, 2, 1, 1], you will get five J values, with the "largest" J
        repeated in the positions marked with '2' and the smaller J in the
        positions marked with '1'.
    """
    # 1) Sort peaks
    peaks = np.sort(peaks)

    # 2) Convert the multiplicity string to line counts
    #    e.g. 'td' -> [3, 2], 'dt' -> [2, 3], 'ddd' -> [2, 2, 2], etc.
    line_counts = multiplicity_to_line_count(multiplicity)

    # 3) Reverse them, so we do the "rightmost" splitting first (smallest J),
    #    then move outward to the leftmost splitting (largest J).
    line_counts_reversed = line_counts[::-1]

    # Keep track of the J-values we find (from smallest to largest)
    dimension_js_ppm = []
    
    # We'll iteratively reduce `current_peaks` by grouping and averaging
    current_peaks = peaks

    # 4) For each splitting dimension in reverse
    for n in line_counts_reversed:
        # Number of groups at this splitting
        n_groups = len(current_peaks) // n
        if n_groups * n != len(current_peaks):
            raise ValueError("Mismatch between multiplicity and peak list length.")

        # Split into subgroups
        groups = [current_peaks[i*n : (i+1)*n] for i in range(n_groups)]

        # Compute the "internal" J for this layer:
        #   - For each group, take the adjacent differences
        #   - Average them -> that group's J
        #   - Then average across all groups -> dimension_j
        per_group_j = []
        for g in groups:
            diffs = np.diff(g)              # consecutive differences in ppm
            avg_diff = np.mean(diffs)       # average difference for this group
            per_group_j.append(avg_diff)

        dimension_j_ppm = np.mean(per_group_j)  # overall average for that splitting
        dimension_js_ppm.append(dimension_j_ppm)

        # Reduce each group to its "group centre" for the next outer dimension
        group_means = [np.mean(g) for g in groups]
        current_peaks = np.array(group_means)

    # Now dimension_js_ppm[0] is the smallest J (rightmost in the multiplicity),
    # dimension_js_ppm[-1] is the largest J (leftmost in the multiplicity).
    #
    # Convert them to Hz: J(Hz) = J(ppm) * frequency (MHz)
    dimension_js_hz = [j_ppm * frequency for j_ppm in dimension_js_ppm]

    # 5) Retrieve the rank order from largest to smallest J via `generate_j_pattern`.
    #    e.g. for 'dt' you might get [1,1,2,1,1], which means we have 2 distinct J's:
    #    - "rank=1" is the smaller J
    #    - "rank=2" is the larger J
    rank_pattern = generate_j_pattern(multiplicity)

    # Identify distinct ranks in ascending order
    # e.g. rank_pattern=[1,1,2,1,1] => unique_ranks=[1,2]
    unique_ranks = sorted(set(rank_pattern))

    # The first element in dimension_js_hz is the smallest J, the last is largest.
    # We map rank=1 -> dimension_js_hz[0], rank=2 -> dimension_js_hz[1], etc.
    if len(unique_ranks) != len(dimension_js_hz):
        raise ValueError("Number of distinct ranks does not match the number of computed J's.")

    rank_to_j = {}
    for i, rank in enumerate(unique_ranks, start=0):
        rank_to_j[rank] = dimension_js_hz[i]

    # 6) Build final array with the correct J in each position
    final_j_vals = [rank_to_j[r] for r in rank_pattern]

    return final_j_vals

# --- Example usage ---
td_peaks = [
    3.98875,  # 4.00 - 0.00875 - 0.0025  # 4.00 - 0.00875 + 0.0025
]
frequency = 400.0  # MHz
test_multiplicity = "s"

calculated_j_td = calculate_j_vals(td_peaks, test_multiplicity, frequency)
print("Peaks (td):", td_peaks)
print("Calculated J values (td):", calculated_j_td)
