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



td_peaks = [
    3.99885,  # 4.00 - 0.00875 - 0.0025
    3.99374,  # 4.00 - 0.00875 + 0.0025
    3.99750,  # 4.00 - 0.0025
    4.00250,  # 4.00 + 0.0025
    4.00625,  # 4.00 + 0.00875 - 0.0025
    4.01125,  # 4.00 + 0.00875 + 0.0025
]

td_peaks = [4.10525, 4.11628, 4.11996, 4.13062]

frequency = 500.0  # MHz
test_multiplicity = "q"

calculated_j_td = calculate_j_vals(td_peaks, test_multiplicity, frequency)
print("Peaks (td):", td_peaks)
print("Calculated J values (td):", calculated_j_td)
