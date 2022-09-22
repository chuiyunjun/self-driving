from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment




def greedy_matching(cost_matrix: np.ndarray) -> Tuple[List, List]:
    """Perform matching based on the greedy matching algorithm.

    Args:
        cost matrix of shape [M, N], where cost[i, j] is the cost of matching i to j
    Returns:
        (row_ids, col_ids), where row_ids and col_ids are lists of the same length,
        and each (row_ids[k], col_ids[k]) is a match.

        Example: if M = 3, N = 4, then the return values of ([0, 1, 2], [3, 1, 0]) means the final
        assignment corresponds to costs[0, 3], costs[1, 1] and costs[2, 0].
    """
    # TODO: Replace this stub code.
    row_ids, col_ids = [], []
    M, N = cost_matrix.shape

    cost_matrix_copy = cost_matrix.copy()
    # np.inf can't be converted to int, so use this way to substitute 
    over_max = np.max(cost_matrix_copy) + 1


    iter_num = min(N, M)
    for i in range(iter_num):
        min_row_ix, min_col_ix = np.unravel_index(np.argmin(cost_matrix_copy, axis=None), cost_matrix_copy.shape)
        row_ids.append(min_row_ix)
        col_ids.append(min_col_ix)
        cost_matrix_copy[min_row_ix,:] =over_max
        cost_matrix_copy[:, min_col_ix] =over_max
    # (List[int], List[int])
    return (row_ids, col_ids)



def hungarian_matching(cost_matrix: np.ndarray) -> Tuple[List, List]:
    """Perform matching based on the Hungarian matching algorithm.
    For simplicity, we just call the scipy `linear_sum_assignment` function. Please refer to
    https://en.wikipedia.org/wiki/Hungarian_algorithm and
    https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html
    for more details of the hungarian matching implementation.

    Args:
        cost matrix of shape [M, N], where cost[i, j] is the cost of matching i to j
    Returns:
        (row_ids, col_ids), where row_ids and col_ids are lists of the same length,
        and each (row_ids[k], col_ids[k]) is a match.

        Example: if M = 3, N = 4, then the return values of ([0, 1, 2], [3, 1, 0]) means the final
        assignment corresponds to costs[0, 3], costs[1, 1] and costs[2, 0].
    """
    # TODO: Replace this stub code.
    row_ids, col_ids = linear_sum_assignment(cost_matrix)
    return row_ids, col_ids
