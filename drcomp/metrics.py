"""
Implementation of metrics for evaluating the performance of dimensionality reduction techniques.

Taken from https://timsainburg.com/coranking-matrix-python-numba.html
"""
import numba
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances_chunked
from tqdm.autonotebook import tqdm


def _compute_ranking_matrix_parallel(D):
    """Compute ranking matrix in parallel. Input (D) is distance matrix."""
    # if data is small, no need for parallel
    if len(D) > 1000:
        n_jobs = -1
    else:
        n_jobs = 1
    r1 = Parallel(n_jobs, prefer="threads")(
        delayed(np.argsort)(i)
        for i in tqdm(D.T, desc="computing rank matrix", leave=False)
    )
    r2 = Parallel(n_jobs, prefer="threads")(
        delayed(np.argsort)(i)
        for i in tqdm(r1, desc="computing rank matrix", leave=False)
    )
    # write as a single array
    r2_array = np.zeros((len(r2), len(r2[0])), dtype="int32")
    for i, r2row in enumerate(tqdm(r2, desc="concatenating rank matrix", leave=False)):
        r2_array[i] = r2row
    return r2_array


@numba.njit(fastmath=True)
def _populate_Q(Q, i, m, R1, R2):
    """Populate coranking matrix using numba for speed."""
    for j in range(m):
        k = R1[i, j]
        u = R2[i, j]
        Q[k, u] += 1
    return Q


def iterate_compute_distances(data):
    """Compute pairwise distance matrix iteratively."""
    n = len(data)
    D = np.zeros((n, n), dtype="float32")
    col = 0
    with tqdm(desc="computing pairwise distances", leave=False) as pbar:
        for i, distances in enumerate(
            pairwise_distances_chunked(data, n_jobs=-1),
        ):
            D[col : col + len(distances)] = distances
            col += len(distances)
            if i == 0:
                pbar.total = int(len(data) / len(distances))
            pbar.update(1)
    return D


def compute_coranking_matrix(data_ld, data_hd=None, D_hd=None):
    """Compute the full coranking matrix."""

    # compute pairwise probabilities
    if D_hd is None:
        D_hd = iterate_compute_distances(data_hd)

    D_ld = iterate_compute_distances(data_ld)
    # n = len(D_ld)

    # compute the ranking matrix for high and low D
    rm_hd = _compute_ranking_matrix_parallel(D_hd)
    rm_ld = _compute_ranking_matrix_parallel(D_ld)

    # compute coranking matrix from_ranking matrix
    m = len(rm_hd)
    Q = np.zeros(rm_hd.shape, dtype="int16")
    for i in tqdm(range(m), desc="computing coranking matrix"):
        Q = _populate_Q(Q, i, m, rm_hd, rm_ld)

    Q = Q[1:, 1:]
    return Q
