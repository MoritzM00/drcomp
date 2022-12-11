"""
Efficient Implementation of co-ranking metrics for evaluating the performance of dimensionality reduction techniques.

Taken from https://timsainburg.com/coranking-matrix-python-numba.html
"""
import numba
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances_chunked
from tqdm.autonotebook import tqdm


def _compute_ranking_matrix_parallel(D, n_jobs=None, verbose=0):
    """Compute ranking matrix in parallel. Input (D) is distance matrix."""
    # if data is small, no need for parallel
    if n_jobs is None:
        if len(D) < 1000:
            n_jobs = 1
        else:
            n_jobs = -1
    r1 = Parallel(n_jobs, prefer="threads", verbose=verbose)(
        delayed(np.argsort)(i)
        for i in tqdm(D.T, desc="computing rank matrix", leave=False)
    )
    r2 = Parallel(n_jobs, prefer="threads", verbose=verbose)(
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


def iterate_compute_distances(X):
    """Compute pairwise distance matrix iteratively."""
    n = len(X)
    D = np.zeros((n, n), dtype="float32")
    col = 0
    with tqdm(desc="computing pairwise distances", leave=False) as pbar:
        for i, distances in enumerate(
            pairwise_distances_chunked(X, n_jobs=-1),
        ):
            D[col : col + len(distances)] = distances
            col += len(distances)
            if i == 0:
                pbar.total = int(len(X) / len(distances))
            pbar.update(1)
    return D


def compute_coranking_matrix(X, Y, X_distances=None, n_jobs=None):
    """Compute the full coranking matrix."""

    # compute pairwise distances
    if X_distances is None:
        X_distances = iterate_compute_distances(X)

    Y_distances = iterate_compute_distances(Y)
    # n = len(D_ld)

    # compute the ranking matrix for high and low D
    ranking_X = _compute_ranking_matrix_parallel(X_distances, n_jobs=n_jobs)
    ranking_Y = _compute_ranking_matrix_parallel(Y_distances, n_jobs=n_jobs)

    # compute coranking matrix from_ranking matrix
    m = len(ranking_X)
    Q = np.zeros(ranking_X.shape, dtype="int16")
    for i in tqdm(range(m), desc="computing coranking matrix"):
        Q = _populate_Q(Q, i, m, ranking_X, ranking_Y)

    Q = Q[1:, 1:]
    return Q
