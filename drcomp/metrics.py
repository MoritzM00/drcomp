import numba
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances_chunked
from tqdm.autonotebook import tqdm

"""
This file contains functions for computing the coranking matrix and metrics based on it.
Taken and adapted from: https://timsainburg.com/coranking-matrix-python-numba.html
and https://github.com/zhangys11/pyDRMetrics
"""


def compute_quality_criteria(Q, max_K: int = None):
    """Compute the quality criteria from the coranking matrix.

    Taken and adapted from the package pyDRMetrics [1], which is licensed
    under the Creative Commons Attribution 4.0 International licence.

    [1] https://github.com/zhangys11/pyDRMetrics

    Parameters
    ----------
    Q : np.ndarray of shape (n_samples - 1, n_samples - 1)
        Coranking matrix.
    max_k : int, default=None
        If None (default), then use the whole co ranking matrix to compute the metrics. Otherwise only up to max_k.

    Returns
    -------
    T : np.ndarray
        Trustworthiness.
    C : np.ndarray
        Continuity.
    LCMC : np.ndarray
        Local Continuity Meta Criterion.
    """

    m = len(Q)
    if max_K is None:
        max_K = m - 1
    else:
        assert max_K < m, "max_k must be smaller than the number of data points"

    T = np.zeros(max_K)  # trustworthiness
    C = np.zeros(max_K)  # continuity
    QNN = np.zeros(max_K)  # Co-k-nearest neighbor size
    LCMC = np.zeros(max_K)  # Local Continuity Meta Criterion

    for k in range(max_K):
        Qs = Q[k:, :k]
        W = np.arange(Qs.shape[0]).reshape(
            -1, 1
        )  # a column vector of weights. weight = rank error = actual_rank - k
        T[k] = 1 - np.sum(Qs * W) / (k + 1) / m / (
            m - 1 - k
        )  # 1 - normalized hard-k-intrusions. lower-left region. weighted by rank error (rank - k)
        Qs = Q[:k, k:]
        W = np.arange(Qs.shape[1]).reshape(
            1, -1
        )  # a row vector of weights. weight = rank error = actual_rank - k
        C[k] = 1 - np.sum(Qs * W) / (k + 1) / m / (
            m - 1 - k
        )  # 1 - normalized hard-k-extrusions. upper-right region

    for k in range(max_K):
        QNN[k] = np.sum(Q[: k + 1, : k + 1]) / (
            (k + 1) * m
        )  # Q[0,0] is always m. 0-th nearest neighbor is always the point itself. Exclude Q[0,0]
        LCMC[k] = QNN[k] - (k + 1) / (m - 1)

    return (
        T,
        C,
        LCMC,
    )


"""
The following functions are taken and adapted from https://timsainburg.com/coranking-matrix-python-numba.html
"""


def _compute_ranking_matrix_parallel(D, n_jobs=None):
    """Compute ranking matrix in parallel. Input (D) is distance matrix

    Parameters
    ----------
    D : np.ndarray
        Distance matrix.
    n_jobs : int, default=None
        Number of jobs to use for parallel computation. If None, then use 1 if the number of data points is less than 1000, otherwise use all available cores."""
    # if data is small, no need for parallel
    if n_jobs is None:
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
        l = R2[i, j]
        Q[k, l] += 1
    return Q


def _iterate_compute_distances(data, n_jobs=None):
    """Compute pairwise distance matrix iteratively.

    Parameters
    ----------
    data : np.ndarray
        Data matrix.
    n_jobs : int, default=None
        Number of jobs to use for parallel computation. If None, then use 1 if the number of data points is less than 1000, otherwise use all available cores.
    """
    if n_jobs is None:
        if len(data) > 1000:
            n_jobs = -1
        else:
            n_jobs = 1
    n = len(data)
    D = np.zeros((n, n), dtype="float32")
    col = 0
    with tqdm(desc="computing pairwise distances", leave=False) as pbar:
        for i, distances in enumerate(
            pairwise_distances_chunked(data, n_jobs=n_jobs),
        ):
            D[col : col + len(distances)] = distances
            col += len(distances)
            if i == 0:
                pbar.total = int(len(data) / len(distances))
            pbar.update(1)
    return D


def compute_coranking_matrix(X, Y, n_jobs=None):
    """Compute the full coranking matrix.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        High dimensional data.
    Y : np.ndarray of shape (n_samples, n_features)
        Low dimensional data.
    n_jobs : int, default=None
        Number of jobs to use for parallel computation. If None, then use all available cores if the dataset has more than 1000 samples, else n_jobs is set to 1.
    """
    D_hd = _iterate_compute_distances(X)
    D_ld = _iterate_compute_distances(Y)

    # compute the ranking matrix for high and low D
    rm_hd = _compute_ranking_matrix_parallel(D_hd, n_jobs=n_jobs)
    rm_ld = _compute_ranking_matrix_parallel(D_ld, n_jobs=n_jobs)

    # compute coranking matrix from_ranking matrix
    m = len(rm_hd)
    Q = np.zeros(rm_hd.shape, dtype="int16")
    for i in tqdm(range(m), desc="computing coranking matrix"):
        Q = _populate_Q(Q, i, m, rm_hd, rm_ld)

    Q = Q[1:, 1:]
    return Q
