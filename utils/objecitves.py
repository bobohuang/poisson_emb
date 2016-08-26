"""
Objective functions that are used in my work.
The methods should be accessed through the factory obj_func
"""
from __future__ import division
import numpy as np
import time

from . import log_utils as log

from scipy.sparse import find
from scipy.sparse import coo_matrix

def _obj_logp(mult, test):
    """
    Returns the logP of the test data.

     INPUT:
    -------
        1. user_mult:   <(L, )> ndarray>    probabilities. Sums to 1.
        2. test:        <(N_te, ) ndarray>  test points.

     OUTPUT:
    --------
        1. logP:    <(N_te, ) ndarray>     logP of the test points
    """
    return np.log(mult[test.astype(int)])


def _obj_erank(scores, test):
    """
    Returns the expected rank of the test points. 1 is the best.
    The expected part in it means that if several objects has the same score in the u_scores vector we will return
    the average of them.

    For example:
        u_scores = [2, 2, 0, 3]
        u_test = [1]    (This is the indexes, so it corresponds to the score 2 in the u_scores.

        First step would be to find the avg. ranks of the scores --> [ 2.5, 2.5, 4, 1]  (1 is the best)
        Then the avg score of u_test is 2.5 (only one point :) )

        The returned results is than (L - 2.5 + 1) / L (where L is the size of u_scores) --> (4 - 2.5 + 1) / 4 = 0.65.

        According to this 0 is the worst eRank (but you can't really get 0) and 1 is the best.


     INPUT:
    -------
        1. u_scores:    <(L, )> ndarray>    user scores. Doesn't have to be probability (to accommodate for SVD like)
        2. u_test:      <(N_te, ) ndarray>  user test points.

     OUTPUT:
    --------
        1. test_ranks:    <(N_te, ) ndarray>     (0, 1] erank of the test points (1 is the best)
    """
    L = scores.shape[0]
    rank = np.zeros(L)

    vals, idxs, counts = np.unique(scores, return_index=True, return_counts=True)

    # The expected thing makes non-trivial.
    # I need to find ranks that are equal, and average across their indexes. Only then I can look
    # at the test points
    prev_rank = 0
    for i in range(idxs.shape[0]):
        c = counts[i]
        if c == 1:
            rank[idxs[i]] = prev_rank + 1
            prev_rank += 1
        else:
            curr_rank = prev_rank + (1 + c) / 2
            mask = np.where(scores == vals[i])[0]
            rank[mask] = curr_rank
            prev_rank += c

    e_ranks = L - rank + 1
    test_ranks = e_ranks[test.astype(int)]  # The best score is here 1 and the worst is L
    test_ranks = (L - test_ranks + 1) / L   # Converting to [1, 0). 1 is perfect.

    return test_ranks


def _points_erank(score_mat, test_data):
    """
    Computes the expected rank of the test data. The returned value is the average across all points' erank.

     INPUT:
    -------
        1. score_mat:    <(I, L) csr_mat>  users scores. Doesn't have to be probabilities (for SVD)
        2. test_data:    <(I, L) csr_mat>  users test observations.

     OUTPUT:
    --------
        1. avg_erank:   <float>  avg. erank of all the test data.
    """
    sum_eranks = 0
    sum_counts = 0

    N = test_data.sum()

    log.info('Computing erank for %d points' % N)
    start = time.time()

    I = score_mat.shape[0]
    ind = 0
    for i in range(I):
        ind += 1
        # Converting it to a testable form [i, l, counts]
        i_test, i_counts = np.vstack(find(test_data[i]))[1:]
        i_erank = _obj_erank(score_mat[i], i_test)

        sum_eranks += np.sum(i_erank * i_counts)
        sum_counts += np.sum(i_counts)

        if ind % 1000 == 0:
            log.debug('Done testing %d out of %d points' % (ind, N))

    total = time.time() - start
    log.info('Erank for points took %d seconds. %.2f seconds on avg for point' % (total, total / N))
    return sum_eranks / sum_counts


def _individuals_erank(score_mat, test_data):
    """
    Computes the expected rank of the test data. The returned value is the average across all individuals.
    For each individual we average the rank first on hers test data.

     INPUT:
    -------
        1. score_mat:    <(I, L) csr_mat>  users scores. Doesn't have to be probabilities (for SVD)
        2. test_data:    <(I, L) csr_mat>  users test observations.

     OUTPUT:
    --------
        1. avg_erank:   <float>  avg. erank of all the individual.
    """
    avg_erank = 0

    start = time.time()
    I = score_mat.shape[0]
    for i in range(I):
        i_test, i_counts = np.vstack(find(test_data[i]))[1:]
        i_erank = _obj_erank(score_mat[i], i_test)
        i_erank *= i_counts

        avg_erank += np.sum(i_erank) / np.sum(i_counts)

        if i % 200 == 0:
            log.debug('Done testing %d out of %d users' % (i, I))

    total = time.time() - start
    log.info('Erank for individuals took %d seconds. %.2f secs on avg for indiv' % (total, total / I))
    return avg_erank / I


def _points_logp(score_mat, test_data):
    """
    Computes the expected rank of the test data. The returned value is the average across all points' erank.

     INPUT:
    -------
        1. score_mat:    <(I, L) csr_mat>  users scores. Doesn't have to be probabilities (for SVD)
        2. test_data:    <(I, L) csr_mat>  users test observations.

     OUTPUT:
    --------
        1. avg_erank:   <float>  avg. erank of all the test data.
    """
    score_mat += 0.0001
    score_mat /= np.sum(score_mat)
    N = test_data.sum()
    sum_logp = 0
    test_data = coo_matrix(test_data)

    for i,j,v in zip(test_data.row, test_data.col, test_data.data):
        sum_logp += v * np.log(score_mat[i,j])

    return sum_logp / N


def _individuals_logp(score_mat, test_data):
    """
    Computes the expected rank of the test data. The returned value is the average across all individuals.
    For each individual we average the rank first on hers test data.

     INPUT:
    -------
        1. score_mat:    <(I, L) csr_mat>  users scores. Doesn't have to be probabilities (for SVD)
        2. test_data:    <(I, L) csr_mat>  users test observations.

     OUTPUT:
    --------
        1. avg_erank:   <float>  avg. erank of all the individual.
    """
    score_mat += 0.0001
    row_sums = score_mat.sum(axis=1)
    score_mat = score_mat / row_sums[:, np.newaxis]
    I = score_mat.shape[0]
    logp_indiv = np.zeros(test_data.shape[0])
    n_train = np.array([int(test_data.sum(axis=1)[i][0]) for i in range(I)])

    test_data = coo_matrix(test_data)

    for i,j,v in zip(test_data.row, test_data.col, test_data.data):
        logp_indiv[i] += v * np.log(score_mat[i,j])
    logp_indiv /= n_train

    return sum(logp_indiv) / I


# factory
obj_func = {'ind_erank': _individuals_erank,
            'p_erank': _points_erank,
            'ind_logp': _individuals_logp,
            'p_logp':_points_logp}
