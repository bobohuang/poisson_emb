"""
Small functions that I tend to use in multiple places or wrappers with better names :)

Author: Moshe Lichman
"""
from __future__ import division
import numpy as np
from scipy.sparse import find


def normalize_mat_row(mat, norm=1):
    """
    Normalizing the rows of the matrix.

     INPUT:
    -------
        1. mat:     <(N, D) ndarray>    matrix
        2. norm:    <float>             sum of each row after normalizing (default = 1)

     OUTPUT:
    --------
        1. norm_mat:    <(N, D) ndarray>    row-normalized matrix. Each row sums to norm
    """
    n, d = mat.shape
    tmp = mat / col_vector(np.sum(mat, axis=1))
    return tmp * norm


def convert_sparse_to_coo(s_mat):
    """
    Converts a scipy.sparse (and even dense) matrix to a coo_matrix form where each row is [row, col, value].
    This is used to allow faster evaluation and optimization.

     INPUT:
    -------
        1. s_mat:       <(N, D) sparse_mat>     sparse/dense matrix or vector. It works with all!

     OUTPUT:
    --------
        1. coo_form:    <(nnz, 3) ndarray>      nnz is the number of non-zero elements.
                                                Each row is [row, col, val]. If the input is a vector all row values
                                                will be 0.
    """
    return np.vstack(find(s_mat)).T


def col_vector(row_vect):
    """
    Converting np array to a column vector. Useful when we want to do fast element wise
    product or division.

     INPUT:
    -------
        1. row_vect:        <(N, ) ndarray>     row vector

     OUTPUT:
    --------
        1. col_vect:        <(N, 1) ndarray>    column vector
    """
    return np.reshape(row_vect, [row_vect.shape[0], 1])

