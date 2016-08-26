"""
File methods wrapper.

Author: Moshe Lichman
"""
from __future__ import division
import numpy as np
import time
from utils import log_utils as log
from scipy.sparse import coo_matrix

from os.path import join

def valid_areas():
    return ['tw_oc','go_sf','go_ny']

def load_data(area):
    """
    Loads train, validation and test data for the given area.
    When testing, train and val should be combined (train += val).

     OUTPUT:
    --------
        1. train:   <(I, L) csr_mat>    sparse counts matrix. Rows are individuals, columns are locations.
        2. val:     <(I, L) csr_mat>    sparse counts matrix. Rows are individuals, columns are locations.
        3. test:    <(I, L) csr_mat>    sparse counts matrix. Rows are individuals, columns are locations.

     RAISE:
    -------
        1.  IOError:              Area or one of the files does not exist.
    """
    root_folder = ''

    log.info('Loading all data for area %s' % area)
    train_file = join(root_folder, area, 'train.csv')
    val_file = join(root_folder, area, 'val.csv')
    test_file = join(root_folder, area, 'test.csv')

    train_data = np.loadtxt(train_file,delimiter=',')
    val_data = np.loadtxt(val_file,delimiter=',')
    test_data = np.loadtxt(test_file,delimiter=',')

    # In order to create the coo_matrix we need to have the number of rows and columns in the matrix
    # All individuals will have data in train, val and test so it's enough to check how many uses are in train.
    I = np.unique(train_data[:, 0]).shape[0]

    # For location that is not tha case. We need to check the maximum location across all 3.
    L = np.max(train_data[:, 1])
    L = np.max([L, np.max(val_data[:, 1])])
    L = np.max([L, np.max(test_data[:, 1])])
    L += 1  # It's all 0 based

    train = coo_matrix((train_data[:, 2], (train_data[:, 0], train_data[:, 1])), shape=(I, L)).tocsr()
    val = coo_matrix((val_data[:, 2], (val_data[:, 0], val_data[:, 1])), shape=(I, L)).tocsr()
    test = coo_matrix((test_data[:, 2], (test_data[:, 0], test_data[:, 1])), shape=(I, L)).tocsr()

    return train, val, test