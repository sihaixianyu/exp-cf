import os
import sys
import time
from collections import Iterable
from pprint import pprint

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

# Timer for supervising function
from tqdm import tqdm


def timer(func):
    def calc_time(*args):
        start_time = time.time()
        res = func(*args)
        end_time = time.time()
        if isinstance(res, Iterable) and not isinstance(res, str):
            return (*res, end_time - start_time)
        else:
            return res, end_time - start_time

    return calc_time


def check_file(*files: str):
    for file in files:
        if not os.path.exists(file):
            raise FileNotFoundError(
                '{} not exist, please confirm the presence of the target!'.format(file))


def color_print(content: str):
    print(f'\033[0;30;43m{content}\033[0m')


def sep_print(obj: any, start=True, end=True, desc=None, num=80):
    if start:
        print('-' * num)
    if desc:
        print(desc + ':')
    if isinstance(obj, str):
        print(obj)
    else:
        pprint(obj)
    if end:
        print('-' * num)


def calc_cosine_similarity(mat: np.ndarray):
    sim_mat = cosine_similarity(mat)
    np.fill_diagonal(sim_mat, 0)

    return sim_mat


def calc_pearson_similarity(mat: np.ndarray):
    sim_mat = np.zeros(shape=(mat.shape[0], mat.shape[0]))
    for i in tqdm(range(mat.shape[0]), file=sys.stdout):
        for j in range(i + 1, mat.shape[0]):
            sim = pearsonr(mat[i], mat[j])[0]
            sim_mat[i][j] = sim
    sim_mat += sim_mat.T - np.diag(sim_mat.diagonal())

    return sim_mat
