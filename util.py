import os
import pickle
from pprint import pprint

import numpy as np
import pandas as pd
import scipy.sparse as sp


def load_data(data_dir):
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    neg_path = os.path.join(data_dir, 'neg_dict.pkl')
    info_path = os.path.join(data_dir, 'info_dict.pkl')
    sp_ui_path = os.path.join(data_dir, 'ui_csr_mat.npz')

    check_file(train_path, test_path, sp_ui_path, neg_path, info_path)

    train_arr = pd.read_csv(train_path).to_numpy()
    test_arr = pd.read_csv(test_path).to_numpy()

    ui_csr_mat = sp.load_npz(sp_ui_path)

    with open(neg_path, 'rb') as f:
        neg_dict = pickle.load(f)
    with open(info_path, 'rb') as f:
        info_dict = pickle.load(f)

    return train_arr, test_arr, ui_csr_mat, neg_dict, info_dict


def check_file(*files: str):
    for file in files:
        if not os.path.exists(file):
            raise FileNotFoundError('{} not exist, please confirm the presence of the target!'.format(file))


def shuffle(*arrs):
    shuffle_idxs = np.arange(len(arrs[0]))
    np.random.shuffle(shuffle_idxs)

    if len(arrs) == 1:
        result = arrs[0][shuffle_idxs]
    else:
        result = tuple(x[shuffle_idxs] for x in arrs)

    return result


def mini_batch(*arrs, batch_size):
    if len(arrs) == 1:
        arr = arrs[0]
        for i in range(0, len(arr), batch_size):
            yield arr[i:i + batch_size]
    else:
        for i in range(0, len(arrs[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in arrs)


def color_print(words):
    print(f"\033[0;30;43m{words}\033[0m")


def sep_print(content, desc=None, split='-', num=75):
    print(split * num)
    if desc:
        print(desc + ':')
    pprint(content)
    print(split * num)


if __name__ == '__main__':
    load_data('data/ml-1m/')
