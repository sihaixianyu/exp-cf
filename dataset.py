import pickle
import sys
from os import path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from numpy import random

import util


class Dataset:
    def __init__(self, data_dir: str, config: dict):
        self.data_dir = data_dir
        self.data_name = config['data_name']
        self.neighbor_num = config['neighbor_num']
        self.sample_method = config['sample_method']

        self.train_arr, self.test_arr, self.ui_csr_mat, self.neg_dict = self.__load_data()

        self.ui_mat = self.ui_csr_mat.toarray()
        self.iu_mat = self.ui_mat.T

        self.user_num = self.ui_mat.shape[0]
        self.item_num = self.iu_mat.shape[0]

        self.user_pos_items = self.__build_user_pos_items()
        self.user_sim_mat = self.__build_user_sim_mat()
        self.item_sim_mat = self.__build_item_sim_mat()
        self.ui_exp_mat = self.__build_ui_exp_mat()

        if self.sample_method == 'random':
            self.sample = self.__random_sample
        elif self.sample_method == 'explainable':
            self.sample = self.__explainable_sample
        else:
            raise ValueError('No such sample method: %s' % self.sample_method)

    def get_train_data(self):
        print('Building train data...')
        train_arr = self.sample()
        return train_arr

    def get_test_data(self):
        print('Building test data...')
        test_list = []
        for user, pos_item in tqdm(self.test_arr, file=sys.stdout):
            test_list.append([user, pos_item])
            for i in range(99):
                test_list.append([user, self.neg_dict[user][i]])
        return np.array(test_list)

    def __load_data(self):
        train_path = path.join(self.data_dir, 'train.csv')
        test_path = path.join(self.data_dir, 'test.csv')
        neg_path = path.join(self.data_dir, 'neg_dict.pkl')
        sp_ui_path = path.join(self.data_dir, 'ui_csr_mat.npz')

        util.check_file(train_path, test_path, sp_ui_path, neg_path)

        train_arr = pd.read_csv(train_path).to_numpy()
        test_arr = pd.read_csv(test_path).to_numpy()

        ui_csr_mat = sp.load_npz(sp_ui_path)

        with open(neg_path, 'rb') as f:
            neg_dict = pickle.load(f)

        return train_arr, test_arr, ui_csr_mat, neg_dict

    def __build_user_pos_items(self):
        users = list(range(self.user_num))
        user_pos_items = []
        for user in users:
            user_pos_items.append(self.ui_csr_mat[user].nonzero()[1])

        return user_pos_items

    def __build_user_sim_mat(self):
        user_sim_mat = cosine_similarity(self.ui_mat)
        np.fill_diagonal(user_sim_mat, 0)
        return user_sim_mat

    def __build_item_sim_mat(self):
        item_sim_mat = cosine_similarity(self.iu_mat)
        np.fill_diagonal(item_sim_mat, 0)
        return item_sim_mat

    def __build_ui_exp_mat(self):
        exp_mat_path = path.join(self.data_dir, 'ui_exp_mat_{}.npy'.format(self.neighbor_num))
        if path.exists(exp_mat_path):
            print('Loading explainable matrix...')
            ui_exp_mat = np.load(exp_mat_path)
        else:
            neighbors = [np.argpartition(row, - self.neighbor_num)[- self.neighbor_num:]
                         for row in self.item_sim_mat]

            print('Building explainable matrix...')
            ui_exp_mat = np.zeros((self.user_num, self.item_num), np.float32)
            for user in tqdm(range(self.user_num), file=sys.stdout):
                for item in range(self.item_num):
                    ui_exp_mat[user][item] = sum(
                        [self.ui_mat[user][neighbor] for neighbor in neighbors[item]]) / self.neighbor_num

            np.save(exp_mat_path, ui_exp_mat)

        return ui_exp_mat

    def __random_sample(self):
        train_list = []
        for user, pos_item in tqdm(self.train_arr, file=sys.stdout):
            pos_items = self.user_pos_items[user]

            neg_item = random.randint(0, self.item_num)
            while neg_item in pos_items:
                neg_item = random.randint(0, self.item_num)

            train_list.append([user, pos_item, neg_item])

        return np.array(train_list)

    def __explainable_sample(self):
        train_list = []
        for user, pos_item in tqdm(self.train_arr, file=sys.stdout):
            pos_items = self.user_pos_items[user]

            inv_exp_arr = np.reciprocal(self.ui_exp_mat[user])
            inv_exp_arr[np.isinf(inv_exp_arr)] = 0.
            prob_arr = inv_exp_arr / sum(inv_exp_arr)
            neg_item = random.choice(np.arange(self.item_num), p=prob_arr)
            while neg_item in pos_items:
                neg_item = random.randint(0, self.item_num)

            train_list.append([user, pos_item, neg_item])

        return np.array(train_list)
