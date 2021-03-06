import pickle
import sys
from os import path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy import random
from tqdm import tqdm

import util


class Dataset:
    def __init__(self, data_dir: str, config: dict):
        self.data_dir = data_dir
        self.data_name = config['data_name']
        self.nbr_num = config['nbr_num']
        self.similarity = config['similarity']

        self.train_arr, self.test_arr, self.neg_dict, self.train_csrmat, self.full_csrmat = self.__load_data()

        self.train_mat = self.train_csrmat.toarray()
        self.full_mat = self.full_csrmat.toarray()

        self.user_num = self.train_mat.shape[0]
        self.item_num = self.train_mat.shape[1]

        if self.similarity == 'cosine':
            self.similarity_func = util.calc_cosine_similarity
        elif self.similarity == 'pearson':
            self.similarity_func = util.calc_pearson_similarity
        elif self.similarity == 'jaccard':
            self.similarity_func = util.calc_jaccard_similarity
        else:
            raise ValueError('The target similarity function {} not exist!'.format(self.similarity))

        self.temp_dir = path.join(data_dir, 'temp/')
        util.check_dir(self.temp_dir)

        self.user_pos_items = self.__build_user_pos_items()

        self.item_sim_mat = self.__build_item_sim_mat()
        self.item_nbr_mat = self.__build_item_nbr_mat()

        self.train_exp_mat = self.__build_train_exp_mat()
        self.full_exp_mat = self.__build_full_exp_mat()

    def get_train_data(self):
        train_list = []
        for user, pos_item in self.train_arr:
            pos_items = self.user_pos_items[user]

            neg_item = random.randint(0, self.item_num)
            while neg_item in pos_items:
                neg_item = random.randint(0, self.item_num)

            train_list.append([user, pos_item, neg_item])

        return np.array(train_list)

    def get_test_data(self):
        test_list = []
        for user, pos_item in self.test_arr:
            test_list.append([user, pos_item])
            for i in range(99):
                test_list.append([user, self.neg_dict[user][i]])

        return np.array(test_list)

    def __load_data(self):
        train_path = path.join(self.data_dir, 'train.csv')
        test_path = path.join(self.data_dir, 'test.csv')
        neg_path = path.join(self.data_dir, 'neg_dict.pkl')
        train_csrmat_path = path.join(self.data_dir, 'train_csrmat.npz')
        full_csrmat_path = path.join(self.data_dir, 'full_csrmat.npz')

        util.check_file(train_path, test_path, neg_path, train_csrmat_path, full_csrmat_path)

        train_arr = pd.read_csv(train_path).to_numpy()
        test_arr = pd.read_csv(test_path).to_numpy()

        train_csrmat = sp.load_npz(train_csrmat_path)
        full_csrmat = sp.load_npz(full_csrmat_path)

        with open(neg_path, 'rb') as f:
            neg_dict = pickle.load(f)

        return train_arr, test_arr, neg_dict, train_csrmat, full_csrmat

    def __build_user_pos_items(self):
        users = list(range(self.user_num))
        user_pos_items = []
        for user in users:
            user_pos_items.append(self.train_csrmat[user].nonzero()[1])

        return user_pos_items

    def __build_item_sim_mat(self):
        item_sim_path = path.join(self.temp_dir, 'item_sim_mat_{}{}.npy'.format(self.similarity, self.nbr_num))
        if path.exists(item_sim_path):
            print('Loading item similarity matrix...')
            item_sim_mat = np.load(item_sim_path)
        else:
            print('Building item similarity matrix...')
            item_sim_mat = self.similarity_func(self.train_mat.T)
            np.save(item_sim_path, item_sim_mat)

        return item_sim_mat

    def __build_item_nbr_mat(self):
        item_nbr_list = [np.argpartition(row, - self.nbr_num)[- self.nbr_num:]
                         for row in self.item_sim_mat]

        return np.array(item_nbr_list)

    def __build_train_exp_mat(self):
        train_exp_path = path.join(self.temp_dir, 'train_exp_mat_{}{}.npy'.format(self.similarity, self.nbr_num))
        if path.exists(train_exp_path):
            print('Loading train explainable matrix...')
            train_exp_mat = np.load(train_exp_path)
        else:
            print('Building train explainable matrix...')
            train_exp_mat = np.zeros((self.user_num, self.item_num), np.float32)
            for user in tqdm(range(self.user_num), file=sys.stdout):
                for item in range(self.item_num):
                    train_exp_mat[user][item] = sum(
                        [self.train_mat[user][neighbor] for neighbor in self.item_nbr_mat[item]]) / self.nbr_num
            np.save(train_exp_path, train_exp_mat)

        return train_exp_mat

    def __build_full_exp_mat(self):
        full_exp_path = path.join(self.temp_dir, 'full_exp_mat_{}{}.npy'.format(self.similarity, self.nbr_num))
        if path.exists(full_exp_path):
            print('Loading full explainable matrix...')
            full_exp_mat = np.load(full_exp_path)
        else:
            print('Building full explainable matrix...')
            full_exp_mat = np.zeros((self.user_num, self.item_num), np.float32)
            for user in tqdm(range(self.user_num), file=sys.stdout):
                for item in range(self.item_num):
                    full_exp_mat[user][item] = sum(
                        [self.train_mat[user][neighbor] for neighbor in self.item_nbr_mat[item]]) / self.nbr_num
            np.save(full_exp_path, full_exp_mat)

        return full_exp_mat
