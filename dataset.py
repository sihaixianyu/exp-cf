import sys

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class Dataset:
    def __init__(self, train_arr, test_arr, ui_csr_mat, neg_dict, info_dict):
        self.train_arr = train_arr
        self.test_arr = test_arr
        self.ui_csr_mat = ui_csr_mat
        self.iu_csr_mat = ui_csr_mat.T
        self.neg_dict = neg_dict

        self.user_num = info_dict['user_num']
        self.item_num = info_dict['item_num']

        self.ui_mat = ui_csr_mat.toarray()
        self.iu_mat = ui_csr_mat.toarray().T

        self.user_pos_items = self.__build_user_pos_items()
        self.item_sim_mat = self.__build_item_sim_mat()

        # Normal sample strategy by default
        self.sample = self.__random_sample

    def set_sample_method(self, method_name):
        if method_name == 'random':
            self.sample = self.__random_sample
        elif method_name == 'explainable':
            self.sample = self.__explainable_sample
        else:
            raise ValueError('No such sample method: %s' % method_name)
            # Normal sample method, which sets neg_num = 1 by default

    def get_train_data(self):
        print('Building train data...')
        return self.sample()

    def get_test_data(self):
        print('Building test data...')
        test_list = []
        for user, pos_item in tqdm(self.test_arr, file=sys.stdout):
            test_list.append([user, pos_item])
            for i in range(99):
                test_list.append([user, self.neg_dict[user][i]])
        return np.array(test_list)

    def __build_user_pos_items(self):
        users = list(range(self.user_num))
        user_pos_items = []
        for user in users:
            user_pos_items.append(self.ui_csr_mat[user].nonzero()[1])

        return user_pos_items

    def __build_item_sim_mat(self):
        item_sim_mat = cosine_similarity(self.iu_mat)
        np.fill_diagonal(item_sim_mat, 0)
        return item_sim_mat

    def __random_sample(self):
        train_list = []
        for user, pos_item in tqdm(self.train_arr, file=sys.stdout):
            pos_items = self.user_pos_items[user]

            neg_item = np.random.randint(0, self.item_num)
            while neg_item in pos_items:
                neg_item = np.random.randint(0, self.item_num)

            train_list.append([user, pos_item, neg_item])

        return np.array(train_list)

    def __explainable_sample(self):
        train_list = []
        for user, pos_item in tqdm(self.train_arr, file=sys.stdout):
            neg_item_idxs = np.where(self.ui_mat[user] == 0)[0]
            neg_item_sims = self.item_sim_mat[pos_item][neg_item_idxs]
            neg_item = np.argsort(neg_item_sims)[0]

            train_list.append([user, pos_item, neg_item])

        return np.array(train_list)
