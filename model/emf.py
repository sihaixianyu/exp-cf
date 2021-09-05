import sys
from os import path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

import util
from dataset import Dataset
from model import BaseModel


class EMF(BaseModel):
    def __init__(self, dataset: Dataset, model_config: dict, data_dir: str):
        super(EMF, self).__init__(dataset)
        self.ui_mat = dataset.ui_mat
        self.item_sim_mat = dataset.item_sim_mat

        self.latent_dim = model_config['latent_dim']
        self.neighbor_num = model_config['neighbor_num']

        self.embed_user = nn.Embedding(self.user_num, self.latent_dim)
        self.embed_item = nn.Embedding(self.item_num, self.latent_dim)

        self.ui_exp_mat = self.__build_exp_mat(data_dir)

        self.to(self.device)

    def forward(self, data):
        users = data[:, 0]
        pos_items = data[:, 1]
        neg_items = data[:, 2]

        users = torch.LongTensor(users).to(self.device)
        pos_items = torch.LongTensor(pos_items).to(self.device)
        neg_items = torch.LongTensor(neg_items).to(self.device)

        user_embs = self.embed_user(users)
        pos_item_embs = self.embed_item(pos_items)
        neg_item_embs = self.embed_item(neg_items)

        pos_ratings = torch.sum(user_embs * pos_item_embs, dim=1)
        neg_ratings = torch.sum(user_embs * neg_item_embs, dim=1)

        exp_coef = self.ui_exp_mat[users, pos_items] * (1 - self.ui_exp_mat[users, neg_items])
        loss = torch.mean(F.softplus((neg_ratings - pos_ratings)) * exp_coef)

        return loss

    def predict(self, users, items):
        users = torch.LongTensor(users).to(self.device)
        items = torch.LongTensor(items).to(self.device)

        user_embs = self.embed_user(users)
        item_embs = self.embed_item(items)

        pred_ratings = torch.sum(user_embs * item_embs, dim=1)
        pred_ratings = torch.sigmoid(pred_ratings)

        return pred_ratings

    def __build_exp_mat(self, data_dir):
        exp_mat_path = path.join(data_dir, 'ui_exp_mat_{}.npy'.format(self.neighbor_num))
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

        return torch.from_numpy(ui_exp_mat).to(self.device)
