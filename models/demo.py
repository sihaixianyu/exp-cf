from os import path

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import Dataset

from models import BaseModel


class Demo(BaseModel):
    def __init__(self, dataset: Dataset, config: dict):
        super(Demo, self).__init__(dataset)
        self.model_name = config['model_name']
        self.latent_dim = config['latent_dim']
        self.weight_decay = config['weight_decay']

        self.embed_user = nn.Embedding(self.user_num, self.latent_dim)
        self.embed_item = nn.Embedding(self.item_num, self.latent_dim)

        self.item_sim_mat = self.__build_item_sim_mat(dataset.item_sim_mat)
        self.ui_exp_mat = self.__build_ui_exp_mat(dataset.train_exp_mat)

        self.to(self.device)

    def forward(self, batch_data):
        users = batch_data[:, 0]
        pos_items = batch_data[:, 1]
        neg_items = batch_data[:, 2]

        users = torch.LongTensor(users).to(self.device)
        pos_items = torch.LongTensor(pos_items).to(self.device)
        neg_items = torch.LongTensor(neg_items).to(self.device)

        user_embs = self.embed_user(users)
        pos_item_embs = self.embed_item(pos_items)
        neg_item_embs = self.embed_item(neg_items)

        pos_ratings = torch.sum(user_embs * pos_item_embs, dim=1)
        neg_ratings = torch.sum(user_embs * neg_item_embs, dim=1)

        sim_mat = self.item_sim_mat[pos_items, neg_items]
        sim_mat[sim_mat <= 0.4] = 1
        sim_mat[sim_mat != 1] = 0

        exp_coef = self.ui_exp_mat[users, pos_items] * (1 - self.ui_exp_mat[users, neg_items]) * sim_mat
        loss = torch.mean(- F.logsigmoid((pos_ratings - neg_ratings)) * exp_coef)

        reg_term = (1 / 2) * (
                user_embs.norm(2).pow(2) + pos_item_embs.norm(2).pow(2) + neg_item_embs.norm(2).pow(2)) / float(
            len(users))

        return loss + self.weight_decay * reg_term

    def predict(self, batch_users, batch_items):
        batch_users = torch.LongTensor(batch_users).to(self.device)
        batch_items = torch.LongTensor(batch_items).to(self.device)

        user_embs = self.embed_user(batch_users)
        item_embs = self.embed_item(batch_items)

        pred_ratings = torch.sum(user_embs * item_embs, dim=1)
        pred_ratings = torch.sigmoid(pred_ratings)

        return pred_ratings

    def get_model_path(self, model_dir: str):
        return path.join(model_dir, '{}_ld{}_wd{}.pth'.format(self.model_name,
                                                              self.latent_dim,
                                                              self.weight_decay))

    def __build_ui_exp_mat(self, ui_exp_mat):
        return torch.from_numpy(ui_exp_mat).to(self.device)

    def __build_item_sim_mat(self, item_sim_mat):
        return torch.from_numpy(item_sim_mat).to(self.device)
