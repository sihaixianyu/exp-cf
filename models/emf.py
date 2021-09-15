from os import path

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Dataset
from models import BaseModel


class EMF(BaseModel):
    def __init__(self, dataset: Dataset, config: dict):
        super(EMF, self).__init__(dataset)
        self.model_name = config['model_name']
        self.latent_dim = config['latent_dim']
        self.weight_decay = config['weight_decay']

        self.embed_user = nn.Embedding(self.user_num, self.latent_dim)
        self.embed_item = nn.Embedding(self.item_num, self.latent_dim)

        self.ui_exp_tsr = self.__build_ui_exp_tsr(dataset.ui_exp_mat)

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

        exp_coef = self.ui_exp_tsr[users, pos_items] * (1 - self.ui_exp_tsr[users, neg_items])
        loss = torch.mean(F.softplus((neg_ratings - pos_ratings)) * exp_coef)

        reg_term = (1 / 2) * (user_embs.norm(2).pow(2) +
                              pos_item_embs.norm(2).pow(2) +
                              neg_item_embs.norm(2).pow(2)) / float(len(users))

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
        return path.join(model_dir, '{}_ld{}_n{}.pth'.format(self.model_name,
                                                             self.latent_dim,
                                                             self.neighbor_num))

    def __build_ui_exp_tsr(self, ui_exp_mat):
        return torch.from_numpy(ui_exp_mat).to(self.device)
