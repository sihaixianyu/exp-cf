from os import path

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import Dataset

from models import BaseModel


class NEMF(BaseModel):
    def __init__(self, dataset: Dataset, config: dict):
        super(NEMF, self).__init__(dataset)
        self.model_name = config['model_name']
        self.latent_dim = config['latent_dim']
        self.weight_decay = config['weight_decay']

        self.theta = config['theta']
        self.alpha = config['alpha']
        self.beta = config['beta']

        self.embed_user = nn.Embedding(self.user_num, self.latent_dim)
        self.embed_item = nn.Embedding(self.item_num, self.latent_dim)

        self.item_sim_tsr = self.__build_item_sim_tsr(dataset.item_sim_mat)
        self.item_nbr_tsr = self.__build_item_nbr_tsr(dataset.item_nbr_mat)
        self.ui_exp_tsr = self.__build_ui_exp_tsr(dataset.train_exp_mat)

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
        loss = - (F.logsigmoid((pos_ratings - neg_ratings)) * exp_coef).mean()

        reg_term = (1 / 2) * (
                user_embs.norm(2).pow(2) + pos_item_embs.norm(2).pow(2) + neg_item_embs.norm(2).pow(2)) / float(
            len(users))

        W_pos = self.ui_exp_tsr[users, pos_items]
        W_pos[W_pos < self.theta] = 0

        exp_reg = (user_embs - pos_item_embs).pow(2).sum(dim=1)
        exp_reg = (1 / 2) * (W_pos * exp_reg).sum() / float(len(users))

        nbr_reg = self.__calc_item_nbr_reg(user_embs, pos_items)

        return loss + self.weight_decay * reg_term + self.alpha * exp_reg + self.beta * nbr_reg

    def predict(self, batch_users, batch_items):
        batch_users = torch.LongTensor(batch_users).to(self.device)
        batch_items = torch.LongTensor(batch_items).to(self.device)

        user_embs = self.embed_user(batch_users)
        item_embs = self.embed_item(batch_items)

        pred_ratings = torch.sum(user_embs * item_embs, dim=1)
        pred_ratings = torch.sigmoid(pred_ratings)

        return pred_ratings

    def get_model_path(self, model_dir: str):
        return path.join(model_dir, '{}_ld{}_wd{}_t{}_a{}_b{}.pth'.format(self.model_name,
                                                                          self.latent_dim,
                                                                          self.weight_decay,
                                                                          self.theta,
                                                                          self.alpha,
                                                                          self.beta))

    def __build_item_sim_tsr(self, item_sim_mat):
        return torch.from_numpy(item_sim_mat).to(self.device)

    def __build_item_nbr_tsr(self, item_nbr_mat):
        return torch.from_numpy(item_nbr_mat).to(self.device)

    def __build_ui_exp_tsr(self, ui_exp_mat):
        return torch.from_numpy(ui_exp_mat).to(self.device)

    def __calc_item_nbr_reg(self, b_user_embs, b_items):
        with torch.no_grad():
            reg_sum = 0
            b_item_nbr_tsr = self.item_nbr_tsr[b_items]
            for i, item_nbrs in enumerate(b_item_nbr_tsr):
                user_embs = b_user_embs[i].repeat(self.neighbor_num, 1)
                item_embs = self.embed_item(item_nbrs)

                emb_diffs = user_embs - item_embs
                nbr_regs = emb_diffs.pow(2).sum(dim=1)

                sim_arr = self.item_sim_tsr[b_items[i], item_nbrs]
                nbr_regs = (nbr_regs * sim_arr).sum(dim=0)

                reg_sum += nbr_regs

        return reg_sum
