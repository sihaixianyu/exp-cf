from os import path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Dataset
from models import BaseModel


class CEGCN(BaseModel):
    def __init__(self, dataset: Dataset, config: dict):
        super(CEGCN, self).__init__(dataset)
        self.model_name = config['model_name']
        self.latent_dim = config['latent_dim']
        self.layer_num = config['layer_num']
        self.weight_decay = config['weight_decay']

        self.theta = config['theta']
        self.alpha = config['alpha']
        self.beta = config['beta']

        self.embed_user = nn.Embedding(self.user_num, self.latent_dim)
        self.embed_item = nn.Embedding(self.item_num, self.latent_dim)

        self.graph = self.__build_graph(dataset.train_csrmat)
        self.ui_exp_mat = self.__build_ui_exp_mat(dataset.train_exp_mat)

        self.to(self.device)

    def forward(self, batch_data):
        users = batch_data[:, 0]
        pos_items = batch_data[:, 1]
        neg_items = batch_data[:, 2]

        all_user_embs, all_item_embs = self.__compute()

        user_embs = all_user_embs[users]
        pos_item_embs = all_item_embs[pos_items]
        neg_item_embs = all_item_embs[neg_items]

        users = torch.LongTensor(users).to(self.device)
        pos_items = torch.LongTensor(pos_items).to(self.device)
        neg_items = torch.LongTensor(neg_items).to(self.device)

        user_egos = self.embed_user(users)
        pos_item_egos = self.embed_item(pos_items)
        neg_item_egos = self.embed_item(neg_items)

        pos_ratings = torch.sum(user_embs * pos_item_embs, dim=1)
        neg_ratings = torch.sum(user_embs * neg_item_embs, dim=1)

        pos_exp_mat = self.ui_exp_mat[users, pos_items]
        neg_exp_mat = self.ui_exp_mat[users, neg_items]

        exp_coef = pos_exp_mat * (1 - neg_exp_mat)
        loss = - (F.logsigmoid((pos_ratings - neg_ratings)) * exp_coef).mean()

        reg_term = (1 / 2) * (
                user_egos.norm(2).pow(2) + pos_item_egos.norm(2).pow(2) + neg_item_egos.norm(2).pow(2)) / float(
            len(users))

        pos_exp_mat[pos_exp_mat < self.theta] = 0
        neg_exp_mat[neg_exp_mat < self.theta] = 0

        pos_emb_diffs = torch.sum((user_embs - pos_item_embs), dim=1)
        neg_emb_diffs = torch.sum((user_embs - neg_item_embs), dim=1)

        pos_exp_reg = (1 / 2) * (pos_emb_diffs * pos_exp_mat).norm().pow(2) / float(len(users))
        neg_exp_reg = (1 / 2) * (neg_emb_diffs * neg_exp_mat).norm().pow(2) / float(len(users))

        return loss + self.weight_decay * reg_term + self.alpha * pos_exp_reg + self.beta * neg_exp_reg

    def predict(self, batch_users, batch_items):
        all_user_embs, all_item_embs = self.__compute()

        user_embs = all_user_embs[batch_users]
        item_embs = all_item_embs[batch_items]

        pred_ratings = torch.mul(user_embs, item_embs)
        pred_ratings = torch.sum(pred_ratings, dim=1)
        pred_ratings = torch.sigmoid(pred_ratings)

        return pred_ratings

    def get_model_path(self, model_dir: str):
        return path.join(model_dir, '{}_ld{}_ln{}_wd{}_t{}_a{}.pth'.format(self.model_name,
                                                                           self.latent_dim,
                                                                           self.layer_num,
                                                                           self.weight_decay,
                                                                           self.theta,
                                                                           self.alpha,
                                                                           self.beta))

    def __compute(self):
        embed_user_weight = self.embed_user.weight
        embed_item_weight = self.embed_item.weight
        emb_weight = torch.cat([embed_user_weight, embed_item_weight])

        embs = [emb_weight]
        for i in range(self.layer_num):
            emb_weight = torch.sparse.mm(self.graph, emb_weight)
            embs.append(emb_weight)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        all_user_embs, all_item_embs = torch.split(light_out, [self.user_num, self.item_num])

        return all_user_embs, all_item_embs

    def __build_graph(self, ui_csr_mat):
        adj_mat = sp.dok_matrix((self.user_num + self.item_num,
                                 self.user_num + self.item_num), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = ui_csr_mat.tolil()
        adj_mat[:self.user_num, self.user_num:] = R
        adj_mat[self.user_num:, :self.user_num] = R.T
        adj_mat.todok()

        row_sum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(row_sum, -0.5).flatten()
        # Solve the devide by 0 problem
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj_mat = d_mat.dot(adj_mat)
        norm_adj_mat = norm_adj_mat.dot(d_mat)
        norm_adj_mat = norm_adj_mat.tocsr()

        coo_mat = norm_adj_mat.tocoo().astype(np.float32)
        row_tsr = torch.Tensor(coo_mat.row).long()
        col_tsr = torch.Tensor(coo_mat.col).long()
        idx_tsr = torch.stack([row_tsr, col_tsr])
        val_tsr = torch.FloatTensor(coo_mat.data)
        graph = torch.sparse.FloatTensor(idx_tsr, val_tsr, torch.Size(coo_mat.shape))

        return graph.coalesce().to(self.device)

    def __build_ui_exp_mat(self, ui_exp_mat):
        return torch.from_numpy(ui_exp_mat).to(self.device)
