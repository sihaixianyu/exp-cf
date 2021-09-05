import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor

from dataset import Dataset
from model import BaseModel


class LGCN(BaseModel):
    def __init__(self, dataset: Dataset, model_config: dict):
        super(LGCN, self).__init__(dataset)
        self.ui_csr_mat = dataset.ui_csr_mat

        self.latent_dim = model_config['latent_dim']
        self.layer_num = model_config['layer_num']
        self.weight_decay = model_config['weight_decay']

        self.embed_user = nn.Embedding(self.user_num, self.latent_dim)
        self.embed_item = nn.Embedding(self.item_num, self.latent_dim)

        self.graph = self.__build_graph()

        self.to(self.device)

    def forward(self, data):
        users = data[:, 0]
        pos_items = data[:, 1]
        neg_items = data[:, 2]

        all_user_embs, all_item_embs = self.compute()

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

        bpr_loss = torch.mean(F.softplus(neg_ratings - pos_ratings))
        reg_term = (1 / 2) * (user_egos.norm(2).pow(2) +
                              pos_item_egos.norm(2).pow(2) +
                              neg_item_egos.norm(2).pow(2)) / float(len(users))

        return bpr_loss + reg_term * self.weight_decay

    def predict(self, users, items):
        all_user_embs, all_item_embs = self.compute()

        user_embs = all_user_embs[users]
        item_embs = all_item_embs[items]

        pred_ratings = torch.mul(user_embs, item_embs)
        pred_ratings = torch.sum(pred_ratings, dim=1)
        pred_ratings = torch.sigmoid(pred_ratings)

        return pred_ratings

    def get_embs(self, users, items):
        with torch.no_grad():
            all_user_embs, all_item_embs = self.compute()
            user_embs = all_user_embs[users]
            item_embs = all_item_embs[items]
            embs = user_embs * item_embs

        return embs

    def compute(self) -> (FloatTensor, FloatTensor):
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

    def __build_graph(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        adj_mat = sp.dok_matrix((self.user_num + self.item_num, self.user_num + self.item_num),
                                dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.ui_csr_mat.tolil()
        adj_mat[:self.user_num, self.user_num:] = R
        adj_mat[self.user_num:, :self.user_num] = R.T
        adj_mat.todok()

        row_sum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(row_sum, -0.5).flatten()
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
        graph = graph.coalesce().to(device)

        return graph
