import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Dataset
from model import BaseModel


class MF(BaseModel):
    def __init__(self, dataset: Dataset, model_config: dict):
        super(MF, self).__init__(dataset)
        self.latent_dim = model_config['latent_dim']

        self.embed_user = nn.Embedding(self.user_num, self.latent_dim)
        self.embed_item = nn.Embedding(self.item_num, self.latent_dim)

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

        loss = torch.mean(F.softplus(neg_ratings - pos_ratings))

        return loss

    def predict(self, users, items):
        users = torch.LongTensor(users).to(self.device)
        items = torch.LongTensor(items).to(self.device)

        user_embs = self.embed_user(users)
        item_embs = self.embed_item(items)

        pred_ratings = torch.sum(user_embs * item_embs, dim=1)
        pred_ratings = torch.sigmoid(pred_ratings)

        return pred_ratings
