import torch
import torch.nn as nn

from dataset import Dataset


class BaseModel(nn.Module):
    def __init__(self, dataset: Dataset):
        super(BaseModel, self).__init__()
        self.user_num = dataset.user_num
        self.item_num = dataset.item_num

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, *args):
        raise NotImplementedError

    def predict(self, *args):
        raise NotImplementedError
