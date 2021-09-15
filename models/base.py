import torch
import torch.nn as nn

from dataset import Dataset


class BaseModel(nn.Module):
    def __init__(self, dataset: Dataset):
        super(BaseModel, self).__init__()
        self.data_name = dataset.data_name
        self.sample_method = dataset.sample_method
        self.neighbor_num = dataset.neighbor_num
        self.user_num = dataset.user_num
        self.item_num = dataset.item_num

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, batch_data):
        raise NotImplementedError('A models must implement forward() function!')

    def predict(self, batch_users, batch_items):
        raise NotImplementedError('A models must implement predict() function!')

    def get_model_path(self, model_dir: str):
        raise NotImplementedError('A models must implement get_model_suffix() function!')
