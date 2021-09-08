import numpy as np
import torch
from torch.utils.data import DataLoader

from model import BaseModel
from util import timer


class Tester:
    def __init__(self, loader: DataLoader, model: BaseModel, topk=10):
        self.loader = loader
        self.model = model
        self.topk = topk

        self.test = self.__leave_one_out_test

    # Todo: alternative evaluation strategies setting
    def set_test_method(self, test_method: str):
        pass

    @timer
    def __leave_one_out_test(self) -> (float, float, float):
        self.model.eval()

        hr_list, ndcg_list = [], []
        for batch_data in self.loader:
            users = batch_data[:, 0]
            items = batch_data[:, 1]

            with torch.no_grad():
                pred_ratings = self.model.predict(users, items)
                pred_ratings = torch.squeeze(pred_ratings)
                _, idxs = torch.topk(pred_ratings, self.topk)
                items = torch.LongTensor(items).to(self.model.device)
                rec_list = torch.take(items, idxs).cpu().numpy().tolist()

            pos_item = items[0].item()
            hr_list.append(self.calc_hr(pos_item, rec_list))
            ndcg_list.append(self.calc_ndcg(pos_item, rec_list))

        return np.mean(hr_list).item(), np.mean(ndcg_list).item()

    @staticmethod
    def calc_hr(iid, rec_list) -> float:
        if iid in rec_list:
            return 1
        return 0

    @staticmethod
    def calc_ndcg(iid, rec_list) -> float:
        if iid in rec_list:
            idx = rec_list.index(iid)
            return np.reciprocal(np.log2(idx + 2))
        return 0
