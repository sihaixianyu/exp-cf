from typing import Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader

from models import BaseModel
from util import timer


class Tester:
    def __init__(self, loader: DataLoader, model: BaseModel, full_exp_mat: np.ndarray, topk=10):
        self.loader = loader
        self.model = model
        self.full_exp_mat = full_exp_mat
        self.topk = topk

        self.test = self.__leave_one_out_test

    # Todo: alternative evaluation strategies setting
    def set_test_method(self, test_method: str):
        pass

    @timer
    def __leave_one_out_test(self):
        self.model.eval()

        hr_list, ndcg_list, mep_list, wmep_list = [], [], [], []
        for batch_data in self.loader:
            users = batch_data[:, 0]
            items = batch_data[:, 1]

            with torch.no_grad():
                pred_ratings = self.model.predict(users, items)
                pred_ratings = torch.squeeze(pred_ratings)
                _, idxs = torch.topk(pred_ratings, self.topk)
                items = torch.LongTensor(items).to(self.model.device)
                rec_list = torch.take(items, idxs).cpu().numpy().tolist()
                exp_list = self.full_exp_mat[users.cpu().numpy()[0]].tolist()

            pos_item = items[0].item()
            hr_list.append(self.calc_hr(pos_item, rec_list))
            ndcg_list.append(self.calc_ndcg(pos_item, rec_list))
            mep_list.append(self.calc_mep(rec_list, exp_list))
            wmep_list.append(self.calc_wmep(rec_list, exp_list))

        avg_hr = np.mean(hr_list).item()
        avg_ndcg = np.mean(ndcg_list).item()
        avg_mep = np.mean(mep_list).item()
        avg_wmep = np.mean(wmep_list).item()

        return avg_hr, avg_ndcg, avg_mep, avg_wmep

    @staticmethod
    def calc_hr(tar_item, rec_list):
        if tar_item in rec_list:
            return 1

        return 0

    @staticmethod
    def calc_ndcg(tar_item, rec_list):
        if tar_item in rec_list:
            idx = rec_list.index(tar_item)
            return np.reciprocal(np.log2(idx + 2))

        return 0

    @staticmethod
    def calc_mep(rec_list, exp_list):
        exp_sum = 0
        for item in rec_list:
            exp_val = exp_list[item]
            exp_sum += 1 if exp_val > 0 else 0

        return exp_sum / len(rec_list)

    @staticmethod
    def calc_wmep(rec_list, exp_list):
        exp_sum = 0
        for item in rec_list:
            exp_val = exp_list[item]
            exp_sum += exp_val if exp_val > 0 else 0

        return exp_sum / len(rec_list)
