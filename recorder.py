from typing import List, Tuple

import numpy as np

import util


class Recorder:
    def __init__(self, interval: int, cmp_key='hr'):
        self.interval = interval
        self.cmp_key = cmp_key

        self.epoch_list = []
        self.hr_list = []
        self.ndcg_list = []
        self.mep_list = []
        self.wmep_list = []

        self.best_epoch = 0
        self.best_hr = 0
        self.best_ndcg = 0
        self.best_mep = 0
        self.best_wmep = 0

        self.is_update = True
        self.no_improve_cnt = 0

        if cmp_key not in ('hr', 'ndcg', 'mep', 'wmep'):
            raise ValueError('Metric: {} is not available'.format(cmp_key))

    def record(self, epoch: int, hr: float, ndcg: float, mep: float, wmep: float) -> Tuple[bool, int]:
        self.epoch_list.append(epoch)
        self.hr_list.append(hr)
        self.ndcg_list.append(ndcg)
        self.mep_list.append(mep)
        self.wmep_list.append(wmep)

        self.is_update = False
        if self.cmp_key == 'hr':
            if self.best_hr <= hr:
                self.__update(epoch, hr, ndcg, mep, wmep)
        elif self.cmp_key == 'ndcg':
            if self.best_ndcg <= ndcg:
                self.__update(epoch, hr, ndcg, mep, wmep)
        elif self.cmp_key == 'mep':
            if self.best_mep <= mep:
                self.__update(epoch, hr, ndcg, mep, wmep)
        else:
            if self.best_wmep <= wmep:
                self.__update(epoch, hr, ndcg, mep, wmep)

        if self.is_update:
            self.no_improve_cnt = 0
        else:
            self.no_improve_cnt += self.interval

        return self.is_update, self.no_improve_cnt

    def print_best(self, model_name: str, keys: List[str]):
        for key in keys:
            if key not in ('hr', 'ndcg', 'mep', 'wmep'):
                raise ValueError('Metric: {} is not available'.format(key))

        for idx, key in enumerate(keys):
            epoch, hr, ndcg, mep, wmep = self.__find_best(key)
            res_str = '{} Best {:>4}: epoch={:>3d}, hr={:.4f}, ndcg={:.4f}, mep={:.4f}, wmep={:.4f}'.format(
                model_name.upper(), key.upper(), epoch, hr, ndcg, mep, wmep)
            util.sep_print(res_str, end=False) if idx != len(keys) - 1 else util.sep_print(res_str)

    def __find_best(self, key='hr') -> Tuple[int, float, float, float, float]:
        if key == 'hr':
            idx = np.argsort(self.hr_list)[-1]
        elif key == 'ndcg':
            idx = np.argsort(self.ndcg_list)[-1]
        elif key == 'mep':
            idx = np.argsort(self.mep_list)[-1]
        elif key == 'wmep':
            idx = np.argsort(self.wmep_list)[-1]
        else:
            raise ValueError('Metric: {} is not available'.format(key))

        epoch = self.epoch_list[idx]
        hr = self.hr_list[idx]
        ndcg = self.ndcg_list[idx]
        mep = self.mep_list[idx]
        wmep = self.wmep_list[idx]

        return epoch, hr, ndcg, mep, wmep

    def __update(self, epoch, hr, ndcg, mep, wmep):
        self.best_epoch = epoch
        self.best_hr = hr
        self.best_ndcg = ndcg
        self.best_mep = mep
        self.best_wmep = wmep
        # We have updated the best result
        self.is_update = True
