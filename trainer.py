import time
from typing import Tuple

import numpy as np
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from models import BaseModel
from util import timer


class Trainer:
    def __init__(self, loader: DataLoader, model: BaseModel, optimizer: Optimizer):
        self.loader = loader
        self.model = model
        self.optimizer = optimizer

    @timer
    def train(self) -> (float, float):
        self.model.train()

        loss_list = []
        for batch_data in self.loader:
            loss = self.model.forward(batch_data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_list.append(loss.cpu().item())

        return np.mean(loss_list).item()
