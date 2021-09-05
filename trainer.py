import time

import numpy as np
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, loader: DataLoader, model: Module, optimizer: Optimizer):
        self.loader = loader
        self.model = model
        self.optimizer = optimizer

    def train(self) -> (float, float):
        train_start = time.time()
        self.model.train()

        loss_list = []
        for batch_data in self.loader:
            loss = self.model.forward(batch_data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_list.append(loss.cpu().item())

        train_time = time.time() - train_start
        return np.mean(loss_list).item(), train_time
