import time
from typing import Tuple

import numpy as np
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from dataset import Dataset
from models import BaseModel
from util import timer


class Trainer:
    def __init__(self, dataset: Dataset, model: BaseModel, optimizer: Optimizer, batch_size=512):
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size

    @timer
    def train(self):
        self.model.train()
        
        # Sample the training data in every epoch.
        train_loader = DataLoader(self.dataset.get_train_data(), batch_size=self.batch_size, shuffle=True)

        loss_list = []
        for batch_data in train_loader:
            loss = self.model.forward(batch_data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_list.append(loss.cpu().item())

        return np.mean(loss_list).item()
