import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys
sys.path.append("..")
from dataclasses import dataclass
from typing import Callable


@dataclass
class Config:
    train_dataloader: DataLoader
    test_dataloader: DataLoader
    logging_fn: Callable
    save_path: str
    lr: float = 1e-2
    epochs: int = 10
    batch_size: int = 64

def probabilistic_xentropy(y_pred_batch, y_batch):
    y_pred_batch, _ = y_pred_batch
    return nn.CrossEntropyLoss()(y_pred_batch, y_batch)

def accuracy(y_pred_batch, y_batch):
    _, y_pred_batch = y_pred_batch.topk(1, dim=1)
    y_pred_batch = y_pred_batch.squeeze()
    y_batch = y_batch.squeeze()
    return torch.sum(y_pred_batch == y_batch)

def probabilistic_accuracy(y_pred_batch, y_batch):
    y_pred_batch, _ = y_pred_batch
    return accuracy(y_pred_batch, y_batch)

class Trainer():
    def __init__(self, base_model, config, loss = nn.CrossEntropyLoss(), accuracy_metric = accuracy):
        self.base_model = base_model
        self.config = config
        self.optimizer = torch.optim.SGD(self.base_model.parameters(), lr = self.config.lr)
        self.loss = loss
        self.accuracy_metric = accuracy_metric
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    
    def train(self):
        sum_loss = 0
        for i, (x_batch, y_batch) in tqdm(enumerate(self.config.train_dataloader)):
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            self.optimizer.zero_grad()
            y_pred_batch = self.base_model(x_batch)
            loss = self.loss(y_pred_batch, y_batch)
            loss.backward()
            self.optimizer.step()
            sum_loss += loss
        return sum_loss / i
    
    def test(self):
        sum_loss = 0
        accuracy = 0
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(self.config.test_dataloader):
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                y_pred_batch = self.base_model(x_batch)
                loss = self.loss(y_pred_batch, y_batch)
                sum_loss += loss
                accuracy += self.accuracy_metric(y_pred_batch, y_batch)
        accuracy = accuracy/(i*self.config.batch_size)
        return sum_loss/i, accuracy
    
    def run_training_loop(self):
        for epoch in range(self.config.epochs):
            print("training on epoch", epoch, "out of", self.config.epochs)
            train_loss = self.train()
            test_loss, test_acc = self.test()
            self.config.logging_fn(train_loss, test_loss, test_acc)
            torch.save(self.base_model.state_dict(), self.config.save_path)
