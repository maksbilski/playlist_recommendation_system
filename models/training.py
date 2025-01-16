from typing import List, Tuple
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, predictions: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        loss = weights * (labels - predictions) ** 2
        return loss.mean()


class Trainer:
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device
                 ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def train_with_weight_decay(self, 
                                epochs: int, 
                                learning_rate: float, 
                                weight_decay: float = 0.01
                                ) -> Tuple[List[float], List[float]]:
        self.model = self.model.to(self.device)
        criterion = WeightedMSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        epoch_train_losses, epoch_val_losses = [], []
        
        for epoch in range(epochs):
            train_losses, val_losses = [], []
            self.model.train()

            for user_ids, item_ids, labels, weights in self.train_loader:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                labels = labels.to(self.device)
                weights = weights.to(self.device)

                predictions = self.model(user_ids, item_ids)
                train_loss = criterion(predictions, labels, weights)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                train_loss = criterion(predictions, labels, weights)
                train_losses.append(train_loss.item())

            self.model.eval()
            for user_ids, item_ids, labels, weights in self.val_loader:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                labels = labels.to(self.device)
                weights = weights.to(self.device)

                with torch.no_grad():
                    predictions = self.model(user_ids, item_ids)

                val_loss = criterion(predictions, labels, weights)
                val_losses.append(val_loss.item())

            epoch_train_loss = np.mean(train_losses)
            epoch_val_loss = np.mean(val_losses)
            epoch_train_losses.append(epoch_train_loss)
            epoch_val_losses.append(epoch_val_loss)
            print(f'Epoch: {epoch}; Train loss: {epoch_train_loss}; Val loss: {epoch_val_loss}')
        return epoch_train_losses, epoch_val_losses

            


