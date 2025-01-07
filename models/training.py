import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


class GMFTrainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def train_with_weight_decay(self, epochs, learning_rate, weight_decay=0.01):
        self.model = self.model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        epoch_train_losses, epoch_val_losses = [], []
        
        for epoch in range(epochs):
            train_losses, val_losses = [], []
            self.model.train()

            for user_ids, item_ids, labels in self.train_loader:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                labels = labels.to(self.device)

                predictions = self.model(user_ids, item_ids)
                train_loss = criterion(predictions, labels)
                train_losses.append(train_loss.item())

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

            self.model.eval()
            for user_ids, item_ids, labels in self.val_loader:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                labels = labels.to(self.device)
                with torch.no_grad():
                    predictions = self.model(user_ids, item_ids)

                val_loss = criterion(predictions, labels)
                val_losses.append(val_loss.item())

            epoch_train_loss = np.mean(train_losses)
            epoch_val_loss = np.mean(val_losses)
            epoch_train_losses.append(epoch_train_loss)
            epoch_val_losses.append(epoch_val_loss)
            print(f'Epoch: {epoch}; Train loss: {epoch_train_loss}; Val loss: {epoch_val_loss}')
        return epoch_train_losses, epoch_val_losses

            


