import torch.nn as nn
import torch.optim as optim


class GMFTrainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def train(self, epochs, learning_rate):
        self.model = self.model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)    
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, (user_ids, item_ids, labels) in enumerate(self.train_loader):
                predictions = self.model(user_ids, item_ids)
                loss = criterion(predictions, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(self.train_loader)
            print(f'Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}')
