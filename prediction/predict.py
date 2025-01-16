import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

def predict(model: nn.Module, data_loader: DataLoader, device: torch.device) -> np.ndarray:
    model = model.to(device)
    model.eval()
    predictions = []
    for user_ids, item_ids in data_loader:
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        with torch.no_grad():
            batch_predictions = model(user_ids, item_ids)
        predictions.append(batch_predictions.cpu())

    return torch.cat(predictions).numpy()


