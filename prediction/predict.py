import torch

def predict(model, data_loader, device):
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


