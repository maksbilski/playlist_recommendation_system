import numpy as np
import pandas as pd
import torch
from itertools import product
from models.advanced_model import WMF
from prediction.predict import predict


class AdvancedModel:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_data = torch.load(model_path)
        config = model_data["model_config"]
        self.user_encoder = model_data["user_encoder"]
        self.track_encoder = model_data["track_decoder"]

        self.model = WMF(
                n_users=len(self.user_encoder),
                n_items=len(self.track_encoder),
                embedding_dim=config["embedding_dim"],
                dropout_rate=config["dropout_rate"],
                bias=config["bias"],
                sigmoid=config["sigmoid"],
                init=False
                )
        self.model.load_state_dict(model_data["model_state_dict"])
        self.unique_tracks_count = len(self.track_encoder)

    def get_predictions(self, user_ids, n_recommendations):
        users = np.array(self.user_encoder.encode(user_ids))
        tracks = np.arange(self.unique_tracks_count)
        df = pd.DataFrame(
                product(users, tracks),
                columns=['user_id', 'track_id']
                )
        pred_data = torch.utils.data.TensorDataset(
            torch.LongTensor(df['user_id'].values),
            torch.LongTensor(df['track_id'].values),
            )

        pred_loader = torch.utils.data.DataLoader(
            pred_data,
            batch_size=512
            shuffle=False
            )

        df["score_pred"] = predict(self.model, pred_loader, self.device)

        group_recommendations = (df
                               .groupby('track_id')['score_pred']
                               .mean()
                               .reset_index()
                               .sort_values('score_pred', ascending=False)
                               .head(n_recommendations))

        group_recommendations["track_id"] = self.track_encoder.decode(
                group_recommendations["track_id"].astype(int).tolist()
                )

        return group_recommendations["track_id"].tolist()
