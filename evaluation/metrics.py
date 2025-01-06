import numpy as np
import torch
from sklearn.metrics import ndcg_score

def calculate_ndcg(user_df, k):
    top_k = user_df.nlargest(k, 'score_pred')

    sigmoid = torch.nn.Sigmoid()

    normalized_pred = sigmoid(torch.tensor(user_df['score_pred'].values)).numpy()
    normalized_gt = sigmoid(torch.tensor(user_df['score'].values)).numpy()

    return ndcg_score(y_true=normalized_gt.reshape(1, -1),
                      y_score=normalized_pred.reshape(1, -1),
                      k=k)


def calculate_recall(user_df, k, relevance_threshold):
    relevant_items = set(user_df[user_df['score'] > relevance_threshold]['track_id'])
    if not relevant_items:
        return 0

    top_k_items = set(user_df.nlargest(k, 'score_pred')['track_id'])
    hits = len(relevant_items & top_k_items)

    return hits / len(relevant_items)


def calculate_precision(user_df, k, relevance_threshold):
    top_k_items = user_df.nlargest(k, 'score_pred')
    hits = len(top_k_items[top_k_items['score'] > relevance_threshold])

    return hits / len(top_k_items)
