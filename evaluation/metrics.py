import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


def calculate_ndcg(user_df: pd.DataFrame, k: int) -> float:
    pred_scores = user_df['score_pred'].values
    gt_scores = user_df['score'].values
    
    pred_range = pred_scores.max() - pred_scores.min()
    if pred_range == 0:
        pred_normalized = np.zeros_like(pred_scores)
    else:
        pred_normalized = (pred_scores - pred_scores.min()) / pred_range
    
    gt_range = gt_scores.max() - gt_scores.min()
    if gt_range == 0:
        gt_normalized = np.zeros_like(gt_scores)
    else:
        gt_normalized = (gt_scores - gt_scores.min()) / gt_range
        
    return ndcg_score(y_true=gt_normalized.reshape(1, -1),
                     y_score=pred_normalized.reshape(1, -1),
                     k=k)


def calculate_recall(user_df: pd.DataFrame, k: int, relevance_threshold: float) -> float:
    relevant_items = set(user_df[user_df['score'] > relevance_threshold]['track_id'])
    if not relevant_items:
        return 0

    top_k_items = set(user_df.nlargest(k, 'score_pred')['track_id'])
    hits = len(relevant_items & top_k_items)

    return hits / len(relevant_items)


def calculate_precision(user_df: pd.DataFrame, k: int, relevance_threshold: float) -> float:
    top_k_items = user_df.nlargest(k, 'score_pred')
    hits = len(top_k_items[top_k_items['score'] > relevance_threshold])

    return hits / len(top_k_items)


def print_metrics(metrics, k_list):
    print("\nModel Performance Metrics:")
    print("=" * 50)
    
    for k in k_list:
        print(f"\nMetrics for top-{k} recommendations:")
        print("-" * 35)
        
        metrics_at_k = {
            "NDCG": (metrics[f'NDCG@{k}_mean'] * 100, metrics[f'NDCG@{k}_std'] * 100),
            "Precision": (metrics[f'Precision@{k}_mean'] * 100, metrics[f'Precision@{k}_std'] * 100),
            "Recall": (metrics[f'Recall@{k}_mean'] * 100, metrics[f'Recall@{k}_std'] * 100)
        }
        
        for metric_name, (mean, std) in metrics_at_k.items():
            print(f"{metric_name:10} = {mean:6.2f}% ± {std:6.2f}%")
