from typing import Dict, List
import pandas as pd
from .metrics import calculate_ndcg, calculate_recall, calculate_precision


class Evaluator:
    def __init__(self, k_values: List[int], relevance_threshold: float) -> None:
        self.k_values = k_values
        self.relevance_threshold = relevance_threshold 

    def evaluate(self, pred_sessions_df: pd.DataFrame) -> Dict:
        metrics = {}
        for k in self.k_values:
            ndcg_scores = pred_sessions_df.groupby('user_id').apply(
                lambda x: calculate_ndcg(x, k)   
            )

            recall_scores = pred_sessions_df.groupby('user_id').apply(
                lambda x: calculate_recall(x, k, self.relevance_threshold)
            )

            precision_scores = pred_sessions_df.groupby('user_id').apply(
                lambda x: calculate_precision(x, k, self.relevance_threshold)
            )

            metrics[f'NDCG@{k}'] = ndcg_scores
            metrics[f'NDCG@{k}_mean'] = ndcg_scores.mean()
            metrics[f'NDCG@{k}_std'] = ndcg_scores.std()
            metrics[f'Recall@{k}'] = recall_scores
            metrics[f'Recall@{k}_mean'] = recall_scores.mean()
            metrics[f'Recall@{k}_std'] = recall_scores.std()
            metrics[f'Precision@{k}'] = precision_scores
            metrics[f'Precision@{k}_mean'] = precision_scores.mean()
            metrics[f'Precision@{k}_std'] = precision_scores.std()

        return metrics


