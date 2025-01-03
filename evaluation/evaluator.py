from metrics import calculate_ndcg, calculate_recall, calculate_precision


class Evaluator:
    def __init__(self, k_values):
        self.k_values = k_values

    def evaluate(self,
                 predictions_df,
                 ground_truth_df):
        merged_df = predictions_df.merge(
            ground_truth_df,
            on=['user_id', 'item_id'],
            how='left',
            suffixes=('_pred', '')
        ).fillna({'score': 0})

        metrics = {}

        for k in self.k_values:
            ndcg_scores = merged_df.groupby('user_id').apply(
                lambda x: calculate_ndcg(x, k)   
            )

            recall_scores = merged_df.groupby('user_id').apply(
                lambda x: calculate_recall(x, k)
            )

            precision_scores = merged_df.groupby('user_id').apply(
                lambda x: calculate_precision(x, k)
            )

            metrics[f'NDCG@{k}'] = ndcg_scores.mean()
            metrics[f'NDCG@{k}_std'] = ndcg_scores.std()
            metrics[f'Recall@{k}'] = recall_scores.mean()
            metrics[f'Recall@{k}_std'] = recall_scores.std()
            metrics[f'Precision@{k}'] = precision_scores.mean()
            metrics[f'Precision@{k}_std'] = precision_scores.std()

        return metrics


