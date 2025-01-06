import numpy as np

def calculate_ndcg(user_df, k):
    top_k = user_df.nlargest(k, 'score_pred')
    dcg = np.sum(top_k['score'] / np.log2(np.arange(1, k + 1) + 1))

    ideal_order = top_k.sort_values(by='score', ascending=False)
    idcg = np.sum(ideal_order['score'] / np.log2(np.arange(1, k + 1) + 1))

    if idcg == 0:
        return 0.0

    return dcg / idcg


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
