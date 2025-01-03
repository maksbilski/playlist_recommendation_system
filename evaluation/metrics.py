import numpy as np

def calculate_ndcg(user_df, k):
    top_k = user_df.nlargest(k, 'score_pred')
    dcg = np.sum(top_k['score'] / np.log2(np.arange(1, len(top_k) + 1)))

    ideal_order = user_df.nlargest(k, 'score')
    idcg = np.sum(ideal_order['score'] / np.log2(np.arange(1, len(ideal_order) + 1)))

    return dcg / idcg


def calculate_recall(user_df, k):
    relevant_items = set(user_df[user_df['score'] > 0.6]['item_id'])
    if not relevant_items:
        return 0

    top_k_items = set(user_df.nlargest(k, 'score_pred')['item_id'])
    hits = len(relevant_items & top_k_items)

    return hits / len(relevant_items)


def calculate_precision(user_df, k):
    top_k_items = user_df.nlargest(k, 'score_pred')
    hits = len(top_k_items[top_k_items['score'] > 0.6])

    return hits / len(top_k_items)
