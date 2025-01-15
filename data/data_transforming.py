from sklearn.preprocessing import LabelEncoder
import numpy as np

class IDEncoder:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self._is_fitted = False
    
    def fit(self, hash_ids):
        self.label_encoder.fit(hash_ids)
        self._is_fitted = True

    def encode(self, hash_ids):
        if not self._is_fitted:
            raise ValueError("Encoder must be trained first using method fit()")

        numeric_ids = self.label_encoder.transform(hash_ids)

        return numeric_ids.tolist()

    def decode(self, numeric_ids):
        if not self._is_fitted:
            raise ValueError("Encoder must be trained first using method fit()")

        hash_ids = self.label_encoder.inverse_transform(np.array(numeric_ids))
        return hash_ids.tolist()

    def __len__(self):
        if not self._is_fitted:
            return 0
        return len(self.label_encoder.classes_)


def calculate_score(events):
    score = sum(1.0 if e == 'like' else 0.1 if e == 'play' 
                   else -1.0 if e == 'skip' else 0.0 for e in events)
    return score


def aggregate_interactions(df):
    grouped = df.groupby(['user_id', 'track_id'])['event_type'].agg(list).reset_index()
    grouped['score'] = grouped['event_type'].apply(calculate_score)
    return grouped[['user_id', 'track_id', 'score']]
