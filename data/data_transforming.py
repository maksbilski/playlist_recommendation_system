from typing import List
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

class IDEncoder:
    def __init__(self) -> None:
        self.label_encoder = LabelEncoder()
        self._is_fitted = False
    
    def fit(self, hash_ids: List[str]) -> None:
        self.label_encoder.fit(hash_ids)
        self._is_fitted = True

    def encode(self, hash_ids: List[str]) -> List[int]:
        if not self._is_fitted:
            raise ValueError("Encoder must be trained first using method fit()")

        numeric_ids = self.label_encoder.transform(hash_ids)

        return numeric_ids.tolist()

    def decode(self, numeric_ids: List[int]) -> List[str]:
        if not self._is_fitted:
            raise ValueError("Encoder must be trained first using method fit()")

        hash_ids = self.label_encoder.inverse_transform(np.array(numeric_ids))
        return hash_ids.tolist()

    def __len__(self) -> int:
        if not self._is_fitted:
            return 0
        return len(self.label_encoder.classes_)


def calculate_score(events: List[str]) -> float:
    score = sum(1.0 if e == 'like' else 1.00 if e == 'play' 
                   else 0.0 if e == 'skip' else 0.0 for e in events)
    return score


def aggregate_interactions(df: pd.DataFrame) -> pd.Series | pd.DataFrame:
    grouped = df.groupby(['user_id', 'track_id'])['event_type'].agg(list).reset_index()
    grouped['score'] = grouped['event_type'].apply(calculate_score)
    return grouped[['user_id', 'track_id', 'score']]
