import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

class SessionsEncoder:
    def __init__(self):
        self.encoders = defaultdict(LabelEncoder)
    
    def fit_transform(self, df):
        """Fits encoders and transforms the IDs in the dataframe."""
        df_copy = df.copy()
        id_columns = ['user_id', 'track_id']
        for col in id_columns:
            self.encoders[col].fit(df_copy[col].unique())
            df_copy[col] = self.encoders[col].transform(df_copy[col])
        return df_copy
    
    def transform(self, df):
        """Transforms IDs using previously fitted encoders."""
        df_copy = df.copy()
        for col in ['user_id', 'track_id']:
            df_copy[col] = self.encoders[col].transform(df_copy[col])
        return df_copy
    
    @property
    def n_users(self):
        return len(self.encoders['user_id'].classes_) if 'user_id' in self.encoders else 0
    
    @property
    def n_tracks(self):
        return len(self.encoders['track_id'].classes_) if 'track_id' in self.encoders else 0


def calculate_score(events):
    score = sum(1.0 if e == 'like' else 0.1 if e == 'play' 
                   else -1.0 if e == 'skip' else 0.0 for e in events)
    return score

def aggregate_interactions(df):
    grouped = df.groupby(['user_id', 'track_id'])['event_type'].agg(list).reset_index()
    grouped['score'] = grouped['event_type'].apply(calculate_score)
    return grouped[['user_id', 'track_id', 'score']]
