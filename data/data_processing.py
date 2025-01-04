import pandas as pd
import numpy as np

class SessionDataProcessor:
    def __init__(self):
        self.user_mapping = {}
        self.track_mapping = {}
        self.next_user_id = 0
        self.next_track_id = 0
        
    def _map_user_id(self, original_user_id):
       if original_user_id not in self.user_mapping:
           self.user_mapping[original_user_id] = self.next_user_id
           self.next_user_id += 1
       return self.user_mapping[original_user_id]
    
    def _map_track_id(self, original_track_id):
       if original_track_id not in self.track_mapping:
           self.track_mapping[original_track_id] = self.next_track_id 
           self.next_track_id += 1
       return self.track_mapping[original_track_id]

    def prepare_training_data(self, train_df):
        grouped = train_df.groupby(['user_id', 'track_id'])['event_type'].agg(list).reset_index()
        
        processed_data = []
        for _, row in grouped.iterrows():
            mapped_user_id = self._map_user_id(row['user_id'])
            mapped_track_id = self._map_track_id(row['track_id'])
            score = calculate_interaction_score(row['event_type'])
            processed_data.append((mapped_user_id, mapped_track_id, score))
        
        return pd.DataFrame(processed_data, columns=['user_id', 'track_id', 'score'])    

    @property
    def n_users(self):
        return len(self.user_mapping)
        
    @property
    def n_tracks(self):
        return len(self.track_mapping)


def calculate_interaction_score(events):
    score = sum(-1.0 if e == 'skip' else 0.1 if e == 'play' else 1.0 if e == 'like' else 0.0 for e in events)
    return score
