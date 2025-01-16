import pandas as pd
import random
from typing import List
from .model_interface import ModelInterface

class BasicModel(ModelInterface):
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        self.user_plays = self._prepare_user_data()

    def _prepare_user_data(self) -> pd.DataFrame:
        df = pd.read_json(self.data_path, lines=True)
        # play_counts = df[df['event_type'] == 'play'].groupby(
        #     ['user_id', 'track_id']
        # ).size().reset_index(name='play_count')
        # return play_counts
        return df[['user_id', 'track_id', 'score']].rename(columns={'score': 'play_count'})

    def get_recommendations(self, user_ids: List[str], n: int = 30) -> List[str]:
        target_group_ids = [int(uid) for uid in user_ids]

        top_tracks_per_user = {}
        for user_id in target_group_ids:
            user_plays = self.user_plays[self.user_plays['user_id'] == user_id]
            if not user_plays.empty:
                top_50 = user_plays.nlargest(50, 'play_count')['track_id'].tolist()
                top_tracks_per_user[user_id] = set(top_50)

        if not top_tracks_per_user:
            return []

        track_frequency = {}
        all_top_tracks = set().union(*top_tracks_per_user.values())

        for track_id in all_top_tracks:
            frequency = sum(1 for user_tracks in top_tracks_per_user.values() 
                          if track_id in user_tracks)
            track_frequency[track_id] = frequency

        tracks_by_frequency = {}
        for track_id, freq in track_frequency.items():
            tracks_by_frequency.setdefault(freq, []).append(track_id)

        recommended_tracks = []
        frequencies = sorted(tracks_by_frequency.keys(), reverse=True)

        for freq in frequencies:
            tracks = tracks_by_frequency[freq]
            if len(recommended_tracks) + len(tracks) <= n:
                recommended_tracks.extend(tracks)
            else:
                needed = n - len(recommended_tracks)
                recommended_tracks.extend(random.sample(all_top_tracks, needed))
                break

            if len(recommended_tracks) >= n:
                break

        return [str(track_id) for track_id in recommended_tracks[:n]]

    def update_data(self, new_data_path: str) -> None:
        self.data_path = new_data_path
        self.user_plays = self._prepare_user_data()
