from typing import List, Dict, Any
import pandas as pd
from collections import defaultdict
from datetime import datetime
import random

class BasicModel:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.users_data = self._prepare_user_data()
    
    def _prepare_user_data(self) -> Dict:
        df = pd.read_csv(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        play_counts = df[df['event_type'] == 'play'].groupby(
            ['user_id', 'track_id']
        ).size().reset_index(name='play_count')
        
        latest_timestamps = df.groupby(['user_id', 'track_id'])['timestamp'].max().reset_index()
        
        user_history = pd.merge(play_counts, latest_timestamps, on=['user_id', 'track_id'])
        
        users_data = {}
        for user_id, group in user_history.groupby('user_id'):
            users_data[user_id] = [
                (row['track_id'], row['timestamp'], row['play_count'])
                for _, row in group.iterrows()
            ]
        
        return users_data

    def get_recommendations(self, user_ids: List[str], n: int = 30) -> List[str]:
        target_group_ids = [int(uid) for uid in user_ids]
        return [str(track_id) for track_id in self._get_group_playlist_recommendations(
            target_group_ids=target_group_ids,
            playlist_size=n
        )]

    def _get_group_playlist_recommendations(
        self,
        target_group_ids: List[int],
        playlist_size: int = 50,
        strategy: str = "average"
    ) -> List[int]:
        users_top_tracks = {}
        current_time = datetime.now()
        
        for user_id in target_group_ids:
            if user_id not in self.users_data:
                continue
                
            user_history = self.users_data[user_id]
            sorted_tracks = sorted(user_history, key=lambda x: x[2], reverse=True)
            top_50_tracks = sorted_tracks[:50]
            
            weighted_tracks = []
            max_weight = 0
            
            for track_id, timestamp, play_count in top_50_tracks:
                time_diff = (current_time - timestamp).total_seconds() / (24 * 3600)
                time_weight = 1.0 / (1.0 + time_diff/365.0)
                
                combined_weight = time_weight * play_count
                max_weight = max(max_weight, combined_weight)
                weighted_tracks.append((track_id, combined_weight))
            
            if max_weight > 0:
                normalized_tracks = [(track_id, weight/max_weight) 
                                   for track_id, weight in weighted_tracks]
            else:
                normalized_tracks = weighted_tracks
                
            users_top_tracks[user_id] = normalized_tracks
        
        group_track_ratings = defaultdict(lambda: defaultdict(float))
        for user_id in target_group_ids:
            if user_id in users_top_tracks:
                for track_id, weight in users_top_tracks[user_id]:
                    group_track_ratings[track_id][user_id] = weight
        
        track_scores = self._calculate_track_scores(
            group_track_ratings,
            target_group_ids,
            strategy
        )
        
        sorted_tracks = sorted(track_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommended_tracks = self._select_recommended_tracks(
            sorted_tracks,
            group_track_ratings,
            target_group_ids,
            users_top_tracks,
            playlist_size
        )
        
        return recommended_tracks

    def _calculate_track_scores(
        self,
        group_track_ratings: Dict,
        target_group_ids: List[int],
        strategy: str
    ) -> Dict[int, float]:
        track_scores = {}
        
        if strategy == "average":
            for track_id, user_ratings in group_track_ratings.items():
                if user_ratings:
                    track_scores[track_id] = sum(user_ratings.values()) / len(target_group_ids)
        
        elif strategy == "least_misery":
            for track_id, user_ratings in group_track_ratings.items():
                if len(user_ratings) >= len(target_group_ids) * 0.5:
                    track_scores[track_id] = min(user_ratings.values())
        
        elif strategy == "maximum_satisfaction":
            for track_id, user_ratings in group_track_ratings.items():
                if user_ratings:
                    track_scores[track_id] = max(user_ratings.values())
                    
        return track_scores

    def _select_recommended_tracks(
        self,
        sorted_tracks: List[tuple],
        group_track_ratings: Dict,
        target_group_ids: List[int],
        users_top_tracks: Dict,
        playlist_size: int
    ) -> List[int]:
        recommended_tracks = []
        
        for track_id, score in sorted_tracks:
            if len(group_track_ratings[track_id]) >= len(target_group_ids) * 0.5:
                recommended_tracks.append(track_id)
                if len(recommended_tracks) == playlist_size:
                    break
        
        if len(recommended_tracks) < playlist_size:
            other_track_scores = defaultdict(float)
            other_track_users = defaultdict(int)
            
            for user_id, tracks in users_top_tracks.items():
                if user_id not in target_group_ids:
                    for track_id, norm_weight in tracks:
                        if track_id not in recommended_tracks:
                            other_track_scores[track_id] += norm_weight
                            other_track_users[track_id] += 1
            
            avg_other_scores = {
                track_id: score/other_track_users[track_id]
                for track_id, score in other_track_scores.items()
                if other_track_users[track_id] >= 2
            }
            
            remaining_tracks = sorted(avg_other_scores.items(), key=lambda x: x[1], reverse=True)
            for track_id, _ in remaining_tracks:
                if track_id not in recommended_tracks:
                    recommended_tracks.append(track_id)
                    if len(recommended_tracks) == playlist_size:
                        break
        
        return recommended_tracks

    def update_data(self, new_data_path: str) -> None:
        self.data_path = new_data_path
        self.users_data = self._prepare_user_data()
