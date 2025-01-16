from numpy import void
import pandas as pd
import json


def remove_sessions_with_null(df: pd.DataFrame):
    df = df[df['event_type'] != 'advertisment']
    df = df[df['track_id'].notna()]
    df['user_id'] = df['user_id'].fillna(method='ffill')
    df = df.dropna(subset=['user_id', 'track_id', 'event_type'])
    df = df.sort_values(['user_id', 'timestamp'])
    return df


def remove_sessions_with_wrong_track_id(tracks_df: pd.DataFrame, sessions_df: pd.DataFrame) -> pd.DataFrame:
    valid_track_ids = set(tracks_df['id'].dropna().unique())
    return sessions_df[sessions_df['track_id'].isin(valid_track_ids)]


def clean_sessions_jsonl(tracks_filepath: str, sessions_filepath: str, output_filepath:str, chunk_size: int = 100000) -> None:
    tracks_df = pd.read_json(tracks_filepath, lines=True)
    with open(sessions_filepath, 'r') as fin, open(output_filepath, 'w') as fout:
        chunk = []
        for line in fin:
            chunk.append(json.loads(line))
            if len(chunk) >= chunk_size:
                df = remove_sessions_with_null(pd.DataFrame(chunk))
                df = remove_sessions_with_wrong_track_id(tracks_df, df)
                if df is not None:
                    df.to_json(fout, orient='records', lines=True, mode='a')
                chunk.clear()
        if chunk:
            df = remove_sessions_with_null(pd.DataFrame(chunk))
            df = remove_sessions_with_wrong_track_id(tracks_df, df)
            if df is not None:
                df.to_json(fout, orient='records', lines=True, mode='a')
