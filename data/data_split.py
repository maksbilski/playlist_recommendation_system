import pandas as pd

def split_data(sessions_df: pd.DataFrame, train_size: float = 0.6, val_size: float = 0.2):
    sessions_sorted = sessions_df.sort_values('timestamp')
    
    n = len(sessions_sorted)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    
    train_data = sessions_sorted[:train_end]
    val_data = sessions_sorted[train_end:val_end]
    test_data = sessions_sorted[val_end:]
    
    return train_data, val_data, test_data
