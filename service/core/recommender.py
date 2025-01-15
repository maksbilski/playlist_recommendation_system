import logging
import json
from datetime import datetime
import random
from typing import List, Dict
from service_models import AdvancedModel

class GroupPlaylistService:
    def __init__(self, data_path: str, model_path: str, log_path: str):
        self.models = {
            'basic': BasicModel(data_path),
            'advanced': AdvancedModel(model_path)
        }
        self.weights = {'advanced': 0.5, 'basic': 0.5}
        self.log_path = log_path
        
        logging.basicConfig(
            filename=f"{log_path}/ab_test_{datetime.now().strftime('%Y%m%d')}.log",
            level=logging.INFO,
            format='%(asctime)s\t%(message)s'
        )
    
    def get_recommendations(self, user_ids: List[str], n: int = 30):
        model_type = random.choices(
            list(self.weights.keys()), 
            weights=list(self.weights.values())
        )[0]
        
        model = self.models[model_type]
        tracks = model.get_recommendations(user_ids, n)
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'model': model_type,
            'group_size': len(user_ids),
            'user_ids': user_ids,
            'recommended_tracks': tracks,
            'n_requested': n
        }
        logging.info(json.dumps(log_entry))
        
        return {
            "tracks": tracks,
            "group_size": len(user_ids)
        }
