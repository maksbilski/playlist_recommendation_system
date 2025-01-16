from abc import ABC, abstractmethod
from typing import List

class ModelInterface(ABC):
    @abstractmethod
    def get_recommendations(self, user_ids: List[str], n: int = 30) -> List[str]:
        pass
