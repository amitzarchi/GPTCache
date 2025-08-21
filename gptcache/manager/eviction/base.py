from abc import ABCMeta, abstractmethod
from typing import Any, List


class EvictionBase(metaclass=ABCMeta):
    """
    Eviction base.
    """

    @abstractmethod
    def put(self, objs: List[Any]):
        pass

    @abstractmethod
    def get(self, obj: Any):
        pass

    @property
    @abstractmethod
    def policy(self) -> str:
        pass

    def update_quality(self, obj: Any, similarity_score: float):
        """Optional: Update quality metrics based on similarity score.
        
        This method allows quality-aware eviction policies to learn from 
        similarity evaluation results and make more intelligent eviction decisions.
        
        Default implementation does nothing for backward compatibility.
        Quality-aware eviction policies should override this method.
        
        :param obj: Object ID whose quality to update
        :type obj: Any
        :param similarity_score: Similarity score from cache hit evaluation  
        :type similarity_score: float
        """
        pass