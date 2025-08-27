# pylint: disable=import-outside-toplevel
from typing import Callable, List, Any

from gptcache.utils.error import NotFoundError


class EvictionBase:
    """
    EvictionBase to evict the cache data.
    """

    def __init__(self):
        raise EnvironmentError(
            "EvictionBase is designed to be instantiated, "
            "please using the `EvictionBase.get(name, policy, maxsize, clean_size)`."
        )

    @staticmethod
    def get(
        name: str,
        policy: str = "LRU",
        maxsize: int = 1000,
        clean_size: int = 0,
        on_evict: Callable[[List[Any]], None] = None,
        **kwargs
    ):
        if not clean_size:
            clean_size = int(maxsize * 0.2)
        if name in "memory":
            from gptcache.manager.eviction.memory_cache import MemoryCacheEviction

            eviction_base = MemoryCacheEviction(
                policy, maxsize, clean_size, on_evict, **kwargs
            )
            return eviction_base
        if name == "redis":
            from gptcache.manager.eviction.redis_eviction import RedisCacheEviction
            if policy == "LRU":
                policy = None
            eviction_base = RedisCacheEviction(policy=policy, **kwargs)
            return eviction_base
        if name == "no_op_eviction":
            from gptcache.manager.eviction.distributed_cache import NoOpEviction
            eviction_base = NoOpEviction()
            return eviction_base
        if name == "quality_score":
            from gptcache.manager.eviction.quality_score import QualityScoreEviction
            
            # Only pass quality score parameters if they're not None
            quality_params = {}
            if 'learning_rate' in kwargs and kwargs['learning_rate'] is not None:
                quality_params['learning_rate'] = kwargs['learning_rate']
            if 'quality_weight' in kwargs and kwargs['quality_weight'] is not None:
                quality_params['quality_weight'] = kwargs['quality_weight']
            if 'recency_weight' in kwargs and kwargs['recency_weight'] is not None:
                quality_params['recency_weight'] = kwargs['recency_weight']
            if 'frequency_weight' in kwargs and kwargs['frequency_weight'] is not None:
                quality_params['frequency_weight'] = kwargs['frequency_weight']
            
            # Get remaining kwargs excluding quality score parameters
            other_params = {k: v for k, v in kwargs.items() if k not in ['learning_rate', 'quality_weight', 'recency_weight', 'frequency_weight']}
            
            eviction_base = QualityScoreEviction(
                maxsize=maxsize, 
                clean_size=clean_size, 
                on_evict=on_evict, 
                **quality_params,
                **other_params
            )
            return eviction_base

        else:
            raise NotFoundError("eviction base", name)
