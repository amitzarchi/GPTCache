import time
import heapq
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Dict, Tuple
import threading

from gptcache.manager.eviction.base import EvictionBase
from gptcache.utils.log import gptcache_log


@dataclass
class EntryMetadata:
    """Metadata for tracking cache entry quality and access patterns."""
    obj_id: Any
    quality_score: float = 0.0  # Exponential moving average of similarity scores
    access_count: int = 0
    last_access_time: float = field(default_factory=time.time)
    creation_time: float = field(default_factory=time.time)
    similarity_history: List[float] = field(default_factory=list)  # Recent scores for variance tracking
    
    def __post_init__(self):
        if not self.similarity_history:
            self.similarity_history = []


class QualityScoreEviction(EvictionBase):
    """Quality-based eviction policy that considers similarity scores, recency, and frequency.
    
    This eviction policy makes intelligent decisions by considering:
    - Quality: Based on similarity evaluation scores over time
    - Recency: How recently items were accessed
    - Frequency: How often items are accessed
    
    :param maxsize: Maximum number of items in cache
    :type maxsize: int
    :param clean_size: Number of items to evict when cache is full
    :type clean_size: int
    :param learning_rate: Rate at which quality scores are updated (0.0-1.0)
    :type learning_rate: float  
    :param quality_weight: Weight for quality component in composite score (default 0.6)
    :type quality_weight: float
    :param recency_weight: Weight for recency component in composite score (default 0.3)
    :type recency_weight: float
    :param frequency_weight: Weight for frequency component in composite score (default 0.1)
    :type frequency_weight: float
    :param on_evict: Callback function when items are evicted
    :type on_evict: Callable[[List[Any]], None]
    :param history_size: Number of recent similarity scores to keep for variance calculation
    :type history_size: int
    :param min_quality_threshold: Minimum quality score for new entries (default 0.1)
    :type min_quality_threshold: float
    """
    
    def __init__(
        self,
        maxsize: int = 1000,
        clean_size: int = None,
        learning_rate: float = 0.3,
        quality_weight: float = 0.6,
        recency_weight: float = 0.3,
        frequency_weight: float = 0.1,
        on_evict: Callable[[List[Any]], None] = None,
        history_size: int = 10,
        min_quality_threshold: float = 0.1,
        **kwargs
    ):
        if not (0.0 <= learning_rate <= 1.0):
            raise ValueError("learning_rate must be between 0.0 and 1.0")
        
        if abs(quality_weight + recency_weight + frequency_weight - 1.0) > 1e-6:
            raise ValueError("Quality, recency, and frequency weights must sum to 1.0")
            
        self.maxsize = maxsize
        self.clean_size = clean_size if clean_size else max(1, int(maxsize * 0.2))
        self.learning_rate = learning_rate
        self.history_size = history_size
        self.min_quality_threshold = min_quality_threshold
        
        # Composite score weights
        self.weights = {
            'quality': quality_weight,
            'recency': recency_weight,
            'frequency': frequency_weight
        }
        
        # Callback for eviction
        self.on_evict = on_evict
        
        # Core data structures
        self.entries: Dict[Any, EntryMetadata] = {}
        self.access_order = deque()  # For tracking access order
        self._lock = threading.RLock()  # Thread safety
        
        # Statistics for normalization
        self.stats = {
            'max_quality': 1.0,
            'min_quality': 0.0,
            'max_frequency': 1,
            'total_accesses': 0
        }
        
        gptcache_log.info(
            f"Initialized QualityScoreEviction: maxsize={maxsize}, "
            f"weights=Q{quality_weight}:R{recency_weight}:F{frequency_weight}, "
            f"learning_rate={learning_rate}"
        )
    
    def put(self, objs: List[Any]):
        """Add new objects to the cache.
        
        :param objs: List of object IDs to add to cache
        :type objs: List[Any]
        """
        with self._lock:
            current_time = time.time()
            
            for obj in objs:
                if obj in self.entries:
                    # Update existing entry
                    self.entries[obj].last_access_time = current_time
                    self.entries[obj].access_count += 1
                    self._update_access_order(obj)
                else:
                    # Add new entry
                    self.entries[obj] = EntryMetadata(
                        obj_id=obj,
                        quality_score=self.min_quality_threshold,  # Start with minimum quality
                        creation_time=current_time,
                        last_access_time=current_time,
                        access_count=1
                    )
                    self.access_order.append(obj)
            
            self.stats['total_accesses'] += len(objs)
            
            # Check if eviction is needed
            if len(self.entries) > self.maxsize:
                self._evict_items()
    
    def get(self, obj: Any) -> Optional[bool]:
        """Access an object in the cache, updating its access metrics.
        
        :param obj: Object ID to access  
        :type obj: Any
        :return: True if object exists, None otherwise
        :rtype: Optional[bool]
        """
        with self._lock:
            if obj not in self.entries:
                return None
                
            # Update access metrics
            entry = self.entries[obj]
            entry.last_access_time = time.time()
            entry.access_count += 1
            self.stats['total_accesses'] += 1
            
            # Update max frequency for normalization
            self.stats['max_frequency'] = max(self.stats['max_frequency'], entry.access_count)
            
            # Update access order
            self._update_access_order(obj)
            
            return True
    
    def update_quality(self, obj: Any, similarity_score: float):
        """Update the quality score of an object based on similarity evaluation.
        
        :param obj: Object ID whose quality to update
        :type obj: Any  
        :param similarity_score: Similarity score from cache hit evaluation
        :type similarity_score: float
        """
        with self._lock:
            if obj not in self.entries:
                return
                
            entry = self.entries[obj]
            
            # Update quality using exponential moving average
            if entry.quality_score == 0.0:
                # First similarity score
                entry.quality_score = similarity_score
            else:
                entry.quality_score = (
                    (1 - self.learning_rate) * entry.quality_score + 
                    self.learning_rate * similarity_score
                )
            
            # Track similarity history for variance analysis
            entry.similarity_history.append(similarity_score)
            if len(entry.similarity_history) > self.history_size:
                entry.similarity_history.pop(0)
            
            # Update global quality statistics
            self.stats['max_quality'] = max(self.stats['max_quality'], entry.quality_score)
            self.stats['min_quality'] = min(self.stats['min_quality'], entry.quality_score)
            
            gptcache_log.debug(
                f"Updated quality for {obj}: score={similarity_score:.3f}, "
                f"avg_quality={entry.quality_score:.3f}, access_count={entry.access_count}"
            )
    
    def _update_access_order(self, obj: Any):
        """Update the access order by moving object to end of deque."""
        # Remove from current position and add to end
        try:
            self.access_order.remove(obj)
        except ValueError:
            pass  # Object not in deque yet
        self.access_order.append(obj)
    
    def _calculate_composite_score(self, entry: EntryMetadata) -> float:
        """Calculate composite score for eviction decision.
        
        Lower scores are evicted first.
        
        :param entry: Entry metadata
        :type entry: EntryMetadata
        :return: Composite score (lower = more likely to evict)
        :rtype: float
        """
        current_time = time.time()
        
        # Quality component (0-1, higher is better)
        quality_range = self.stats['max_quality'] - self.stats['min_quality']
        if quality_range > 0:
            quality_norm = (entry.quality_score - self.stats['min_quality']) / quality_range
        else:
            quality_norm = 0.5  # Default when all qualities are the same
        
        # Recency component (0-1, more recent is better)
        time_since_access = current_time - entry.last_access_time
        max_time_since = max(1.0, current_time - min(e.last_access_time for e in self.entries.values()))
        recency_norm = 1.0 - (time_since_access / max_time_since)
        
        # Frequency component (0-1, higher frequency is better)  
        if self.stats['max_frequency'] > 0:
            frequency_norm = entry.access_count / self.stats['max_frequency']
        else:
            frequency_norm = 0.0
        
        # Calculate weighted composite score
        composite_score = (
            self.weights['quality'] * quality_norm +
            self.weights['recency'] * recency_norm +
            self.weights['frequency'] * frequency_norm
        )
        
        return composite_score
    
    def _evict_items(self):
        """Evict items with lowest composite scores."""
        if len(self.entries) <= self.maxsize:
            return
            
        # Calculate scores for all entries
        scored_entries = []
        for obj_id, entry in self.entries.items():
            score = self._calculate_composite_score(entry)
            scored_entries.append((score, obj_id, entry))
        
        # Sort by score (ascending - lowest scores first)
        scored_entries.sort(key=lambda x: x[0])
        
        # Determine how many items to evict
        num_to_evict = min(self.clean_size, len(self.entries) - self.maxsize + self.clean_size)
        items_to_evict = scored_entries[:num_to_evict]
                
        # Remove evicted items
        evicted_ids = []
        for score, obj_id, entry in items_to_evict:
            evicted_ids.append(obj_id)
            del self.entries[obj_id]
            
            # Remove from access order
            try:
                self.access_order.remove(obj_id)
            except ValueError:
                pass
                
            gptcache_log.debug(
                f"Evicted {obj_id}: composite_score={score:.3f}, "
                f"quality={entry.quality_score:.3f}, access_count={entry.access_count}"
            )
        
        if evicted_ids and self.on_evict:
            self.on_evict(evicted_ids)
            
        gptcache_log.info(f"Evicted {len(evicted_ids)} items, cache size now: {len(self.entries)}")
    
    @property
    def policy(self) -> str:
        """Return the policy name."""
        return "QUALITY_SCORE"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current cache statistics for monitoring.
        
        :return: Dictionary of cache statistics
        :rtype: Dict[str, Any]
        """
        with self._lock:
            if not self.entries:
                return {
                    'size': 0,
                    'max_size': self.maxsize,
                    'avg_quality': 0.0,
                    'avg_access_count': 0.0,
                    'total_accesses': self.stats['total_accesses'],
                    'weights': self.weights.copy(),
                    'learning_rate': self.learning_rate
                }
                
            qualities = [entry.quality_score for entry in self.entries.values()]
            access_counts = [entry.access_count for entry in self.entries.values()]
            
            return {
                'size': len(self.entries),
                'max_size': self.maxsize,
                'avg_quality': sum(qualities) / len(qualities),
                'min_quality': min(qualities),
                'max_quality': max(qualities),
                'avg_access_count': sum(access_counts) / len(access_counts),
                'total_accesses': self.stats['total_accesses'],
                'weights': self.weights.copy(),
                'learning_rate': self.learning_rate
            } 