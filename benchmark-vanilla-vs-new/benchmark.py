import json
import os
import time
import sys
import psutil
import threading

# Add the path to cloned GPTCache repo
sys.path.insert(0, r'C:\Users\yossi_1wprdhx\Desktop\llmcache\GPTCache')

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from gptcache.adapter import openai
from gptcache import cache, Config
from gptcache.similarity_evaluation.onnx import OnnxModelEvaluation
from gptcache.embedding import Onnx as EmbeddingOnnx
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.manager import manager_factory


@dataclass
class DistanceBenchmarkConfig:
    """Configuration for benchmark tests for semantic caching"""
    policy = "distance"
    cache_size: int = 100000
    similarity_threshold: float = 0.95
    data_dir: str = "./cache_data"
    mock_data_path: str = r"..\..\GPTCache\examples\benchmark\mock_data.json"
    clean_cache: bool = True

    def __init__(self, cache_size, similarity_threshold, data_dir, mock_data_path, clean_cache):
        self.cache_size = cache_size
        self.similarity_threshold = similarity_threshold
        self.data_dir = data_dir
        self.mock_data_path = mock_data_path
        self.clean_cache = clean_cache


@dataclass
class QualityScoreBenchmarkConfig(DistanceBenchmarkConfig):
    """Configuration for benchmark tests for semantic caching using quality score"""
    # Quality score eviction params (only used for quality_score mode)
    policy = "quality"
    learning_rate: float = 0.3
    quality_weight: float = 0.8
    recency_weight: float = 0.15
    frequency_weight: float = 0.05
    clean_size: int = 1

    def __init__(self, cache_size, similarity_threshold, data_dir, mock_data_path, clean_cache, learning_rate,
                 quality_weight, recency_weight, frequency_weight, clean_size):
        super().__init__(cache_size, similarity_threshold, data_dir, mock_data_path, clean_cache)
        self.learning_rate = learning_rate
        self.quality_weight = quality_weight
        self.recency_weight = recency_weight
        self.frequency_weight = frequency_weight
        self.clean_size = clean_size


class SystemMetrics:
    """System resource monitoring"""

    def __init__(self):
        self.cpu_samples = []
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        """Start collecting system metrics in background"""
        self.monitoring = True
        self.cpu_samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop collecting metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()

            self.cpu_samples.append({
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
            })

            time.sleep(0.1)  # Sample every 100ms

    def get_cpu_stats(self) -> Optional[Dict]:
        """Calculate CPU statistics"""
        if not self.cpu_samples:
            return None

        cpu_values = [sample['cpu_percent'] for sample in self.cpu_samples]
        cpu_values.sort()
        n = len(cpu_values)

        return {
            'mean': sum(cpu_values) / n,
            'p95': cpu_values[int(0.95 * n)] if n > 0 else 0,
            'p99': cpu_values[int(0.99 * n)] if n > 0 else 0,
            'max': max(cpu_values) if cpu_values else 0,
            'samples': n
        }


class Benchmark():
    """this class allows to benchmark quality score and distance based eviction policies.
    the dataset provided should be a json file with pairs of origin and similar.
    see mock_data.json for an example."""

    def __init__(self, config: DistanceBenchmarkConfig):
        self.config = config
        self.system_metrics = SystemMetrics()

    def _setup_cache(self):
        cache.set_openai_key()

        class WrapEvaluation(SearchDistanceEvaluation):
            def evaluation(self, src_dict, cache_dict, **kwargs):  # eval distance
                return super().evaluation(src_dict, cache_dict, **kwargs)

            def range(self):
                return super().range()

        eviction_policy = self.config.policy
        if eviction_policy == "quality":
            eviction_params = {
                "maxsize": self.config.cache_size,
                "clean_size": 1,  # Evict 1 item at a time for clear demonstration
                "learning_rate": 0.3,  # Learn reasonably fast
                "quality_weight": 0.8,  # Heavily prioritize quality
                "recency_weight": 0.15,  # Some recency consideration
                "frequency_weight": 0.05  # Minimal frequency impact
            }
            eviction_manager = "quality_score"
        elif eviction_policy == "distance":
            eviction_params = {
                "maxsize": self.config.cache_size,
                "clean_size": 1  # Evict 1 item at a time for clear demonstration
            }
            eviction_manager = "LRU"
        else:
            raise ValueError("Incorrect eviction policy")

        data_manager = manager_factory(
            manager="sqlite,faiss",
            data_dir="./cache_data",
            eviction_manager="quality_score",
            vector_params={"dimension": 768},
            eviction_params=eviction_params
        )

        embedding_onnx = EmbeddingOnnx()

        cache.init(
            embedding_func=embedding_onnx.to_embeddings,
            data_manager=data_manager,
            similarity_evaluation=WrapEvaluation(),
            config=Config(similarity_threshold=self.config.similarity_threshold),
        )

    def _clean_cache(self):
        if os.path.exists("./cache_data"):
            import shutil

            shutil.rmtree("./cache_data")
            print("🧹 Cleaned up existing cache data")

    def _load_json_data(self):
        with open(self.config.mock_data_path, "r") as mock_file:
            mock_data = json.load(mock_file)

        # add id to each pair
        i = 0
        for pair in mock_data:
            pair["id"] = str(i)
            i += 1

        return mock_data

    def _populate_cache(self, mock_data):
        """ when using mock_data we use this function to populate the cache if needed """
        print("insert data")
        start_time = time.time()
        # storing prompts and ids as answers to test for cache hit ratio and speed, no need to keep actual response
        questions, answers = map(
            list, zip(*((pair["origin"], pair["id"]) for pair in mock_data))
        )
        cache.import_data(questions=questions, answers=answers)
        print(
            "end insert data, time consuming: {:.2f}s".format(time.time() - start_time)
        )

    def run(self):
        self._setup_cache()
        if self.config.clean_cache:
            self._clean_cache()
        mock_data = self._load_json_data()
        sqlite_file = "./cache_data/sqlite.db"
        faiss_file = "./cache_data/faiss.index"
        has_data = os.path.isfile(sqlite_file) and os.path.isfile(faiss_file)
        if has_data:
            self._populate_cache(mock_data)

        print(f"🚀 Running benchmark with {len(mock_data)} queries...")

        # Initialize metrics
        all_time = 0.0
        hit_cache_positive = 0
        cache_hit_negative = 0
        fail_count = 0
        cache_miss = 0
        latency_samples = []

        # Start system monitoring
        self.system_metrics.start_monitoring()
        benchmark_start = time.time()

        for pair in mock_data:
            mock_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": pair["similar"]},
            ]
            try:
                start_time = time.time()
                res = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=mock_messages,
                )
                res_text = openai.get_message_from_openai_answer(res)
                consume_time = time.time() - start_time
                all_time += consume_time
                latency_samples.append(consume_time * 1000)  # Convert to ms

                # check if cache hit
                if res_text == pair["id"]:
                    hit_cache_positive += 1
                elif res_text == "Hello! How can I help you today?":
                    print("Cache miss!")
                    cache_miss += 1
                else:
                    cache_hit_negative += 1

                print("cache hint time consuming: {:.2f}s".format(consume_time))
            except:
                fail_count += 1

        cpu_stats = self.system_metrics.get_cpu_stats()
        print("printing cp stats....")
        print('\n'.join("%s: %s" % item for item in cpu_stats.items()))
        print("=" * 60)
        print("\n")

        attrs = cache.report
        print("printing cache report....")
        print('\n'.join("%s: %s" % item for item in attrs.items()))
        print("=" * 60)
        print("\n")
        print("average time: {:.2f}s".format(all_time / len(mock_data)))
        print("cache_hit_positive:", hit_cache_positive)
        print("cache_hit_negative:", cache_hit_negative)
        print("cache_miss:", cache_miss)
        print("fail_count:", fail_count)
        print("average embedding time: ", cache.report.average_embedding_time())
        print("average search time: ", cache.report.average_search_time())


data_path = r"C:\Users\yossi_1wprdhx\Desktop\llmcache\GPTCache\examples\benchmark\mock_data.json"
config = QualityScoreBenchmarkConfig(100000, 0.95, "./cache_data", data_path, True, 0.2, 0.8, 0.15, 0.05)
