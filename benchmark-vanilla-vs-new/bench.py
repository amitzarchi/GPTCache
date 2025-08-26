import json
import os
import time
import sys
import shutil
import psutil
import threading

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add the path to cloned GPTCache repo
sys.path.insert(0, r'C:\Users\yossi_1wprdhx\Desktop\llmcache\GPTCache')

from gptcache.adapter import openai
from gptcache import cache, Config
from gptcache.embedding import Onnx as EmbeddingOnnx
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.manager import manager_factory


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests"""
    cache_size: int = 100000
    similarity_threshold: float = 0.95
    data_dir: str = "./cache_data"
    mock_data_path: str = r"C:\Users\yossi_1wprdhx\Desktop\llmcache\GPTCache\examples\benchmark\mock_data.json"
    clean_cache: bool = True

    # Quality score eviction params (only used for quality_score mode)
    learning_rate: float = 0.3
    quality_weight: float = 0.8
    recency_weight: float = 0.15
    frequency_weight: float = 0.05
    clean_size: int = 1


@dataclass
class BenchmarkResults:
    """Results from benchmark run"""
    eviction_policy: str
    total_requests: int
    cache_hits_positive: int
    cache_hits_negative: int
    cache_miss: int
    fail_count: int
    average_latency_ms: float
    total_time_seconds: float
    average_embedding_time_ms: float
    average_search_time_ms: float
    cache_hit_rate_percent: float
    cpu_utilization: Optional[Dict] = None
    throughput_rps: float = 0.0



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


class CacheEvictionBenchmark:
    """Benchmark class for testing different eviction policies"""

    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.system_metrics = SystemMetrics()

        # Setup OpenAI mockup
        os.environ['OPENAI_API_KEY'] = 'mockup'
        cache.set_openai_key()

    def _setup_cache(self, eviction_policy: str) -> None:
        """Setup cache with specified eviction policy"""

        class WrapEvaluation(SearchDistanceEvaluation):
            def evaluation(self, src_dict, cache_dict, **kwargs):
                return super().evaluation(src_dict, cache_dict, **kwargs)

            def range(self):
                return super().range()

        embedding_onnx = EmbeddingOnnx()

        # Configure eviction manager based on policy
        if eviction_policy == "quality_score":
            print("choosing quality score")
            print({
                    "maxsize": self.config.cache_size,
                    "clean_size": self.config.cache_size,  # Evict 1 item at a time for clear demonstration
                    "learning_rate": self.config.learning_rate,  # Learn reasonably fast
                    "quality_weight": self.config.quality_weight,  # Heavily prioritize quality
                    "recency_weight": self.config.recency_weight,  # Some recency consideration
                    "frequency_weight": self.config.frequency_weight  # Minimal frequency impact
                })
            data_manager = manager_factory(
                manager="sqlite,faiss",
                data_dir="./cache_data",
                eviction_manager="quality_score",
                vector_params={"dimension": 768},
                eviction_params={
                    "maxsize": self.config.cache_size,
                    "clean_size": self.config.cache_size,  # Evict 1 item at a time for clear demonstration
                    "learning_rate": self.config.learning_rate,  # Learn reasonably fast
                    "quality_weight": self.config.quality_weight,  # Heavily prioritize quality
                    "recency_weight": self.config.recency_weight,  # Some recency consideration
                    "frequency_weight": self.config.frequency_weight  # Minimal frequency impact
                }
            )
        elif eviction_policy == "vanilla":
            from gptcache.manager import get_data_manager, CacheBase, VectorBase
            cache_base = CacheBase("./cache/sqlite")
            vector_base = VectorBase("./cache/faiss", dimension=embedding_onnx.dimension)
            data_manager = get_data_manager(cache_base, vector_base, max_size=100000)

        else:
            raise ValueError(f"Unsupported eviction policy: {eviction_policy}")

        cache.init(
            embedding_func=embedding_onnx.to_embeddings,
            data_manager=data_manager,
            similarity_evaluation=SearchDistanceEvaluation(),
            config=Config(similarity_threshold=self.config.similarity_threshold),
        )

    def _cleanup_cache(self) -> None:
        """Clean up cache data directory"""
        if self.config.clean_cache and os.path.exists(self.config.data_dir):
            shutil.rmtree(self.config.data_dir)
            print(f"🧹 Cleaned up cache directory: {self.config.data_dir}")

    def _load_mock_data(self) -> List[Dict]:
        """Load and prepare mock data"""
        with open(self.config.mock_data_path, "r") as mock_file:
            mock_data = json.load(mock_file)

        # Add id to each pair
        for i, pair in enumerate(mock_data):
            pair["id"] = str(i)

        return mock_data

    def _populate_initial_cache(self, mock_data: List[Dict]) -> None:
        """Populate cache with initial data if needed"""
        sqlite_file = os.path.join(self.config.data_dir, "./cache_data/sqlite.db")
        faiss_file = os.path.join(self.config.data_dir, "./cache_data/faiss.index")
        has_data = os.path.isfile(sqlite_file) and os.path.isfile(faiss_file)

        if not has_data:
            print("📥 Populating initial cache data...")
            start_time = time.time()

            # Extract questions and answers (using IDs as answers for benchmarking)
            questions, answers = map(
                list, zip(*((pair["origin"], pair["id"]) for pair in mock_data))
            )

            cache.import_data(questions=questions, answers=answers)

            populate_time = time.time() - start_time
            print(f"✅ Cache populated with {len(questions)} entries in {populate_time:.2f}s")
        else:
            print("📋 Using existing cache data")

    def _run_benchmark_queries(self, mock_data: List[Dict]) -> BenchmarkResults:
        """Run benchmark queries and collect metrics"""
        print(f"🚀 Running benchmark with {len(mock_data)} queries...")

        # Initialize metrics
        all_time = 0.0
        hit_cache_positive = 0
        hit_cache_negative = 0
        fail_count = 0
        cache_miss = 0
        latency_samples = []

        # Start system monitoring
        self.system_metrics.start_monitoring()
        benchmark_start = time.time()

        try:
            for i, pair in enumerate(mock_data):
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

                    # Record metrics
                    all_time += consume_time
                    latency_samples.append(consume_time * 1000)  # Convert to ms
                    print("cache hint time consuming: {:.2f}s".format(consume_time))

                    # Check cache hit accuracy
                    if res_text == pair["id"]:
                        print("cache hit!")
                        hit_cache_positive += 1
                    else:
                        hit_cache_negative += 1

                    if res_text == "Hello! How can I help you today?":
                        print("Cache miss!")
                        cache_miss += 1

                    if (i + 1) % 100 == 0:
                        print(f"  Processed {i + 1}/{len(mock_data)} queries...")


                except Exception as e:
                    fail_count += 1
                    print(f"  Query {i + 1} failed: {e}")

        finally:
            # Stop monitoring
            self.system_metrics.stop_monitoring()

        benchmark_duration = time.time() - benchmark_start

        # Calculate final metrics
        total_requests = len(mock_data)
        cache_hit_rate = (hit_cache_positive / total_requests * 100) if total_requests > 0 else 0
        throughput = total_requests / benchmark_duration if benchmark_duration > 0 else 0

        # Get system metrics
        cpu_stats = self.system_metrics.get_cpu_stats()

        # Get cache-specific metrics
        avg_embedding_time = cache.report.average_embedding_time()
        avg_search_time = cache.report.average_search_time()

        return BenchmarkResults(
            eviction_policy="unknown",  # Will be set by caller
            total_requests=total_requests,
            cache_hits_positive=hit_cache_positive,
            cache_hits_negative=hit_cache_negative,
            cache_miss=cache_miss,
            fail_count=fail_count,
            average_latency_ms=sum(latency_samples) / len(latency_samples) if latency_samples else 0.0,
            total_time_seconds=all_time,
            average_embedding_time_ms=avg_embedding_time * 1000 if avg_embedding_time else 0.0,
            average_search_time_ms=avg_search_time * 1000 if avg_search_time else 0.0,
            cache_hit_rate_percent=cache_hit_rate,
            cpu_utilization=cpu_stats,
            throughput_rps=throughput
        )

    def benchmark_eviction_policy(self, eviction_policy: str) -> BenchmarkResults:
        """Run complete benchmark for a specific eviction policy"""
        print(f"\n{'=' * 60}")
        print(f"🧪 BENCHMARKING: {eviction_policy.upper()} EVICTION POLICY")
        print(f"{'=' * 60}")

        # Cleanup and setup
        self._cleanup_cache()
        self._setup_cache(eviction_policy)

        # Load data and populate cache
        mock_data = self._load_mock_data()
        self._populate_initial_cache(mock_data)

        # Run benchmark
        results = self._run_benchmark_queries(mock_data)
        results.eviction_policy = eviction_policy

        # Print results
        self._print_results(results)

        return results

    def compare_policies(self, policies: List[str]) -> Dict[str, BenchmarkResults]:
        """Compare multiple eviction policies"""
        print(f"\n{'=' * 80}")
        print(f"🔬 COMPARATIVE BENCHMARK: {', '.join(policies).upper()}")
        print(f"{'=' * 80}")

        results = {}

        for policy in policies:
            results[policy] = self.benchmark_eviction_policy(policy)
            time.sleep(5)

        # Print comparison summary
        self._print_comparison(results)

        return results

    def _print_results(self, results: BenchmarkResults) -> None:
        """Print detailed benchmark results"""
        print(f"\n📊 BENCHMARK RESULTS - {results.eviction_policy.upper()}")
        print("-" * 50)

        print(f"📈 Performance Metrics:")
        print(f"   Total Requests: {results.total_requests}")
        print(f"   Cache Hit Rate: {results.cache_hit_rate_percent:.1f}%")
        print(f"   Average Latency: {results.average_latency_ms:.2f}ms")
        print(f"   Throughput: {results.throughput_rps:.1f} requests/second")

        print(f"\n🎯 Cache Accuracy:")
        print(f"   Positive Hits: {results.cache_hits_positive}")
        print(f"   Negative Hits: {results.cache_hits_negative}")
        print(f"   Cache miss: {results.cache_miss}")
        print(f"   Failed Requests: {results.fail_count}")

        print(f"\n⚡ Internal Timing:")
        print(f"   Average Embedding Time: {results.average_embedding_time_ms:.2f}ms")
        print(f"   Average Search Time: {results.average_search_time_ms:.2f}ms")
        print(f"   Total Benchmark Time: {results.total_time_seconds:.2f}s")

        if results.cpu_utilization:
            cpu = results.cpu_utilization
            print(f"\n🖥️  CPU Utilization:")
            print(f"   Mean: {cpu['mean']:.1f}%")
            print(f"   P95: {cpu['p95']:.1f}%")
            print(f"   P99: {cpu['p99']:.1f}%")
            print(f"   Max: {cpu['max']:.1f}%")

    def _print_comparison(self, results: Dict[str, BenchmarkResults]) -> None:
        """Print comparative analysis of multiple policies"""
        print(f"\n🏆 COMPARATIVE ANALYSIS")
        print("=" * 80)

        policies = list(results.keys())

        # Create comparison table
        print(f"{'Metric':<25} | ", end="")
        for policy in policies:
            print(f"{policy.upper():<15} | ", end="")
        print()
        print("-" * (27 + len(policies) * 18))

        # Cache Hit Rate
        print(f"{'Cache Hit Rate (%)':<25} | ", end="")
        for policy in policies:
            print(f"{results[policy].cache_hit_rate_percent:>13.1f}% | ", end="")
        print()

        # Average Latency
        print(f"{'Avg Latency (ms)':<25} | ", end="")
        for policy in policies:
            print(f"{results[policy].average_latency_ms:>13.2f}ms | ", end="")
        print()

        # Throughput
        print(f"{'Throughput (req/s)':<25} | ", end="")
        for policy in policies:
            print(f"{results[policy].throughput_rps:>13.1f} | ", end="")
        print()

        # CPU Utilization (mean)
        print(f"{'CPU Usage (mean %)':<25} | ", end="")
        for policy in policies:
            cpu_mean = results[policy].cpu_utilization['mean'] if results[policy].cpu_utilization else 0
            print(f"{cpu_mean:>13.1f}% | ", end="")
        print()

        print("\n💡 Key Insights:")

        # Find best performing policy for each metric
        best_hit_rate = max(results.items(), key=lambda x: x[1].cache_hit_rate_percent)
        best_latency = min(results.items(), key=lambda x: x[1].average_latency_ms)
        best_throughput = max(results.items(), key=lambda x: x[1].throughput_rps)

        print(f"   🎯 Best Cache Hit Rate: {best_hit_rate[0].upper()} ({best_hit_rate[1].cache_hit_rate_percent:.1f}%)")
        print(f"   ⚡ Lowest Latency: {best_latency[0].upper()} ({best_latency[1].average_latency_ms:.2f}ms)")
        print(f"   🚀 Highest Throughput: {best_throughput[0].upper()} ({best_throughput[1].throughput_rps:.1f} req/s)")

    def save_results(self, results: Dict[str, BenchmarkResults], filename: str) -> None:
        """Save benchmark results to JSON file"""
        serializable_results = {}

        for policy, result in results.items():
            serializable_results[policy] = {
                'eviction_policy': result.eviction_policy,
                'total_requests': result.total_requests,
                'cache_hits_positive': result.cache_hits_positive,
                'cache_hits_negative': result.cache_hits_negative,
                'fail_count': result.fail_count,
                'average_latency_ms': result.average_latency_ms,
                'total_time_seconds': result.total_time_seconds,
                'average_embedding_time_ms': result.average_embedding_time_ms,
                'average_search_time_ms': result.average_search_time_ms,
                'cache_hit_rate_percent': result.cache_hit_rate_percent,
                'throughput_rps': result.throughput_rps,
                'cpu_utilization': result.cpu_utilization
            }

        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"📁 Results saved to: {filename}")


# Example usage
if __name__ == "__main__":
    # Create benchmark configuration
    config = BenchmarkConfig(
        cache_size=1000,  # Smaller cache for testing eviction
        clean_cache=True,
        similarity_threshold=0.85
    )

    # Initialize benchmark
    benchmark = CacheEvictionBenchmark(config)

    # Compare vanilla vs quality score eviction
    policies_to_test = ["quality_score", "vanilla"]

    # Run comparative benchmark
    results = benchmark.compare_policies(policies_to_test)

    # Save results
    timestamp = int(time.time())
    benchmark.save_results(results, f"benchmark_results_{timestamp}.json")

    print(f"\n🎉 Benchmark complete! Results saved.")