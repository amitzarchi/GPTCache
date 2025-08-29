# GPTCache: Quality Score Eviction Implementation

A semantic caching library for LLM queries with advanced quality-based eviction policies.

## Project Overview

This project implements and benchmarks a novel quality score-based eviction policy for GPTCache, comparing it against traditional memory-based policies (LRU, LFU, FIFO, RR). The quality score considers multiple factors including recency, frequency, and semantic similarity to make intelligent caching decisions.

## Features

- **Quality Score Eviction**: Advanced eviction policy using weighted scoring of recency, frequency, and quality factors
- **Traditional Policies**: LRU, LFU, FIFO, and Random Replacement for comparison
- **Comprehensive Benchmarking**: Automated testing across different question sets and cache configurations
- **Performance Metrics**: Hit rate, throughput (requests/second), and CPU usage analysis

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/zilliztech/GPTCache.git
cd GPTCache

# Install dependencies
pip install -r requirements.txt

# Install the package
python setup.py install
```

### Running the Benchmark

Execute the quality score eviction benchmark:

```bash
python examples/benchmark/quality_score/benchmark.py
```

**⚠️ Performance Warning**: The full benchmark runs 270 different configurations and can take **several days** on a standard machine. Consider either:
- Running on a powerful VM with multiple cores
- Reducing the number of configurations by modifying the constants in `benchmark.py`

The benchmark will:
- Test different question sets (HIGH, LOW, MIXED repetition patterns)
- Vary cache sizes (5-300 entries depending on question count)
- Compare quality score vs traditional eviction policies
- Generate comprehensive performance metrics

Results are saved to `examples/benchmark/quality_score/benchmark_results.csv`

## Benchmark Configuration

The benchmark tests multiple configurations:

**Question Sets:**
- 500, 1000, 3000 questions each with HIGH/LOW/MIXED repetition patterns

**Quality Score Parameters:**
- Learning rates: 0.3, 0.5, 0.7
- Weight combinations: (0.6, 0.3, 0.1) and (0.8, 0.1, 0.1) for quality/recency/frequency

**Traditional Policies:**
- LRU (Least Recently Used)
- LFU (Least Frequently Used) 
- FIFO (First In, First Out)
- RR (Random Replacement)

**Cache Sizes:**
- 500 questions: 5, 10, 20 entries
- 1000 questions: 10, 50, 100 entries  
- 3000 questions: 30, 150, 300 entries

## Architecture

The system uses:
- **Embedding**: ONNX-based semantic embeddings
- **Vector Store**: FAISS for similarity search
- **Cache Storage**: SQLite for response storage
- **Similarity Evaluation**: Distance-based semantic matching

## Key Components

- `examples/benchmark/quality_score/benchmark.py`: Main benchmark runner
- `examples/benchmark/quality_score/test.py`: Individual test execution
- `gptcache/manager/eviction/quality_score.py`: Eviction policy implementation

## Results Analysis

The benchmark generates metrics for:
- **Hit Rate**: Percentage of requests served from cache
- **Throughput**: Requests processed per second
- **CPU Usage**: Average CPU utilization during testing

Compare quality score performance against traditional policies to evaluate the effectiveness of semantic-aware eviction.

## Requirements

- Python >= 3.8.1
- Dependencies listed in `requirements.txt`

## Project Structure

```
GPTCache/
├── gptcache/           # Core library code
│   └── manager/
│       └── eviction/
│           └── quality_score.py   # Eviction policy implementation
├── examples/
│   └── benchmark/
│       └── quality_score/  # Benchmark implementation
├── requirements.txt    # Python dependencies
├── setup.py           # Package installation
└── README.md          # This file
```

This implementation demonstrates advanced caching strategies that consider semantic relationships and usage patterns to optimize cache performance for LLM applications.


