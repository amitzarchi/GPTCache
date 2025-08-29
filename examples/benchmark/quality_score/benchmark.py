import os
import uuid
import csv
import threading
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from test import test
from question_sampler import sample_questions

QUESTION_CONFIGS = [
    ('HIGH', 500), ('HIGH', 1000), ('HIGH', 3000),
    ('LOW', 500), ('LOW', 1000), ('LOW', 3000),
    ('MIXED', 500), ('MIXED', 1000), ('MIXED', 3000),
]
QUALITY_WEIGHTS = [(0.6, 0.3, 0.1), (0.8, 0.1, 0.1)]
LEARNING_RATES = [0.3, 0.5, 0.7]
MEMORY_POLICIES = ['LRU', 'LFU', 'FIFO', 'RR']
MAX_SIZES_FOR_NUM_QUESTIONS = {
    500: [5, 10, 20],
    1000: [10, 50, 100],
    3000: [30, 150, 300],
}

CSV_HEADERS = [
    'number_of_questions', 'degree_of_repetition', 'eviction_base', 'eviction_policy',
    'max_size', 'learning_rate', 'quality_weight', 'recency_weight', 'frequency_weight',
    'hit_rate', 'requests_per_second', 'average_cpu_usage'
]

csv_lock = threading.Lock()


def extract_result_data(result):
    """Extract and format data from test result"""
    config_details = result.get('config_details', {})
    
    eviction_base = result.get('eviction_base', 'memory')
    eviction_policy = ('quality_score' if eviction_base == 'quality_score' 
                      else config_details.get('memory_policy') or result.get('policy'))
    
    quality_weights = config_details.get('quality_weights')
    quality_weight, recency_weight, frequency_weight = (quality_weights if quality_weights 
                                                       else (None, None, None))
    
    throughput = result.get('throughput', {})
    cpu_metrics = result.get('cpu_metrics', {})
    
    return [
        config_details.get('num_questions'),
        config_details.get('degree_of_repetition'),
        eviction_base,
        eviction_policy,
        result.get('max_size'),
        config_details.get('learning_rate'),
        quality_weight,
        recency_weight,
        frequency_weight,
        result.get('cache_hit_rate'),
        throughput.get('requests_per_second'),
        cpu_metrics.get('average_cpu_usage')
    ]


def save_result_to_csv(csv_file_path, result):
    """Save test result to CSV file with thread safety"""
    row_data = extract_result_data(result)
    
    with csv_lock:
        file_exists = os.path.exists(csv_file_path)
        
        with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            if not file_exists:
                writer.writerow(CSV_HEADERS)
            
            writer.writerow(row_data)


def create_file_extension(degree_of_repetition, num_questions, max_size, test_type, unique_id):
    """Create unique file extension for test data"""
    return f"{degree_of_repetition.lower()}_{num_questions}_{max_size}_{test_type}_{unique_id}"


def run_quality_score_test(questions, max_size, quality_weights, learning_rate, file_extension):
    """Run quality score eviction test"""
    quality_weight, recency_weight, frequency_weight = quality_weights
    return test(
        questions=questions,
        max_size=max_size,
        eviction_base='quality_score',
        learning_rate=learning_rate,
        quality_weight=quality_weight,
        recency_weight=recency_weight,
        frequency_weight=frequency_weight,
        file_extension=file_extension
    )


def run_memory_test(questions, max_size, memory_policy, file_extension):
    """Run memory eviction test"""
    return test(
        questions=questions,
        max_size=max_size,
        eviction_base='memory',
        policy=memory_policy,
        file_extension=file_extension
    )


def run_single_test(config):
    """Run a single test configuration"""
    question_config, max_size, quality_weights, learning_rate, memory_policy = config
    degree_of_repetition, num_questions = question_config
    
    try:
        questions = sample_questions(degree_of_repetition, num_questions)
        unique_id = str(uuid.uuid4())
        
        if quality_weights and learning_rate:
            file_extension = create_file_extension(
                degree_of_repetition, num_questions, max_size, 
                f"qs_{learning_rate}", unique_id
            )
            result = run_quality_score_test(questions, max_size, quality_weights, learning_rate, file_extension)
            eviction_base = 'quality_score'
        else:
            file_extension = create_file_extension(
                degree_of_repetition, num_questions, max_size, 
                memory_policy.lower(), unique_id
            )
            result = run_memory_test(questions, max_size, memory_policy, file_extension)
            eviction_base = 'memory'
        
        result.update({
            'question_config': f"{degree_of_repetition}_{num_questions}",
            'max_size': max_size,
            'eviction_base': eviction_base,
            'config_details': {
                'degree_of_repetition': degree_of_repetition,
                'num_questions': num_questions,
                'quality_weights': quality_weights,
                'learning_rate': learning_rate,
                'memory_policy': memory_policy
            }
        })
        
        return result
        
    except Exception as e:
        return {
            'error': str(e),
            'question_config': f"{degree_of_repetition}_{num_questions}",
            'max_size': max_size,
            'config_details': {
                'degree_of_repetition': degree_of_repetition,
                'num_questions': num_questions,
                'quality_weights': quality_weights,
                'learning_rate': learning_rate,
                'memory_policy': memory_policy
            }
        }


def generate_test_configs():
    """Generate all test configurations"""
    configs = []
    
    for question_config in QUESTION_CONFIGS:
        degree_of_repetition, num_questions = question_config
        max_sizes = MAX_SIZES_FOR_NUM_QUESTIONS[num_questions]
        
        for max_size in max_sizes:
            for quality_weights in QUALITY_WEIGHTS:
                for learning_rate in LEARNING_RATES:
                    configs.append((question_config, max_size, quality_weights, learning_rate, None))
            
            for memory_policy in MEMORY_POLICIES:
                configs.append((question_config, max_size, None, None, memory_policy))
    
    return configs


def setup_csv_file(csv_file_path):
    """Setup CSV file for results"""
    if os.path.exists(csv_file_path):
        os.remove(csv_file_path)
        print("Removed existing results file")
    print(f"Results will be saved to: {csv_file_path}")


def run_benchmark_tests(test_configs, csv_file_path):
    """Run all benchmark tests in parallel"""
    max_workers = min(mp.cpu_count(), 4)
    print(f"Running tests with {max_workers} parallel workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_config = {
            executor.submit(run_single_test, config): config
            for config in test_configs
        }
        
        completed = 0
        for future in as_completed(future_to_config):
            try:
                result = future.result()
                
                if 'error' in result:
                    print(f"Test failed: {result['error']}")
                else:
                    save_result_to_csv(csv_file_path, result)
                    question_config_key = result['question_config']
                    max_size = result['max_size']
                    print(f"Completed {completed + 1}/{len(test_configs)} tests - {question_config_key}, max_size: {max_size}")
                
                completed += 1
                
            except Exception as e:
                print(f"Test failed: {e}")
                completed += 1


def main():
    """Main function to run all benchmark tests"""
    csv_file_path = os.path.join(os.path.dirname(__file__), "benchmark_results.csv")
    
    setup_csv_file(csv_file_path)
    
    test_configs = generate_test_configs()
    print(f"Total test configurations: {len(test_configs)}")
    
    run_benchmark_tests(test_configs, csv_file_path)
    
    print(f"Benchmark complete! Results saved to: {csv_file_path}")


if __name__ == "__main__":
    main()