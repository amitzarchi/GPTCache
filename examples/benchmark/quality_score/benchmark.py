import json
import uuid
import os
import sqlite3
import multiprocessing as mp
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from test import test
from question_sampler import sample_questions

# Test configurations
QUESTION_CONFIGS = [ 
    ('HIGH', 500), ('HIGH', 1000), ('HIGH', 3000),
    ('LOW', 500), ('LOW', 1000), ('LOW', 3000),
    ('MIXED', 500), ('MIXED', 1000), ('MIXED', 3000),
]

QUALITY_WEIGHTS = [(0.6, 0.3, 0.1)]
LEARNING_RATES = [0.3, 0.5, 0.7]
MEMORY_POLICIES = ['LRU', 'LFU', 'FIFO', 'RR']
MAX_SIZES_FOR_NUM_QUESTIONS = {
    500: [5, 10, 20],
    1000: [10, 50, 100],
    3000: [30, 150, 300],
}


def create_database_if_not_exists(db_path):
    """
    Create SQLite database and table if they don't exist
    Args:
        db_path: Path to the SQLite database file
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table with all required columns
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS benchmark_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            number_of_questions INTEGER,
            degree_of_repetition TEXT,
            eviction_base TEXT,
            eviction_policy TEXT,
            max_size INTEGER,
            learning_rate REAL,
            quality_weight REAL,
            recency_weight REAL,
            frequency_weight REAL,
            hit_rate REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()


def save_result_to_db(db_path, result):
    """
    Save a single test result to the SQLite database
    Args:
        db_path: Path to the SQLite database file
        result: Dictionary containing test results and configuration
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    config_details = result.get('config_details', {})
    
    # Extract values, handling both quality_score and memory eviction cases
    number_of_questions = config_details.get('num_questions')
    degree_of_repetition = config_details.get('degree_of_repetition')
    eviction_base = result.get('eviction_base', 'memory')  # Default to memory if not specified
    
    # For eviction_policy, use the policy from result or memory_policy from config
    if eviction_base == 'quality_score':
        eviction_policy = 'quality_score'
    else:
        eviction_policy = config_details.get('memory_policy') or result.get('policy')
    
    max_size = result.get('max_size')
    learning_rate = config_details.get('learning_rate')
    
    # Quality weights
    quality_weights = config_details.get('quality_weights')
    if quality_weights:
        quality_weight, recency_weight, frequency_weight = quality_weights
    else:
        quality_weight = recency_weight = frequency_weight = None
    
    hit_rate = result.get('cache_hit_rate')
    
    cursor.execute('''
        INSERT INTO benchmark_results (
            number_of_questions, degree_of_repetition, eviction_base, eviction_policy,
            max_size, learning_rate, quality_weight, recency_weight, frequency_weight, hit_rate
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        number_of_questions, degree_of_repetition, eviction_base, eviction_policy,
        max_size, learning_rate, quality_weight, recency_weight, frequency_weight, hit_rate
    ))
    
    conn.commit()
    conn.close()


def run_single_test(config):
    """
    Run a single test configuration
    Args:
        config: tuple containing (question_config, max_size, quality_weights, learning_rate, memory_policy)
    Returns:
        dict: test results with configuration info
    """
    question_config, max_size, quality_weights, learning_rate, memory_policy = config

    try:
        # Sample questions for this configuration
        degree_of_repetition, num_questions = question_config
        questions = sample_questions(degree_of_repetition, num_questions)
        unique_file_extension_id = str(uuid.uuid4())

        # Determine if this is a quality_score or memory test
        if quality_weights and learning_rate:
            # Quality score eviction test
            eviction_base = 'quality_score'
            quality_weight, recency_weight, frequency_weight = quality_weights
            result = test(
                questions=questions,
                max_size=max_size,
                eviction_base=eviction_base,
                learning_rate=learning_rate,
                quality_weight=quality_weight,
                recency_weight=recency_weight,
                frequency_weight=frequency_weight,
                file_extension=f"{degree_of_repetition.lower()}_{num_questions}_{max_size}_qs_{learning_rate}_{unique_file_extension_id}"
            )
        else:
            # Memory eviction test
            eviction_base = 'memory'
            result = test(
                questions=questions,
                max_size=max_size,
                eviction_base=eviction_base,
                policy=memory_policy,
                file_extension=f"{degree_of_repetition.lower()}_{num_questions}_{max_size}_{memory_policy.lower()}_{unique_file_extension_id}"
            )

        # Add configuration info to result
        result['question_config'] = f"{degree_of_repetition}_{num_questions}"
        result['max_size'] = max_size
        result['eviction_base'] = eviction_base
        result['config_details'] = {
            'degree_of_repetition': degree_of_repetition,
            'num_questions': num_questions,
            'quality_weights': quality_weights,
            'learning_rate': learning_rate,
            'memory_policy': memory_policy
        }

        return result

    except Exception as e:
        return {
            'error': str(e),
            'question_config': f"{question_config[0]}_{question_config[1]}",
            'max_size': max_size,
            'config_details': {
                'degree_of_repetition': question_config[0],
                'num_questions': question_config[1],
                'quality_weights': quality_weights,
                'learning_rate': learning_rate,
                'memory_policy': memory_policy
            }
        }


def generate_test_configs():
    """
    Generate all test configurations
    Returns:
        list: List of tuples containing all parameter combinations
    """
    test_configs = []

    for question_config in QUESTION_CONFIGS:
        degree_of_repetition, num_questions = question_config
        max_sizes = MAX_SIZES_FOR_NUM_QUESTIONS[num_questions]

        for max_size in max_sizes:
            # Quality score configurations
            for quality_weights in QUALITY_WEIGHTS:
                for learning_rate in LEARNING_RATES:
                    test_configs.append((
                        question_config,
                        max_size,
                        quality_weights,
                        learning_rate,
                        None  # memory_policy not used for quality_score
                    ))

            # Memory policy configurations
            for memory_policy in MEMORY_POLICIES:
                test_configs.append((
                    question_config,
                    max_size,
                    None,  # quality_weights not used for memory
                    None,  # learning_rate not used for memory
                    memory_policy
                ))

    return test_configs


def main():
    """Main function to run all benchmark tests in parallel"""

    print("Generating test configurations...")
    test_configs = generate_test_configs()
    print(f"Total test configurations: {len(test_configs)}")

    # Create SQLite database
    db_path = "/home/amitzarchi/Limodim/CacheLLM/new/GPTCache/benchmark_results.db"
    print(f"Creating database at {db_path}...")
    create_database_if_not_exists(db_path)

    # Use ProcessPoolExecutor for parallel execution
    max_workers = min(mp.cpu_count(), 4)  # Use up to 8 workers for better performance
    print(f"Running tests with {max_workers} parallel workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all test configurations
        future_to_config = {
            executor.submit(run_single_test, config): config
            for config in test_configs
        }

        # Process results as they complete
        completed = 0
        for future in as_completed(future_to_config):
            config = future_to_config[future]
            try:
                result = future.result()
                
                # Skip if there was an error
                if 'error' in result:
                    print(f"Test failed: {result['error']}")
                    completed += 1
                    continue
                
                question_config_key = result['question_config']
                max_size = result['max_size']

                # Save result to database
                save_result_to_db(db_path, result)

                completed += 1
                print(f"Completed {completed}/{len(test_configs)} tests - {question_config_key}, max_size: {max_size}")

            except Exception as e:
                print(f"Test failed: {e}")
                completed += 1

    print("Benchmark complete!")
    print(f"Results saved to SQLite database: {db_path}")


if __name__ == "__main__":
    main()

