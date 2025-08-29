from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.manager import get_data_manager, VectorBase, CacheBase
from gptcache.embedding import Onnx
from gptcache import cache, Config
from gptcache.adapter import openai
from question_sampler import sample_questions
from common_func import remove_data_files, mock_chat_completion, ensure_directory_permissions, ensure_file_permissions, CPUMonitor
import os
import time


def test(
        questions,
        max_size,
        eviction_base,
        policy = None,
        learning_rate = None,
        quality_weight = None, 
        recency_weight = None, 
        frequency_weight = None,
        file_extension = None
        ):
    file_path = os.path.join(os.path.dirname(__file__), "cache_data")
    ensure_directory_permissions(file_path)
    remove_data_files(file_extension, file_path)
    onnx = Onnx()
    sql_url = f'sqlite:///{os.path.join(os.path.dirname(__file__), f"cache_data/sqlite_{file_extension}.db")}'
    index_path = os.path.join(os.path.dirname(__file__), f"cache_data/faiss_{file_extension}.index")
    vector_base = VectorBase('faiss', dimension=onnx.dimension, index_path=index_path)
    cache_base = CacheBase('sqlite', sql_url=sql_url)

    # Ensure proper permissions on database and index files
    db_path = os.path.join(os.path.dirname(__file__), f"cache_data/sqlite_{file_extension}.db")
    ensure_file_permissions(db_path)
    ensure_file_permissions(index_path)
    if eviction_base == 'quality_score':
        data_manager = get_data_manager(cache_base, 
                                        vector_base, 
                                        max_size=max_size, 
                                        eviction_base='quality_score',
                                        learning_rate=learning_rate,
                                        quality_weight=quality_weight,
                                        recency_weight=recency_weight,
                                        frequency_weight=frequency_weight,
                                        )
    elif eviction_base == 'memory':
        data_manager = get_data_manager(cache_base, 
                                        vector_base, 
                                        max_size=max_size, 
                                        eviction_base='memory',
                                        eviction=policy,
                                        )
    else:
        raise ValueError(f'Invalid eviction base: {eviction_base}')
    
    cache.init(
        embedding_func=onnx.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=SearchDistanceEvaluation(),
        config=Config(),
    )
    
    openai.ChatCompletion.llm = mock_chat_completion
    cache_hit = 0
    request_times = []
    cpu_usages = []
    
    # Start measuring total execution time
    start_time = time.time()
    
    # Initialize CPU monitor
    cpu_monitor = CPUMonitor()
    
    for question_data in questions:
        # Measure time for each individual request
        request_start_time = time.time()
        
        # Start CPU monitoring for this request
        cpu_monitor.start_monitoring()
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": question_data['question']}],
        )
        
        # Stop CPU monitoring and get average CPU usage for this request
        avg_cpu_usage = cpu_monitor.stop_monitoring()
        cpu_usages.append(avg_cpu_usage)
        
        request_end_time = time.time()
        request_times.append(request_end_time - request_start_time)
        
        if response.get('gptcache', False):
            # print("hit")
            cache_hit += 1
        # else:
            # print("miss")
    
    # End measuring total execution time
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate throughput metrics
    num_requests = len(questions)
    requests_per_second = num_requests / total_time if total_time > 0 else 0
    average_request_time = sum(request_times) / num_requests if num_requests > 0 else 0
    min_request_time = min(request_times) if request_times else 0
    max_request_time = max(request_times) if request_times else 0
    
    # Calculate CPU metrics
    average_cpu_usage = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0
    min_cpu_usage = min(cpu_usages) if cpu_usages else 0
    max_cpu_usage = max(cpu_usages) if cpu_usages else 0
    
    remove_data_files(file_extension, file_path)
    return {
        'policy': 'quality_score' if eviction_base == 'quality_score' else policy,
        'requests': len(questions),
        'cache_hit': cache_hit,
        'cache_hit_rate': cache_hit / len(questions),
        'throughput': {
            'total_time': total_time,
            'requests_per_second': requests_per_second,
            'average_request_time': average_request_time,
            'min_request_time': min_request_time,
            'max_request_time': max_request_time,
            'total_requests': num_requests
        },
        'cpu_metrics': {
            'average_cpu_usage': average_cpu_usage,
            'min_cpu_usage': min_cpu_usage,
            'max_cpu_usage': max_cpu_usage,
            'cpu_readings_count': len(cpu_usages)
        },
        'params': {
            'learning_rate': learning_rate,
            'quality_weight': quality_weight,
            'recency_weight': recency_weight,
            'frequency_weight': frequency_weight,
        } if eviction_base == 'quality_score' else {
        }
    }

