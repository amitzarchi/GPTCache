from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.manager import get_data_manager, VectorBase, CacheBase
from gptcache.embedding import Onnx
from gptcache import cache, Config
from gptcache.adapter import openai
from question_sampler import sample_questions
from common_func import remove_data_files, mock_chat_completion, ensure_directory_permissions, ensure_file_permissions
import os

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
    for question_data in questions:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": question_data['question']}],
        )
        if response.get('gptcache', False):
            cache_hit += 1
    remove_data_files(file_extension, file_path)
    return {
        'policy': 'quality_score' if eviction_base == 'quality_score' else policy,
        'requests': len(questions),
        'cache_hit': cache_hit,
        'cache_hit_rate': cache_hit / len(questions),
        'params': {
            'learning_rate': learning_rate,
            'quality_weight': quality_weight,
            'recency_weight': recency_weight,
            'frequency_weight': frequency_weight,
        } if eviction_base == 'quality_score' else {
        }
    }


