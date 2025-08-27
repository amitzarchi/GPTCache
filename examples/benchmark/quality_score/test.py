from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.manager import get_data_manager, VectorBase, CacheBase
from gptcache.embedding import Onnx
from gptcache import cache, Config
from gptcache.adapter import openai
from question_sampler import sample_questions
from common_func import remove_data_files, mock_chat_completion

def test(
        questions,
        max_size,
        eviction_base,
        policy = None,
        learning_rate = None,
        quality_weight = None, 
        recency_weight = None, 
        frequency_weight = None
        ):
    remove_data_files()
    onnx = Onnx()
    vector_base = VectorBase('faiss', dimension=onnx.dimension)
    if eviction_base == 'quality_score':
        data_manager = get_data_manager('sqlite', 
                                        vector_base, 
                                        max_size=max_size, 
                                        eviction_base='quality_score',
                                        learning_rate=learning_rate,
                                        quality_weight=quality_weight,
                                        recency_weight=recency_weight,
                                        frequency_weight=frequency_weight,
                                        )
    elif eviction_base == 'memory':
        data_manager = get_data_manager('sqlite', 
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

    remove_data_files()
    return {
        'requests': len(questions),
        'cache_hit': cache_hit,
        'cache_hit_rate': cache_hit / len(questions),
    }


questions = sample_questions('MIXED', 300)
print(f"sampled {len(questions)} questions")

quality_score_result = test(
        questions=questions,
        eviction_base='quality_score',
        max_size=5,
        learning_rate=0.3,
        quality_weight=0.7,
        recency_weight=0.2,
        frequency_weight=0.1,
    )

LRU_result = test(
        questions=questions,
        eviction_base='memory',
        policy='LRU',
        max_size=5,
    )

print(f"quality_score_result: {quality_score_result}")
print(f"LRU_result: {LRU_result}")







