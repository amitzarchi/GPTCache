from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.manager import get_data_manager, VectorBase, CacheBase
from gptcache.embedding import Onnx
from gptcache import cache, Config
from gptcache.adapter import openai
from question_sampler import sample_questions
from common_func import remove_data_files, mock_chat_completion, ensure_directory_permissions, ensure_file_permissions
import os
import gc
import time

def close_connections(data_manager, cache_base, vector_base):
    """Explore GPTCache structure and close all connections systematically"""
    try:
        # Debug data_manager structure
        if data_manager:

            # Look for scalar storage (SQL)
            scalar_storage = None
            for attr_name in ['s', 'scalar_data', 'cache_storage']:
                if hasattr(data_manager, attr_name):
                    scalar_storage = getattr(data_manager, attr_name)
                    break

            if scalar_storage:

                # Try to find the actual SQLAlchemy session/engine
                if hasattr(scalar_storage, 'session'):
                    if hasattr(scalar_storage.session, 'close'):
                        scalar_storage.session.close()

                if hasattr(scalar_storage, 'engine'):
                    if hasattr(scalar_storage.engine, 'dispose'):
                        scalar_storage.engine.dispose()

                # Check for other connection-related attributes
                for attr in ['connection', 'conn', '_engine', '_session', 'db_engine']:
                    if hasattr(scalar_storage, attr):
                        obj = getattr(scalar_storage, attr)
                        if hasattr(obj, 'close'):
                            obj.close()
                        if hasattr(obj, 'dispose'):
                            obj.dispose()

                # Finally, try generic close on storage
                if hasattr(scalar_storage, 'close'):
                    scalar_storage.close()

        # Debug cache_base structure
        if cache_base:

            # Look for storage within cache_base
            if hasattr(cache_base, 'storage'):
                storage = cache_base.storage

                # Close all possible connections in storage
                for attr in ['session', 'engine', '_session', '_engine', 'connection', 'conn']:
                    if hasattr(storage, attr):
                        obj = getattr(storage, attr)
                        if hasattr(obj, 'close'):
                            obj.close()
                        if hasattr(obj, 'dispose'):
                            obj.dispose()

                # Close storage itself
                if hasattr(storage, 'close'):
                    storage.close()

            # Close cache_base itself
            if hasattr(cache_base, 'close'):
                cache_base.close()

    except Exception as e:
        print(f"Error during close: {e}")
        import traceback
        traceback.print_exc()


def test(
        questions,
        max_size,
        eviction_base,
        policy=None,
        learning_rate=None,
        quality_weight=None,
        recency_weight=None,
        frequency_weight=None,
        file_extension=None
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

    data_manager = None
    result = None

    try:
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
        cache_hit_positive = 0
        cache_hit_negative = 0
        for question_data in questions:
            # question_cluster_id = f"{question_data['cluster_id']}"
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": question_data['question']}],
                # messages=[{"role": "user", "content": question_cluster_id}], #save cluster id - we don't need the actual answer for the test
            )
            if response.get('gptcache', False):
                cache_hit += 1
                # print(response['choices'][0]['message']['content'], question_cluster_id)
                # if response['choices'][0]['message']['content'] == question_cluster_id:
                #     cache_hit_positive += 1
                # else:
                #     cache_hit_negative += 1

        result = {
            'policy': 'quality_score' if eviction_base == 'quality_score' else policy,
            'requests': len(questions),
            'cache_hit': cache_hit,
            # 'cache_hit_positive': cache_hit_positive,
            # 'cache_hit_negative': cache_hit_negative,
            # 'precision': cache_hit_positive/(cache_hit_positive+cache_hit_negative),
            'cache_hit_rate': cache_hit / len(questions),
            'params': {
                'learning_rate': learning_rate,
                'quality_weight': quality_weight,
                'recency_weight': recency_weight,
                'frequency_weight': frequency_weight,
            } if eviction_base == 'quality_score' else {
            }
        }

    finally:
        # Comprehensive debug and cleanup
        try:
            # Flush cache first
            if hasattr(cache, 'flush'):
                cache.flush()
                print("Flushed cache")

            # Debug and close all connections systematically
            close_connections(data_manager, cache_base, vector_base)

            # Force garbage collection multiple times
            for _ in range(3):
                gc.collect()
                time.sleep(0.1)

            time.sleep(0.5)  # Give Windows more time to release file handles

        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")

        # Clean up files
        remove_data_files(file_extension, file_path)

    print(result)
    return result