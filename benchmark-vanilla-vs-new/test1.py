import json
import os
import time
import sys

# Add the path to cloned GPTCache repo
sys.path.insert(0, r'C:\Users\yossi_1wprdhx\Desktop\llmcache\GPTCache')

from gptcache.adapter import openai
from gptcache import cache, Config
from gptcache.similarity_evaluation.onnx import OnnxModelEvaluation
from gptcache.embedding import Onnx as EmbeddingOnnx
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.manager import manager_factory

# setup to mockup
os.environ['OPENAI_API_KEY'] = 'mockup'
cache.set_openai_key()


#
def response_text(openai_resp):
    return openai_resp['choices'][0]['message']['content']


# Clean up any existing cache data
if os.path.exists("./cache_data"):
    import shutil

    shutil.rmtree("./cache_data")
    print("🧹 Cleaned up existing cache data")

# mode = os.environ["TESTING_MODE"]  # either vanilla or quality

cache_size = 100000


def run():
    sys.path.insert(0, r'C:\Users\yossi_1wprdhx\Desktop\llmcache\GPTCache')

    class WrapEvaluation(SearchDistanceEvaluation):
        def evaluation(self, src_dict, cache_dict, **kwargs):  # eval distance
            return super().evaluation(src_dict, cache_dict, **kwargs)

        def range(self):
            return super().range()

    embedding_onnx = EmbeddingOnnx()

    sqlite_file = "./cache_data/sqlite.db"
    faiss_file = "./cache_data/faiss.index"
    has_data = os.path.isfile(sqlite_file) and os.path.isfile(faiss_file)

    data_manager = manager_factory(
        manager="sqlite,faiss",
        data_dir="./cache_data",
        eviction_manager="quality_score",
        vector_params={"dimension": 768},
        eviction_params={
            "maxsize": cache_size,
            "clean_size": 1,  # Evict 1 item at a time for clear demonstration
            "learning_rate": 0.3,  # Learn reasonably fast
            "quality_weight": 0.8,  # Heavily prioritize quality
            "recency_weight": 0.15,  # Some recency consideration
            "frequency_weight": 0.05  # Minimal frequency impact
        }
    )

    cache.init(
        embedding_func=embedding_onnx.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=WrapEvaluation(),
        config=Config(similarity_threshold=0.95),
    )

    with open(r"C:\Users\yossi_1wprdhx\Desktop\llmcache\GPTCache\examples\benchmark\mock_data.json", "r") as mock_file:
        mock_data = json.load(mock_file)  # using gptcahce data for benchmarking

    # add id to each pair
    i = 0
    for pair in mock_data:
        pair["id"] = str(i)
        i += 1

    # populating the cache to baseline meaningfully
    if not has_data:
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

    all_time = 0.0
    hit_cache_positive, hit_cache_negative = 0, 0
    fail_count = 0
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
            if res_text == pair["id"]:
                hit_cache_positive += 1
            else:
                hit_cache_negative += 1
            consume_time = time.time() - start_time
            all_time += consume_time
            print("cache hint time consuming: {:.2f}s".format(consume_time))
        except:
            fail_count += 1

    print("average time: {:.2f}s".format(all_time / len(mock_data)))
    print("cache_hint_positive:", hit_cache_positive)
    print("hit_cache_negative:", hit_cache_negative)
    print("fail_count:", fail_count)
    print("average embedding time: ", cache.report.average_embedding_time())
    print("average search time: ", cache.report.average_search_time())


run()
