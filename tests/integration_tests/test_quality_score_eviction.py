import pytest

from base.client_base import Base
from tests.integration_tests.common import common_func as cf
from gptcache import cache, Config
from gptcache.adapter import openai
from gptcache.embedding import Onnx
from gptcache.manager import get_data_manager, VectorBase
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from utils.util_log import test_log as log

class TestQualityScoreEviction(Base):
    def test_quality_score_eviction(self):
        """
        target: test quality score eviction
        method: test quality score eviction
        expected: quality score eviction
        """
        onnx = Onnx()
        vector_base = VectorBase("faiss", dimension=onnx.dimension)
        data_manager = get_data_manager("sqlite", vector_base, max_size=2000, eviction_base="quality_score")
        cache.init(
            embedding_func=onnx.to_embeddings,
            data_manager=data_manager,
            similarity_evaluation=SearchDistanceEvaluation(),
            config=Config(
                log_time_func=cf.log_time_func,
            ),
        )

        question = "what do you think about chatgpt"
        answer = "chatgpt is a good application"
        
        openai.ChatCompletion.llm = cf.mock_chat_completion
        cache.data_manager.save(question, answer, cache.embedding_func(question))

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "what do you feel like chatgpt"},
            ],
        )
        assert response.get("gptcache", False)
