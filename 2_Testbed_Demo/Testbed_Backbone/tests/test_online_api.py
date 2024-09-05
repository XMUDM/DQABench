import sys
from pathlib import Path
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from configs import ONLINE_LLM_MODEL
from server.model_workers.base import *
from server.utils import get_model_worker_config, list_config_llm_models
from pprint import pprint
import pytest


workers = []
for x in list_config_llm_models()["online"]:
    if x in ONLINE_LLM_MODEL and x not in workers:
        workers.append(x)
print(f"all workers to test: {workers}")

# workers = ["fangzhou-api"]


@pytest.mark.parametrize("worker", workers)
def test_chat(worker):
    params = ApiChatParams(
        messages = [
            {"role": "user", "content": "Who are you"},
        ],
    )
    print(f"\nchat with {worker} \n")

    if worker_class := get_model_worker_config(worker).get("worker_class"):
        for x in worker_class().do_chat(params):
            pprint(x)
            assert isinstance(x, dict)
            assert x["error_code"] == 0


@pytest.mark.parametrize("worker", workers)
def test_embeddings(worker):
    params = ApiEmbeddingsParams(
        texts = [
            "LangChain-Chatchat (formerly Langchain-Chatglm): Implementation of local knowledge base Q&A application based on Langchain and ChatGLM and other large language models.",
            "A Q&A application based on local knowledge base is realized by using langchain idea. The goal is to build a Q&A solution of knowledge base that is friendly to Chinese scene and open source model and can run offline.",
        ]
    )

    if worker_class := get_model_worker_config(worker).get("worker_class"):
        if worker_class.can_embedding():
            print(f"\embeddings with {worker} \n")
            resp = worker_class().do_embeddings(params)

            pprint(resp, depth=2)
            assert resp["code"] == 200
            assert "data" in resp
            embeddings = resp["data"]
            assert isinstance(embeddings, list) and len(embeddings) > 0
            assert isinstance(embeddings[0], list) and len(embeddings[0]) > 0
            assert isinstance(embeddings[0][0], float)
            print("Vector length:", len(embeddings[0]))

