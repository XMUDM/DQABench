import requests
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from configs import BING_SUBSCRIPTION_KEY
from server.utils import api_address

from pprint import pprint


api_base_url = api_address()


def dump_input(d, title):
    print("\n")
    print("=" * 30 + title + "  input " + "="*30)
    pprint(d)


def dump_output(r, title):
    print("\n")
    print("=" * 30 + title + "  output" + "="*30)
    for line in r.iter_content(None, decode_unicode=True):
        print(line, end="", flush=True)


headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}

data = {
    "query": "Please introduce yourself in about 100 words",
    "history": [
        {
            "role": "user",
            "content": "Hello"
        },
        {
            "role": "assistant",
            "content": "Hi, I'm a big model of artificial intelligence"
        }
    ],
    "stream": True,
    "temperature": 0.7,
}


def test_chat_chat(api="/chat/chat"):
    url = f"{api_base_url}{api}"
    dump_input(data, api)
    response = requests.post(url, headers=headers, json=data, stream=True)
    dump_output(response, api)
    assert response.status_code == 200


def test_knowledge_chat(api="/chat/knowledge_base_chat"):
    url = f"{api_base_url}{api}"
    data = {
        "query": "How to ask questions to get quality answers",
        "knowledge_base_name": "samples",
        "history": [
            {
                "role": "user",
                "content": "Hello"
            },
            {
                "role": "assistant",
                "content": "Hi, I'm ChatGLM"
            }
        ],
        "stream": True
    }
    dump_input(data, api)
    response = requests.post(url, headers=headers, json=data, stream=True)
    print("\n")
    print("=" * 30 + api + "  output" + "="*30)
    for line in response.iter_content(None, decode_unicode=True):
        data = json.loads(line[6:])
        if "answer" in data:
            print(data["answer"], end="", flush=True)
    pprint(data)
    assert "docs" in data and len(data["docs"]) > 0
    assert response.status_code == 200


def test_search_engine_chat(api="/chat/search_engine_chat"):
    global data

    data["query"] = "What are the latest advances in room temperature superconductivity?"

    url = f"{api_base_url}{api}"
    for se in ["bing", "duckduckgo"]:
        data["search_engine_name"] = se
        dump_input(data, api + f" by {se}")
        response = requests.post(url, json=data, stream=True)
        if se == "bing" and not BING_SUBSCRIPTION_KEY:
            data = response.json()
            assert data["code"] == 404
            assert data["msg"] == f"To use the Bing search engine, set `BING_SUBSCRIPTION_KEY`"

        print("\n")
        print("=" * 30 + api + f" by {se}  output" + "="*30)
        for line in response.iter_content(None, decode_unicode=True):
            data = json.loads(line[6:])
            if "answer" in data:
                print(data["answer"], end="", flush=True)
        assert "docs" in data and len(data["docs"]) > 0
        pprint(data["docs"])
        assert response.status_code == 200

