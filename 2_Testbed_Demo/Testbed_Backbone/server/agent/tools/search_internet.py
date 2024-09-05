import json
from server.chat.search_engine_chat import search_engine_chat
from configs import VECTOR_SEARCH_TOP_K, MAX_TOKENS
import asyncio
from server.agent import model_container
from pydantic import BaseModel, Field

async def search_engine_iter(query: str):
    response = await search_engine_chat(query=query,
                                         search_engine_name="bing", # Switch search engine here
                                         model_name=model_container.MODEL.model_name,
                                         temperature=0.01, # When the Agent searches the Internet, the temperature is set to 0.01
                                         history=[],
                                         top_k = VECTOR_SEARCH_TOP_K,
                                         max_tokens= MAX_TOKENS,
                                         prompt_name = "default",
                                         stream=False)

    contents = ""

    async for data in response.body_iterator:
        data = json.loads(data)
        contents = data["answer"]
        docs = data["docs"]

    return contents

def search_internet(query: str):
    return asyncio.run(search_engine_iter(query))

class SearchInternetInput(BaseModel):
    location: str = Field(description="Query for Internet search")


if __name__ == "__main__":
    result = search_internet("What day is today?")
    print("Answer:",result)
