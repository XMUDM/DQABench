from langchain.docstore.document import Document
from configs import EMBEDDING_MODEL, logger
from server.model_workers.base import ApiEmbeddingsParams
from server.utils import BaseResponse, get_model_worker_config, list_embed_models, list_online_embed_models
from fastapi import Body
from fastapi.concurrency import run_in_threadpool
from typing import Dict, List

online_embed_models = list_online_embed_models()


def embed_texts(
        texts: List[str],
        embed_model: str = EMBEDDING_MODEL,
        to_query: bool = False,
) -> BaseResponse:
    '''
    Vectorize the text. Return data format: BaseResponse(data=List[List[float]])
    TODO: It may be necessary to add a caching mechanism to reduce token consumption
    '''
    try:
        if embed_model in list_embed_models():  # Use the local Embeddings model
            from server.utils import load_local_embeddings

            embeddings = load_local_embeddings(model=embed_model)
            return BaseResponse(data=embeddings.embed_documents(texts))

        if embed_model in list_online_embed_models():  # Using online apis
            config = get_model_worker_config(embed_model)
            worker_class = config.get("worker_class")
            embed_model = config.get("embed_model")
            worker = worker_class()
            if worker_class.can_embedding():
                params = ApiEmbeddingsParams(texts=texts, to_query=to_query, embed_model=embed_model)
                resp = worker.do_embeddings(params)
                return BaseResponse(**resp)

        return BaseResponse(code=500, msg=f"The specified model {embed_model} does not support Embeddings functionality.")
    except Exception as e:
        logger.error(e)
        return BaseResponse(code=500, msg=f"An error occurred during text vectorization:{e}")


async def aembed_texts(
    texts: List[str],
    embed_model: str = EMBEDDING_MODEL,
    to_query: bool = False,
) -> BaseResponse:
    '''
    Vectorize the text. Return data format: BaseResponse(data=List[List[float]])
    '''
    try:
        if embed_model in list_embed_models(): # Use the local Embeddings model
            from server.utils import load_local_embeddings

            embeddings = load_local_embeddings(model=embed_model)
            return BaseResponse(data=await embeddings.aembed_documents(texts))

        if embed_model in list_online_embed_models(): # Using online apis
            return await run_in_threadpool(embed_texts,
                                           texts=texts,
                                           embed_model=embed_model,
                                           to_query=to_query)
    except Exception as e:
        logger.error(e)
        return BaseResponse(code=500, msg=f"An error occurred during text vectorization:{e}")


def embed_texts_endpoint(
        texts: List[str] = Body(..., description="A list of text to embed", examples=[["hello", "world"]]),
        embed_model: str = Body(EMBEDDING_MODEL,
                                description=f"The Embedding model used, in addition to the locally deployed embedding model, also supports embedding services provided by the online API({online_embed_models})."),
        to_query: bool = Body(False, description="Whether the vector is used for the query. Some models such as Minimax are optimized for storage/query vectors."),
) -> BaseResponse:
    '''
    Vectorize the text, returning BaseResponse(data=List[List[float]])
    '''
    return embed_texts(texts=texts, embed_model=embed_model, to_query=to_query)


def embed_documents(
        docs: List[Document],
        embed_model: str = EMBEDDING_MODEL,
        to_query: bool = False,
) -> Dict:
    """
    VectorStore.add_embeddings converts the List[Document] to an acceptable parameter
    """
    texts = [x.page_content for x in docs]
    metadatas = [x.metadata for x in docs]
    embeddings = embed_texts(texts=texts, embed_model=embed_model, to_query=to_query).data
    if embeddings is not None:
        return {
            "texts": texts,
            "embeddings": embeddings,
            "metadatas": metadatas,
        }
