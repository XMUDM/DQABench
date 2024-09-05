import nltk
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from configs import VERSION
from configs.model_config import NLTK_DATA_PATH
from configs.server_config import OPEN_CROSS_DOMAIN
import argparse
import uvicorn
from fastapi import Body
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from server.chat.chat import chat
from server.chat.search_engine_chat import search_engine_chat
from server.chat.completion import completion
from server.chat.feedback import chat_feedback
from server.embeddings_api import embed_texts_endpoint
from server.chat.db_chat import db_chat
from server.llm_api import (list_running_models, list_config_models,
                            change_llm_model, stop_llm_model,
                            get_model_config, list_search_engines)
from server.utils import (BaseResponse, ListResponse, FastAPI, MakeFastAPIOffline,
                          get_server_configs, get_prompt_template)
from typing import List, Literal

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path


async def document():
    return RedirectResponse(url="/docs")


def create_app(run_mode: str = None):
    app = FastAPI(
        title="Langchain-Chatchat API Server",
        version=VERSION
    )
    MakeFastAPIOffline(app)
    # Add CORS middleware to allow all origins
    # Set OPEN_DOMAIN=True in config.py to allow cross-domain
    # set OPEN_DOMAIN=True in config.py to allow cross-domain
    if OPEN_CROSS_DOMAIN:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    mount_app_routes(app, run_mode=run_mode)
    return app


def mount_app_routes(app: FastAPI, run_mode: str = None):
    app.get("/",
            response_model=BaseResponse,
            summary="swagger Documentation")(document)

    # Tag: Chat
    app.post("/chat/chat",
             tags=["Chat"],
             summary="Talk to llm models (via LLMChain)",
             )(chat)

    app.post("/chat/search_engine_chat",
             tags=["Chat"],
             summary="Talk to search engines",
             )(search_engine_chat)

    app.post("/chat/feedback",
             tags=["Chat"],
             summary="Return llm model session score",
             )(chat_feedback)
    
    app.post("/chat/db_chat",
            tags=["Chat"],
            summary="Talk to db",
            )(db_chat)

    # Knowledge base correlation interface
    mount_knowledge_routes(app)
    # Abstract correlation interface
    mount_filename_summary_routes(app)

    # LLM related interface
    app.post("/llm_model/list_running_models",
             tags=["LLM Model Management"],
             summary="Lists the currently loaded models",
             )(list_running_models)

    app.post("/llm_model/list_config_models",
             tags=["LLM Model Management"],
             summary="Lists models configured with configs",
             )(list_config_models)

    app.post("/llm_model/get_model_config",
             tags=["LLM Model Management"],
             summary="Get the model configuration (after merge)",
             )(get_model_config)

    app.post("/llm_model/stop",
             tags=["LLM Model Management"],
             summary="Stop the specified LLM Model (Model Worker)",
             )(stop_llm_model)

    app.post("/llm_model/change",
             tags=["LLM Model Management"],
             summary="Toggle the specified LLM Model (Model Worker)",
             )(change_llm_model)

    # Server dependent interface
    app.post("/server/configs",
             tags=["Server State"],
             summary="Obtain the original configuration information of the server",
             )(get_server_configs)

    app.post("/server/list_search_engines",
             tags=["Server State"],
             summary="Obtain the search engine supported by the server",
             )(list_search_engines)

    @app.post("/server/get_prompt_template",
             tags=["Server State"],
             summary="Gets the prompt template for the service area configuration")
    def get_server_prompt_template(
        type: Literal["llm_chat", "knowledge_base_chat", "search_engine_chat", "agent_chat"]=Body("llm_chat", description="The value can be llm_chat, knowledge_base_chat, search_engine_chat, or agent_chat"),
        name: str = Body("default", description="Template name"),
    ) -> str:
        return get_prompt_template(type=type, name=name)

    # Other interface
    app.post("/other/completion",
             tags=["Other"],
             summary="Require llm model completion (via LLMChain)",
             )(completion)

    app.post("/other/embed_texts",
            tags=["Other"],
            summary="Vectorization of text supports both local and online models",
            )(embed_texts_endpoint)


def mount_knowledge_routes(app: FastAPI):
    from server.chat.knowledge_base_chat import knowledge_base_chat
    from server.chat.file_chat import upload_temp_docs, file_chat
    from server.chat.agent_chat import agent_chat
    from server.knowledge_base.kb_api import list_kbs, create_kb, delete_kb
    from server.knowledge_base.kb_doc_api import (list_files, upload_docs, delete_docs,
                                                update_docs, download_doc, recreate_vector_store,
                                                search_docs, DocumentWithVSId, update_info,
                                                update_docs_by_id,)

    app.post("/chat/knowledge_base_chat",
             tags=["Chat"],
             summary="Talk to knowledge base")(knowledge_base_chat)

    app.post("/chat/file_chat",
             tags=["Knowledge Base Management"],
             summary="File chat"
             )(file_chat)

    app.post("/chat/agent_chat",
             tags=["Chat"],
             summary="Chat with agent")(agent_chat)

    # Tag: Knowledge Base Management
    app.get("/knowledge_base/list_knowledge_bases",
            tags=["Knowledge Base Management"],
            response_model=ListResponse,
            summary="Get the knowledge base list")(list_kbs)

    app.post("/knowledge_base/create_knowledge_base",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="Create a knowledge base"
             )(create_kb)

    app.post("/knowledge_base/delete_knowledge_base",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="Delete a knowledge base"
             )(delete_kb)

    app.get("/knowledge_base/list_files",
            tags=["Knowledge Base Management"],
            response_model=ListResponse,
            summary="Get a list of files in the Knowledge base"
            )(list_files)

    app.post("/knowledge_base/search_docs",
             tags=["Knowledge Base Management"],
             response_model=List[DocumentWithVSId],
             summary="Search knowledge base"
             )(search_docs)

    app.post("/knowledge_base/update_docs_by_id",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="Update the knowledge base document directly"
             )(update_docs_by_id)


    app.post("/knowledge_base/upload_docs",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="Upload files to the knowledge base and/or vectorize"
             )(upload_docs)

    app.post("/knowledge_base/delete_docs",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="Delete a specified file from the knowledge base"
             )(delete_docs)

    app.post("/knowledge_base/update_info",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="Update Knowledge base Introduction"
             )(update_info)
    app.post("/knowledge_base/update_docs",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="Update existing files to the knowledge base"
             )(update_docs)

    app.get("/knowledge_base/download_doc",
            tags=["Knowledge Base Management"],
            summary="Download the knowledge file")(download_doc)

    app.post("/knowledge_base/recreate_vector_store",
             tags=["Knowledge Base Management"],
             summary="According to the document reconstruction vector library in content, the processing progress is streamed out."
             )(recreate_vector_store)

    app.post("/knowledge_base/upload_temp_docs",
             tags=["Knowledge Base Management"],
             summary="Upload files to a temporary directory for file conversations."
             )(upload_temp_docs)


def mount_filename_summary_routes(app: FastAPI):
    from server.knowledge_base.kb_summary_api import (summary_file_to_vector_store, recreate_summary_vector_store,
                                                      summary_doc_ids_to_vector_store)

    app.post("/knowledge_base/kb_summary_api/summary_file_to_vector_store",
             tags=["Knowledge kb_summary_api Management"],
             summary="Individual knowledge bases are summarized according to file names"
             )(summary_file_to_vector_store)
    app.post("/knowledge_base/kb_summary_api/summary_doc_ids_to_vector_store",
             tags=["Knowledge kb_summary_api Management"],
             summary="Individual knowledge base is summarized according to doc_ids",
             response_model=BaseResponse,
             )(summary_doc_ids_to_vector_store)
    app.post("/knowledge_base/kb_summary_api/recreate_summary_vector_store",
             tags=["Knowledge kb_summary_api Management"],
             summary="Rebuild a single KB file summary"
             )(recreate_summary_vector_store)



def run_api(host, port, **kwargs):
    if kwargs.get("ssl_keyfile") and kwargs.get("ssl_certfile"):
        uvicorn.run(app,
                    host=host,
                    port=port,
                    ssl_keyfile=kwargs.get("ssl_keyfile"),
                    ssl_certfile=kwargs.get("ssl_certfile"),
                    )
    else:
        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='langchain-ChatGLM',
                                     description='About langchain-ChatGLM, local knowledge based ChatGLM with langchain')
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--ssl_keyfile", type=str)
    parser.add_argument("--ssl_certfile", type=str)
    # Initialize message
    args = parser.parse_args()
    args_dict = vars(args)

    app = create_app()

    run_api(host=args.host,
            port=args.port,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            )
