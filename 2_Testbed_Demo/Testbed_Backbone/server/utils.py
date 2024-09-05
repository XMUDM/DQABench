import pydantic
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI
from pathlib import Path
import asyncio
from configs import (LLM_MODELS, LLM_DEVICE, EMBEDDING_DEVICE,
                     MODEL_PATH, MODEL_ROOT_PATH, ONLINE_LLM_MODEL, logger, log_verbose,
                     FSCHAT_MODEL_WORKERS, HTTPX_DEFAULT_TIMEOUT)
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import httpx
from typing import Literal, Optional, Callable, Generator, Dict, Any, Awaitable, Union, Tuple
import logging
import torch


async def wrap_done(fn: Awaitable, event: asyncio.Event):
    """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
    try:
        await fn
    except Exception as e:
        logging.exception(e)
        # TODO: handle exception
        msg = f"Caught exception: {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
    finally:
        # Signal the aiter to stop.
        event.set()


def get_ChatOpenAI(
        model_name: str,
        temperature: float,
        max_tokens: int = None,
        streaming: bool = True,
        callbacks: List[Callable] = [],
        verbose: bool = True,
        **kwargs: Any,
) -> ChatOpenAI:
    config = get_model_worker_config(model_name)
    if model_name == "openai-api":
        model_name = config.get("model_name")

    model = ChatOpenAI(
        streaming=streaming,
        verbose=verbose,
        callbacks=callbacks,
        openai_api_key=config.get("api_key", "EMPTY"),
        openai_api_base=config.get("api_base_url", fschat_openai_api_address()),
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_proxy=config.get("openai_proxy"),
        **kwargs
    )
    return model


def get_OpenAI(
        model_name: str,
        temperature: float,
        max_tokens: int = None,
        streaming: bool = True,
        echo: bool = True,
        callbacks: List[Callable] = [],
        verbose: bool = True,
        **kwargs: Any,
) -> OpenAI:
    config = get_model_worker_config(model_name)
    if model_name == "openai-api":
        model_name = config.get("model_name")
    model = OpenAI(
        streaming=streaming,
        verbose=verbose,
        callbacks=callbacks,
        openai_api_key=config.get("api_key", "EMPTY"),
        openai_api_base=config.get("api_base_url", fschat_openai_api_address()),
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_proxy=config.get("openai_proxy"),
        echo=echo,
        **kwargs
    )
    return model


class BaseResponse(BaseModel):
    code: int = pydantic.Field(200, description="API status code")
    msg: str = pydantic.Field("success", description="API status message")
    data: Any = pydantic.Field(None, description="API data")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }


class ListResponse(BaseResponse):
    data: List[str] = pydantic.Field(..., description="List of names")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
                "data": ["doc1.docx", "doc2.pdf", "doc3.txt"],
            }
        }


class ChatMessage(BaseModel):
    question: str = pydantic.Field(..., description="Question text")
    response: str = pydantic.Field(..., description="Response text")
    history: List[List[str]] = pydantic.Field(..., description="History text")
    source_documents: List[str] = pydantic.Field(
        ..., description="List of source documents and their scores"
    )

    class Config:
        schema_extra = {
            "example": {
                "question": "How to apply for industrial injury insurance?",
                "response": "According to the known information, it can be summarized as follows: \n\n1. The insured unit pays the injury insurance premium for the employees to ensure that the employees can get the corresponding treatment in the event of injury. \n"
                            "2. The payment rules of work-related injury insurance may be different in different regions, so you need to consult the local social security department for the specific payment standards and regulations. \n"
                            "3. Employees injured at work and their close relatives need to apply for identification of injuries, confirm the eligibility for benefits, and pay work-related injury insurance premiums on time. \n"
                            "4. Industrial injury insurance benefits include industrial injury medical treatment, rehabilitation, auxiliary equipment configuration costs, disability benefits, workers' death benefits, one-time workers' death benefits, etc. \n"
                            "5. The qualification certification for receiving industrial injury insurance benefits includes the certification for receiving long-term benefits and the certification for receiving one-time benefits. \n"
                            "6. The benefits paid by the industrial injury insurance fund include medical treatment for industrial injuries, rehabilitation benefits, costs for the allocation of assistive devices, one-off workers' death benefits, funeral benefits, etc.",
                "history": [
                    [
                        "What is injury insurance?",
                        "Work-related injury insurance refers to the payment of work-related injury insurance premiums by the employing unit for its own employees and other personnel in accordance with state regulations."
                        "A social insurance system in which workers' compensation is granted by the insurance institution according to the standards set by the State.",
                    ]
                ],
                "source_documents": [
                    "Source [1] Guangzhou units engaged in specific personnel to participate in industrial injury insurance guidelines.docx: \n\n\t"
                    "(1) The employer (organization) shall, in accordance with the principle of \"voluntary participation in insurance\", participate in industrial injury insurance and pay industrial injury insurance premiums for specific employees who have not established labor relations.",
                    "Source [2]...",
                    "Source [3]...",
                ],
            }
        }


def torch_gc():
    try:
        if torch.cuda.is_available():
            # with torch.cuda.device(DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif torch.backends.mps.is_available():
            try:
                from torch.mps import empty_cache
                empty_cache()
            except Exception as e:
                msg = ("If you are using macOS, it is recommended to upgrade the pytorch version to 2.0.0 or later "
                       " to support timely cleaning of the memory footprint generated by torch.")
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
    except Exception:
        ...


def run_async(cor):
    '''
    Running asynchronous code in a synchronous environment.
    '''
    try:
        loop = asyncio.get_event_loop()
    except:
        loop = asyncio.new_event_loop()
    return loop.run_until_complete(cor)


def iter_over_async(ait, loop=None):
    '''
    Encapsulate an asynchronous generator into a synchronous generator.
    '''
    ait = ait.__aiter__()

    async def get_next():
        try:
            obj = await ait.__anext__()
            return False, obj
        except StopAsyncIteration:
            return True, None

    if loop is None:
        try:
            loop = asyncio.get_event_loop()
        except:
            loop = asyncio.new_event_loop()

    while True:
        done, obj = loop.run_until_complete(get_next())
        if done:
            break
        yield obj


def MakeFastAPIOffline(
        app: FastAPI,
        static_dir=Path(__file__).parent / "static",
        static_url="/static-offline-docs",
        docs_url: Optional[str] = "/docs",
        redoc_url: Optional[str] = "/redoc",
) -> None:
    """patch the FastAPI obj that doesn't rely on CDN for the documentation page"""
    from fastapi import Request
    from fastapi.openapi.docs import (
        get_redoc_html,
        get_swagger_ui_html,
        get_swagger_ui_oauth2_redirect_html,
    )
    from fastapi.staticfiles import StaticFiles
    from starlette.responses import HTMLResponse

    openapi_url = app.openapi_url
    swagger_ui_oauth2_redirect_url = app.swagger_ui_oauth2_redirect_url

    def remove_route(url: str) -> None:
        '''
        remove original route from app
        '''
        index = None
        for i, r in enumerate(app.routes):
            if r.path.lower() == url.lower():
                index = i
                break
        if isinstance(index, int):
            app.routes.pop(index)

    # Set up static file mount
    app.mount(
        static_url,
        StaticFiles(directory=Path(static_dir).as_posix()),
        name="static-offline-docs",
    )

    if docs_url is not None:
        remove_route(docs_url)
        remove_route(swagger_ui_oauth2_redirect_url)

        # Define the doc and redoc pages, pointing at the right files
        @app.get(docs_url, include_in_schema=False)
        async def custom_swagger_ui_html(request: Request) -> HTMLResponse:
            root = request.scope.get("root_path")
            favicon = f"{root}{static_url}/favicon.png"
            return get_swagger_ui_html(
                openapi_url=f"{root}{openapi_url}",
                title=app.title + " - Swagger UI",
                oauth2_redirect_url=swagger_ui_oauth2_redirect_url,
                swagger_js_url=f"{root}{static_url}/swagger-ui-bundle.js",
                swagger_css_url=f"{root}{static_url}/swagger-ui.css",
                swagger_favicon_url=favicon,
            )

        @app.get(swagger_ui_oauth2_redirect_url, include_in_schema=False)
        async def swagger_ui_redirect() -> HTMLResponse:
            return get_swagger_ui_oauth2_redirect_html()

    if redoc_url is not None:
        remove_route(redoc_url)

        @app.get(redoc_url, include_in_schema=False)
        async def redoc_html(request: Request) -> HTMLResponse:
            root = request.scope.get("root_path")
            favicon = f"{root}{static_url}/favicon.png"

            return get_redoc_html(
                openapi_url=f"{root}{openapi_url}",
                title=app.title + " - ReDoc",
                redoc_js_url=f"{root}{static_url}/redoc.standalone.js",
                with_google_fonts=False,
                redoc_favicon_url=favicon,
            )


# Get model information from model_config

def list_embed_models() -> List[str]:
    '''
    get names of configured embedding models
    '''
    return list(MODEL_PATH["embed_model"])


def list_config_llm_models() -> Dict[str, Dict]:
    '''
    get configured llm models with different types.
    return {config_type: {model_name: config}, ...}
    '''
    workers = FSCHAT_MODEL_WORKERS.copy()
    workers.pop("default", None)

    return {
        "local": MODEL_PATH["llm_model"].copy(),
        "online": ONLINE_LLM_MODEL.copy(),
        "worker": workers,
    }


def get_model_path(model_name: str, type: str = None) -> Optional[str]:
    if type in MODEL_PATH:
        paths = MODEL_PATH[type]
    else:
        paths = {}
        for v in MODEL_PATH.values():
            paths.update(v)

    if path_str := paths.get(model_name):  # Take "chatglm-6b": "THUDM/chatglm-6b-new" as an example. The following paths are supported
        path = Path(path_str)
        if path.is_dir():  # Arbitrary absolute path
            return str(path)

        root_path = Path(MODEL_ROOT_PATH)
        if root_path.is_dir():
            path = root_path / model_name
            if path.is_dir():  # use key, {MODEL_ROOT_PATH}/chatglm-6b
                return str(path)
            path = root_path / path_str
            if path.is_dir():  # use value, {MODEL_ROOT_PATH}/THUDM/chatglm-6b-new
                return str(path)
            path = root_path / path_str.split("/")[-1]
            if path.is_dir():  # use value split by "/", {MODEL_ROOT_PATH}/chatglm-6b-new
                return str(path)
        return path_str  # THUDM/chatglm06b


# Get service information from server_config

def get_model_worker_config(model_name: str = None) -> dict:
    '''
    Load the configuration items for the model worker.
    Priority:FSCHAT_MODEL_WORKERS[model_name] > ONLINE_LLM_MODEL[model_name] > FSCHAT_MODEL_WORKERS["default"]
    '''
    from configs.model_config import ONLINE_LLM_MODEL, MODEL_PATH
    from configs.server_config import FSCHAT_MODEL_WORKERS
    from server import model_workers

    config = FSCHAT_MODEL_WORKERS.get("default", {}).copy()
    config.update(ONLINE_LLM_MODEL.get(model_name, {}).copy())
    config.update(FSCHAT_MODEL_WORKERS.get(model_name, {}).copy())

    if model_name in ONLINE_LLM_MODEL:
        config["online_api"] = True
        if provider := config.get("provider"):
            try:
                config["worker_class"] = getattr(model_workers, provider)
            except Exception as e:
                msg = f"The provider of the online model \'{model_name}\' is not configured correctly"
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
    # Local model
    if model_name in MODEL_PATH["llm_model"]:
        path = get_model_path(model_name)
        config["model_path"] = path
        if path and os.path.isdir(path):
            config["model_path_exists"] = True
        config["device"] = llm_device(config.get("device"))
    return config


def get_all_model_worker_configs() -> dict:
    result = {}
    model_names = set(FSCHAT_MODEL_WORKERS.keys())
    for name in model_names:
        if name != "default":
            result[name] = get_model_worker_config(name)
    return result


def fschat_controller_address() -> str:
    from configs.server_config import FSCHAT_CONTROLLER

    host = FSCHAT_CONTROLLER["host"]
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = FSCHAT_CONTROLLER["port"]
    return f"http://{host}:{port}"


def fschat_model_worker_address(model_name: str = LLM_MODELS[0]) -> str:
    if model := get_model_worker_config(model_name):  # TODO: depends fastchat
        host = model["host"]
        if host == "0.0.0.0":
            host = "127.0.0.1"
        port = model["port"]
        return f"http://{host}:{port}"
    return ""


def fschat_openai_api_address() -> str:
    from configs.server_config import FSCHAT_OPENAI_API

    host = FSCHAT_OPENAI_API["host"]
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = FSCHAT_OPENAI_API["port"]
    return f"http://{host}:{port}/v1"


def api_address() -> str:
    from configs.server_config import API_SERVER

    host = API_SERVER["host"]
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = API_SERVER["port"]
    return f"http://{host}:{port}"


def webui_address() -> str:
    from configs.server_config import WEBUI_SERVER

    host = WEBUI_SERVER["host"]
    port = WEBUI_SERVER["port"]
    return f"http://{host}:{port}"


def get_prompt_template(type: str, name: str) -> Optional[str]:
    '''
    The template content was loaded from prompt_config
    type: one of "llm_chat","agent_chat","knowledge_base_chat","search_engine_chat". If new functions are available, they should be added.
    '''

    from configs import prompt_config
    import importlib
    importlib.reload(prompt_config)  # TODO: Check whether the configs/prompt_config.py file is modified and reload it again
    return prompt_config.PROMPT_TEMPLATES[type].get(name)


def set_httpx_config(
        timeout: float = HTTPX_DEFAULT_TIMEOUT,
        proxy: Union[str, Dict] = None,
):
    '''
    Set the default httpx timeout. The default timeout of httpx is 5 seconds, which is not sufficient when requesting an LLM answer.
    Add services related to this project to the no-proxy list to avoid server request errors for fastchat. (Not valid on windows)
    For online apis such as chatgpt, you need to configure the proxy manually if you want to use it. How to deal with the agent of search engine still needs to be considered.
    '''

    import httpx
    import os

    httpx._config.DEFAULT_TIMEOUT_CONFIG.connect = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.read = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.write = timeout

    # Set system-level proxies at process scope
    proxies = {}
    if isinstance(proxy, str):
        for n in ["http", "https", "all"]:
            proxies[n + "_proxy"] = proxy
    elif isinstance(proxy, dict):
        for n in ["http", "https", "all"]:
            if p := proxy.get(n):
                proxies[n + "_proxy"] = p
            elif p := proxy.get(n + "_proxy"):
                proxies[n + "_proxy"] = p

    for k, v in proxies.items():
        os.environ[k] = v

    # set host to bypass proxy
    no_proxy = [x.strip() for x in os.environ.get("no_proxy", "").split(",") if x.strip()]
    no_proxy += [
        # do not use proxy for locahost
        "http://127.0.0.1",
        "http://localhost",
    ]
    # do not use proxy for user deployed fastchat servers
    for x in [
        fschat_controller_address(),
        fschat_model_worker_address(),
        fschat_openai_api_address(),
    ]:
        host = ":".join(x.split(":")[:2])
        if host not in no_proxy:
            no_proxy.append(host)
    os.environ["NO_PROXY"] = ",".join(no_proxy)

    # TODO: Simply clearing the system agent is not a good option because it has too many impacts. It seems better to modify the bypass list of the proxy server.
    # patch requests to use custom proxies instead of system settings
    def _get_proxies():
        return proxies

    import urllib.request
    urllib.request.getproxies = _get_proxies

    # Automatically checks torch for available devices. In distributed deployment, torch can not be installed on machines that do not run LLM


def is_mps_available():
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def is_cuda_available():
    return torch.cuda.is_available()


def detect_device() -> Literal["cuda", "mps", "cpu"]:
    try:
        if torch.cuda.is_available():
            return "cuda"
        if is_mps_available():
            return "mps"
    except:
        pass
    return "cpu"


def llm_device(device: str = None) -> Literal["cuda", "mps", "cpu", "xpu"]:
    device = device or LLM_DEVICE
    if device not in ["cuda", "mps", "cpu", "xpu"]:
        logging.warning(f"device not in ['cuda', 'mps', 'cpu','xpu'], device = {device}")
        device = detect_device()
    elif device == 'cuda' and not is_cuda_available() and is_mps_available():
        logging.warning("cuda is not available, fallback to mps")
        return "mps"
    if device == 'mps' and not is_mps_available() and is_cuda_available():
        logging.warning("mps is not available, fallback to cuda")
        return "cuda"

    # auto detect device if not specified
    if device not in ["cuda", "mps", "cpu", "xpu"]:
        return detect_device()
    return device


def embedding_device(device: str = None) -> Literal["cuda", "mps", "cpu", "xpu"]:
    device = device or LLM_DEVICE
    if device not in ["cuda", "mps", "cpu"]:
        logging.warning(f"device not in ['cuda', 'mps', 'cpu','xpu'], device = {device}")
        device = detect_device()
    elif device == 'cuda' and not is_cuda_available() and is_mps_available():
        logging.warning("cuda is not available, fallback to mps")
        return "mps"
    if device == 'mps' and not is_mps_available() and is_cuda_available():
        logging.warning("mps is not available, fallback to cuda")
        return "cuda"

    # auto detect device if not specified
    if device not in ["cuda", "mps", "cpu"]:
        return detect_device()
    return device


def run_in_thread_pool(
        func: Callable,
        params: List[Dict] = [],
) -> Generator:
    '''
    Run tasks in batches in a thread pool and return the results as a generator.
    Make sure that all operations in the task are thread-safe and that all task functions use keyword arguments.
    '''
    tasks = []
    with ThreadPoolExecutor() as pool:
        for kwargs in params:
            thread = pool.submit(func, **kwargs)
            tasks.append(thread)

        for obj in as_completed(tasks):  # TODO: Ctrl+c cannot stop
            yield obj.result()


def get_httpx_client(
        use_async: bool = False,
        proxies: Union[str, Dict] = None,
        timeout: float = HTTPX_DEFAULT_TIMEOUT,
        **kwargs,
) -> Union[httpx.Client, httpx.AsyncClient]:
    '''
    helper to get httpx client with default proxies that bypass local addesses.
    '''
    default_proxies = {
        # do not use proxy for locahost
        "all://127.0.0.1": None,
        "all://localhost": None,
    }
    # do not use proxy for user deployed fastchat servers
    for x in [
        fschat_controller_address(),
        fschat_model_worker_address(),
        fschat_openai_api_address(),
    ]:
        host = ":".join(x.split(":")[:2])
        default_proxies.update({host: None})

    # get proxies from system envionrent
    # proxy not str empty string, None, False, 0, [] or {}
    default_proxies.update({
        "http://": (os.environ.get("http_proxy")
                    if os.environ.get("http_proxy") and len(os.environ.get("http_proxy").strip())
                    else None),
        "https://": (os.environ.get("https_proxy")
                     if os.environ.get("https_proxy") and len(os.environ.get("https_proxy").strip())
                     else None),
        "all://": (os.environ.get("all_proxy")
                   if os.environ.get("all_proxy") and len(os.environ.get("all_proxy").strip())
                   else None),
    })
    for host in os.environ.get("no_proxy", "").split(","):
        if host := host.strip():
            # default_proxies.update({host: None}) # Origin code
            default_proxies.update({'all://' + host: None})  # PR 1838 fix, if not add 'all://', httpx will raise error

    # merge default proxies with user provided proxies
    if isinstance(proxies, str):
        proxies = {"all://": proxies}

    if isinstance(proxies, dict):
        default_proxies.update(proxies)

    # construct Client
    kwargs.update(timeout=timeout, proxies=default_proxies)

    if log_verbose:
        logger.info(f'{get_httpx_client.__class__.__name__}:kwargs: {kwargs}')

    if use_async:
        return httpx.AsyncClient(**kwargs)
    else:
        return httpx.Client(**kwargs)


def get_server_configs() -> Dict:
    '''
    Gets the original configuration item from configs for use in the front-end
    '''
    from configs.kb_config import (
        DEFAULT_KNOWLEDGE_BASE,
        DEFAULT_SEARCH_ENGINE,
        DEFAULT_VS_TYPE,
        CHUNK_SIZE,
        OVERLAP_SIZE,
        SCORE_THRESHOLD,
        VECTOR_SEARCH_TOP_K,
        SEARCH_ENGINE_TOP_K,
        ZH_TITLE_ENHANCE,
        text_splitter_dict,
        TEXT_SPLITTER_NAME,
    )
    from configs.model_config import (
        LLM_MODELS,
        HISTORY_LEN,
        TEMPERATURE,
    )
    from configs.prompt_config import PROMPT_TEMPLATES

    _custom = {
        "controller_address": fschat_controller_address(),
        "openai_api_address": fschat_openai_api_address(),
        "api_address": api_address(),
    }

    return {**{k: v for k, v in locals().items() if k[0] != "_"}, **_custom}


def list_online_embed_models() -> List[str]:
    from server import model_workers

    ret = []
    for k, v in list_config_llm_models()["online"].items():
        if provider := v.get("provider"):
            worker_class = getattr(model_workers, provider, None)
            if worker_class is not None and worker_class.can_embedding():
                ret.append(k)
    return ret


def load_local_embeddings(model: str = None, device: str = embedding_device()):
    '''
    Load embeddings from the cache to avoid competing loads when multithreading.
    '''
    from server.knowledge_base.kb_cache.base import embeddings_pool
    from configs import EMBEDDING_MODEL

    model = model or EMBEDDING_MODEL
    return embeddings_pool.load_embeddings(model=model, device=device)


def get_temp_dir(id: str = None) -> Tuple[str, str]:
    '''
    Create a temporary directory, return (path, folder name)
    '''
    from configs.basic_config import BASE_TEMP_DIR
    import tempfile

    if id is not None:  # If the specified temporary directory already exists, return it directly
        path = os.path.join(BASE_TEMP_DIR, id)
        if os.path.isdir(path):
            return path, id

    path = tempfile.mkdtemp(dir=BASE_TEMP_DIR)
    return path, os.path.basename(path)
