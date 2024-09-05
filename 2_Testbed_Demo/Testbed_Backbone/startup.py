import asyncio
import multiprocessing as mp
import os
import subprocess
import sys
from multiprocessing import Process
from datetime import datetime
from pprint import pprint

# numexpr Specifies the maximum number of threads, which is the number of CPU cores by default
try:
    import numexpr

    n_cores = numexpr.utils.detect_number_of_cores()
    os.environ["NUMEXPR_MAX_THREADS"] = str(n_cores)
except:
    pass

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs import (
    LOG_PATH,
    log_verbose,
    logger,
    LLM_MODELS,
    EMBEDDING_MODEL,
    TEXT_SPLITTER_NAME,
    FSCHAT_CONTROLLER,
    FSCHAT_OPENAI_API,
    FSCHAT_MODEL_WORKERS,
    API_SERVER,
    WEBUI_SERVER,
    HTTPX_DEFAULT_TIMEOUT,
)
from server.utils import (fschat_controller_address, fschat_model_worker_address,
                          fschat_openai_api_address, set_httpx_config, get_httpx_client,
                          get_model_worker_config, get_all_model_worker_configs,
                          MakeFastAPIOffline, FastAPI, llm_device, embedding_device)
from server.knowledge_base.migrate import create_tables
import argparse
from typing import Tuple, List, Dict
from configs import VERSION


def create_controller_app(
        dispatch_method: str,
        log_level: str = "INFO",
) -> FastAPI:
    import fastchat.constants
    fastchat.constants.LOGDIR = LOG_PATH
    from fastchat.serve.controller import app, Controller, logger
    logger.setLevel(log_level)

    controller = Controller(dispatch_method)
    sys.modules["fastchat.serve.controller"].controller = controller

    MakeFastAPIOffline(app)
    app.title = "FastChat Controller"
    app._controller = controller
    return app


def create_model_worker_app(log_level: str = "INFO", **kwargs) -> FastAPI:
    """
    kwargs contains the following fields:
    host:
    port:
    model_names:[`model_name`]
    controller_address:
    worker_address:

    For Langchain supported models:
        langchain_model:True
        fschat will not be used
    For online_api:
        online_api:True
        worker_class: `provider`
    For offline models:
        model_path: 'model_name_or_path',huggingface's repo-id or local path
        device:`LLM_DEVICE`
    """
    import fastchat.constants
    fastchat.constants.LOGDIR = LOG_PATH
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args([])

    for k, v in kwargs.items():
        setattr(args, k, v)
    if worker_class := kwargs.get("langchain_model"): #Langchian supports a model with no action
        from fastchat.serve.base_model_worker import app
        worker = ""
    # Online model API
    elif worker_class := kwargs.get("worker_class"):
        from fastchat.serve.base_model_worker import app

        worker = worker_class(model_names=args.model_names,
                              controller_addr=args.controller_address,
                              worker_addr=args.worker_address)
        # sys.modules["fastchat.serve.base_model_worker"].worker = worker
        sys.modules["fastchat.serve.base_model_worker"].logger.setLevel(log_level)
    # Local model
    else:
        from configs.model_config import VLLM_MODEL_DICT
        if kwargs["model_names"][0] in VLLM_MODEL_DICT and args.infer_turbo == "vllm":
            import fastchat.serve.vllm_worker
            from fastchat.serve.vllm_worker import VLLMWorker, app, worker_id
            from vllm import AsyncLLMEngine
            from vllm.engine.arg_utils import AsyncEngineArgs,EngineArgs

            args.tokenizer = args.model_path # Add here if tokenizer is inconsistent with model_path
            args.tokenizer_mode = 'auto'
            args.trust_remote_code= True
            args.download_dir= None
            args.load_format = 'auto'
            args.dtype = 'auto'
            args.seed = 0
            args.worker_use_ray = False
            args.pipeline_parallel_size = 1
            args.tensor_parallel_size = 1
            args.block_size = 16
            args.swap_space = 4  # GiB
            args.gpu_memory_utilization = 0.90
            args.max_num_batched_tokens = None # The maximum number of tokens in a batch depends on your graphics card and large model Settings, setting too large memory will not be enough
            args.max_num_seqs = 256
            args.disable_log_stats = False
            args.conv_template = None
            args.limit_worker_concurrency = 5
            args.no_register = False
            args.num_gpus = 1 # The segmentation of the vllm worker is tensor parallel, where the number of graphics cards is filled in
            args.engine_use_ray = False
            args.disable_log_requests = False

            # The parameter to be added after 0.2.1 vllm is not needed here
            args.max_model_len = None
            args.revision = None
            args.quantization = None
            args.max_log_len = None
            args.tokenizer_revision = None
            
            # 0.2.2 vllm requires new parameters
            args.max_paddings = 256
            
            if args.model_path:
                args.model = args.model_path
            if args.num_gpus > 1:
                args.tensor_parallel_size = args.num_gpus

            for k, v in kwargs.items():
                setattr(args, k, v)

            engine_args = AsyncEngineArgs.from_cli_args(args)
            engine = AsyncLLMEngine.from_engine_args(engine_args)

            worker = VLLMWorker(
                        controller_addr = args.controller_address,
                        worker_addr = args.worker_address,
                        worker_id = worker_id,
                        model_path = args.model_path,
                        model_names = args.model_names,
                        limit_worker_concurrency = args.limit_worker_concurrency,
                        no_register = args.no_register,
                        llm_engine =  engine,
                        conv_template = args.conv_template,
                        )
            sys.modules["fastchat.serve.vllm_worker"].engine = engine
            sys.modules["fastchat.serve.vllm_worker"].worker = worker
            sys.modules["fastchat.serve.vllm_worker"].logger.setLevel(log_level)

        else:
            from fastchat.serve.model_worker import app, GptqConfig, AWQConfig, ModelWorker, worker_id

            args.gpus = "0" # The number of the GPU, if there are multiple Gpus, can be set to "0,1,2,3"
            args.max_gpu_memory = "22GiB"
            args.num_gpus = 1  # model worker segmentation is model parallel, here fill in the number of graphics cards

            args.load_8bit = False
            args.cpu_offloading = None
            args.gptq_ckpt = None
            args.gptq_wbits = 16
            args.gptq_groupsize = -1
            args.gptq_act_order = False
            args.awq_ckpt = None
            args.awq_wbits = 16
            args.awq_groupsize = -1
            args.model_names = [""]
            args.conv_template = None
            args.limit_worker_concurrency = 5
            args.stream_interval = 2
            args.no_register = False
            args.embed_in_truncate = False
            for k, v in kwargs.items():
                setattr(args, k, v)
            if args.gpus:
                if args.num_gpus is None:
                    args.num_gpus = len(args.gpus.split(','))
                if len(args.gpus.split(",")) < args.num_gpus:
                    raise ValueError(
                        f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
                    )
                os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
            gptq_config = GptqConfig(
                ckpt=args.gptq_ckpt or args.model_path,
                wbits=args.gptq_wbits,
                groupsize=args.gptq_groupsize,
                act_order=args.gptq_act_order,
            )
            awq_config = AWQConfig(
                ckpt=args.awq_ckpt or args.model_path,
                wbits=args.awq_wbits,
                groupsize=args.awq_groupsize,
            )

            worker = ModelWorker(
                controller_addr=args.controller_address,
                worker_addr=args.worker_address,
                worker_id=worker_id,
                model_path=args.model_path,
                model_names=args.model_names,
                limit_worker_concurrency=args.limit_worker_concurrency,
                no_register=args.no_register,
                device=args.device,
                num_gpus=args.num_gpus,
                max_gpu_memory=args.max_gpu_memory,
                load_8bit=args.load_8bit,
                cpu_offloading=args.cpu_offloading,
                gptq_config=gptq_config,
                awq_config=awq_config,
                stream_interval=args.stream_interval,
                conv_template=args.conv_template,
                embed_in_truncate=args.embed_in_truncate,
            )
            sys.modules["fastchat.serve.model_worker"].args = args
            sys.modules["fastchat.serve.model_worker"].gptq_config = gptq_config
            # sys.modules["fastchat.serve.model_worker"].worker = worker
            sys.modules["fastchat.serve.model_worker"].logger.setLevel(log_level)

    MakeFastAPIOffline(app)
    app.title = f"FastChat LLM Server ({args.model_names[0]})"
    app._worker = worker
    return app


def create_openai_api_app(
        controller_address: str,
        api_keys: List = [],
        log_level: str = "INFO",
) -> FastAPI:
    import fastchat.constants
    fastchat.constants.LOGDIR = LOG_PATH
    from fastchat.serve.openai_api_server import app, CORSMiddleware, app_settings
    from fastchat.utils import build_logger
    logger = build_logger("openai_api", "openai_api.log")
    logger.setLevel(log_level)

    app.add_middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    sys.modules["fastchat.serve.openai_api_server"].logger = logger
    app_settings.controller_address = controller_address
    app_settings.api_keys = api_keys

    MakeFastAPIOffline(app)
    app.title = "FastChat OpeanAI API Server"
    return app


def _set_app_event(app: FastAPI, started_event: mp.Event = None):
    @app.on_event("startup")
    async def on_startup():
        if started_event is not None:
            started_event.set()


def run_controller(log_level: str = "INFO", started_event: mp.Event = None):
    import uvicorn
    import httpx
    from fastapi import Body
    import time
    import sys
    from server.utils import set_httpx_config
    set_httpx_config()

    app = create_controller_app(
        dispatch_method=FSCHAT_CONTROLLER.get("dispatch_method"),
        log_level=log_level,
    )
    _set_app_event(app, started_event)

    # add interface to release and load model worker
    @app.post("/release_worker")
    def release_worker(
            model_name: str = Body(..., description="The name of the model to release", samples=["chatglm-6b"]),
            # worker_address: str = Body(None, description="To release the address of the model, choose either the name or the address", samples=[FSCHAT_CONTROLLER_address()]),
            new_model_name: str = Body(None, description="Load the model after release"),
            keep_origin: bool = Body(False, description="Load a new model without releasing the original model")
    ) -> Dict:
        available_models = app._controller.list_models()
        if new_model_name in available_models:
            msg = f"The LLM model to be switched {new_model_name} already exists"
            logger.info(msg)
            return {"code": 500, "msg": msg}

        if new_model_name:
            logger.info(f"Start switching LLM models: from {model_name} to {new_model_name}")
        else:
            logger.info(f"LLM model to be stopped: {model_name}")

        if model_name not in available_models:
            msg = f"the model {model_name} is not available"
            logger.error(msg)
            return {"code": 500, "msg": msg}

        worker_address = app._controller.get_worker_address(model_name)
        if not worker_address:
            msg = f"can not find model_worker address for {model_name}"
            logger.error(msg)
            return {"code": 500, "msg": msg}

        with get_httpx_client() as client:
            r = client.post(worker_address + "/release",
                        json={"new_model_name": new_model_name, "keep_origin": keep_origin})
            if r.status_code != 200:
                msg = f"failed to release model: {model_name}"
                logger.error(msg)
                return {"code": 500, "msg": msg}

        if new_model_name:
            timer = HTTPX_DEFAULT_TIMEOUT  # wait for new model_worker register
            while timer > 0:
                models = app._controller.list_models()
                if new_model_name in models:
                    break
                time.sleep(1)
                timer -= 1
            if timer > 0:
                msg = f"sucess change model from {model_name} to {new_model_name}"
                logger.info(msg)
                return {"code": 200, "msg": msg}
            else:
                msg = f"failed change model from {model_name} to {new_model_name}"
                logger.error(msg)
                return {"code": 500, "msg": msg}
        else:
            msg = f"sucess to release model: {model_name}"
            logger.info(msg)
            return {"code": 200, "msg": msg}

    host = FSCHAT_CONTROLLER["host"]
    port = FSCHAT_CONTROLLER["port"]

    if log_level == "ERROR":
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())


def run_model_worker(
        model_name: str = LLM_MODELS[0],
        controller_address: str = "",
        log_level: str = "INFO",
        q: mp.Queue = None,
        started_event: mp.Event = None,
):
    import uvicorn
    from fastapi import Body
    import sys
    from server.utils import set_httpx_config
    set_httpx_config()

    kwargs = get_model_worker_config(model_name)
    host = kwargs.pop("host")
    port = kwargs.pop("port")
    kwargs["model_names"] = [model_name]
    kwargs["controller_address"] = controller_address or fschat_controller_address()
    kwargs["worker_address"] = fschat_model_worker_address(model_name)
    model_path = kwargs.get("model_path", "")
    kwargs["model_path"] = model_path

    app = create_model_worker_app(log_level=log_level, **kwargs)
    _set_app_event(app, started_event)
    if log_level == "ERROR":
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    # add interface to release and load model
    @app.post("/release")
    def release_model(
        new_model_name: str = Body(None, description="Load the model after release"),
        keep_origin: bool = Body(False, description="Load a new model without releasing the original model")
    ) -> Dict:
        if keep_origin:
            if new_model_name:
                q.put([model_name, "start", new_model_name])
        else:
            if new_model_name:
                q.put([model_name, "replace", new_model_name])
            else:
                q.put([model_name, "stop", None])
        return {"code": 200, "msg": "done"}

    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())


def run_openai_api(log_level: str = "INFO", started_event: mp.Event = None):
    import uvicorn
    import sys
    from server.utils import set_httpx_config
    set_httpx_config()

    controller_addr = fschat_controller_address()
    app = create_openai_api_app(controller_addr, log_level=log_level)  # TODO: not support keys yet.
    _set_app_event(app, started_event)

    host = FSCHAT_OPENAI_API["host"]
    port = FSCHAT_OPENAI_API["port"]
    if log_level == "ERROR":
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    uvicorn.run(app, host=host, port=port)


def run_api_server(started_event: mp.Event = None, run_mode: str = None):
    from server.api import create_app
    import uvicorn
    from server.utils import set_httpx_config
    set_httpx_config()

    app = create_app(run_mode=run_mode)
    _set_app_event(app, started_event)

    host = API_SERVER["host"]
    port = API_SERVER["port"]

    uvicorn.run(app, host=host, port=port)


def run_webui(started_event: mp.Event = None, run_mode: str = None):
    from server.utils import set_httpx_config
    set_httpx_config()

    host = WEBUI_SERVER["host"]
    port = WEBUI_SERVER["port"]

    cmd = ["streamlit", "run", "webui.py",
            "--server.address", host,
            "--server.port", str(port),
            "--theme.base", "light",
            "--theme.primaryColor", "#165dff",
            "--theme.secondaryBackgroundColor", "#f5f5f5",
            "--theme.textColor", "#000000",
        ]
    if run_mode == "lite":
        cmd += [
            "--",
            "lite",
        ]
    p = subprocess.Popen(cmd)
    started_event.set()
    p.wait()


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--all-webui",
        action="store_true",
        help="run fastchat's controller/openai_api/model_worker servers, run api.py and webui.py",
        dest="all_webui",
    )
    parser.add_argument(
        "--all-api",
        action="store_true",
        help="run fastchat's controller/openai_api/model_worker servers, run api.py",
        dest="all_api",
    )
    parser.add_argument(
        "--llm-api",
        action="store_true",
        help="run fastchat's controller/openai_api/model_worker servers",
        dest="llm_api",
    )
    parser.add_argument(
        "-o",
        "--openai-api",
        action="store_true",
        help="run fastchat's controller/openai_api servers",
        dest="openai_api",
    )
    parser.add_argument(
        "-m",
        "--model-worker",
        action="store_true",
        help="run fastchat's model_worker server with specified model name. "
             "specify --model-name if not using default LLM_MODELS",
        dest="model_worker",
    )
    parser.add_argument(
        "-n",
        "--model-name",
        type=str,
        nargs="+",
        default=LLM_MODELS,
        help="specify model name for model worker. "
             "add addition names with space seperated to start multiple model workers.",
        dest="model_name",
    )
    parser.add_argument(
        "-c",
        "--controller",
        type=str,
        help="specify controller address the worker is registered to. default is FSCHAT_CONTROLLER",
        dest="controller_address",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="run api.py server",
        dest="api",
    )
    parser.add_argument(
        "-p",
        "--api-worker",
        action="store_true",
        help="run online model api such as zhipuai",
        dest="api_worker",
    )
    parser.add_argument(
        "-w",
        "--webui",
        action="store_true",
        help="run webui.py server",
        dest="webui",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Reduce the log information of the fastchat service",
        dest="quiet",
    )
    parser.add_argument(
        "-i",
        "--lite",
        action="store_true",
        help="Lite mode: Supports only LLM chats and search engine chats of online apis",
        dest="lite",
    )
    args = parser.parse_args()
    return args, parser


def dump_server_info(after_start=False, args=None):
    import platform
    import langchain
    import fastchat
    from server.utils import api_address, webui_address

    print("\n")
    print("=" * 30 + "Langchain-Chatchat Configuration" + "=" * 30)
    print(f"Operating system:{platform.platform()}.")
    print(f"python version:{sys.version}")
    print(f"Project version:{VERSION}")
    print(f"langchain version:{langchain.__version__}. fastchat version:{fastchat.__version__}")
    print("\n")

    models = LLM_MODELS
    if args and args.model_name:
        models = args.model_name

    print(f"Current use of word segmentation:{TEXT_SPLITTER_NAME}")
    print(f"Currently started LLM model:{models} @ {llm_device()}")

    for model in models:
        pprint(get_model_worker_config(model))
    print(f"Current Embbedings model:{EMBEDDING_MODEL} @ {embedding_device()}")

    if after_start:
        print("\n")
        print(f"Server running information:")
        if args.openai_api:
            print(f"    OpenAI API Server: {fschat_openai_api_address()}")
        if args.api:
            print(f"    Chatchat  API  Server: {api_address()}")
        if args.webui:
            print(f"    Chatchat WEBUI Server: {webui_address()}")
    print("=" * 30 + "Langchain-Chatchat Configuration" + "=" * 30)
    print("\n")


async def start_main_server():
    import time
    import signal

    def handler(signalname):
        """
        Python 3.9 has `signal.strsignal(signalnum)` so this closure would not be needed.
        Also, 3.8 includes `signal.valid_signals()` that can be used to create a mapping for the same purpose.
        """
        def f(signal_received, frame):
            raise KeyboardInterrupt(f"{signalname} received")
        return f

    # This will be inherited by the child process if it is forked (not spawned)
    signal.signal(signal.SIGINT, handler("SIGINT"))
    signal.signal(signal.SIGTERM, handler("SIGTERM"))

    mp.set_start_method("spawn")
    manager = mp.Manager()
    run_mode = None

    queue = manager.Queue()
    args, parser = parse_args()

    if args.all_webui:
        args.openai_api = True
        args.model_worker = True
        args.api = True
        args.api_worker = True
        args.webui = True

    elif args.all_api:
        args.openai_api = True
        args.model_worker = True
        args.api = True
        args.api_worker = True
        args.webui = False

    elif args.llm_api:
        args.openai_api = True
        args.model_worker = True
        args.api_worker = True
        args.api = False
        args.webui = False

    if args.lite:
        args.model_worker = False
        run_mode = "lite"

    dump_server_info(args=args)

    if len(sys.argv) > 1:
        logger.info(f"Starting the service:")
        logger.info(f"To view the llm_api log, go to {LOG_PATH}")

    processes = {"online_api": {}, "model_worker": {}}

    def process_count():
        return len(processes) + len(processes["online_api"]) + len(processes["model_worker"]) - 2

    if args.quiet or not log_verbose:
        log_level = "ERROR"
    else:
        log_level = "INFO"

    controller_started = manager.Event()
    if args.openai_api:
        process = Process(
            target=run_controller,
            name=f"controller",
            kwargs=dict(log_level=log_level, started_event=controller_started),
            daemon=True,
        )
        processes["controller"] = process

        process = Process(
            target=run_openai_api,
            name=f"openai_api",
            daemon=True,
        )
        processes["openai_api"] = process

    model_worker_started = []
    if args.model_worker:
        for model_name in args.model_name:
            config = get_model_worker_config(model_name)
            if not config.get("online_api"):
                e = manager.Event()
                model_worker_started.append(e)
                process = Process(
                    target=run_model_worker,
                    name=f"model_worker - {model_name}",
                    kwargs=dict(model_name=model_name,
                                controller_address=args.controller_address,
                                log_level=log_level,
                                q=queue,
                                started_event=e),
                    daemon=True,
                )
                processes["model_worker"][model_name] = process

    if args.api_worker:
        for model_name in args.model_name:
            config = get_model_worker_config(model_name)
            if (config.get("online_api")
                and config.get("worker_class")
                and model_name in FSCHAT_MODEL_WORKERS):
                e = manager.Event()
                model_worker_started.append(e)
                process = Process(
                    target=run_model_worker,
                    name=f"api_worker - {model_name}",
                    kwargs=dict(model_name=model_name,
                                controller_address=args.controller_address,
                                log_level=log_level,
                                q=queue,
                                started_event=e),
                    daemon=True,
                )
                processes["online_api"][model_name] = process

    api_started = manager.Event()
    if args.api:
        process = Process(
            target=run_api_server,
            name=f"API Server",
            kwargs=dict(started_event=api_started, run_mode=run_mode),
            daemon=True,
        )
        processes["api"] = process

    webui_started = manager.Event()
    if args.webui:
        process = Process(
            target=run_webui,
            name=f"WEBUI Server",
            kwargs=dict(started_event=webui_started, run_mode=run_mode),
            daemon=True,
        )
        processes["webui"] = process

    if process_count() == 0:
        parser.print_help()
    else:
        try:
            # Ensure that tasks can exit normally after receiving SIGINT
            if p:= processes.get("controller"):
                p.start()
                p.name = f"{p.name} ({p.pid})"
                controller_started.wait() # Wait for controller to complete startup

            if p:= processes.get("openai_api"):
                p.start()
                p.name = f"{p.name} ({p.pid})"

            for n, p in processes.get("model_worker", {}).items():
                p.start()
                p.name = f"{p.name} ({p.pid})"

            for n, p in processes.get("online_api", []).items():
                p.start()
                p.name = f"{p.name} ({p.pid})"

            # Wait for all model_worker to finish starting
            for e in model_worker_started:
                e.wait()

            if p:= processes.get("api"):
                p.start()
                p.name = f"{p.name} ({p.pid})"
                api_started.wait() # Wait for api.py to complete

            if p:= processes.get("webui"):
                p.start()
                p.name = f"{p.name} ({p.pid})"
                webui_started.wait() # Wait for webu.py to start

            dump_server_info(after_start=True, args=args)

            while True:
                cmd = queue.get() # Received a message to switch models
                e = manager.Event()
                if isinstance(cmd, list):
                    model_name, cmd, new_model_name = cmd
                    if cmd == "start": # Run the new model
                        logger.info(f"Ready to start the new model process:{new_model_name}")
                        process = Process(
                            target=run_model_worker,
                            name=f"model_worker - {new_model_name}",
                            kwargs=dict(model_name=new_model_name,
                                        controller_address=args.controller_address,
                                        log_level=log_level,
                                        q=queue,
                                        started_event=e),
                            daemon=True,
                        )
                        process.start()
                        process.name = f"{process.name} ({process.pid})"
                        processes["model_worker"][new_model_name] = process
                        e.wait()
                        logger.info(f"Successfully start the new model process:{new_model_name}")
                    elif cmd == "stop":
                        if process := processes["model_worker"].get(model_name):
                            time.sleep(1)
                            process.terminate()
                            process.join()
                            logger.info(f"To stop the model process:{model_name}")
                        else:
                            logger.error(f"Model process not found:{model_name}")
                    elif cmd == "replace":
                        if process := processes["model_worker"].pop(model_name, None):
                            logger.info(f"To stop the model process:{model_name}")
                            start_time = datetime.now()
                            time.sleep(1)
                            process.terminate()
                            process.join()
                            process = Process(
                                target=run_model_worker,
                                name=f"model_worker - {new_model_name}",
                                kwargs=dict(model_name=new_model_name,
                                            controller_address=args.controller_address,
                                            log_level=log_level,
                                            q=queue,
                                            started_event=e),
                                daemon=True,
                            )
                            process.start()
                            process.name = f"{process.name} ({process.pid})"
                            processes["model_worker"][new_model_name] = process
                            e.wait()
                            timing = datetime.now() - start_time
                            logger.info(f"New model process successfully started: {new_model_name}. When used: {timing}.")
                        else:
                            logger.error(f"Model process not found:{model_name}")

        except Exception as e:
            logger.error(e)
            logger.warning("Caught KeyboardInterrupt! Setting stop event...")
        finally:
            # Send SIGINT if process doesn't exit quickly enough, and kill it as last resort
            # .is_alive() also implicitly joins the process (good practice in linux)
            # while alive_procs := [p for p in processes.values() if p.is_alive()]:

            for p in processes.values():
                logger.warning("Sending SIGKILL to %s", p)
                # Queues and other inter-process communication primitives can break when
                # process is killed, but we don't care here

                if isinstance(p, dict):
                    for process in p.values():
                        process.kill()
                else:
                    p.kill()

            for p in processes.values():
                logger.info("Process status: %s", p)

if __name__ == "__main__":
    # Ensure that database tables are created
    create_tables()

    if sys.version_info < (3, 10):
        loop = asyncio.get_event_loop()
    else:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()

        asyncio.set_event_loop(loop)
    # Call the coroutine code synchronously
    loop.run_until_complete(start_main_server())


# Example of interface call after service start:
# import openai
# openai.api_key = "EMPTY" # Not support yet
# openai.api_base = "http://localhost:8888/v1"

# model = "chatglm2-6b"

# # create a chat completion
# completion = openai.ChatCompletion.create(
#   model=model,
#   messages=[{"role": "user", "content": "Hello! What is your name?"}]
# )
# # print the completion
# print(completion.choices[0].message.content)
