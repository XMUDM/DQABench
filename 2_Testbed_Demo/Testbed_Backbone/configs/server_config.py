import sys
from configs.model_config import LLM_DEVICE

# Default httpx request timeout (seconds). If the model loads or conversations are slow and timeout errors occur, you can increase this value appropriately.
HTTPX_DEFAULT_TIMEOUT = 300.0

# API Whether to enable cross-domain. The default value is False. If you need to enable cross-domain, set it to True
# is open cross domain
OPEN_CROSS_DOMAIN = False

# Each server is bound to a host by default. To change to "0.0.0.0", you need to change the host of all XX_servers below
DEFAULT_BIND_HOST = "0.0.0.0" if sys.platform != "win32" else "127.0.0.1"

# webui.py server
WEBUI_SERVER = {
    "host": DEFAULT_BIND_HOST,
    "port": 8501,
}

# api.py server
API_SERVER = {
    "host": DEFAULT_BIND_HOST,
    "port": 7861,
}

# fastchat openai_api server
FSCHAT_OPENAI_API = {
    "host": DEFAULT_BIND_HOST,
    "port": 20000,
}

# fastchat model_worker server
# These models must be configured correctly in model_config.MODEL_PATH or ONLINE_MODEL.
# When starting startup.py, you can specify the model by '--model-name xxxx yyyy', or LLM_MODELS if not specified
FSCHAT_MODEL_WORKERS = {
    # The default configuration shared by all models can be overwritten in the model-specific configuration.
    "default": {
        "host": DEFAULT_BIND_HOST,
        "port": 20311,
        "device": LLM_DEVICE,
        # False,'vllm', uses the inference acceleration framework
        # vllm support for some models is not yet mature and is temporarily turned off by default
        # fschat=0.2.33 code has a bug, if you need to use, modify the source code fastchat.server.vllm_worker,
        # Change stop=list(stop) of sampling_params = SamplingParams in line 103 to stop= [i for i in stop if i!=""]
        "infer_turbo": False,

        # model_worker Specifies the parameters to be configured for multi-card loading
        "gpus": "0,1,2,3", # Specifies the GPU to be used in the str format, such as "0,1". If the GPU is invalid, specify it in the CUDA_VISIBLE_DEVICES="0,1" format
        "num_gpus": 4, # Number of Gpus used
        "max_gpu_memory": "15GiB", # Maximum memory per GPU

        # The following parameters are used by model_worker and can be configured as required
        # "load_8bit": False, # Enable 8bit quantization
        # "cpu_offloading": None,
        # "gptq_ckpt": None,
        # "gptq_wbits": 16,
        # "gptq_groupsize": -1,
        # "gptq_act_order": False,
        # "awq_ckpt": None,
        # "awq_wbits": 16,
        # "awq_groupsize": -1,
        # "model_names": LLM_MODELS,
        # "conv_template": None,
        # "limit_worker_concurrency": 5,
        # "stream_interval": 2,
        # "no_register": False,
        # "embed_in_truncate": False,

        # The following are vllm_worker configuration parameters. Note that a gpu is required to use vllm, and it only passes the Linux test

        # tokenizer = model_path # If the tokenizer is inconsistent with model_path add it here
        # 'tokenizer_mode':'auto',
        # 'trust_remote_code':True,
        # 'download_dir':None,
        # 'load_format':'auto',
        # 'dtype':'auto',
        # 'seed':0,
        # 'worker_use_ray':False,
        # 'pipeline_parallel_size':1,
        # 'tensor_parallel_size':1,
        # 'block_size':16,
        # 'swap_space':4 , # GiB
        # 'gpu_memory_utilization':0.90,
        # 'max_num_batched_tokens':2560,
        # 'max_num_seqs':256,
        # 'disable_log_stats':False,
        # 'conv_template':None,
        # 'limit_worker_concurrency':5,
        # 'no_register':False,
        # 'num_gpus': 1
        # 'engine_use_ray': False,
        # 'disable_log_requests': False

    },
    # You can change the default configuration in the following example
    # "Qwen-1_8B-Chat": { # Use the IP and port in default
    #    "device": "cpu",
    # },
    # "chatglm3-6b": {  # Use the IP and port in default
    #     "port": 23001,
    #     "device": "cuda",
    # },

    # "baichuan-13b": { # Use the IP and port in default
    #     "port": 23002,
    #    "device": "cuda",
    # },
    
    "baichuan-13b-chat": { # 使用default中的IP和端口
        "port": 23003,
       "device": "cuda",
    },

    # The following configuration allows you to set the launched model in model_config without modification
    "zhipu-api": {
        "port": 21001,
    },
    # "minimax-api": {
    #     "port": 21002,
    # },
    # "xinghuo-api": {
    #     "port": 21003,
    # },
    # "qianfan-api": {
    #     "port": 21004,
    # },
    # "fangzhou-api": {
    #     "port": 21005,
    # },
    # "qwen-api": {
    #     "port": 21006,
    # },
    # "baichuan-api": {
    #     "port": 21007,
    # },
    # "azure-api": {
    #     "port": 21008,
    # },
    # "tiangong-api": {
    #     "port": 21009,
    # },
}

# fastchat multi model worker server
FSCHAT_MULTI_MODEL_WORKERS = {
    # TODO:
}

# fastchat controller server
FSCHAT_CONTROLLER = {
    "host": DEFAULT_BIND_HOST,
    "port": 20071,
    "dispatch_method": "shortest_queue",
}
