import os

# You can specify an absolute path to store all Embedding and LLM models.
# Each model can be a separate directory or a secondary subdirectory within a directory.
# If the model directory name is the same as the key or value in MODEL_PATH, the program automatically detects loading without modifying the path in MODEL_PATH.
MODEL_ROOT_PATH = ""

# The chosen Embedding name
EMBEDDING_MODEL = "bge-large-zh"

# The Embedding model runs the device. Set to" auto" to automatically detect, or manually set to" cuda","mps","cpu" one of them.
EMBEDDING_DEVICE = "auto"

# Selected reranker model
RERANKER_MODEL = "bge-reranker-large"
# Whether to enable the reranker model
USE_RERANKER = False
RERANKER_MAX_LENGTH = 1024

# It is configured for EMBEDDING_MODEL to add user-defined keywords
EMBEDDING_KEYWORD_FILE = "keywords.txt"
EMBEDDING_MODEL_OUTPUT_PATH = "output"

# The name of the LLM to run, which can include both local and online models. The local models in the list are all loaded when the project is started.
# The first model in the list will serve as the default model for the API and WEBUI.
# Here, we use the two mainstream offline models, of which chatglm3-6b is the default loading model.
# If you're short on video memory, you can use Qwen-1_8B-Chat, which only needs 3.8GB of video memory.

# chatglm3-6b outputs the role tag <|user|> and asks and answers questions in the project wiki-> Frequently Asked Questions ->Q20.

LLM_MODELS = ["zhipu-api"]  # "Qwen-1_8B-Chat",

# AgentLM model name (optionally, if specified, it locks the model that enters the Chain behind the Agent, otherwise it is LLM_MODELS[0])
Agent_MODEL = None

# LLM Runs the device. Set to" auto" to automatically detect, or manually set to" cuda","mps","cpu" one of them.
LLM_DEVICE = "auto"

# Number of historical session rounds
HISTORY_LEN = 3

# The maximum length supported by the large model, if not filled, the default maximum length of the model is used, if filled, the maximum length set by the user
MAX_TOKENS = None

# LLM Common session parameters
TEMPERATURE = 0.3
# TOP_P = 0.95 # ChatOpenAI does not support this parameter yet

ONLINE_LLM_MODEL = {
    # Online model. Set a different port for each online API in server_config

    "openai-api": {
        "model_name": "gpt-3.5-turbo",
        "api_base_url": "https://api.openai.com/v1",
        "api_key": "",
        "openai_proxy": "",
    },

    # To register and obtain an api key, go to http://open.bigmodel.cn
    "zhipu-api": {
        "api_key": "",
        "version": "chatglm_turbo",  # Optional include "chatglm_turbo"
        "provider": "ChatGLMWorker",
    },

    # specific registration and API key access please go to https://api.minimax.chat/
    "minimax-api": {
        "group_id": "",
        "api_key": "",
        "is_pro": False,
        "provider": "MiniMaxWorker",
    },


    # specific registration and API key access please go to https://xinghuo.xfyun.cn/
    "xinghuo-api": {
        "APPID": "",
        "APISecret": "",
        "api_key": "",
        "version": "v1.5",  # The version of IFlystar Fire you are using, optionally including "v3.0", "v1.5", "v2.0"
        "provider": "XingHuoWorker",
    },

    # Baidu Qianfan API, please refer to the https://cloud.baidu.com/doc/WENXINWORKSHOP/s/4lilb2lpf application way
    "qianfan-api": {
        "version": "ERNIE-Bot",  # Case awareness. Currently supports "Erne-bot" or "Erne-bot-turbo", see the official documentation for more information.
        "version_url": "",  # You can also do not fill in the version, directly fill in the application model in Qianfan API address
        "api_key": "",
        "secret_key": "",
        "provider": "QianFanWorker",
    },

    # https://www.volcengine.com/docs/82379 # volcanic ark API, document reference
    "fangzhou-api": {
        "version": "chatglm-6b-model",  # Currently supports "chatglm-6b-model", see the Ark section of the document Model Support list for more information.
        "version_url": "",  # Do not fill in the version, directly fill in the ark application model release API address
        "api_key": "",
        "secret_key": "",
        "provider": "FangZhouWorker",
    },

    # ali cloud righteousness qian asked API, document reference https://help.aliyun.com/zh/dashscope/developer-reference/api-details
    "qwen-api": {
        "version": "qwen-turbo",  # Options include "qwen-turbo", "qwen-plus"
        "api_key": "",  # Please create in Ali Cloud console model service Spirit product API-KEY management page
        "provider": "QwenWorker",
        "embed_model": "text-embedding-v1" # embedding model name
    },

    # Baichuan API, please refer to https://www.baichuan-ai.com/home#api-enter to apply
    "baichuan-api": {
        "version": "Baichuan2-53B",  # Currently supports "Baichuan2-53B", see official documentation.
        "api_key": "",
        "secret_key": "",
        "provider": "BaiChuanWorker",
    },

    # Azure API
    "azure-api": {
        "deployment_name": "",  # Deployment container name
        "resource_name": "",  # {resource_name # https://}. Openai.azure.com/openai/ fill resource_name part, don't fill in the rest
        "api_version": "",  # API version, not model version
        "api_key": "",
        "provider": "AzureWorker",
    },

    # Kunlun Wanwei Tiangong API https://model-platform.tiangong.cn/
    "tiangong-api": {
        "version": "SkyChat-MegaVerse",
        "api_key": "",
        "secret_key": "",
        "provider": "TianGongWorker",
    },

}

# Modify the property value in the following dictionary to specify the local embedding model storage location. Three setting methods are supported:
# 1. Modify the corresponding value to the model absolute path
# 2. Do not change the value here (take text2vec as an example) :
#       2.1 If any of the following subdirectories exist under {MODEL_ROOT_PATH} :
#           - text2vec
#           - GanymedeNil/text2vec-large-chinese
#           - text2vec-large-chinese
#       2.2 If the above local path does not exist, the huggingface model is used
MODEL_PATH = {
    "embed_model": {
        "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
        "ernie-base": "nghuyong/ernie-3.0-base-zh",
        "text2vec-base": "shibing624/text2vec-base-chinese",
        "text2vec": "GanymedeNil/text2vec-large-chinese",
        "text2vec-paraphrase": "shibing624/text2vec-base-chinese-paraphrase",
        "text2vec-sentence": "shibing624/text2vec-base-chinese-sentence",
        "text2vec-multilingual": "shibing624/text2vec-base-multilingual",
        "text2vec-bge-large-chinese": "shibing624/text2vec-bge-large-chinese",
        "m3e-small": "moka-ai/m3e-small",
        "m3e-base": "moka-ai/m3e-base",
        "m3e-large": "moka-ai/m3e-large",
        "bge-small-zh": "BAAI/bge-small-zh",
        "bge-base-zh": "BAAI/bge-base-zh",
        "bge-large-zh": "BAAI/bge-large-zh",
        "bge-large-zh-noinstruct": "BAAI/bge-large-zh-noinstruct",
        "bge-base-zh-v1.5": "BAAI/bge-base-zh-v1.5",
        "bge-large-zh-v1.5": "BAAI/bge-large-zh-v1.5",
        "piccolo-base-zh": "sensenova/piccolo-base-zh",
        "piccolo-large-zh": "sensenova/piccolo-large-zh",
        "nlp_gte_sentence-embedding_chinese-large": "damo/nlp_gte_sentence-embedding_chinese-large",
        "text-embedding-ada-002": "your OPENAI_API_KEY",
    },

    "llm_model": {
        # Some of the models below have not been fully tested and are only presumed to be supported based on the list of models for fastchat and vllm models
        "chatglm2-6b": "THUDM/chatglm2-6b",
        "chatglm2-6b-32k": "THUDM/chatglm2-6b-32k",

        "chatglm3-6b": "THUDM/chatglm3-6b",
        "chatglm3-6b-32k": "THUDM/chatglm3-6b-32k",
        "chatglm3-6b-base": "THUDM/chatglm3-6b-base",

        "Qwen-1_8B": "Qwen/Qwen-1_8B",
        "Qwen-1_8B-Chat": "Qwen/Qwen-1_8B-Chat",
        "Qwen-1_8B-Chat-Int8": "Qwen/Qwen-1_8B-Chat-Int8",
        "Qwen-1_8B-Chat-Int4": "Qwen/Qwen-1_8B-Chat-Int4",

        "Qwen-7B": "Qwen/Qwen-7B",
        "Qwen-7B-Chat": "Qwen/Qwen-7B-Chat",

        "Qwen-14B": "Qwen/Qwen-14B",
        "Qwen-14B-Chat": "Qwen/Qwen-14B-Chat",

        "Qwen-14B-Chat-Int8": "Qwen/Qwen-14B-Chat-Int8",
        # In the new transformers version, you need to manually modify the model's config.json file in the quantization_config dictionary
        # Add the 'disable_exllama:true' field to enable qwen's quantitative model
        "Qwen-14B-Chat-Int4": "Qwen/Qwen-14B-Chat-Int4",

        "Qwen-72B": "Qwen/Qwen-72B",
        "Qwen-72B-Chat": "Qwen/Qwen-72B-Chat",
        "Qwen-72B-Chat-Int8": "Qwen/Qwen-72B-Chat-Int8",
        "Qwen-72B-Chat-Int4": "Qwen/Qwen-72B-Chat-Int4",

        "baichuan2-13b": "baichuan-inc/Baichuan2-13B-Chat",
        "baichuan2-7b": "baichuan-inc/Baichuan2-7B-Chat",

        "baichuan-7b": "baichuan-inc/Baichuan-7B",
        "baichuan-13b": "baichuan-inc/Baichuan2-13B-Base",
        "baichuan-13b-chat": "baichuan-inc/Baichuan2-13B-Chat",

        "aquila-7b": "BAAI/Aquila-7B",
        "aquilachat-7b": "BAAI/AquilaChat-7B",

        "internlm-7b": "internlm/internlm-7b",
        "internlm-chat-7b": "internlm/internlm-chat-7b",

        "falcon-7b": "tiiuae/falcon-7b",
        "falcon-40b": "tiiuae/falcon-40b",
        "falcon-rw-7b": "tiiuae/falcon-rw-7b",

        "gpt2": "gpt2",
        "gpt2-xl": "gpt2-xl",

        "gpt-j-6b": "EleutherAI/gpt-j-6b",
        "gpt4all-j": "nomic-ai/gpt4all-j",
        "gpt-neox-20b": "EleutherAI/gpt-neox-20b",
        "pythia-12b": "EleutherAI/pythia-12b",
        "oasst-sft-4-pythia-12b-epoch-3.5": "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
        "dolly-v2-12b": "databricks/dolly-v2-12b",
        "stablelm-tuned-alpha-7b": "stabilityai/stablelm-tuned-alpha-7b",

        "Llama-2-13b-hf": "meta-llama/Llama-2-13b-hf",
        "Llama-2-70b-hf": "meta-llama/Llama-2-70b-hf",
        "open_llama_13b": "openlm-research/open_llama_13b",
        "vicuna-13b-v1.3": "lmsys/vicuna-13b-v1.3",
        "koala": "young-geng/koala",

        "mpt-7b": "mosaicml/mpt-7b",
        "mpt-7b-storywriter": "mosaicml/mpt-7b-storywriter",
        "mpt-30b": "mosaicml/mpt-30b",
        "opt-66b": "facebook/opt-66b",
        "opt-iml-max-30b": "facebook/opt-iml-max-30b",

        "agentlm-7b": "THUDM/agentlm-7b",
        "agentlm-13b": "THUDM/agentlm-13b",
        "agentlm-70b": "THUDM/agentlm-70b",

        "Yi-34B-Chat": "01-ai/Yi-34B-Chat",
    },
    "reranker":{
        "bge-reranker-large":"BAAI/bge-reranker-large",
        "bge-reranker-base":"BAAI/bge-reranker-base",
    }
}


# In general, you do not need to change the following

# nltk model storage path
NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")

VLLM_MODEL_DICT = {
    "aquila-7b": "BAAI/Aquila-7B",
    "aquilachat-7b": "BAAI/AquilaChat-7B",

    "baichuan-7b": "baichuan-inc/Baichuan-7B",
    "baichuan-13b": "baichuan-inc/Baichuan-13B",
    "baichuan-13b-chat": "baichuan-inc/Baichuan-13B-Chat",

    "chatglm2-6b": "THUDM/chatglm2-6b",
    "chatglm2-6b-32k": "THUDM/chatglm2-6b-32k",
    "chatglm3-6b": "THUDM/chatglm3-6b",
    "chatglm3-6b-32k": "THUDM/chatglm3-6b-32k",

    "BlueLM-7B-Chat": "vivo-ai/BlueLM-7B-Chat",
    "BlueLM-7B-Chat-32k": "vivo-ai/BlueLM-7B-Chat-32k",

    # Note: The bloom tokenizer is separate from the model, so while vllm is supported, it is not compatible with the fschat framework
    # "bloom": "bigscience/bloom",
    # "bloomz": "bigscience/bloomz",
    # "bloomz-560m": "bigscience/bloomz-560m",
    # "bloomz-7b1": "bigscience/bloomz-7b1",
    # "bloomz-1b7": "bigscience/bloomz-1b7",

    "internlm-7b": "internlm/internlm-7b",
    "internlm-chat-7b": "internlm/internlm-chat-7b",
    "falcon-7b": "tiiuae/falcon-7b",
    "falcon-40b": "tiiuae/falcon-40b",
    "falcon-rw-7b": "tiiuae/falcon-rw-7b",
    "gpt2": "gpt2",
    "gpt2-xl": "gpt2-xl",
    "gpt-j-6b": "EleutherAI/gpt-j-6b",
    "gpt4all-j": "nomic-ai/gpt4all-j",
    "gpt-neox-20b": "EleutherAI/gpt-neox-20b",
    "pythia-12b": "EleutherAI/pythia-12b",
    "oasst-sft-4-pythia-12b-epoch-3.5": "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
    "dolly-v2-12b": "databricks/dolly-v2-12b",
    "stablelm-tuned-alpha-7b": "stabilityai/stablelm-tuned-alpha-7b",
    "Llama-2-13b-hf": "meta-llama/Llama-2-13b-hf",
    "Llama-2-70b-hf": "meta-llama/Llama-2-70b-hf",
    "open_llama_13b": "openlm-research/open_llama_13b",
    "vicuna-13b-v1.3": "lmsys/vicuna-13b-v1.3",
    "koala": "young-geng/koala",
    "mpt-7b": "mosaicml/mpt-7b",
    "mpt-7b-storywriter": "mosaicml/mpt-7b-storywriter",
    "mpt-30b": "mosaicml/mpt-30b",
    "opt-66b": "facebook/opt-66b",
    "opt-iml-max-30b": "facebook/opt-iml-max-30b",

    "Qwen-1_8B": "Qwen/Qwen-1_8B",
    "Qwen-1_8B-Chat": "Qwen/Qwen-1_8B-Chat",
    "Qwen-1_8B-Chat-Int8": "Qwen/Qwen-1_8B-Chat-Int8",
    "Qwen-1_8B-Chat-Int4": "Qwen/Qwen-1_8B-Chat-Int4",

    "Qwen-7B": "Qwen/Qwen-7B",
    "Qwen-7B-Chat": "Qwen/Qwen-7B-Chat",

    "Qwen-14B": "Qwen/Qwen-14B",
    "Qwen-14B-Chat": "Qwen/Qwen-14B-Chat",
    "Qwen-14B-Chat-Int8": "Qwen/Qwen-14B-Chat-Int8",
    "Qwen-14B-Chat-Int4": "Qwen/Qwen-14B-Chat-Int4",

    "Qwen-72B": "Qwen/Qwen-72B",
    "Qwen-72B-Chat": "Qwen/Qwen-72B-Chat",
    "Qwen-72B-Chat-Int8": "Qwen/Qwen-72B-Chat-Int8",
    "Qwen-72B-Chat-Int4": "Qwen/Qwen-72B-Chat-Int4",

    "agentlm-7b": "THUDM/agentlm-7b",
    "agentlm-13b": "THUDM/agentlm-13b",
    "agentlm-70b": "THUDM/agentlm-70b",

}

# You think that support Agent capabilities of the model, you can add here, after the addition will not appear visual interface warning
# After our testing, there are only a few models that natively support Agent
SUPPORT_AGENT_MODEL = [
    "azure-api",
    "openai-api",
    "qwen-api",
    "Qwen",
    "chatglm3",
    "xinghuo-api",
]
