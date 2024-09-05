import os

# The default knowledge base
DEFAULT_KNOWLEDGE_BASE = "OpenGauss"

# Default vector library/full-text search engine type. Optional: faiss, milvus(offline) & zilliz(online), pgvector, full-text search engine es
DEFAULT_VS_TYPE = "faiss"

# Cache vector library number (for FAISS)
CACHED_VS_NUM = 1

# Cache the number of temporary vector libraries (for FAISS) for file conversations
CACHED_MEMO_VS_NUM = 10

# In the knowledge base single text length (not applicable MarkdownHeaderTextSplitter)
CHUNK_SIZE = 250

# The knowledge base in the adjacent text overlap length (not applicable MarkdownHeaderTextSplitter)
OVERLAP_SIZE = 50

# Number of knowledge base matching vectors
VECTOR_SEARCH_TOP_K = 3

# The distance threshold of knowledge base matching ranges from 0 to 1. The smaller the SCORE, the smaller the distance, and thus the higher the correlation.
# Taking 1 is equivalent to not screening, and most of the measured bge-large distance scores are between 0.01 and 0.7.
# Similar texts have a maximum score of around 0.55, so it is recommended to set the score to 0.6 for bge
SCORE_THRESHOLD = 0.6

# Default search engine. Optional: bing, duckduckgo, metaphor
DEFAULT_SEARCH_ENGINE = "duckduckgo"

# Search engines match the number of completed questions
SEARCH_ENGINE_TOP_K = 3


# Bing Search required variable
# Bing Subscription Key is required to use bing search and you need to request a trial of bing Search in the azure port
# For details on how to apply, please see
# https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/create-bing-search-service-resource
# To create a bing api search instance using python, see:
# https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/quickstarts/rest/python
BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"
# Note that it is not the api key of bing Webmaster Tools,

# In addition, if it is on the server, the Failed to establish a new connection: [Errno 110] Connection timed out message is displayed
# Because a firewall is added to the server, contact the administrator to add the whitelist
BING_SUBSCRIPTION_KEY = ""

# metaphor search requires a KEY
METAPHOR_API_KEY = ""


# Whether to enable Chinese title enhancement and related configuration of title enhancement
# By adding title judgment, determine which text is the title, and mark it in metadata;
# Then the text is combined with the title of the upper level to achieve the enhancement of the text information.
ZH_TITLE_ENHANCE = False


# The initialization introduction of each knowledge base is used to display and invoke the Agent when initializing the knowledge base. If it is not written, it is not introduced and will not be invoked by the Agent.
KB_INFO = {
    "Knowledge base name": "Introduction to knowledge base",
    "samples": "Answer to issue about this item",
}


# In general, you do not need to change the following

# Default path for storing the knowledge base
KB_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base")
if not os.path.exists(KB_ROOT_PATH):
    os.mkdir(KB_ROOT_PATH)
# Default database storage path.
# If you use sqlite, you can directly modify DB_ROOT_PATH. If you are using a different database, modify SQLALCHEMY_DATABASE_URI directly.
DB_ROOT_PATH = os.path.join(KB_ROOT_PATH, "info.db")
SQLALCHEMY_DATABASE_URI = f"sqlite:///{DB_ROOT_PATH}"

# Optional vector library types and corresponding configurations
kbs_config = {
    "faiss": {
    },
    "milvus": {
        "host": "127.0.0.1",
        "port": "19530",
        "user": "",
        "password": "",
        "secure": False,
    },
    "zilliz": {
        "host": "in01-a7ce524e41e3935.ali-cn-hangzhou.vectordb.zilliz.com.cn",
        "port": "19530",
        "user": "",
        "password": "",
        "secure": True,
        },
    "pg": {
        "connection_uri": "",
    },

    "es": {
        "host": "127.0.0.1",
        "port": "9200",
        "index_name": "test_index",
        "user": "",
        "password": ""
    },
    "milvus_kwargs":{
        "search_params":{"metric_type": "L2"}, # Add search_params here
        "index_params":{"metric_type": "L2","index_type": "HNSW"} # Add index_params here
    }
}

# TextSplitter configuration item. If you do not understand its meaning, do not change it.
text_splitter_dict = {
    "ChineseRecursiveTextSplitter": {
        "source": "huggingface",   # Select tiktoken and use openai's method
        "tokenizer_name_or_path": "",
    },
    "SpacyTextSplitter": {
        "source": "huggingface",
        "tokenizer_name_or_path": "gpt2",
    },
    "RecursiveCharacterTextSplitter": {
        "source": "tiktoken",
        "tokenizer_name_or_path": "cl100k_base",
    },
    "MarkdownHeaderTextSplitter": {
        "headers_to_split_on":
            [
                ("#", "head1"),
                ("##", "head2"),
                ("###", "head3"),
                ("####", "head4"),
            ]
    },
}

# TEXT_SPLITTER Name
TEXT_SPLITTER_NAME = "ChineseRecursiveTextSplitter"

# Glossary file for the Embedding model custom words
EMBEDDING_KEYWORD_FILE = "embedding_keywords.txt"
