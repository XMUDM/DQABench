from fastchat.conversation import Conversation
from configs import LOG_PATH, TEMPERATURE
import fastchat.constants
fastchat.constants.LOGDIR = LOG_PATH
from fastchat.serve.base_model_worker import BaseModelWorker
import uuid
import json
import sys
from pydantic import BaseModel, root_validator
import fastchat
import asyncio
from server.utils import get_model_worker_config
from typing import Dict, List, Optional


__all__ = ["ApiModelWorker", "ApiChatParams", "ApiCompletionParams", "ApiEmbeddingsParams"]


class ApiConfigParams(BaseModel):
    '''
    Online API configuration parameters, values not provided are automatically read from model_config.ONLINE_LLM_MODEL
    '''
    api_base_url: Optional[str] = None
    api_proxy: Optional[str] = None
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    group_id: Optional[str] = None # for minimax
    is_pro: bool = False # for minimax

    APPID: Optional[str] = None # for xinghuo
    APISecret: Optional[str] = None # for xinghuo
    is_v2: bool = False # for xinghuo

    worker_name: Optional[str] = None

    class Config:
        extra = "allow"

    @root_validator(pre=True)
    def validate_config(cls, v: Dict) -> Dict:
        if config := get_model_worker_config(v.get("worker_name")):
            for n in cls.__fields__:
                if n in config:
                    v[n] = config[n]
        return v

    def load_config(self, worker_name: str):
        self.worker_name = worker_name
        if config := get_model_worker_config(worker_name):
            for n in self.__fields__:
                if n in config:
                    setattr(self, n, config[n])
        return self


class ApiModelParams(ApiConfigParams):
    '''
    Model configuration parameter
    '''
    version: Optional[str] = None
    version_url: Optional[str] = None
    api_version: Optional[str] = None # for azure
    deployment_name: Optional[str] = None # for azure
    resource_name: Optional[str] = None # for azure

    temperature: float = TEMPERATURE
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0


class ApiChatParams(ApiModelParams):
    '''
    chat request parameters
    '''
    messages: List[Dict[str, str]]
    system_message: Optional[str] = None # for minimax
    role_meta: Dict = {} # for minimax


class ApiCompletionParams(ApiModelParams):
    prompt: str


class ApiEmbeddingsParams(ApiConfigParams):
    texts: List[str]
    embed_model: Optional[str] = None
    to_query: bool = False # for minimax


class ApiModelWorker(BaseModelWorker):
    DEFAULT_EMBED_MODEL: str = None # None means not support embedding

    def __init__(
        self,
        model_names: List[str],
        controller_addr: str = None,
        worker_addr: str = None,
        context_len: int = 2048,
        no_register: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("worker_id", uuid.uuid4().hex[:8])
        kwargs.setdefault("model_path", "")
        kwargs.setdefault("limit_worker_concurrency", 5)
        super().__init__(model_names=model_names,
                        controller_addr=controller_addr,
                        worker_addr=worker_addr,
                        **kwargs)
        import fastchat.serve.base_model_worker
        import sys
        self.logger = fastchat.serve.base_model_worker.logger
        # Restore standard output overwritten by fastchat
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)

        self.context_len = context_len
        self.semaphore = asyncio.Semaphore(self.limit_worker_concurrency)
        self.version = None

        if not no_register and self.controller_addr:
            self.init_heart_beat()


    def count_token(self, params):
        prompt = params["prompt"]
        return {"count": len(str(prompt)), "error_code": 0}

    def generate_stream_gate(self, params: Dict):
        self.call_ct += 1

        try:
            prompt = params["prompt"]
            if self._is_chat(prompt):
                messages = self.prompt_to_messages(prompt)
                messages = self.validate_messages(messages)
            else: # Use chat to mimic the write continuation function and do not support historical messages
                messages = [{"role": self.user_role, "content": f"please continue writing from here: {prompt}"}]

            p = ApiChatParams(
                messages=messages,
                temperature=params.get("temperature"),
                top_p=params.get("top_p"),
                max_tokens=params.get("max_new_tokens"),
                version=self.version,
            )
            for resp in self.do_chat(p):
                yield self._jsonify(resp)
        except Exception as e:
            yield self._jsonify({"error_code": 500, "text": f"{self.model_names[0]}An error occurred while requesting the API: {e}"})

    def generate_gate(self, params):
        try:
            for x in self.generate_stream_gate(params):
                ...
            return json.loads(x[:-1].decode())
        except Exception as e:
            return {"error_code": 500, "text": str(e)}


    # Requires user-defined methods

    def do_chat(self, params: ApiChatParams) -> Dict:
        '''
        The Chat method is executed by default using the chat function in the module.
        Return form required:{"error_code": int, "text": str}
        '''
        return {"error_code": 500, "text": f"{self.model_names[0]} didn't implement chat function"} 

    def do_embeddings(self, params: ApiEmbeddingsParams) -> Dict:
        '''
        Execute the Embeddings method, using the embed_documents function in the module by default.
        Return form: {"code": int, "data": List[List[float]], "msg": str}
        '''
        return {"code": 500, "msg": f"{self.model_names[0]} didn't implement embeddings function"}

    def get_embeddings(self, params):
        print("get_embedding")
        print(params)

    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        raise NotImplementedError

    def validate_messages(self, messages: List[Dict]) -> List[Dict]:
        '''
        Some apis have a special format for mesages that you can override to replace the default messages.
        The value of prompt_to_messages is different from that of prompt_to_messages because of different application scenarios and parameters
        '''
        return messages


    # help methods
    @property
    def user_role(self):
        return self.conv.roles[0]

    @property
    def ai_role(self):
        return self.conv.roles[1]

    def _jsonify(self, data: Dict) -> str:
        '''
        Return the results from the chat function in the fastchat openai-api-server format
        '''
        return json.dumps(data, ensure_ascii=False).encode() + b"\0"

    def _is_chat(self, prompt: str) -> bool:
        '''
        Check whether the prompt is spliced from chat messages
        '''
        key = f"{self.conv.sep}{self.user_role}:"
        return key in prompt

    def prompt_to_messages(self, prompt: str) -> List[Dict]:
        '''
        Split the prompt string into messages.
        '''
        result = []
        user_role = self.user_role
        ai_role = self.ai_role
        user_start = user_role + ":"
        ai_start = ai_role + ":"
        for msg in prompt.split(self.conv.sep)[1:-1]:
            if msg.startswith(user_start):
                if content := msg[len(user_start):].strip():
                    result.append({"role": user_role, "content": content})
            elif msg.startswith(ai_start):
                if content := msg[len(ai_start):].strip():
                    result.append({"role": ai_role, "content": content})
            else:
                raise RuntimeError(f"unknown role in msg: {msg}")
        return result

    @classmethod
    def can_embedding(cls):
        return cls.DEFAULT_EMBED_MODEL is not None
