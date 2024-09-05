import requests
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM
from transformers.generation.utils import GenerationConfig
from time import sleep
import torch

class LLMAPI(object):
    def __new__(cls, model_name):
        if "gpt" in model_name:
            return CustomOpenAI(model_name)
        elif "glm" in model_name:
            return CustomZhipuAPI(model_name)
        else:
            return super(LLMAPI, cls).__new__(cls)  
    
    def __init__(self, model_name):
        print("Error: Do not support LLM API: " + str(model_name))

class LLMLocal(object):
    def __new__(cls, model_name):
        if "Llama2" in model_name:
            return CustomLlama2(model_name)
        elif "Llama3" in model_name:
            return CustomLlama3(model_name)
        elif "Baichuan2" in model_name:
            return CustomBaichuan2(model_name)
        elif "mistral" in model_name:
            return CustomMistral(model_name)
        elif "Qwen" in model_name:
            return CustomQwen(model_name)
        elif "Yuan" in model_name:
            return CustomYuan(model_name)
        else:
            return super(LLMAPI, cls).__new__(cls)  
    
    def __init__(self, model_name):
        print("Error: Do not support Local LLM Model: " + str(model_name))

class CustomZhipuAPI(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.api_key = ""
        self.model_name = model # example: 'glm-4'
            
    def load_model(self):
        from zhipuai import ZhipuAI
        return ZhipuAI(api_key= self.api_key)

    def generate(self, prompt: str) -> str:
        client = self.load_model()
        
        messages = [{"role":"user","content":prompt}]

        response = client.chat.completions.create(
            model= self.model_name,  
            messages= messages
        )

        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Custom Zhipu API Model"
    
    def __call__(self, prompt: str) -> str:
        return self.generate(prompt)

class CustomOpenAI(DeepEvalBaseLLM):
    def __init__(
        self,
        model,
    ):
        self.api_key = ""
        self.model = model # example: 'gpt-3.5-turbo-0125'

    def load_model(self):
        pass

    def generate(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }
        url = "https://api.openai.com/v1/completions"

        messages = [{"role":"user","content":prompt}]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
        except:
            sleep(0.2)
            response = requests.post(url, json=payload, headers=headers)
        try:
            return str(response.json()["choices"][0]["message"]['content'])
        except:
            print('Error: ' + str(response.json()))
            return ""

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Custom OpenAI Model"
    
    def __call__(self, prompt: str) -> str:
        return self.generate(prompt)
    
class CustomMistral(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.model_name = model
        self.model = self.load_model()

    def load_model(self):
        if self.model_name == "Mistral-7B":
            self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
            self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        elif self.model_name == "Mistral-7B-Instruct":
            self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
            self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        return self.model

    def generate(self, prompt: str) -> str:
        device = "cuda" # the device to load the model onto

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        self.model.to(device)

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=4000, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)[0]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Mistral"
    

class CustomLlama2(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.model_name = model
        self.model = self.load_model()

    def load_model(self):
        if self.model_name == "Llama-2-7b-chat":
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        elif self.model_name == "Llama-2-7b":
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        elif self.model_name == "Llama-2-13b":
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf")
        elif self.model_name == "Llama-2-13b-chat":
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
        return self.model

    def generate(self, prompt: str) -> str:
        device = "cuda" # the device to load the model onto
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        self.model.to(device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=4000, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)[0]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Llama2"
    
    def __call__(self, prompt: str) -> str:
        return self.generate(prompt)

class CustomLlama3(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.model_name = model
        self.model = self.load_model()

    def load_model(self):
        if self.model_name == "Llama3-8B-Instruct":
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16)
        return self.model

    def generate(self, prompt: str) -> str:
        device = "cuda"
        self.model.to(device)
        input_ids = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=4000,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.01
        )
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Llama3"
    
    def __call__(self, prompt: str) -> str:
        return self.generate(prompt)

class CustomBaichuan2(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.model_name = model
        self.model = self.load_model()

    def load_model(self):
        if self.model_name == "Baichuan2-7B-Chat":
            self.tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-7B-Chat", use_fast=False, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan2-7B-Chat", trust_remote_code=True)
            self.model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan2-7B-Chat")
        elif self.model_name == "Baichuan-7B":
            self.tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan-7B", trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan-7B", trust_remote_code=True)
        elif self.model_name == "Baichuan2-13B-Chat":
            self.tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-13B-Chat",
                revision="v2.0",
                use_fast=False,
                trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan2-13B-Chat",
                revision="v2.0",
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True)
            self.model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan2-13B-Chat", revision="v2.0")
        return self.model

    def generate(self, prompt: str) -> str:
        if "chat" or "Chat" in self.model_name:
            messages = [{"role": "user", "content": prompt}]
            device = "cuda" # the device to load the model onto
            self.model.to(device)
            response = self.model.chat(self.tokenizer, messages)
        else:
            inputs = self.tokenizer([prompt], return_tensors='pt')
            device = "cuda" # the device to load the model onto
            inputs.to(device)
            self.model.to(device)
            pred = self.model.generate(**inputs, max_new_tokens=4000,repetition_penalty=1.1, temperature=0.01)
            response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        return response

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Baichuan2"

    def __call__(self, prompt: str) -> str:
        return self.generate(prompt)

class CustomQwen(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.model_name = model
        self.model = self.load_model()

    def load_model(self):
        if self.model_name == "Qwen1.5-14B-Chat":
            self.model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen1.5-14B-Chat",
                torch_dtype="auto",
            
            )
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-14B-Chat")
        elif self.model_name == "Qwen/Qwen1.5-14B":
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-14B")
            self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-14B")
        elif self.model_name == "Qwen1.5-7B-Chat":
            self.model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen1.5-7B-Chat",
                torch_dtype="auto",
            
            )
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat")
        elif self.model_name == "Qwen/Qwen1.5-7B":
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B")
            self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-7B")
        return self.model

    def generate(self, prompt: str) -> str:
        messages = {"role": "user", "content": prompt}
        device = "cuda" # the device to load the model onto
        self.model.to(device)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=4000
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Qwen"

    def __call__(self, prompt: str) -> str:
        return self.generate(prompt)

class CustomYuan(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.model_name = model
        self.model = self.load_model()

    def load_model(self):
        if self.model_name == "Yuan2-2B-hf":
            self.tokenizer = LlamaTokenizer.from_pretrained('IEITYuan/Yuan2-2B-hf', add_eos_token=False, add_bos_token=False, eos_token='<eod>')
            self.tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)
            self.model = AutoModelForCausalLM.from_pretrained('IEITYuan/Yuan2-2B-hf', trust_remote_code=True)
        if self.model_name == "Yuan2-2B-Janus-hf":
            self.tokenizer = LlamaTokenizer.from_pretrained('IEITYuan/Yuan2-2B-Janus-hf', add_eos_token=False, add_bos_token=False, eos_token='<eod>')
            self.tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)
            self.model = AutoModelForCausalLM.from_pretrained('IEITYuan/Yuan2-2B-Janus-hf', trust_remote_code=True)
        if self.model_name == "Yuan2-2B-Februa-hf":
            self.tokenizer = LlamaTokenizer.from_pretrained('IEITYuan/Yuan2-2B-Februa-hf', add_eos_token=False, add_bos_token=False, eos_token='<eod>')
            self.tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)
            self.model = AutoModelForCausalLM.from_pretrained('IEITYuan/Yuan2-2B-Februa-hf', torch_dtype=torch.bfloat16, trust_remote_code=True)

        return self.model

    def generate(self, prompt: str) -> str:
        device = "cuda" # the device to load the model onto
        self.model.to(device)
        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        outputs = self.model.generate(inputs,do_sample=False,max_new_tokens=4000,temperature=0.01)
        response = self.tokenizer.decode(outputs[0])
        
        return response

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Yuan"
    
    def __call__(self, prompt: str) -> str:
        return self.generate(prompt)
