import re
import os
import json

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from llm import LLMAPI, LLMLocal
from jinja2 import Template

class WinRateMetric(BaseMetric):
    # This metric is to evaluate which is better between evaluated LLM's and GPT3.5's Response.
    def __init__(self,base_model,evaluator, threshold: float = 0.0):
        self.threshold = threshold
        self.base_model = base_model
        self.evaluate_general_prompt_template = """
            I will provide you with an `input`, `expected_output`, `now_output`, and `others_output`.Please output a `winner` that you think whether the `now_output` and `others_output` is better, using `expected_output` as the criterion. Also, provide your reasoning.

            Please note that if one of them fabricates facts, that is, it is inconsistent with what is in expected_output. Then it should be lose.

            Please strictly adhere to the following format for your output:
            {
                "reason": "your reason",
                "winner": "now_output or others_output or tie"
            }

            input: {{input}}  

            expected_output: {{expected_output}}  

            now_output: {{now_output}}  

            others_output: {{others_output}}  
        """
        self.evaluate_tool_prompt_template = """
            I will provide you with an `input`, `expected_output`, `now_output`, and `others_output`. Please output a `winner` that you think whether the `now_output` and `others_output` is better, using `expected_output` as the criterion. Also, provide your reasoning.

            Please note that if "调用工具失败" or "tool failure" appeared in one of them, then it should be lose. 
            
            Please strictly adhere to the following format for your output:
            {
                "reason": "your reason",
                "winner": "now_output or others_output or tie",
            }

            input: {{input}}  

            expected_output: {{expected_output}}  

            now_output: {{now_output}}  

            others_output: {{others_output}}  
        """
        self.evaluate_rag_prompt_template = """
            I will provide you with an `input`, `expected_output`, `now_output`, and `others_output`.Please output a `winner` that you think whether the `now_output` and `others_output` is better, using `expected_output` as the criterion. Also, provide your reasoning.

            Please note that if one of them fabricates facts, that is, it is inconsistent with what is in expected_output. Then it should be lose.

            Please strictly adhere to the following format for your output:
            {
                "reason": "your reason",
                "winner": "now_output or others_output or tie"
            }

            input: {{input}}  

            expected_output: {{expected_output}}  

            now_output: {{now_output}}  

            others_output: {{others_output}}  
        """
        self.evaluator = evaluator
        with open('', 'r', encoding='utf-8') as f: # Please fill in the path of the data generated by the base model, such as 'gpt-3.5-turbo-0125_general_zh_cases.json'
            data = json.load(f)
        data_now = []
        for i in data:
            if type(i) == list:
                if i != []:
                    data_now.append(i[-1])
                else:
                    tem = data_now[0]
                    tem['actual_output'] = ""
                    tem['context'] = None
                    tem['retrieval_text'] = None
                    data_now.append(tem)
            else:
                data_now.append(i)
        dict = {d['input']:d['actual_output'] for d in data_now}
        self.baseAnswer = dict

    async def a_measure(
        self, test_case: LLMTestCase, _show_indicator: bool = True
    ):
        return self.measure(test_case)
    
    def measure(self, test_case: LLMTestCase):
        self.error = None
        try:
            others_output = self.baseAnswer[test_case.input]
        except:
            print("None Appeared")
            if "gpt" or "glm" in self.base_model:
                base_gpt_model = LLMAPI(model_name = self.base_model)
                others_output = base_gpt_model(test_case.input)
            else:
                base_gpt_model = LLMLocal(model_name = self.base_model)
                others_output = base_gpt_model(test_case.input)
        print('WinRate now_output:\n' + test_case.actual_output)
        print('WinRate others_output:\n' + others_output)
        now_output = test_case.actual_output
        expected_output = test_case.expected_output
        data = {
            "input": test_case.input,
            "expected_output": expected_output,
            "now_output": now_output,
            "others_output": others_output
        }
        if test_case.context == None:
            test_case.context = []
            test_case.context.append('tool')
        if test_case.context[0] == 'general':
            evaluate_prompt = Template(self.evaluate_general_prompt_template).render(data)
        elif test_case.context[0] == 'tool':
            evaluate_prompt = Template(self.evaluate_tool_prompt_template).render(data)
        else:
            evaluate_prompt = Template(self.evaluate_rag_prompt_template).render(data)
        test_case.context = test_case.context[1:]
        response = self.evaluator(evaluate_prompt)
        print('WinRate result:\n' + response)
        value_pattern = r'"winner":\s*"(.*?)"'
        reason_pattern = r'"reason":\s*"(.*?)"'

        value_match = re.search(value_pattern, response)
        reason_match = re.search(reason_pattern, response)

        if value_match and reason_match:
            if value_match.group(1)=="now_output":
                self.score = 1
            elif value_match.group(1)=="others_output":
                self.score = 0
            else:
                self.score = -8
            self.reason = reason_match.group(1)
        else:
            print("Error: GPT output format error!")
            self.success = False

        self.success = True
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "WinRate"

class ToolSelectionAccuracyMetric(BaseMetric):
    # This metric is to evaluate whether the LLM choose the right tools.
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    async def a_measure(
        self, test_case: LLMTestCase, _show_indicator: bool = True
    ):
        return self.measure(test_case)
    
    def std_string(self,s):
        std_s = re.sub(r'\s+|\\n', '', s).lower()
        parts_after_action = std_s.split("action:", 1)
        desired_substring = parts_after_action[1].split("action_input", 1)[0]
        return desired_substring
    
    def measure(self, test_case: LLMTestCase):
        self.error = None
        now_output = test_case.actual_output
        expected_output = test_case.expected_output

        tool_pattern = r'(Action:\s*.*?\s*Action_Input)'

        now_matches = re.findall(tool_pattern, now_output, re.IGNORECASE)
        expected_matches = re.findall(tool_pattern, expected_output, re.IGNORECASE)

        std_expected_matches = []
        for i in expected_matches:
            std_match = self.std_string(i)
            std_expected_matches.append(std_match)

        cover_count = 0
        for now_match in now_matches:
            std_now_match = self.std_string(now_match)
            if std_now_match in std_expected_matches:
                cover_count = cover_count+1

        if len(expected_matches) == 0:
            self.score = 0
        else:
            self.score = float(cover_count)/len(expected_matches)
        
        if self.score > 1:
            self.score = 1

        self.success = True

        self.reason = None

        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "ToolSelectionAccuracy"
    
class ToolFormatAccuracyMetric(BaseMetric):
    # This metric is to evaluate whether the LLM use tool correctly.
    def __init__(self, evaluator, threshold: float = 0.0):
        self.threshold = threshold
        self.ToolTemplate = """
        I will provide you with an `input` and a `format`. Please output a `value`, which can be either 0 or 1. 

        1. If you believe that the Action_Input in the input adheres to the requirement specified in format, set the value to 1. If it does not adhere to the format, then set the value to 0.
        2. In the format, a # symbol represents a descriptive placeholder. For example, #type indicates that a word representing a type should be entered here.
        
        Please strictly adhere to the following format for your output:
            {
                "reason": "your reason";
                "value": "0 or 1"
            }
        
        input: {{input}}

        format: {{format}}
        """
        self.evaluator = evaluator
    
    async def a_measure(
        self, test_case: LLMTestCase, _show_indicator: bool = True
    ):
        return self.measure(test_case)

    def measure(self, test_case: LLMTestCase):
        self.error = None
        now_output = test_case.actual_output

        tool_pattern = r'(Action:\s*.*?\s*Action_Input:\s*.*?\s*Observation)'

        now_matches = re.findall(tool_pattern, now_output, re.IGNORECASE)

        if len(now_matches) == 0:
            self.score = -8
            self.reason = "Not exist tool usage."
            self.success = False
            return self.score

        count = 0

        for i in now_matches:
            input = i
            format = test_case.context[1:]
            data = {
                "input": input,
                "format": format
            }
            evaluate_prompt = Template(self.ToolTemplate).render(data)
            response = self.evaluator(evaluate_prompt)

            value_pattern = r'"value":\s*"(.*?)"'
            value_match = re.search(value_pattern, response)

            if value_match:
                count = count + int(value_match.group(1))
            else:
                print("Error: GPT output format error in ToolFormatAccuracyMetric!")
                self.success = False
        
        if count == 0:
            self.score = 0
        else:
            self.score = float(count)/len(now_matches)
        self.reason = None
        self.success = True

        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "ToolFormatAccuracy"

class RAGAccuracyMetric(BaseMetric):
    # This metric is to evaluate RAG accuarcy between retrieval_text and expected_retrieval_text
    def __init__(self,evaluator,threshold: float = 0.0):
        self.threshold = threshold
        self.evaluate_prompt_template = """
            I will provide you with an `input`, `retrieval_text`, and `expected_retrieval_text`. Please output a `value`, which ranges between 0 and 1. The higher the value, the more relevant the `retrieval_text` is to the `input`, and the content of the `retrieval_text` is within the `expected_retrieval_text`. Also, provide your reasoning.

            Please strictly adhere to the following format for your output:

            {
                "value": "0-1",
                "reason": "your reason"
            }

            input: {{input}}  
            retrieval_text: {{retrieval_text}}  
            expected_retrieval_text: {{expected_retrieval_text}}  
        """
        self.evaluator = evaluator

    async def a_measure(
        self, test_case: LLMTestCase, _show_indicator: bool = True
    ):
        return self.measure(test_case)
    
    def measure(self, test_case: LLMTestCase):
        self.error = None
        data = {
            "input": test_case.input,
            "retrieval_text": test_case.retrieval_context,
            "expected_retrieval_text": test_case.context,
        }

        evaluate_prompt = Template(self.evaluate_prompt_template).render(data)
        response = self.evaluator(evaluate_prompt)

        value_pattern = r'"value":\s*"(.*?)"'
        reason_pattern = r'"reason":\s*"(.*?)"'

        value_match = re.search(value_pattern, response)
        reason_match = re.search(reason_pattern, response)

        if value_match and reason_match:
            self.score = value_match.group(1)
            self.reason = reason_match.group(1)
        else:
            print("Error: GPT output format error!")
            self.success = False

        self.success = True
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "RAGAccuracy"
    
class SelectAccuracyMetric(BaseMetric):
    # This metric is used to evaluate the accuracy of choice questions
    def __init__(self,threshold: float = 0.0):
        self.threshold = threshold

    async def a_measure(
        self, test_case: LLMTestCase, _show_indicator: bool = True
    ):
        return self.measure(test_case)
    
    def measure(self, test_case: LLMTestCase):
        self.error = None
        data = {
            "actual_output": test_case.actual_output,
            "expected_output": test_case.expected_output
        }

        try:
            if data["actual_output"] == data["expected_output"]:
                self.score = 1
            else:
                self.score = 0
        except:
            self.success = False

        self.success = True
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "SelectAccuracy"
    