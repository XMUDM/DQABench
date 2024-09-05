import argparse
import json
from jinja2 import Template
import os
import re
from tqdm import tqdm
import random

from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.metrics.ragas import RagasMetric
from deepeval.test_case import LLMTestCaseParams
from customMetric import RAGAccuracyMetric, WinRateMetric, ToolSelectionAccuracyMetric, ToolFormatAccuracyMetric, SelectAccuracyMetric
from llm import LLMAPI, LLMLocal
from config import PROMPT, METRIC

class Evaluator(object):
    def __init__(self,args):
        self.model_name = args.model
        self.benchmark_name = args.benchmark
        self.benchmark_language = args.language
        self.evaluator = args.evaluator
        self.action = ""
        if args.action:
            if args.action != "data" and args.action != "eval":
                print("Error: action error. must be one of \"data\" or \"eval\".")
                exit()
            self.action = args.action
        if self.action == "data" or self.action == "":
            self.load_model()
            self.load_benchmark()
    
    def load_model(self):
        # load model into self.model
        if "gpt" in self.model_name or "glm" in self.model_name:
            llm = LLMAPI(self.model_name)
        else:
            llm = LLMLocal(self.model_name)

        self.model = llm
    
    def load_benchmark(self):
        #load benchmark into self.benchmark
        self.benchmark = []
        
        if self.benchmark_language == "zh":
            self.dir = "DQABenchmark_zh"
        else:
            self.dir = "DQABenchmark_en"

        if self.benchmark_name == "select":
            path = os.walk(self.dir + '/General/Select')
        elif self.benchmark_name == "general":
            path = os.walk(self.dir + '/General/Forum_QA')
        elif self.benchmark_name == "gauss":
            path = os.walk(self.dir + '/Gauss')
        elif self.benchmark_name == "tool":
            path = os.walk(self.dir + '/Tool')
        else:
            print("Error: benchmark name error. must be one of \"select\", \"general\", \"gauss\", \"tool\".")
            exit()

        for root, dirs, files in path:
            for file in files:
                if file.endswith("test.json") or file.endswith("validation.json"):
                    with open(os.path.join(root, file), 'r') as f:
                        self.benchmark += json.loads(f.read())

        for i in range(len(self.benchmark)):
            if self.benchmark[i]['field'] == 'select':
                self.benchmark[i] = {'question': self.benchmark[i]['question'], 'answer': self.benchmark[i]['answer'], 'field': 'select', 'doc': None, 'tool': None}
            elif self.benchmark[i]['field'] == 'general':
                self.benchmark[i] = {'question': self.benchmark[i]['question'], 'answer': self.benchmark[i]['answer'], 'field': 'general', 'doc': None, 'tool': None}
            elif self.benchmark[i]['field'] == 'tool':
                self.benchmark[i] = {'question': self.benchmark[i]['question'], 'answer': self.benchmark[i]['answer'], 'field': 'tool', 'doc': self.benchmark[i]['format'], 'tool': self.benchmark[i]['tool']}
            else:
                self.benchmark[i] = {'question': self.benchmark[i]['question'], 'answer': self.benchmark[i]['answer'], 'field': 'RAG', 'doc': self.benchmark[i]['retrieval'], 'tool': None}

        
    def gen_actual_output(self, input, i, agent_scratchpad):
        # generate output by LLM model.
        select_prompt_template = PROMPT[(self.benchmark_language + '_select_prompt').upper()]
        general_prompt_template = PROMPT[(self.benchmark_language + '_general_prompt').upper()]
        RAG_prompt_template = PROMPT[(self.benchmark_language + '_gauss_prompt').upper()]
        if 'sql_executor' in self.benchmark[i]['answer'] and ('data' in self.benchmark[i]['answer'] or '数据' in self.benchmark[i]['answer'] or 'analysis' in self.benchmark[i]['answer'] or '分析' in self.benchmark[i]['answer']):
            tool_prompt_template = PROMPT[(self.benchmark_language + '_analysis_prompt').upper()]
        else:
            tool_prompt_template = PROMPT[(self.benchmark_language + '_management_prompt').upper()]

        if self.benchmark[i]['field'] == "select":
            data = {
                "input": input,
            }
            prompt = Template(select_prompt_template).render(data)
        elif self.benchmark[i]['field'] == "general":
            data = {
                "input": input,
            }
            prompt = Template(general_prompt_template).render(data)

        elif self.benchmark[i]['field'] == "RAG":
            data = {
                "input": input,
                "knowledge": self.benchmark[i]['doc']
            }
            prompt = Template(RAG_prompt_template).render(data)

        elif self.benchmark[i]['field'] == "tool":
            with open(self.dir + "/Tool/tools.json", 'r', encoding='utf-8') as f:
                all_tools = json.loads(f.read())
            tools = []
            tool_names = []
            for t in self.benchmark[i]['doc']:
                tool_names.append(t['tool'])
                tools.append(t['tool'] + ': Content requirement: ' + t['Content requirement'] + ' Format requirement: ' + t['Format requirement'])
            cnt = 0
            while cnt < 2:
                t = random.choice(all_tools['necessary'])
                if t['tool'] not in tool_names:
                    tool_names.append(t['tool'])
                    tools.append(t['tool'] + ': Content requirement: ' + t['Content requirement'] + ' Format requirement: ' + t['Format requirement'])
                    cnt += 1
            while cnt < 4:
                t = random.choice(all_tools['others'])
                if t['tool'] not in tool_names:
                    tool_names.append(t['tool'])
                    format = random.choice(t['format'])
                    tools.append(t['tool'] + ': Content requirement: ' + format['Content requirement'] + ' Format requirement: ' + format['Format requirement'])
                    cnt += 1
            combined_lists = list(zip(tools, tool_names))
            random.shuffle(combined_lists)
            tools, tool_names = zip(*combined_lists)
            data = {
                "tools": '\n'.join(tools),
                "tool_names": ','.join(tool_names),
                "input": input,
                "agent_scratchpad": agent_scratchpad,
            }
            prompt = Template(tool_prompt_template).render(data)
        output = self.model(prompt)
        return output
    
    def gen_dataset(self):
        if self.action == "eval":
            self.dataset = EvaluationDataset()
            self.dataset.add_test_cases_from_json_file(
                # file_path is the absolute path to you .json file
                file_path=self.model_name + '_' + self.benchmark_name + '_' + self.benchmark_language + '_cases.json',
                input_key_name="input",
                actual_output_key_name="actual_output",
                expected_output_key_name="expected_output",
                context_key_name="context",
                retrieval_context_key_name="retrieval_text",
            )
        else:
            cases = []
            test_cases = []
            if os.path.exists(self.model_name + '_' + self.benchmark_name + '_' + self.benchmark_language + '_cases.json'):
                with open(self.model_name + '_' + self.benchmark_name + '_' + self.benchmark_language + '_cases.json', 'r', encoding='utf-8') as f:
                    cases = json.load(f)
                for case in cases:
                    if type(case) != list:
                        test_case = LLMTestCase(input=case['input'], actual_output=case['actual_output'], context=case['context'], expected_output=case['expected_output'], retrieval_context=case['retrieval_text'])
                        test_cases.append(test_case)
                    else:
                        for i in range(len(case)):
                            test_case = LLMTestCase(input=case[i]['input'], actual_output=case[i]['actual_output'], context=case[i]['context'], expected_output=case[i]['expected_output'], retrieval_context=case[i]['retrieval_text'])
                            test_cases.append(test_case)
            for i in tqdm(range(len(cases), len(self.benchmark))):
                if self.benchmark[i]['field'] == 'tool':
                    cases.append([])
                input = self.benchmark[i]['question']
                expected_output = self.benchmark[i]['answer']
                doc = ""
                if self.benchmark[i].get('doc'):
                    if self.benchmark[i]['field'] == 'tool':
                        format = ""
                        for f in self.benchmark[i]['doc']:
                            format += 'tool: ' + f['tool'] + '\n' + 'Content requirement: ' + f['Content requirement'] + '\n' + 'Format requirement: ' + f['Format requirement'] + '\n\n'
                        doc = format
                    else:
                        doc = self.benchmark[i]['doc']

                tool_pattern = r"Action[:：](.*?)(?=Action_[iI]nput[:：])"
                if self.benchmark[i]['field'] == 'tool':
                    tool_list = [i.strip() for i in re.findall(tool_pattern, self.benchmark[i]['answer'], re.DOTALL)]
                    self.benchmark[i]['answer'] = self.benchmark[i]['answer'].replace('：', ':')
                    answer_list = self.benchmark[i]['answer'].split('Observation:')
                    if len(tool_list) + 1 != len(answer_list):
                        continue
                    expected_output = ""
                    count = len(answer_list)
                else:
                    count = 1

                actual_output = ""
                agent_scratchpad = ""
                for j in range(count):
                    if (self.benchmark[i]['field'] == 'tool'):
                        pattern = r"Observation[:：](.*?)(?=Thought[:：]|Final_Answer[:：])"
                        observation_list = re.findall(pattern, self.benchmark[i]['answer'], re.DOTALL)
                    if j:
                        agent_scratchpad = actual_output + observation_list[j-1]

                    now_output = self.gen_actual_output(input, i, agent_scratchpad)
                    if (self.benchmark[i]['field'] == 'tool'):
                        actual_tool_list = [i.strip() for i in re.findall(tool_pattern, now_output, re.DOTALL)]
                        if j < count - 1 and (len(actual_tool_list) == 0 or actual_tool_list[0] != tool_list[j]):
                            actual_output = "tool failure"
                        else:
                            if j < count - 1:
                                actual_output = agent_scratchpad + now_output.split('Observation:')[0] + 'Observation:'
                            else:
                                actual_output = agent_scratchpad + now_output.split('Observation:')[0]
                        if j < count - 1:
                            expected_output += answer_list[j] + 'Observation:'
                        else:
                            expected_output += answer_list[j]
                    else:
                        actual_output = now_output
                    retrieval_text = ""

                    # Warning: Here we did an abnormal decision: we use the "context" attribute to store the "expected_retrieval_output".
                    # Our Dataset is to evaluate Q-A ability which donot use the original function of "context".
                    test_case = LLMTestCase(input=input, actual_output=actual_output, context=[self.benchmark[i]['field'], doc], expected_output=expected_output, retrieval_context=[retrieval_text])

                    if self.benchmark[i]['field'] == 'tool':
                        cases[i].append({'model': self.model_name, 'field': self.benchmark[i]['field'], 'language': self.benchmark_language, 'input': input, 'actual_output': actual_output, 'expected_output': expected_output, 'retrieval_text': [retrieval_text], 'context': [self.benchmark[i]['field'], doc]})
                    else:
                        cases.append({'model': self.model_name, 'field': self.benchmark[i]['field'], 'language': self.benchmark_language, 'input': input, 'actual_output': actual_output, 'expected_output': expected_output, 'retrieval_text': [retrieval_text], 'context': [self.benchmark[i]['field'], doc]})

                    if len(cases) % 5 == 0:
                        with open(self.model_name + '_' + self.benchmark_name + '_' + self.benchmark_language + '_cases.json', 'w', encoding='utf-8') as f:
                            f.write(json.dumps(cases, ensure_ascii=False, indent=4))

                    test_cases.append(test_case)
                    if actual_output == "tool failure":
                        break
            with open(self.model_name + '_' + self.benchmark_name + '_' + self.benchmark_language + '_cases.json', 'w', encoding='utf-8') as f:
                f.write(json.dumps(cases, ensure_ascii=False, indent=4))
            self.dataset = EvaluationDataset(
                test_cases=test_cases
            )

        return self.dataset
    
    def evaluate(self):
        # must run after gen_dataset()
        metrics = []

        evaluator_model = LLMAPI(self.evaluator)

        if METRIC['whether_choose']['AnswerRelevancyMetric'] == "True":
            ar = AnswerRelevancyMetric(
                threshold=0.7,
                model=evaluator_model,
                include_reason=True
            )
            metrics.append(ar)
        if METRIC['whether_choose']['GEvalMetric'] == "True":
            ge = GEval(
                name="Goodness",
                criteria="Goodness - determine the goodness of the actual output using the expected output as the standard.\
                          If there is a factual error, it will be recorded as 0 score.\
                          If it's right, you need to score the actual output based on its specificity and user-friendliness",
                evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
                model=evaluator_model
            )
            metrics.append(ge)
        if METRIC['whether_choose']['RagasMetric'] == "True":
            ragas = RagasMetric(threshold=0.5, model=evaluator_model)
            metrics.append(ragas)
        if METRIC['whether_choose']['WinRateMetric'] == "True":
            base_model = METRIC['WinRateMetric_base_model']
            winrate = WinRateMetric(
                base_model = base_model,
                evaluator = evaluator_model
                )
            metrics.append(winrate)
        if METRIC['whether_choose']['RAGAccuracyMetric'] == "True":
            raga = RAGAccuracyMetric(evaluator=evaluator_model)
            metrics.append(raga)
        if METRIC['whether_choose']['ToolSelectionAccuracyMetric'] == "True":
            tsa = ToolSelectionAccuracyMetric()
            metrics.append(tsa)
        if METRIC['whether_choose']['ToolFormatAccuracyMetric'] == "True":
            tfa = ToolFormatAccuracyMetric(evaluator=evaluator_model)
            metrics.append(tfa)
        if METRIC['whether_choose']['SelectAccuracyMetric'] == "True":
            tfa = SelectAccuracyMetric()
            metrics.append(tfa)
        
        if len(metrics) == 0:
            print("Error: no metric is chosen.")
        else:
            self.dataset.evaluate(metrics)

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='receive evaluated model and benchmark.')
    parser.add_argument('-model', required=True, help='evaluated model name.')
    parser.add_argument('-benchmark', required=True, help='evaluated benchmark name. must be one of \"select\", \"general\", \"gauss\" or \"tool\".')
    parser.add_argument('-language', required=True, help='evaluated benchmark language. must be one of \"en\" or \"zh\".')
    parser.add_argument('-evaluator', required=True, help='evaluator model. suggest to be one of \"gpt-3.5-turbo-0125\" or \"gpt-4-0125-preview\".')
    parser.add_argument('-action', required=False, help='Choose to generate dataset only or evaluate only. must be one of \"data\" or \"eval\".')
    args = parser.parse_args()

    main = Evaluator(args)
    dataset = main.gen_dataset()
    if main.action == "eval" or main.action == "":
        main.evaluate()
