<div align='center'>
    <h1>Additional Materials in Footnotes of Our Paper</h1>
</div>

## Contents

* [Overview](#overview)
* [【7】Implementation Details on PostgreSQL and OpenGauss](#7implementation-details-on-postgresql-and-opengauss)
* [【8】Query Prompt for Question Classification Routing](#8query-prompt-for-question-classification-routing)
* [【9】The sources and statistics of the dataset for these classifiers](#9the-sources-and-statistics-of-the-dataset-for-these-classifiers)
* [【11】Prompts for WinRate](#11prompts-for-winrate)
  * [General Prompt Template](#general-prompt-template)
  * [Rag Prompt Template](#rag-prompt-template)
  * [Tool Prompt Template](#tool-prompt-template)
* [【12】Prompt for General Q&A Classification](#12prompt-for-general-qa-classification)
* [【13】The sources and statistics of the dataset for these classifiers](#13the-sources-and-statistics-of-the-dataset-for-these-classifiers)

## Overview

This section provides additional material to the footnotes in our paper, including additional data, methodological details, or other supplementary information, such as prompts for data collection and experiments used, aiming to provide readers with more comprehensive background and understanding.

## 【7】Implementation Details on PostgreSQL and OpenGauss

Go to the folder [Testbed_Demo/Testbed_Backbone](../2_Testbed_Demo/Testbed_Backbone/README.md) for details.

## 【8】Query Prompt for Question Classification Routing

```
Here are seven labels and their definitions:
general\_db: General database-related questions;
gauss\_db: Database-related questions involving Gauss;
tool\_db: Database-related questions requiring tools;
unsafe: Unsafe inputs, such as inputs containing illegal activities, bias and discrimination, insults language, sensitive topics, etc.;
other: Other questions not covered by the above descriptions.

Please read the following input and select the most appropriate label from the seven categories listed above (do not output anything other than the label). 

Input: {{Q}}
```

## 【9】The sources and statistics of the dataset for these classifiers

Go to the folder [Testbed_Demo/Question_Classification_Model](../2_Testbed_Demo/Question_Classification_Model/README.md) for details.

## 【11】Prompts for WinRate

### General Prompt Template

```
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
```

### Rag Prompt Template

```
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
```

### Tool Prompt Template

```
`I will provide you with an `input`, `expected_output`, `now_output`, and `others_output`. Please output a `winner` that you think whether the `now_output` and `others_output` is better, using `expected_output` as the criterion. Also, provide your reasoning.

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
```

## 【12】Prompt for General Q&A Classification

```
You are a database expert, please analyze which of the following database subfields it belongs to based on the questions and corresponding answers.

Question: {{question}}

Answer: {{answer}}

You can only pick one or more of these subfields, and if None of them belong, output "None".

Subfields: {{subfields}}

Your output should just be a list in the following format.
[
    "subfield1",
    "subfield2",
    ......
]

If None of them belong, output "None".
```

## 【13】The sources and statistics of the dataset for these classifiers

Same as 【9】.
