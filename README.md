<div align='center'>
    <h1>DQA-Bench</h1>
</div>

<p align='center'>
    【English | <a href="README_zh.md">中文</a>】
</p>

DQA is the first comprehensive database question answering benchmark, whose dataset is constructed using Internet data collection and an innovative large language model generation-based method. We also propose a comprehensive LLM-based database Q&A testbed on DQA. This testbed is highly modular and scalable, with both basic and advanced components like Question Classification Routing (QCR), Retrieval-Augmented Generation (RAG), Tool Invocation Generation (TIG) and Prompt Template Engineering (PTE). Besides, DQA provides a complete evaluation pipeline, featuring diverse metrics and a standardized evaluation process to ensure comprehensiveness, accuracy, and fairness. We use DQA to evaluate the database Q&A capabilities under the proposed testbed comprehensively. 

---

## Contents

This repository contains the following contents:

* [Dataset of Benchmark DQA](1_Dataset_of_Benchmark_DQA/README.md)

  This section presents the DQA dataset, which is constructed by collecting Internet data and an innovative method based on large language model generation. The dataset contains more than 240,000 Chinese-English question-answer pairs, covering almost all aspects of database knowledge. The directory contains data examples and full dataset download links, divided into two sub-directories in Chinese and English, each containing three parts: General Knowledge, Specific Product, and Specific Instance.

* [Testbed Demo](2_Testbed_Demo/README.md)

  This section is a specific demonstration of the LLM database question answering testbed. The testbed is highly modular and extensible, with a variety of basic and advanced components, designed to support various LLMs to integrate with these components to handle actual database question answering scenarios. This directory contains the implementation, usage and download link of the question classification model ([Question_Classification_Model](2_Testbed_Demo/Question_Classification_Model/README.md)), as well as the specific implementation and usage of database question classification answering ([Testbed_Backbone](2_Testbed_Demo/Testbed_Backbone/README.md)).

* [Evaluation Code of Benchmark](3_Evaluation_Code_of_Benchmark/README.md)

  This section is the complete evaluation process of DQA. The evaluation process includes a variety of indicators and a standardized evaluation process to ensure the comprehensiveness, accuracy, and fairness of the evaluation. The evaluation process supports multiple mainstream large language models and can support more models and test indicators through simple extensions. This directory provides the specific implementation and usage of the evaluation process.

* [Popular LLMs Response for DQA](4_Popular_LLMs_Response_for_DQA/README.md)

  This section shows the response of multiple popular large language models on DQA. By testing this response dataset, we can comprehensively evaluate the performance of different LLMs in database question answering tasks. This directory contains model response examples and download links for the complete response dataset. It is divided into two sub-directories in Chinese and English, each of which contains three parts: General Knowledge, Specific Product, and Specific Instance.

* [Experimental Results on DQA](5_Experimental_Results_on_DQA/README.md)

  This section presents the experimental results on DQA, including the question classification results of different methods and the answer evaluation results of different models, revealing their advantages and disadvantages.

* [Additional Materials in Footnotes of Our Paper](6_Additional_Materials_in_Footnotes_of_Our_Paper/README.md)

  This section provides additional material to the footnotes in our paper, including additional data, methodological details, or other supplementary information, such as prompts for data collection and experiments used, aiming to provide readers with more comprehensive background and understanding.

Click to view details.
