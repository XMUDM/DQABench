<div align='center'>
    <h1>DQA-Bench</h1>
</div>

<p align='center'>
    【<a href="README.md">English</a> | 中文】
</p>

DQA 是第一个全面的数据库问答基准，其数据集采用互联网数据采集和创新的基于大型语言模型生成的方法构建。我们还提出一个基于 LLM 的全面的 DQA 数据库问答测试平台。该测试平台高度模块化且可扩展，具备各种基本和高级组件，例如问题分类路由 (QCR)、检索增强生成 (RAG)、工具调用生成 (TIG) 和提示模板工程 (PTE)。此外，DQA 提供了完整的评估流程，具有多种指标和标准化评估流程，以确保全面性、准确性和公平性。我们使用 DQA 全面评估所提出的测试平台下的数据库问答功能。

---

## 目录

本仓库共包含以下内容：

* [Dataset of Benchmark DQA](1_Dataset_of_Benchmark_DQA/README.md)

  这部分展示了 DQA 的数据集，其构建采用了互联网数据采集和创新的基于大型语言模型生成的方法。该数据集包含了超过240,000对中英文问答对，覆盖了数据库知识的几乎所有方面。该目录包含数据示例和完整数据集下载链接等内容，分为中英两个子目录，每个分别包含General Knowledge，Specific Product，Specific Instance三个部分。

* [Testbed Demo](2_Testbed_Demo/README.md)

  这部分是 LLM 数据库问答测试平台的具体演示。该测试平台高度模块化和可扩展，具备多种基本和高级组件，旨在支持各种 LLM 与这些组件集成，以处理实际的数据库问答场景。该目录包含问题分类模型的实现，使用方法和下载链接（[Question_Classification_Model](2_Testbed_Demo/Question_Classification_Model/README.md)），以及对数据库问题分类解答的具体实现和使用方法（[Testbed_Backbone](2_Testbed_Demo/Testbed_Backbone/README.md)）。

* [Evaluation Code of Benchmark](3_Evaluation_Code_of_Benchmark/README.md)

  这部分是 DQA 的完整评估流程。评估流程包括多样化的指标和标准化的评估过程，以确保评估的全面性、准确性和公平性。该评估流程支持多个主流的大语言模型，并可以通过简单扩展支持更多模型和测试指标。该目录提供了评估流程的具体实现和使用方法。

* [Popular LLMs Response for DQA](4_Popular_LLMs_Response_for_DQA/README.md)
 
  这部分内容展示了多个主流的大型语言模型在 DQA 上的回答。通过对该回答数据集的测试，可以全面评估不同 LLM 在数据库问答任务中的表现。该目录包含模型回答示例和完整回答数据集下载链接等内容，分为中英两个子目录，每个分别包含General Knowledge，Specific Product，Specific Instance三个部分。

* [Experimental Results on DQA](5_Experimental_Results_on_DQA/README.md)

  这部分展示了在 DQA 上的实验结果，包括不同方法的问题分类结果和不同模型的回答测评结果，揭示了它们的优缺点。

* [Additional Materials in Footnotes of Our Paper](6_Additional_Materials_in_Footnotes_of_Our_Paper/README.md)

  这部分内容提供了我们论文中脚注的附加材料，包括额外的数据、方法细节或其他补充信息，如数据收集和实验中使用的prompt等内容，旨在为读者提供更全面的背景和理解。

点击查看详细内容。

## 引用

如果您喜欢这个项目，欢迎引用我们的论文([paper link](https://arxiv.org/abs/2409.04475))，并为项目加上星标。

```bibtex
@misc{zheng2024dqa,
      title={Revolutionizing Database Q&A with Large Language Models: Comprehensive Benchmark and Evaluation}, 
      author={Yihang Zheng, Bo Li, Zhenghao Lin, Yi Luo, Xuanhe Zhou, Chen Lin, Jinsong Su, Guoliang Li, Shifu Li},
      year={2024},
      eprint={2409.04475},
      archivePrefix={arXiv},
      primaryClass={cs.DB}
}
```
