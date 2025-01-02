<div align='center'>
    <h1>Testbet Demo</h1>
</div>

<p align='center'>
    【<a href="README.md">English</a> | 中文】
</p>

这部分是 LLM 数据库问答测试平台的具体演示。该测试平台高度模块化和可扩展，具备多种基本和高级组件，旨在支持各种 LLM 与这些组件集成，以处理实际的数据库问答场景。该目录包含以下两个部分：

* [Question_Classification_Model](Question_Classification_Model/README.md)
  
  除了使用大型模型进行自主分类之外，我们还实现了另外两种分类方法。该目录包含基于 XLNet Transformer 的两种方法的具体实现和代码使用方法，并提供了模型的下载链接。

* [Testbed_Backbone](Testbed_Backbone/README_zh.md)
 
  我们基于Langchain-Chatchat，编写了对数据库问题自动分类和解答的具体实现和演示，包括General Knowledge，Specific Product 和 Specific Instance 等类别。该目录包含了完整的代码实现和使用方法。
