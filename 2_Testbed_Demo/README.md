<div align='center'>
    <h1>Testbet Demo</h1>
</div>

<p align='center'>
    【English | <a href="README_zh.md">中文</a>】
</p>

This section is a specific demonstration of the LLM database question-answering test platform. The test platform is highly modular and extensible, with a variety of basic and advanced components, designed to support various LLMs to integrate with these components to handle actual database question-answering scenarios. The directory contains the following two parts:

* [Question_Classification_Model](Question_Classification_Model/README.md)
  
  In addition to autonomous classification using large language models, we also implemented two other classification methods. This directory contains the specific implementation and code usage of the two methods based on XLNet Transformer, and provides download links for the models.

* [Testbed_Backbone](Testbed_Backbone/README.md)
 
  Based on Langchain-Chatchat, we have written a specific implementation and demonstration of automatic classification and answering of database questions, including General Knowledge, Specific Product and Specific Instance. This directory contains the complete code implementation and usage.