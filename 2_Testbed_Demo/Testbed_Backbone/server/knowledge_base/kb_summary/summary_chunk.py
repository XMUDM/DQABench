from typing import List, Optional

from langchain.schema.language_model import BaseLanguageModel

from server.knowledge_base.model.kb_document_model import DocumentWithVSId
from configs import (logger)
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.prompts import PromptTemplate

from langchain.docstore.document import Document
from langchain.output_parsers.regex import RegexParser
from langchain.chains.combine_documents.map_reduce import ReduceDocumentsChain, MapReduceDocumentsChain

import sys
import asyncio


class SummaryAdapter:
    _OVERLAP_SIZE: int
    token_max: int
    _separator: str = "\n\n"
    chain: MapReduceDocumentsChain

    def __init__(self, overlap_size: int, token_max: int,
                 chain: MapReduceDocumentsChain):
        self._OVERLAP_SIZE = overlap_size
        self.chain = chain
        self.token_max = token_max

    @classmethod
    def form_summary(cls,
                     llm: BaseLanguageModel,
                     reduce_llm: BaseLanguageModel,
                     overlap_size: int,
                     token_max: int = 1300):
        """
        Get instance
        :param reduce_llm: llm for merging summaries
        :param llm: llm for generating summaries
        :param overlap_size: Overlap size
        :param token_max: The maximum number of chunks. The length of each chunk is less than the token_max length. When the digest is generated for the first time, a digest longer than the token_max length will be reported.
        :return:
        """

        # This controls how each document will be formatted. Specifically,
        document_prompt = PromptTemplate(
            input_variables=["page_content"],
            template="{page_content}"
        )

        # The prompt here should take as an input variable the
        # `document_variable_name`
        prompt_template = (
            "Perform tasks according to the text. The following task information" 
            "{task_briefing}" 
            "The text content is as follows: "
            "\r\n"
            "{context}"
        )
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["task_briefing", "context"]
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        # We now define how to combine these summaries
        reduce_prompt = PromptTemplate.from_template(
            "Combine these summaries: {context}"
        )
        reduce_llm_chain = LLMChain(llm=reduce_llm, prompt=reduce_prompt)

        document_variable_name = "context"
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name
        )
        reduce_documents_chain = ReduceDocumentsChain(
            token_max=token_max,
            combine_documents_chain=combine_documents_chain,
        )
        chain = MapReduceDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name=document_variable_name,
            reduce_documents_chain=reduce_documents_chain,
            return_intermediate_steps=True
        )
        return cls(overlap_size=overlap_size,
                   chain=chain,
                   token_max=token_max)

    def summarize(self,
                  file_description: str,
                  docs: List[DocumentWithVSId] = []
                  ) -> List[Document]:

        if sys.version_info < (3, 10):
            loop = asyncio.get_event_loop()
        else:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()

            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.asummarize(file_description=file_description,
                                                       docs=docs))

    async def asummarize(self,
                         file_description: str,
                         docs: List[DocumentWithVSId] = []) -> List[Document]:

        logger.info("start summary")

        """
        This process is divided into two parts:
        1. Process each document and get a summary of each document
         map_results = self.llm_chain.apply(
            # FYI - this is parallelized and so it is fast.
            [{self.document_variable_name: d.page_content, **kwargs} for d in docs],
            callbacks=callbacks,
        )
        2. Merge the summaries of each document to get the final summary.return_intermediate_steps=True
        result, extra_return_dict = self.reduce_documents_chain.combine_docs(
            result_docs, token_max=token_max, callbacks=callbacks, **kwargs
        )
        """
        summary_combine, summary_intermediate_steps = self.chain.combine_docs(docs=docs,
                                                                              task_briefing="Describe the proximity and similarity between different methods"
                                                                                            "to help readers understand the relationship betwern them.")
        print(summary_combine)
        print(summary_intermediate_steps)

        logger.info("end summary")
        doc_ids = ",".join([doc.id for doc in docs])
        _metadata = {
            "file_description": file_description,
            "summary_intermediate_steps": summary_intermediate_steps,
            "doc_ids": doc_ids
        }
        summary_combine_doc = Document(page_content=summary_combine, metadata=_metadata)

        return [summary_combine_doc]

    def _drop_overlap(self, docs: List[DocumentWithVSId]) -> List[str]:
        """
         # Remove the overlapping part of the page_content sentence in the document
        :param docs:
        :param separator:
        :return:
        """
        merge_docs = []

        pre_doc = None
        for doc in docs:
            if len(merge_docs) == 0:
                pre_doc = doc.page_content
                merge_docs.append(doc.page_content)
                continue

            # The part of the list where the previous end overlaps with the next beginning, delete the part where the next beginning overlaps
            # Iterate and decrement the length of pre_doc, deleting the preceding characters each time.
            # Search for overlapping parts until the length of pre_doc is less than self._OVERLAP_SIZE // 2 - 2len(separator)
            for i in range(len(pre_doc), self._OVERLAP_SIZE // 2 - 2 * len(self._separator), -1):
                pre_doc = pre_doc[1:]
                if doc.page_content[:len(pre_doc)] == pre_doc:
                    merge_docs.append(doc.page_content[len(pre_doc):])
                    break

            pre_doc = doc.page_content

        return merge_docs

    def _join_docs(self, docs: List[str]) -> Optional[str]:
        text = self._separator.join(docs)
        text = text.strip()
        if text == "":
            return None
        else:
            return text


if __name__ == '__main__':

    docs = [

        '梦者有特别的作用，也就是说梦是在预卜未来。因此，梦内容的',

        '梦内容的多彩多姿以及对梦者本身所遗留的特殊印象，使他们很难想象',

        '使他们很难想象出一套系统划一的观念，而需要以其个别的价值与可靠性作各',
        '值与可靠性作各种不同的分化与聚合。因此，古代哲学家们对梦的评价也就完全'
    ]
    _OVERLAP_SIZE = 1
    separator: str = "\n\n"
    merge_docs = []
    # Remove the overlapping part of the page_content sentence in the document,
    # The part of the list where the previous end overlaps with the next beginning, delete the part where the next beginning overlaps
    pre_doc = None
    for doc in docs:
        if len(merge_docs) == 0:
            pre_doc = doc
            merge_docs.append(doc)
            continue

        # The part of the list where the previous end overlaps with the next beginning, delete the part where the next beginning overlaps
        # Iterate and decrement the length of pre_doc, deleting the preceding characters each time.
        # Search for overlapping parts until the length of pre_doc is less than _OVERLAP_SIZE-2len(separator)
        for i in range(len(pre_doc), _OVERLAP_SIZE - 2 * len(separator), -1):
            pre_doc = pre_doc[1:]
            if doc[:len(pre_doc)] == pre_doc:
                page_content = doc[len(pre_doc):]
                merge_docs.append(page_content)

                pre_doc = doc
                break

    # Merge the sentences in merge_docs into one document
    text = separator.join(merge_docs)
    text = text.strip()

    print(text)
