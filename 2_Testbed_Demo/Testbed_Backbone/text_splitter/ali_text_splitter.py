from langchain.text_splitter import CharacterTextSplitter
import re
from typing import List


class AliTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        # The use_document_segmentation parameter specifies whether the document is divided semantically. The document semantic segmentation model adopted here is nlp_bert_document-segmentation_chinese-base, which is open source by Damo Institute. The paper see https://arxiv.org/abs/2107.09278
        # If you use models for document semantic segmentation, you need to install modelscope[nlp] : pip install "modelscope[nlp]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
        # Considering the use of three models, it may not be friendly for low-configuration Gpus, so here the model load into the cpu calculation, if necessary, you can replace the device as your own graphics card id
        if self.pdf:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub('\s', " ", text)
            text = re.sub("\n\n", "", text)
        try:
            from modelscope.pipelines import pipeline
        except ImportError:
            raise ImportError(
                "Could not import modelscope python package. "
                "Please install modelscope with `pip install modelscope`. "
            )


        p = pipeline(
            task="document-segmentation",
            model='damo/nlp_bert_document-segmentation_chinese-base',
            device="cpu")
        result = p(documents=text)
        sent_list = [i for i in result["text"].split("\n\t") if i]
        return sent_list
