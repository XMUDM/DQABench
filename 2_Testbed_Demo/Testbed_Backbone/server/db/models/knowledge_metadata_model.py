from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, JSON, func

from server.db.base import Base


class SummaryChunkModel(Base):
    """
    The chunk summary model is used to store the chunk fragments for each doc_id in file_doc.
    Data Sources:
        User input: Users upload files and can fill in the description of the file. The generated doc_id in file_doc is stored in summary_chunk. The program automatically splits the page number information stored in the meta_data field information of the file_doc table, splits it by page number, generates summary text with a custom prompt, and stores the doc_id associated with the corresponding page number in summary_chunk.
    Follow-up tasks:
        Vector library construction: Create an index for summary_context in the database table summary_chunk to build a vector library. meta_data is the metadata of the vector library (doc_ids)
        Semantic association: Calculate semantic similarity based on the description entered by the user and the automatically segmented summary text

    """
    __tablename__ = 'summary_chunk'
    id = Column(Integer, primary_key=True, autoincrement=True, comment='ID')
    kb_name = Column(String(50), comment='Knowledge Base Name')
    summary_context = Column(String(255), comment='Summary text')
    summary_id = Column(String(255), comment='Summary vector id')
    doc_ids = Column(String(1024), comment="Vector library id association list")
    meta_data = Column(JSON, default={})

    def __repr__(self):
        return (f"<SummaryChunk(id='{self.id}', kb_name='{self.kb_name}', summary_context='{self.summary_context}',"
                f" doc_ids='{self.doc_ids}', metadata='{self.metadata}')>")
