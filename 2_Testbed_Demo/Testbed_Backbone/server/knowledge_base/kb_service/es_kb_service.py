from typing import List
import os
import shutil
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from configs import KB_ROOT_PATH, EMBEDDING_MODEL, EMBEDDING_DEVICE, CACHED_VS_NUM
from server.knowledge_base.kb_service.base import KBService, SupportedVSType
from server.utils import load_local_embeddings
from elasticsearch import Elasticsearch,BadRequestError
from configs import logger
from configs import kbs_config

class ESKBService(KBService):

    def do_init(self):
        self.kb_path = self.get_kb_path(self.kb_name)
        self.index_name = self.kb_path.split("/")[-1]
        self.IP = kbs_config[self.vs_type()]['host']
        self.PORT = kbs_config[self.vs_type()]['port']
        self.user = kbs_config[self.vs_type()].get("user",'')
        self.password = kbs_config[self.vs_type()].get("password",'')
        self.dims_length = kbs_config[self.vs_type()].get("dims_length",None)
        self.embeddings_model = load_local_embeddings(self.embed_model, EMBEDDING_DEVICE)
        try:
            if self.user != "" and self.password != "":
                self.es_client_python =  Elasticsearch(f"http://{self.IP}:{self.PORT}",
                basic_auth=(self.user,self.password))
            else:
                logger.warning("ES user name and password are not configured")
                self.es_client_python = Elasticsearch(f"http://{self.IP}:{self.PORT}")
        except ConnectionError:
            logger.error("Connecting to Elasticsearch failed!")
            raise ConnectionError
        except Exception as e:
            logger.error(f"Error occurred: {e}")
            raise e
        try:
            # First try to create it through es_client_python
            self.es_client_python.indices.create(index=self.index_name)
        except BadRequestError as e:
            logger.error("Failed to create index, try again")
            logger.error(e)

        try:
            if self.user != "" and self.password != "":
                self.db_init = ElasticsearchStore(
                es_url=f"http://{self.IP}:{self.PORT}",
                index_name=self.index_name,
                query_field="context",
                vector_query_field="dense_vector",
                embedding=self.embeddings_model,
                es_user=self.user,
                es_password=self.password
            )
            else:
                logger.warning("ES user name and password are not configured")
                self.db_init = ElasticsearchStore(
                    es_url=f"http://{self.IP}:{self.PORT}",
                    index_name=self.index_name,
                    query_field="context",
                    vector_query_field="dense_vector",
                    embedding=self.embeddings_model,
                )
        except ConnectionError:
            print("### Failed to initialize Elasticsearch!")
            logger.error("### Failed to initialize Elasticsearch!")
            raise ConnectionError
        except Exception as e:
            logger.error(f"Error occurred: {e}")
            raise e
        try:
            # Try creating the index via db_init
            self.db_init._create_index_if_not_exists(
                                                     index_name=self.index_name,
                                                     dims_length=self.dims_length
                                                     )
        except Exception as e:
            logger.error("Failed to create index...")
            logger.error(e)
            # raise e 
            
        

    @staticmethod
    def get_kb_path(knowledge_base_name: str):
        return os.path.join(KB_ROOT_PATH, knowledge_base_name)

    @staticmethod
    def get_vs_path(knowledge_base_name: str):
        return os.path.join(ESKBService.get_kb_path(knowledge_base_name), "vector_store")

    def do_create_kb(self):
        if os.path.exists(self.doc_path):
            if not os.path.exists(os.path.join(self.kb_path, "vector_store")):
                os.makedirs(os.path.join(self.kb_path, "vector_store"))
            else:
                logger.warning("directory `vector_store` already exists.")

    def vs_type(self) -> str:
        return SupportedVSType.ES

    def _load_es(self, docs, embed_model):
        try:
            # Connect + write documents simultaneously
            if self.user != "" and self.password != "":
                self.db = ElasticsearchStore.from_documents(
                        documents=docs,
                        embedding=embed_model,
                        es_url= f"http://{self.IP}:{self.PORT}",
                        index_name=self.index_name,
                        distance_strategy="COSINE",
                        query_field="context",
                        vector_query_field="dense_vector",
                        verify_certs=False,
                        es_user=self.user,
                        es_password=self.password
                    )
            else:
                self.db = ElasticsearchStore.from_documents(
                        documents=docs,
                        embedding=embed_model,
                        es_url= f"http://{self.IP}:{self.PORT}",
                        index_name=self.index_name,
                        distance_strategy="COSINE",
                        query_field="context",
                        vector_query_field="dense_vector",
                        verify_certs=False)
        except ConnectionError as ce:
            print(ce)
            print("Connecting to Elasticsearch failed!")
            logger.error("Connecting to Elasticsearch failed!")
        except Exception as e:
            logger.error(f"Error occurred : {e}")
            print(e)



    def do_search(self, query:str, top_k: int, score_threshold: float):
        # Text Similarity Retrieval
        docs = self.db_init.similarity_search_with_score(query=query,
                                         k=top_k)
        return docs

    def del_doc_by_ids(self, ids: List[str]) -> bool:
        for doc_id in ids:
            try:
                self.es_client_python.delete(index=self.index_name,
                                            id=doc_id,
                                            refresh=True)
            except Exception as e:
                logger.error(f"ES Docs Delete Error! {e}")

    def do_delete_doc(self, kb_file, **kwargs):
        if self.es_client_python.indices.exists(index=self.index_name):
            # Delete the index from the vector database (the document name is Keyword)
            query = {
                "query": {
                    "term": {
                        "metadata.source.keyword": kb_file.filepath
                    }
                }
            }
            # Note that the size is set. By default, 10 items are returned.
            search_results = self.es_client_python.search(body=query, size=50)
            delete_list = [hit["_id"] for hit in search_results['hits']['hits']]
            if len(delete_list) == 0:
                return None
            else:
                for doc_id in delete_list:
                    try:
                        self.es_client_python.delete(index=self.index_name,
                                                     id=doc_id,
                                                     refresh=True)
                    except Exception as e:
                        logger.error(f"ES Docs Delete Error! {e}")

            # self.db_init.delete(ids=delete_list)
            #self.es_client_python.indices.refresh(index=self.index_name)


    def do_add_doc(self, docs: List[Document], **kwargs):
        '''Adding files to the knowledge base'''
        print(f"server.knowledge_base.kb_service.es_kb_service.do_add_doc The input docs parameter length is: {len(docs)}")
        print("*"*100)
        self._load_es(docs=docs, embed_model=self.embeddings_model)
        # Get id and source, format: [{"id": str, "metadata": dict}, ...]
        print("Data was written successfully.")
        print("*"*100)
        
        if self.es_client_python.indices.exists(index=self.index_name):
            file_path = docs[0].metadata.get("source")
            query = {
                "query": {
                    "term": {
                        "metadata.source.keyword": file_path
                    }
                }
            }
            search_results = self.es_client_python.search(body=query)
            if len(search_results["hits"]["hits"]) == 0:
                raise ValueError("The number of recalled elements is 0")
        info_docs = [{"id":hit["_id"], "metadata": hit["_source"]["metadata"]} for hit in search_results["hits"]["hits"]]
        return info_docs


    def do_clear_vs(self):
        """Delete all vectors from the knowledge base"""
        if self.es_client_python.indices.exists(index=self.kb_name):
            self.es_client_python.indices.delete(index=self.kb_name)


    def do_drop_kb(self):
        """Delete a Knowledge Base"""
        # self.kb_file: Knowledge Base Path
        if os.path.exists(self.kb_path):
            shutil.rmtree(self.kb_path)








