

import json
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer
from langchain_milvus import Milvus


# ──────────────────────────────────────────────────────────────────────────────
# WRAPPER FOR SBERT → provides both embed_documents() and embed_query()
# ──────────────────────────────────────────────────────────────────────────────
class SBERTEmbedding:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0]


# ──────────────────────────────────────────────────────────────────────────────
# RAGRetriever USING SBERTEmbedding & storing "text" field in Milvus
# ──────────────────────────────────────────────────────────────────────────────
class RAGRetriever:
    def __init__(
        self,
        document_dir="rag_documents/knowledge.json",
        collection_name="document_collection",
        embedding_model_name="nomic-ai/nomic-embed-text-v1",
        milvus_host="localhost",
        milvus_port="19530",
    ):
        self.document_dir = document_dir
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port

        # ─ SBERTEmbedding instance provides both embed_documents() and embed_query() ─
        self.embedding_function = SBERTEmbedding(self.embedding_model_name)

    def load_and_split_documents(self):
        with open(self.document_dir, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        documents = [
            Document(page_content=entry["content"], metadata={"title": entry["title"]})
            for entry in raw_data
        ]
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(documents)

    def setup_milvus(self, texts: list[str], embeddings: np.ndarray):
        # Connect to Milvus
        connections.connect("default", host=self.milvus_host, port=self.milvus_port)
        dim = embeddings.shape[1]

        if utility.has_collection(self.collection_name):
            collection = Collection(self.collection_name)
            existing_fields = {field.name for field in collection.schema.fields}
            # Ensure schema has "text" and "vector" fields; otherwise drop
            if not {"text", "vector"}.issubset(existing_fields):
                print("Schema mismatch. Dropping existing collection.")
                collection.drop()
                collection = None
        else:
            collection = None

        if collection is None:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            ]
            schema = CollectionSchema(fields)
            collection = Collection(self.collection_name, schema)

        # Insert data in the order: [texts, vectors]
        data = [texts, embeddings.tolist()]
        collection.insert(data)
        collection.flush()

    def index_documents(self):
        docs = self.load_and_split_documents()
        texts = [doc.page_content for doc in docs]
        embeddings = self.embedding_function.embed_documents(texts)
        self.setup_milvus(texts, np.array(embeddings))
        print(f"Indexed {len(texts)} documents into Milvus collection '{self.collection_name}'.")

    def retrieve(self, query, top_k=3):
        connections.connect("default", host=self.milvus_host, port=self.milvus_port)

        # Tell LangChain that our vector column is named "vector" and our text column is "text"
        vector_store = Milvus(
            embedding_function=self.embedding_function,
            collection_name=self.collection_name,
            connection_args={"host": self.milvus_host, "port": self.milvus_port},
            text_field="text",
        )

        retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        retrieved_docs = retriever.invoke(query)

        #print(f"\nTop {top_k} results for query: '{query}'")
        combined_text = ""
        for i, doc in enumerate(retrieved_docs, 1):
            # print(f"\nResult {i}:")
            # print(f"Content: {doc.page_content}...")  # Print a snippet of content
            combined_text += doc.page_content + "\n"

        return combined_text

    def update_documents(self):
        print(f"Updating Milvus collection '{self.collection_name}'...")
        self.delete_collection()
        self.index_documents()
        print("Update completed.")

    def delete_collection(self):
        connections.connect("default", host=self.milvus_host, port=self.milvus_port)
        if utility.has_collection(self.collection_name):
            Collection(self.collection_name).drop()
            print(f"Deleted Milvus collection '{self.collection_name}'.")
        else:
            print(f"Collection '{self.collection_name}' does not exist.")


if __name__ == "__main__":
    import time
    retriever = RAGRetriever()
    # 1) Index documents at least once
    #retriever.delete_collection()
    #retriever.index_documents()
    # 2) Then retrieve
    start_time = time.time()
    retriever.retrieve("How does RSI indicate market trends?")
    print(f'total time: {time.time()-start_time}')
    # retriever.update_documents()
    # 
