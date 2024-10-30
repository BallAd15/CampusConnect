from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from pinecone import Pinecone
from dotenv import load_dotenv
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from sentence_transformers import SentenceTransformer

load_dotenv() # Load .env file

pinecone_api_key = os.getenv("PINECONE_API_KEY")
embedding_model_name = "avsolatorio/GIST-all-MiniLM-L6-v2"
print(embedding_model_name)

# Embedding model
embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)

# Intialize variables
Settings.llm = None
Settings.embed_model = embed_model

# Initialize Pinecone
_pc_new_baliga = Pinecone(api_key=pinecone_api_key)
kb_index = _pc_new_baliga.Index(name="campus-connect")
kb_vector_store = PineconeVectorStore(pinecone_index=kb_index)
vector_index = VectorStoreIndex.from_vector_store(vector_store=kb_vector_store)
print(vector_index, "vector_index")
kb_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=2)
print(kb_retriever, "retriever")

def retrieve_docs(query):
    docs = kb_retriever.retrieve(query)
    return docs