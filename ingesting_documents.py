import os
import torch
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import download_loader
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv() # Load .env file
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index_name = "campus-connect"
pinecone_index = pc.Index(index_name)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)


# Load Documents
PagedCSVReader = download_loader("PagedCSVReader")
loader = PagedCSVReader()
documents = loader.load_data(file="documents/campus_connect_FAQ.csv")
print(documents[0])

# Embedding Model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Pipeline
pipeline = IngestionPipeline(
    transformations=[
        TitleExtractor(),
        embed_model,
    ],
    vector_store=vector_store
)

# Upserting documents
nodes = pipeline.run(documents=documents)
print("Documents upserted successfully")