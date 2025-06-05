import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    HF_TOKEN = os.getenv('HF_TOKEN')
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'gcp-starter')
    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'medical-chatbot-index')
    HUGGINGFACE_REPO_ID = os.getenv('HUGGINGFACE_REPO_ID', "microsoft/DialoGPT-medium")
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    TEMPERATURE = 0.5
    MAX_LENGTH = 512
    TOP_K_RESULTS = 3
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
