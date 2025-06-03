import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

class Config:
    # Vectorstore configuration
    DB_FAISS_PATH = os.environ.get("DB_FAISS_PATH", "vectorstore/db_faiss")
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # LLM configuration
    HUGGINGFACE_REPO_ID = os.environ.get("HUGGINGFACE_REPO_ID", "mistralai/Mixtral-8x7B-Instruct-v0.1")
    HF_TOKEN = os.environ.get("HF_TOKEN")
    
    # API configuration
    MAX_SOURCES = int(os.environ.get("MAX_SOURCES", "5"))
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", "500"))
    CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "").split(",")
    
    # Server configuration
    FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    FLASK_HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
    FLASK_PORT = int(os.environ.get("FLASK_PORT", "5000"))