import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    HF_TOKEN = os.getenv('HF_TOKEN')
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'gcp-starter')
    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'medical-chatbot-index')
    
    # Model configurations - Updated with working alternatives
    # Option 1: Use a text-generation compatible model (recommended)
    HUGGINGFACE_REPO_ID = os.getenv('HUGGINGFACE_REPO_ID', "microsoft/DialoGPT-medium")
    
    # Alternative models you can try:
    # "google/flan-t5-base" - Good for Q&A tasks
    # "HuggingFaceH4/zephyr-7b-beta" - Instruction-following model
    # "microsoft/DialoGPT-medium" - Conversational model
    
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    
    # LLM parameters
    TEMPERATURE = 0.5
    MAX_LENGTH = 512
    
    # Retrieval parameters
    TOP_K_RESULTS = 3
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50