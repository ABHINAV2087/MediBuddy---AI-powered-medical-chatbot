import os
import logging
from pinecone import Pinecone
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from typing import Optional, Dict, Any
from medibuddy.config import Config

logger = logging.getLogger(__name__)

class RAGService:
    _instance = None
    vectorstore: Optional[PineconeVectorStore] = None
    llm: Optional[HuggingFaceEndpoint] = None
    is_initialized = False
    pc: Optional[Pinecone] = None  # Pinecone client instance
   
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls):
        if cls.is_initialized:
            logger.info("✅ RAGService already initialized")
            return True

        try:
            # Initialize Pinecone client
            cls.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
            
            # Initialize vectorstore
            if not cls.initialize_vectorstore():
                raise RuntimeError("Vectorstore initialization failed")
            
            # Initialize LLM
            if not cls.initialize_llm():
                raise RuntimeError("LLM initialization failed")
            
            cls.is_initialized = cls.vectorstore is not None and cls.llm is not None
            
            if cls.is_initialized:
                logger.info("✅ RAGService initialized successfully")
            else:
                logger.error("❌ RAGService initialization failed")
            
            return cls.is_initialized
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAGService: {str(e)}")
            cls.is_initialized = False
            return False

    @classmethod
    def initialize_vectorstore(cls):
        try:
            if not Config.PINECONE_API_KEY:
                raise ValueError("Pinecone API key not configured")
            if not Config.PINECONE_INDEX_NAME:
                raise ValueError("Pinecone index name not configured")

            logger.info("Initializing Pinecone vectorstore...")
            
            embedding_model = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            cls.vectorstore = PineconeVectorStore.from_existing_index(
                index_name=Config.PINECONE_INDEX_NAME,
                embedding=embedding_model
            )
            
            logger.info("✅ Pinecone vectorstore loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading Pinecone vectorstore: {str(e)}")
            cls.vectorstore = None
            return False

    @classmethod
    def initialize_llm(cls):
        try:
            if not Config.HF_TOKEN:
                raise ValueError("HuggingFace token not configured")
            if not Config.HUGGINGFACE_REPO_ID:
                raise ValueError("HuggingFace repo ID not configured")

            logger.info("Initializing LLM...")
            
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = Config.HF_TOKEN
            
            cls.llm = HuggingFaceEndpoint(
                repo_id=Config.HUGGINGFACE_REPO_ID,
                temperature=0.3,
                max_new_tokens=1000,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.03
            )
            
            logger.info("✅ LLM loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading LLM: {str(e)}")
            cls.llm = None
            return False

    @classmethod
    @lru_cache(maxsize=1)
    def get_prompt_template(cls):
        template = """
        You are a medical assistant that provides accurate information based on the provided context.
        Use the medical information in the context to answer the user's question thoroughly.
        
        If you don't know the answer based on the context, say "I don't have enough medical information to answer that."
        Never make up medical information.
        
        Context: {context}
        Question: {question}
        
        Provide a detailed medical answer:
        """
        return PromptTemplate(template=template, input_variables=["context", "question"])

    @classmethod
    def clean_metadata(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        if not metadata:
            return {}
        
        cleaned = {}
        for key, value in metadata.items():
            if value and str(value).strip().lower() not in ['none', 'null', 'unknown', '']:
                cleaned[key] = str(value).strip()
        
        return cleaned

    @classmethod
    def get_retriever(cls, k: int = 5):
        """Get a retriever with configurable number of results"""
        if not cls.vectorstore:
            raise RuntimeError("Vectorstore not initialized")
        return cls.vectorstore.as_retriever(search_kwargs={'k': k})