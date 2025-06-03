import os
import logging
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from typing import Optional, Dict, Any

# Add this import - adjust the path based on your project structure
from medibuddy.config import Config  # or wherever your Config class is defined

logger = logging.getLogger(__name__)

class RAGService:
    _instance = None
    vectorstore = None
    llm = None
    is_initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls):
        if cls.is_initialized:
            return True

        try:
            # Initialize vectorstore
            cls.initialize_vectorstore()
            
            # Initialize LLM
            cls.initialize_llm()
            
            cls.is_initialized = cls.vectorstore is not None and cls.llm is not None
            return cls.is_initialized
        except Exception as e:
            logger.error(f"Failed to initialize RAGService: {str(e)}")
            cls.is_initialized = False
            return False

    @classmethod
    def initialize_vectorstore(cls):
        try:
            embedding_model = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}
            )
            
            # Check if vectorstore path exists
            if not os.path.exists(Config.DB_FAISS_PATH):
                raise FileNotFoundError(f"Vectorstore path not found: {Config.DB_FAISS_PATH}")
            
            # Check for required files
            required_files = ['index.faiss', 'index.pkl']
            for file in required_files:
                if not os.path.exists(os.path.join(Config.DB_FAISS_PATH, file)):
                    raise FileNotFoundError(f"Required vectorstore file missing: {file}")

            cls.vectorstore = FAISS.load_local(
                Config.DB_FAISS_PATH, 
                embedding_model, 
                allow_dangerous_deserialization=True
            )
            logger.info("✅ Vectorstore loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading vectorstore: {str(e)}")
            cls.vectorstore = None
            return False

    @classmethod
    def initialize_llm(cls):
        try:
            if not Config.HF_TOKEN:
                raise ValueError("HuggingFace token not found in environment variables")
            
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