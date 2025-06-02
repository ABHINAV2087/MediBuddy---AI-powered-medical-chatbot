import os
import logging
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_vectorstore():
    """Build vectorstore from PDF documents"""
    
    DATA_PATH = "data/"
    DB_FAISS_PATH = "vectorstore/db_faiss"
    
    # Check if data directory exists
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data directory not found: {DATA_PATH}")
        return False
    
    # Check if PDFs exist
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    if not pdf_files:
        logger.error(f"No PDF files found in {DATA_PATH}")
        return False
    
    logger.info(f"Found {len(pdf_files)} PDF files: {pdf_files}")
    
    try:
        # Load PDF documents
        logger.info("Loading PDF documents...")
        loader = DirectoryLoader(
            DATA_PATH,
            glob='*.pdf',
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from PDFs")
        
        # Create text chunks
        logger.info("Creating text chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        text_chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(text_chunks)} text chunks")
        
        # Create embeddings
        logger.info("Creating embeddings...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create vectorstore
        logger.info("Building FAISS vectorstore...")
        os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
        db = FAISS.from_documents(text_chunks, embedding_model)
        db.save_local(DB_FAISS_PATH)
        
        logger.info(f"✅ Vectorstore successfully created at {DB_FAISS_PATH}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error building vectorstore: {str(e)}")
        return False

if __name__ == "__main__":
    success = build_vectorstore()
    exit(0 if success else 1)