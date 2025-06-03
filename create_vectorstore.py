from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def create_vectorstore():
    # Load PDFs
    loader = DirectoryLoader('data/', glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Create and save FAISS index
    db = FAISS.from_documents(texts, embeddings)
    os.makedirs('vectorstore/db_faiss', exist_ok=True)
    db.save_local('vectorstore/db_faiss')

if __name__ == '__main__':
    create_vectorstore()