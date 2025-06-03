from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from medibuddy.config import Config
import os

def create_vectorstore():
    # Load PDFs
    loader = DirectoryLoader('data/', glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    # Create and save to Pinecone
    PineconeVectorStore.from_documents(
        documents=texts,
        embedding=embeddings,
        index_name=Config.PINECONE_INDEX_NAME,
        pinecone_api_key=Config.PINECONE_API_KEY
    )

if __name__ == '__main__':
    create_vectorstore()