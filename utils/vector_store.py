from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore as LangchainPinecone
import os
import time
from config import Config

class PineconeVectorStore:
    def __init__(self):
        self.config = Config()
        self.pc = Pinecone(api_key=self.config.PINECONE_API_KEY)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.config.EMBEDDING_MODEL_NAME
        )
        self.index = None
        self.vectorstore = None
        
    def initialize_index(self):
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.config.PINECONE_INDEX_NAME not in existing_indexes:
                self.pc.create_index(
                    name=self.config.PINECONE_INDEX_NAME,
                    dimension=384,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                print(f"Created new index: {self.config.PINECONE_INDEX_NAME}")
                print("Waiting for index to be ready...")
                time.sleep(10)
            
            self.index = self.pc.Index(self.config.PINECONE_INDEX_NAME)
            print(f"Connected to index: {self.config.PINECONE_INDEX_NAME}")
            
        except Exception as e:
            print(f"Error initializing Pinecone index: {str(e)}")
            raise e
    
    def load_documents(self, data_path="data/"):
        try:
            if not os.path.exists(data_path):
                print(f"Data directory {data_path} does not exist")
                return []
                
            loader = DirectoryLoader(
                data_path,
                glob='*.pdf',
                loader_cls=PyPDFLoader
            )
            documents = loader.load()
            print(f"Loaded {len(documents)} document pages")
            return documents
        except Exception as e:
            print(f"Error loading documents: {str(e)}")
            return []
    
    def create_chunks(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} text chunks")
        return chunks
    
    def create_vectorstore(self, documents):
        try:
            chunks = self.create_chunks(documents)
            
            if not chunks:
                print("No chunks created from documents")
                return False
            
            self.vectorstore = LangchainPinecone.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                index_name=self.config.PINECONE_INDEX_NAME
            )
            print("Vector store created successfully")
            return True
            
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            return False
    
    def get_vectorstore(self):
        if self.vectorstore is None:
            try:
                index_stats = self.index.describe_index_stats()
                total_vector_count = index_stats.get('total_vector_count', 0)
                
                if total_vector_count == 0:
                    print("Index exists but is empty. Please run with SETUP_PIPELINE=true first.")
                    return None
                
                self.vectorstore = LangchainPinecone.from_existing_index(
                    index_name=self.config.PINECONE_INDEX_NAME,
                    embedding=self.embedding_model
                )
                print(f"Connected to existing vector store with {total_vector_count} vectors")
            except Exception as e:
                print(f"Error connecting to vector store: {str(e)}")
                return None
        
        return self.vectorstore
    
    def setup_complete_pipeline(self, data_path="data/"):
        try:
            self.initialize_index()
            documents = self.load_documents(data_path)
            
            if not documents:
                print("No documents found. Make sure PDF files are in the data/ directory.")
                return False
            
            success = self.create_vectorstore(documents)
            if success:
                print("Complete pipeline setup successful")
                return True
            
            print("Pipeline setup failed")
            return False
            
        except Exception as e:
            print(f"Error in complete pipeline: {str(e)}")
            return False
