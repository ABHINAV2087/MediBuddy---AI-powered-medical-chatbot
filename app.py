import os
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv
import uuid

load_dotenv(find_dotenv())

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

CORS(app, origins=["http://localhost:3000", "http://localhost:3001", "https://medibuddy-ai-powered-medical-chatbot.onrender.com"])

class Config:
    DB_FAISS_PATH = os.environ.get("DB_FAISS_PATH", "vectorstore/db_faiss")
    HUGGINGFACE_REPO_ID = os.environ.get("HUGGINGFACE_REPO_ID", "mistralai/Mixtral-8x7B-Instruct-v0.1")
    HF_TOKEN = os.environ.get("HF_TOKEN")
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    MAX_SOURCES = int(os.environ.get("MAX_SOURCES", "5"))
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", "500"))
    DATA_PATH = os.environ.get("DATA_PATH", "data/")

vectorstore = None
llm = None
is_initialized = False
initialization_errors = []

def build_vectorstore_from_data():
    """Build vectorstore from PDF documents in data directory"""
    try:
        from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        logger.info(f"🔨 Building vectorstore from data in: {Config.DATA_PATH}")
        
        # Check if data directory exists
        if not os.path.exists(Config.DATA_PATH):
            logger.error(f"❌ Data directory not found: {Config.DATA_PATH}")
            return False
        
        # Check for PDF files
        pdf_files = [f for f in os.listdir(Config.DATA_PATH) if f.endswith('.pdf')]
        if not pdf_files:
            logger.error(f"❌ No PDF files found in {Config.DATA_PATH}")
            return False
        
        logger.info(f"📚 Found {len(pdf_files)} PDF files: {pdf_files}")
        
        # Load PDF documents
        loader = DirectoryLoader(
            Config.DATA_PATH,
            glob='*.pdf',
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        logger.info(f"📖 Loaded {len(documents)} pages from PDFs")
        
        if not documents:
            logger.error("❌ No documents loaded from PDFs")
            return False
        
        # Create text chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        text_chunks = text_splitter.split_documents(documents)
        logger.info(f"✂️ Created {len(text_chunks)} text chunks")
        
        if not text_chunks:
            logger.error("❌ No text chunks created")
            return False
        
        # Create embeddings
        logger.info(f"🔗 Creating embeddings with model: {Config.EMBEDDING_MODEL}")
        embedding_model = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL
        )
        
        # Create and save vectorstore
        logger.info("💾 Building and saving FAISS vectorstore...")
        os.makedirs(os.path.dirname(Config.DB_FAISS_PATH), exist_ok=True)
        db = FAISS.from_documents(text_chunks, embedding_model)
        db.save_local(Config.DB_FAISS_PATH)
        
        logger.info(f"✅ Vectorstore successfully created at {Config.DB_FAISS_PATH}")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Missing required libraries for PDF processing: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error building vectorstore: {str(e)}")
        return False

class RAGService:
    
    @staticmethod
    def initialize_vectorstore():
        global vectorstore, initialization_errors
        try:
            logger.info(f"🔍 Attempting to load vectorstore from: {Config.DB_FAISS_PATH}")
            
            # Check if the vectorstore path exists
            if not os.path.exists(Config.DB_FAISS_PATH):
                logger.warning(f"⚠️ Vectorstore directory not found: {Config.DB_FAISS_PATH}")
                logger.info("🔨 Attempting to build vectorstore from source data...")
                
                if not build_vectorstore_from_data():
                    error_msg = "Failed to build vectorstore from source data"
                    logger.error(f"❌ {error_msg}")
                    initialization_errors.append(error_msg)
                    return False
            
            # List contents of the vectorstore directory for debugging
            try:
                contents = os.listdir(Config.DB_FAISS_PATH)
                logger.info(f"📁 Vectorstore directory contents: {contents}")
                
                # Check for required FAISS files
                required_files = ['index.faiss', 'index.pkl']
                missing_files = [f for f in required_files if f not in contents]
                if missing_files:
                    logger.warning(f"⚠️ Missing FAISS files: {missing_files}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not list directory contents: {e}")
            
            # Load embedding model
            embedding_model = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
            logger.info(f"✅ Embedding model loaded: {Config.EMBEDDING_MODEL}")
            
            # Load vectorstore
            vectorstore = FAISS.load_local(
                Config.DB_FAISS_PATH, 
                embedding_model, 
                allow_dangerous_deserialization=True
            )
            logger.info("✅ Vectorstore loaded successfully")
            
            # Test vectorstore with a simple search
            try:
                test_docs = vectorstore.similarity_search("test", k=1)
                logger.info(f"✅ Vectorstore test successful - found {len(test_docs)} documents")
            except Exception as e:
                logger.warning(f"⚠️ Vectorstore test failed: {e}")
            
            return True
            
        except FileNotFoundError as e:
            error_msg = f"Vectorstore files not found: {str(e)}"
            logger.error(f"❌ {error_msg}")
            initialization_errors.append(error_msg)
            return False
        except Exception as e:
            error_msg = f"Error loading vectorstore: {str(e)}"
            logger.error(f"❌ {error_msg}")
            initialization_errors.append(error_msg)
            return False

    @staticmethod
    def initialize_llm():
        global llm, initialization_errors
        try:
            if not Config.HF_TOKEN:
                error_msg = "HuggingFace token not found. Please set HF_TOKEN environment variable."
                logger.error(f"❌ {error_msg}")
                initialization_errors.append(error_msg)
                return False
            
            # Validate token format (should start with 'hf_')
            if not Config.HF_TOKEN.startswith('hf_'):
                error_msg = "Invalid HuggingFace token format. Token should start with 'hf_'"
                logger.error(f"❌ {error_msg}")
                initialization_errors.append(error_msg)
                return False
            
            logger.info(f"🤖 Initializing LLM: {Config.HUGGINGFACE_REPO_ID}")
            
            # Set environment variable for HuggingFace
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = Config.HF_TOKEN
            
            llm = HuggingFaceEndpoint(
                repo_id=Config.HUGGINGFACE_REPO_ID,
                temperature=0.3,
                max_new_tokens=1000,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.03,
                huggingfacehub_api_token=Config.HF_TOKEN
            )
            logger.info("✅ LLM loaded successfully")
            
            # Test LLM with a simple query
            try:
                test_response = llm.invoke("Hello")
                logger.info("✅ LLM test successful")
            except Exception as e:
                logger.warning(f"⚠️ LLM test failed: {e}")
            
            return True
            
        except Exception as e:
            error_msg = f"Error loading LLM: {str(e)}"
            logger.error(f"❌ {error_msg}")
            initialization_errors.append(error_msg)
            return False

    @staticmethod
    def create_custom_prompt():
        template = """
        You are a helpful medical assistant that answers questions based on the provided medical context.
        Use the pieces of information provided in the context to answer the user's question comprehensively and accurately.
        
        If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
        Provide detailed and complete answers when the information is available in the context.
        Always prioritize patient safety and recommend consulting healthcare professionals when appropriate.
        
        Context: {context}
        Question: {question}
        
        Answer:
        """
        return PromptTemplate(template=template, input_variables=["context", "question"])

    @staticmethod
    def clean_metadata(metadata):
        if not metadata:
            return {}
        
        cleaned_metadata = {}
        relevant_fields = {
            'source': 'Source',
            'page': 'Page',
            'title': 'Title',
            'author': 'Author',
            'subject': 'Subject',
            'creation_date': 'Created',
            'modification_date': 'Modified',
            'file_name': 'File Name',
            'document_type': 'Type'
        }
        
        for field, display_name in relevant_fields.items():
            if field in metadata and metadata[field]:
                value = str(metadata[field]).strip()
                
                if field == 'source':
                    if '/' in value or '\\' in value:
                        value = os.path.basename(value)
                    if '.' in value:
                        name, ext = os.path.splitext(value)
                        cleaned_metadata['File Name'] = name
                        cleaned_metadata['Type'] = ext.upper().replace('.', '') if ext else 'Unknown'
                    else:
                        cleaned_metadata[display_name] = value
                
                elif field in ['page', 'page_label']:
                    if value.isdigit():
                        cleaned_metadata['Page'] = f"Page {value}"
                    elif value:
                        cleaned_metadata['Page'] = value
                
                elif field in ['creation_date', 'modification_date', 'creationdate', 'moddate']:
                    if ':' in value and len(value) > 10:
                        date_part = value.split('T')[0] if 'T' in value else value.split()[0]
                        if len(date_part) >= 10:
                            cleaned_metadata[display_name] = date_part
                    elif value:
                        cleaned_metadata[display_name] = value
                
                elif field in ['title', 'subject', 'author']:
                    if len(value) > 0 and value.lower() not in ['unknown', 'null', 'none', '']:
                        cleaned_metadata[display_name] = value
        
        # Remove duplicate source information
        if 'Source' in cleaned_metadata and 'File Name' in cleaned_metadata:
            if cleaned_metadata['Source'].lower() == cleaned_metadata['File Name'].lower():
                del cleaned_metadata['Source']
        
        return cleaned_metadata

def log_environment_info():
    """Log important environment information for debugging"""
    logger.info("🔧 Environment Configuration:")
    logger.info(f"  - DB_FAISS_PATH: {Config.DB_FAISS_PATH}")
    logger.info(f"  - DATA_PATH: {Config.DATA_PATH}")
    logger.info(f"  - HUGGINGFACE_REPO_ID: {Config.HUGGINGFACE_REPO_ID}")
    logger.info(f"  - HF_TOKEN: {'✅ Set' if Config.HF_TOKEN else '❌ Not Set'}")
    logger.info(f"  - EMBEDDING_MODEL: {Config.EMBEDDING_MODEL}")
    logger.info(f"  - Working Directory: {os.getcwd()}")
    
    # List current directory contents
    try:
        contents = os.listdir('.')
        logger.info(f"  - Current Directory Contents: {contents}")
    except Exception as e:
        logger.warning(f"  - Could not list current directory: {e}")
    
    # Check data directory
    if os.path.exists(Config.DATA_PATH):
        try:
            data_contents = os.listdir(Config.DATA_PATH)
            pdf_files = [f for f in data_contents if f.endswith('.pdf')]
            logger.info(f"  - Data Directory Contents: {data_contents}")
            logger.info(f"  - PDF Files Found: {pdf_files}")
        except Exception as e:
            logger.warning(f"  - Could not list data directory: {e}")
    else:
        logger.warning(f"  - Data directory does not exist: {Config.DATA_PATH}")

def initialize_services():
    global is_initialized, initialization_errors
    logger.info("🚀 Initializing RAG Services...")
    
    # Clear previous errors
    initialization_errors = []
    
    # Log environment info
    log_environment_info()
    
    # Initialize vectorstore (will auto-build if needed)
    vectorstore_ok = RAGService.initialize_vectorstore()
    
    # Initialize LLM
    llm_ok = RAGService.initialize_llm()
    
    is_initialized = vectorstore_ok and llm_ok
    
    if is_initialized:
        logger.info("✅ All services initialized successfully!")
    else:
        logger.error("❌ Failed to initialize services")
        logger.error("❌ Initialization errors:")
        for error in initialization_errors:
            logger.error(f"  - {error}")
    
    return is_initialized

def require_initialization(f):
    def decorated_function(*args, **kwargs):
        if not is_initialized:
            return jsonify({
                'success': False,
                'error': 'Services not initialized. Please check server logs.',
                'error_code': 'SERVICE_NOT_INITIALIZED',
                'initialization_errors': initialization_errors
            }), 503
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'success': True,
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'services': {
            'vectorstore_loaded': vectorstore is not None,
            'llm_loaded': llm is not None,
            'initialized': is_initialized
        },
        'config': {
            'db_faiss_path': Config.DB_FAISS_PATH,
            'data_path': Config.DATA_PATH,
            'hf_token_set': Config.HF_TOKEN is not None,
            'working_directory': os.getcwd()
        },
        'initialization_errors': initialization_errors if not is_initialized else []
    })

@app.route('/api/debug', methods=['GET'])
def debug_info():
    """Debug endpoint to help troubleshoot deployment issues"""
    try:
        debug_info = {
            'success': True,
            'environment': {
                'working_directory': os.getcwd(),
                'python_path': os.environ.get('PYTHONPATH', 'Not Set'),
                'hf_token_set': Config.HF_TOKEN is not None,
                'hf_token_valid_format': Config.HF_TOKEN.startswith('hf_') if Config.HF_TOKEN else False
            },
            'file_system': {},
            'config': {
                'db_faiss_path': Config.DB_FAISS_PATH,
                'data_path': Config.DATA_PATH,
                'huggingface_repo': Config.HUGGINGFACE_REPO_ID,
                'embedding_model': Config.EMBEDDING_MODEL
            },
            'initialization_errors': initialization_errors
        }
        
        # Check file system
        try:
            debug_info['file_system']['current_dir_contents'] = os.listdir('.')
        except Exception as e:
            debug_info['file_system']['current_dir_error'] = str(e)
        
        # Check data directory
        try:
            if os.path.exists(Config.DATA_PATH):
                debug_info['file_system']['data_dir_exists'] = True
                debug_info['file_system']['data_dir_contents'] = os.listdir(Config.DATA_PATH)
                pdf_files = [f for f in os.listdir(Config.DATA_PATH) if f.endswith('.pdf')]
                debug_info['file_system']['pdf_files'] = pdf_files
            else:
                debug_info['file_system']['data_dir_exists'] = False
        except Exception as e:
            debug_info['file_system']['data_dir_error'] = str(e)
        
        # Check vectorstore
        try:
            if os.path.exists(Config.DB_FAISS_PATH):
                debug_info['file_system']['vectorstore_exists'] = True
                debug_info['file_system']['vectorstore_contents'] = os.listdir(Config.DB_FAISS_PATH)
            else:
                debug_info['file_system']['vectorstore_exists'] = False
        except Exception as e:
            debug_info['file_system']['vectorstore_error'] = str(e)
        
        return jsonify(debug_info)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Debug endpoint error: {str(e)}'
        }), 500

@app.route('/api/rebuild', methods=['POST'])
def rebuild_vectorstore():
    """Force rebuild vectorstore from data files"""
    try:
        logger.info("🔨 Manual vectorstore rebuild requested...")
        
        if build_vectorstore_from_data():
            # Reinitialize services
            global vectorstore, is_initialized
            vectorstore = None
            vectorstore_ok = RAGService.initialize_vectorstore()
            
            if vectorstore_ok:
                logger.info("✅ Vectorstore rebuilt and reloaded successfully")
                return jsonify({
                    'success': True,
                    'message': 'Vectorstore rebuilt successfully',
                    'timestamp': datetime.utcnow().isoformat()
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to reload vectorstore after rebuild',
                    'error_code': 'RELOAD_FAILED'
                }), 500
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to rebuild vectorstore',
                'error_code': 'REBUILD_FAILED'
            }), 500
            
    except Exception as e:
        logger.error(f"Error in rebuild endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Rebuild error: {str(e)}',
            'error_code': 'INTERNAL_ERROR'
        }), 500

@app.route('/api/chat', methods=['POST'])
@require_initialization
def chat():
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json',
                'error_code': 'INVALID_CONTENT_TYPE'
            }), 400

        data = request.get_json()
        
        if 'message' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: message',
                'error_code': 'MISSING_MESSAGE'
            }), 400

        user_question = data.get('message', '').strip()
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        if not user_question:
            return jsonify({
                'success': False,
                'error': 'Message cannot be empty',
                'error_code': 'EMPTY_MESSAGE'
            }), 400

        max_sources = data.get('max_sources', Config.MAX_SOURCES)
        include_sources = data.get('include_sources', True)
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={'k': max_sources}
            ),
            return_source_documents=include_sources,
            chain_type_kwargs={'prompt': RAGService.create_custom_prompt()}
        )
        
        # Get response
        response = qa_chain.invoke({'query': user_question})
        result = response["result"].strip()
        
        response_data = {
            'success': True,
            'response': result,
            'session_id': session_id,
            'timestamp': datetime.utcnow().isoformat(),
            'question': user_question
        }
        
        # Add sources if requested
        if include_sources and "source_documents" in response:
            sources = []
            for i, doc in enumerate(response["source_documents"], 1):
                raw_metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                cleaned_meta = RAGService.clean_metadata(raw_metadata)
                
                content = doc.page_content.strip()
                if len(content) > Config.MAX_CONTENT_LENGTH:
                    content = content[:Config.MAX_CONTENT_LENGTH] + "..."
                
                source_info = {
                    'index': i,
                    'content': content,
                    'metadata': cleaned_meta,
                    'relevance_score': getattr(doc, 'score', None)
                }
                sources.append(source_info)
            
            response_data['sources'] = sources
            response_data['source_count'] = len(sources)
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'error_code': 'INTERNAL_ERROR'
        }), 500

@app.route('/api/search', methods=['POST'])
@require_initialization
def search_documents():
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json',
                'error_code': 'INVALID_CONTENT_TYPE'
            }), 400

        data = request.get_json()
        query = data.get('query', '').strip()
        max_results = data.get('max_results', Config.MAX_SOURCES)
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query cannot be empty',
                'error_code': 'EMPTY_QUERY'
            }), 400

        # Search documents
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={'k': max_results}
        )
        
        docs = retriever.get_relevant_documents(query)
        
        results = []
        for i, doc in enumerate(docs, 1):
            raw_metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            cleaned_meta = RAGService.clean_metadata(raw_metadata)
            
            content = doc.page_content.strip()
            if len(content) > Config.MAX_CONTENT_LENGTH:
                content = content[:Config.MAX_CONTENT_LENGTH] + "..."
            
            result_info = {
                'index': i,
                'content': content,
                'metadata': cleaned_meta,
                'relevance_score': getattr(doc, 'score', None)
            }
            results.append(result_info)
        
        return jsonify({
            'success': True,
            'query': query,
            'results': results,
            'result_count': len(results),
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'error_code': 'INTERNAL_ERROR'
        }), 500

@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify({
        'success': True,
        'config': {
            'max_sources': Config.MAX_SOURCES,
            'max_content_length': Config.MAX_CONTENT_LENGTH,
            'embedding_model': Config.EMBEDDING_MODEL,
            'llm_model': Config.HUGGINGFACE_REPO_ID,
            'data_path': Config.DATA_PATH,
            'vectorstore_path': Config.DB_FAISS_PATH
        }
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'error_code': 'NOT_FOUND'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'success': False,
        'error': 'Method not allowed',
        'error_code': 'METHOD_NOT_ALLOWED'
    }), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'error_code': 'INTERNAL_ERROR'
    }), 500

if __name__ == '__main__':
    # Initialize services
    logger.info("🏥 Starting MediBuddy RAG API Server...")
    
    success = initialize_services()
    
    if not success:
        logger.warning("⚠️ Services failed to initialize, but starting server for debugging...")
        logger.warning("⚠️ Check /api/health, /api/debug, and /api/rebuild endpoints")
    
    logger.info("🚀 Flask API server starting...")
    logger.info("📱 API available at: http://localhost:5000")
    logger.info("📚 API Documentation:")
    logger.info("  - Health Check: GET /api/health")
    logger.info("  - Debug Info: GET /api/debug")
    logger.info("  - Rebuild Vectorstore: POST /api/rebuild")
    logger.info("  - Chat: POST /api/chat")
    logger.info("  - Search: POST /api/search")
    logger.info("  - Config: GET /api/config")
    
    app.run(
        debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true',
        host=os.environ.get('FLASK_HOST', '0.0.0.0'),
        port=int(os.environ.get('FLASK_PORT', 5000))
    )