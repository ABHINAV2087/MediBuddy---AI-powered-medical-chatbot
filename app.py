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

# Load environment variables
load_dotenv(find_dotenv())

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Updated CORS to include your Render domain
CORS(app, origins=[
    "http://localhost:3000", 
    "http://localhost:3001", 
    "https://medibuddy-ai-powered-medical-chatbot.onrender.com",
    "https://*.onrender.com"  # Allow any Render subdomain
])

class Config:
    # Use absolute paths for Render deployment
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_FAISS_PATH = os.environ.get("DB_FAISS_PATH", os.path.join(BASE_DIR, "vectorstore", "db_faiss"))
    HUGGINGFACE_REPO_ID = os.environ.get("HUGGINGFACE_REPO_ID", "microsoft/DialoGPT-medium")
    HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_TOKEN")
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    MAX_SOURCES = int(os.environ.get("MAX_SOURCES", "5"))
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", "500"))
    
    # Fallback mode - if vectorstore fails, use simple QA
    ENABLE_FALLBACK_MODE = os.environ.get("ENABLE_FALLBACK_MODE", "true").lower() == "true"

vectorstore = None
llm = None
is_initialized = False
initialization_errors = []
fallback_mode = False

class RAGService:
    
    @staticmethod
    def initialize_vectorstore():
        global vectorstore, initialization_errors, fallback_mode
        try:
            logger.info(f"🔍 Attempting to load vectorstore from: {Config.DB_FAISS_PATH}")
            
            # Check if the path exists
            if not os.path.exists(Config.DB_FAISS_PATH):
                error_msg = f"Vectorstore directory not found: {Config.DB_FAISS_PATH}"
                logger.error(f"❌ {error_msg}")
                
                # Try alternative paths
                alternative_paths = [
                    "./vectorstore/db_faiss",
                    "./db_faiss",
                    "vectorstore/db_faiss",
                    "db_faiss"
                ]
                
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        Config.DB_FAISS_PATH = alt_path
                        logger.info(f"✅ Found vectorstore at alternative path: {alt_path}")
                        break
                else:
                    if Config.ENABLE_FALLBACK_MODE:
                        logger.warning("⚠️ Enabling fallback mode - vectorstore not available")
                        fallback_mode = True
                        return True
                    else:
                        initialization_errors.append(error_msg)
                        return False
            
            # List contents of the directory
            try:
                contents = os.listdir(Config.DB_FAISS_PATH)
                logger.info(f"📁 Vectorstore directory contents: {contents}")
                
                # Check for required FAISS files
                required_files = ['index.faiss', 'index.pkl']
                missing_files = [f for f in required_files if f not in contents]
                if missing_files:
                    error_msg = f"Missing FAISS files: {missing_files}"
                    logger.error(f"❌ {error_msg}")
                    if Config.ENABLE_FALLBACK_MODE:
                        logger.warning("⚠️ Enabling fallback mode - required FAISS files missing")
                        fallback_mode = True
                        return True
                    else:
                        initialization_errors.append(error_msg)
                        return False
                        
            except Exception as e:
                logger.warning(f"⚠️ Could not list directory contents: {e}")
            
            # Try to load embeddings
            try:
                embedding_model = HuggingFaceEmbeddings(
                    model_name=Config.EMBEDDING_MODEL,
                    cache_folder=os.path.join(Config.BASE_DIR, "embeddings_cache")
                )
                logger.info(f"✅ Embedding model loaded: {Config.EMBEDDING_MODEL}")
            except Exception as e:
                error_msg = f"Failed to load embedding model: {str(e)}"
                logger.error(f"❌ {error_msg}")
                if Config.ENABLE_FALLBACK_MODE:
                    fallback_mode = True
                    return True
                else:
                    initialization_errors.append(error_msg)
                    return False
            
            # Try to load vectorstore
            try:
                vectorstore = FAISS.load_local(
                    Config.DB_FAISS_PATH, 
                    embedding_model, 
                    allow_dangerous_deserialization=True
                )
                logger.info("✅ Vectorstore loaded successfully")
                return True
            except Exception as e:
                error_msg = f"Failed to load vectorstore: {str(e)}"
                logger.error(f"❌ {error_msg}")
                if Config.ENABLE_FALLBACK_MODE:
                    fallback_mode = True
                    return True
                else:
                    initialization_errors.append(error_msg)
                    return False
            
        except Exception as e:
            error_msg = f"Unexpected error in vectorstore initialization: {str(e)}"
            logger.error(f"❌ {error_msg}")
            if Config.ENABLE_FALLBACK_MODE:
                fallback_mode = True
                return True
            else:
                initialization_errors.append(error_msg)
                return False

    @staticmethod
    def initialize_llm():
        global llm, initialization_errors
        try:
            # Check for HuggingFace token
            if not Config.HF_TOKEN:
                error_msg = "HuggingFace token not found. Please set HF_TOKEN or HUGGINGFACE_API_TOKEN environment variable."
                logger.error(f"❌ {error_msg}")
                initialization_errors.append(error_msg)
                return False
            
            # Validate token format (should start with 'hf_')
            if not Config.HF_TOKEN.startswith('hf_'):
                logger.warning("⚠️ HuggingFace token doesn't start with 'hf_' - this might be okay for some token types")
            
            logger.info(f"🤖 Initializing LLM: {Config.HUGGINGFACE_REPO_ID}")
            
            # Set the token in environment
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = Config.HF_TOKEN
            
            # Use a more reliable model for deployment
            llm = HuggingFaceEndpoint(
                repo_id=Config.HUGGINGFACE_REPO_ID,
                temperature=0.3,
                max_new_tokens=500,  # Reduced for reliability
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.1,
                timeout=60  # Add timeout
            )
            
            # Test the LLM with a simple query
            try:
                test_response = llm.invoke("Hello")
                logger.info("✅ LLM test successful")
            except Exception as e:
                logger.warning(f"⚠️ LLM test failed but continuing: {e}")
            
            logger.info("✅ LLM loaded successfully")
            return True
            
        except Exception as e:
            error_msg = f"Error loading LLM: {str(e)}"
            logger.error(f"❌ {error_msg}")
            initialization_errors.append(error_msg)
            return False

    @staticmethod
    def create_custom_prompt():
        if fallback_mode:
            template = """
            You are a helpful AI assistant. Answer the user's question to the best of your ability.
            If the question is about medical topics, provide general information but always recommend consulting with healthcare professionals for specific medical advice.
            
            Question: {question}
            
            Answer:
            """
            return PromptTemplate(template=template, input_variables=["question"])
        else:
            template = """
            You are a helpful assistant that answers questions based on the provided context.
            Use the pieces of information provided in the context to answer the user's question comprehensively.
            
            If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
            Provide detailed and complete answers when the information is available in the context.
            
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
        
        if 'Source' in cleaned_metadata and 'File Name' in cleaned_metadata:
            if cleaned_metadata['Source'].lower() == cleaned_metadata['File Name'].lower():
                del cleaned_metadata['Source']
        
        return cleaned_metadata

def log_environment_info():
    """Log important environment information for debugging"""
    logger.info("🔧 Environment Configuration:")
    logger.info(f"  - BASE_DIR: {Config.BASE_DIR}")
    logger.info(f"  - DB_FAISS_PATH: {Config.DB_FAISS_PATH}")
    logger.info(f"  - HUGGINGFACE_REPO_ID: {Config.HUGGINGFACE_REPO_ID}")
    logger.info(f"  - HF_TOKEN: {'✅ Set' if Config.HF_TOKEN else '❌ Not Set'}")
    logger.info(f"  - EMBEDDING_MODEL: {Config.EMBEDDING_MODEL}")
    logger.info(f"  - Working Directory: {os.getcwd()}")
    logger.info(f"  - ENABLE_FALLBACK_MODE: {Config.ENABLE_FALLBACK_MODE}")
    
    # List current directory contents
    try:
        contents = os.listdir('.')
        logger.info(f"  - Current Directory Contents: {contents}")
    except Exception as e:
        logger.warning(f"  - Could not list current directory: {e}")
    
    # Check if vectorstore directory exists
    if os.path.exists("vectorstore"):
        try:
            vs_contents = os.listdir("vectorstore")
            logger.info(f"  - Vectorstore Directory Contents: {vs_contents}")
        except Exception as e:
            logger.warning(f"  - Could not list vectorstore directory: {e}")
    else:
        logger.warning("  - Vectorstore directory does not exist")

def initialize_services():
    global is_initialized, initialization_errors, fallback_mode
    logger.info("🚀 Initializing RAG Services...")
    
    # Clear previous errors
    initialization_errors = []
    fallback_mode = False
    
    # Log environment info
    log_environment_info()
    
    vectorstore_ok = RAGService.initialize_vectorstore()
    llm_ok = RAGService.initialize_llm()
    
    is_initialized = vectorstore_ok and llm_ok
    
    if is_initialized:
        if fallback_mode:
            logger.info("✅ Services initialized in FALLBACK MODE (no vectorstore)")
        else:
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
                'initialization_errors': initialization_errors,
                'fallback_mode': fallback_mode
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
            'initialized': is_initialized,
            'fallback_mode': fallback_mode
        },
        'config': {
            'db_faiss_path': Config.DB_FAISS_PATH,
            'hf_token_set': Config.HF_TOKEN is not None,
            'working_directory': os.getcwd(),
            'base_directory': Config.BASE_DIR,
            'enable_fallback': Config.ENABLE_FALLBACK_MODE
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
                'base_directory': Config.BASE_DIR,
                'python_path': os.environ.get('PYTHONPATH', 'Not Set'),
                'hf_token_set': Config.HF_TOKEN is not None,
                'hf_token_valid_format': Config.HF_TOKEN.startswith('hf_') if Config.HF_TOKEN else False,
                'fallback_mode_enabled': Config.ENABLE_FALLBACK_MODE,
                'current_fallback_mode': fallback_mode
            },
            'file_system': {},
            'config': {
                'db_faiss_path': Config.DB_FAISS_PATH,
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
        
        # Check vectorstore
        try:
            if os.path.exists(Config.DB_FAISS_PATH):
                debug_info['file_system']['vectorstore_exists'] = True
                debug_info['file_system']['vectorstore_contents'] = os.listdir(Config.DB_FAISS_PATH)
            else:
                debug_info['file_system']['vectorstore_exists'] = False
                
                # Check alternative locations
                alt_locations = ['./vectorstore', './vectorstore/db_faiss', './db_faiss']
                for loc in alt_locations:
                    if os.path.exists(loc):
                        debug_info['file_system'][f'found_at_{loc}'] = os.listdir(loc)
        except Exception as e:
            debug_info['file_system']['vectorstore_error'] = str(e)
        
        # Check environment variables
        env_vars = ['HF_TOKEN', 'HUGGINGFACE_API_TOKEN', 'DB_FAISS_PATH', 'HUGGINGFACE_REPO_ID']
        debug_info['environment']['env_vars'] = {}
        for var in env_vars:
            debug_info['environment']['env_vars'][var] = 'Set' if os.environ.get(var) else 'Not Set'
        
        return jsonify(debug_info)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Debug endpoint error: {str(e)}'
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
        
        if fallback_mode:
            # Fallback mode - direct LLM without RAG
            try:
                prompt = RAGService.create_custom_prompt()
                formatted_prompt = prompt.format(question=user_question)
                result = llm.invoke(formatted_prompt).strip()
                
                response_data = {
                    'success': True,
                    'response': result,
                    'session_id': session_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'question': user_question,
                    'mode': 'fallback',
                    'sources': [],
                    'source_count': 0
                }
                return jsonify(response_data)
                
            except Exception as e:
                logger.error(f"Error in fallback mode: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': f'Fallback mode error: {str(e)}',
                    'error_code': 'FALLBACK_ERROR'
                }), 500
        else:
            # Normal RAG mode
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
            
            response = qa_chain.invoke({'query': user_question})
            result = response["result"].strip()
            
            response_data = {
                'success': True,
                'response': result,
                'session_id': session_id,
                'timestamp': datetime.utcnow().isoformat(),
                'question': user_question,
                'mode': 'rag'
            }
            
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
            else:
                response_data['sources'] = []
                response_data['source_count'] = 0
            
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
    if fallback_mode:
        return jsonify({
            'success': False,
            'error': 'Search not available in fallback mode (no vectorstore)',
            'error_code': 'FALLBACK_MODE_NO_SEARCH'
        }), 503
        
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
            'fallback_mode': fallback_mode,
            'fallback_enabled': Config.ENABLE_FALLBACK_MODE
        }
    })

# Add a simple test endpoint
@app.route('/api/test', methods=['GET'])
def test_endpoint():
    return jsonify({
        'success': True,
        'message': 'API is working!',
        'timestamp': datetime.utcnow().isoformat()
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
    # Try to initialize services, but don't exit if it fails (for debugging)
    initialize_services()
    
    if not is_initialized:
        logger.warning("⚠️ Services failed to initialize, but starting server for debugging...")
        logger.warning("⚠️ Check /api/health and /api/debug endpoints for more information")
    elif fallback_mode:
        logger.warning("⚠️ Running in FALLBACK MODE - vectorstore not available")
    
    logger.info("🚀 Starting Flask API server...")
    logger.info("📱 API available at: http://localhost:5000")
    logger.info("📚 API Documentation:")
    logger.info("  - Health Check: GET /api/health")
    logger.info("  - Debug Info: GET /api/debug")
    logger.info("  - Test: GET /api/test")
    logger.info("  - Chat: POST /api/chat")
    logger.info("  - Search: POST /api/search")
    logger.info("  - Config: GET /api/config")
    
    app.run(
        debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true',
        host=os.environ.get('FLASK_HOST', '0.0.0.0'),
        port=int(os.environ.get('FLASK_PORT', 5000))
    )