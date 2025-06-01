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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

CORS(app, origins=["http://localhost:3000", "http://localhost:3001", "*"])

class Config:
    DB_FAISS_PATH = os.environ.get("DB_FAISS_PATH", "vectorstore/db_faiss")
    HUGGINGFACE_REPO_ID = os.environ.get("HUGGINGFACE_REPO_ID", "mistralai/Mixtral-8x7B-Instruct-v0.1")
    HF_TOKEN = os.environ.get("HF_TOKEN")
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    MAX_SOURCES = int(os.environ.get("MAX_SOURCES", "5"))
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", "500"))

vectorstore = None
llm = None
is_initialized = False

class RAGService:
    
    @staticmethod
    def initialize_vectorstore():
        global vectorstore
        try:
            embedding_model = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
            vectorstore = FAISS.load_local(
                Config.DB_FAISS_PATH, 
                embedding_model, 
                allow_dangerous_deserialization=True
            )
            logger.info("✅ Vectorstore loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading vectorstore: {str(e)}")
            return False

    @staticmethod
    def initialize_llm():
        global llm
        try:
            if not Config.HF_TOKEN:
                raise ValueError("HuggingFace token not found. Please set HF_TOKEN environment variable.")
            
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = Config.HF_TOKEN
            
            llm = HuggingFaceEndpoint(
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
            return False

    @staticmethod
    def create_custom_prompt():
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

def initialize_services():
    global is_initialized
    logger.info("🚀 Initializing RAG Services...")
    
    vectorstore_ok = RAGService.initialize_vectorstore()
    llm_ok = RAGService.initialize_llm()
    
    is_initialized = vectorstore_ok and llm_ok
    
    if is_initialized:
        logger.info("✅ All services initialized successfully!")
    else:
        logger.error("❌ Failed to initialize services")
    
    return is_initialized

def require_initialization(f):
    def decorated_function(*args, **kwargs):
        if not is_initialized:
            return jsonify({
                'success': False,
                'error': 'Services not initialized. Please check server logs.',
                'error_code': 'SERVICE_NOT_INITIALIZED'
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
        }
    })

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
            'question': user_question
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
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'error_code': 'INTERNAL_ERROR'
        }), 500

@app.route('/api/chat/stream', methods=['POST'])
@require_initialization
def chat_stream():
    return jsonify({
        'success': False,
        'error': 'Streaming not implemented yet',
        'error_code': 'NOT_IMPLEMENTED'
    }), 501

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
            'llm_model': Config.HUGGINGFACE_REPO_ID
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
    if not initialize_services():
        logger.error("❌ Failed to initialize services. Exiting...")
        exit(1)
    
    logger.info("🚀 Starting Flask API server...")
    logger.info("📱 API available at: http://localhost:5000")
    logger.info("📚 API Documentation:")
    logger.info("  - Health Check: GET /api/health")
    logger.info("  - Chat: POST /api/chat")
    logger.info("  - Search: POST /api/search")
    logger.info("  - Config: GET /api/config")
    
    app.run(
        debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true',
        host=os.environ.get('FLASK_HOST', '0.0.0.0'),
        port=int(os.environ.get('FLASK_PORT', 5000))
    )