from flask import jsonify, request
from datetime import datetime
import uuid
import logging
from langchain.chains import RetrievalQA

from .services import RAGService
from .config import Config
from .utils import validate_request

logger = logging.getLogger(__name__)

def init_routes(app):
    @app.route('/api/health', methods=['GET'])
    def health_check():
        return jsonify({
            'success': True,
            'status': 'healthy' if RAGService.is_initialized else 'degraded',
            'timestamp': datetime.utcnow().isoformat(),
            'services': {
                'vectorstore_loaded': RAGService.vectorstore is not None,
                'llm_loaded': RAGService.llm is not None,
                'initialized': RAGService.is_initialized
            }
        })

    @app.route('/api/chat', methods=['POST'])
    def chat():
        # Validate request
        validation = validate_request(request)
        if not validation['success']:
            return jsonify(validation), 400

        data = request.get_json()
        user_question = data['message'].strip()
        session_id = data.get('session_id', str(uuid.uuid4()))
        max_sources = min(int(data.get('max_sources', Config.MAX_SOURCES)), 10)
        include_sources = data.get('include_sources', True)

        try:
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=RAGService.llm,
                chain_type="stuff",
                retriever=RAGService.vectorstore.as_retriever(
                    search_kwargs={'k': max_sources}
                ),
                return_source_documents=include_sources,
                chain_type_kwargs={'prompt': RAGService.get_prompt_template()}
            )

            # Get response
            response = qa_chain.invoke({'query': user_question})
            
            # Prepare response
            result = {
                'success': True,
                'response': response["result"].strip(),
                'session_id': session_id,
                'timestamp': datetime.utcnow().isoformat()
            }

            if include_sources and response.get("source_documents"):
                result['sources'] = [
                    {
                        'content': doc.page_content[:Config.MAX_CONTENT_LENGTH] + 
                                  ('...' if len(doc.page_content) > Config.MAX_CONTENT_LENGTH else ''),
                        'metadata': RAGService.clean_metadata(doc.metadata)
                    }
                    for doc in response["source_documents"]
                ]

            return jsonify(result)

        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e),
                'error_code': 'CHAT_ERROR'
            }), 500