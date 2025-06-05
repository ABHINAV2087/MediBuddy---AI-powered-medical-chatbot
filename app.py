from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from utils.vector_store import PineconeVectorStore
from utils.llm_handler import LLMHandler
from config import Config

app = Flask(__name__)
CORS(app)  # Enable CORS for MERN integration

# Global variables
vector_store = None
llm_handler = None
config = Config()

def initialize_chatbot():
    """Initialize the chatbot components"""
    global vector_store, llm_handler
    
    try:
        # Initialize vector store
        vector_store = PineconeVectorStore()
        
        # Check if we need to setup the pipeline (first time)
        setup_required = os.getenv('SETUP_PIPELINE', 'false').lower() == 'true'
        
        if setup_required:
            print("Setting up complete pipeline...")
            success = vector_store.setup_complete_pipeline()
            if not success:
                raise Exception("Failed to setup pipeline")
        else:
            vector_store.initialize_index()
        
        # Get vectorstore
        vectorstore = vector_store.get_vectorstore()
        if vectorstore is None:
            raise Exception("Failed to get vector store")
        
        # Initialize LLM handler
        llm_handler = LLMHandler()
        success = llm_handler.create_qa_chain(vectorstore)
        if not success:
            raise Exception("Failed to create QA chain")
        
        print("Chatbot initialized successfully!")
        return True
        
    except Exception as e:
        print(f"Error initializing chatbot: {str(e)}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Medical Chatbot API is running"
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        # Get query from request
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' in request body",
                "answer": None,
                "sources": []
            }), 400
        
        query = data['query'].strip()
        
        if not query:
            return jsonify({
                "error": "Query cannot be empty",
                "answer": None,
                "sources": []
            }), 400
        
        # Check if chatbot is initialized
        if not llm_handler:
            return jsonify({
                "error": "Chatbot not initialized",
                "answer": None,
                "sources": []
            }), 500
        
        # Get response
        response = llm_handler.get_response(query)
        
        if response["error"]:
            return jsonify(response), 500
        
        return jsonify({
            "error": None,
            "answer": response["answer"],
            "sources": response["sources"],
            "query": query
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "answer": None,
            "sources": []
        }), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get chatbot status"""
    return jsonify({
        "initialized": llm_handler is not None,
        "vector_store_connected": vector_store is not None,
        "index_name": config.PINECONE_INDEX_NAME
    })

if __name__ == '__main__':
    print("Starting Medical Chatbot API...")
    
    # Initialize chatbot
    success = initialize_chatbot()
    
    if not success:
        print("Failed to initialize chatbot. Check your configuration.")
        exit(1)
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('FLASK_ENV') == 'development'
    )