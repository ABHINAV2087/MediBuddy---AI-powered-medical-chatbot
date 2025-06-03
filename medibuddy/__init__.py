from flask import Flask
from flask_cors import CORS
import logging

def create_app():
    app = Flask(__name__)
    
    # Configure CORS
    CORS(app)  # Adjust origins as needed
    
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize services
    from medibuddy.services import RAGService
    if not RAGService.initialize():
        raise RuntimeError("Failed to initialize services")
    
    # Register routes
    from medibuddy.routes import init_routes
    init_routes(app)
    
    return app