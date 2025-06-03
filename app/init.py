from flask import Flask
from flask_cors import CORS
import logging
from .config import Config
from .services import RAGService

def create_app():
    app = Flask(__name__)
    
    # Configure CORS
    CORS(app, origins=Config.CORS_ORIGINS)
    
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Initialize services
    logger.info("Initializing RAG services...")
    if not RAGService.initialize():
        logger.error("Failed to initialize RAG services")
        raise RuntimeError("Service initialization failed")
    
    # Register routes
    from .routes import init_routes
    init_routes(app)
    
    return app