from flask import request, jsonify
from functools import wraps
import logging

logger = logging.getLogger(__name__)

def validate_request(required_fields=None, optional_fields=None):
    """
    Decorator to validate incoming requests
    
    Args:
        required_fields (list): List of required fields in request JSON
        optional_fields (list): List of optional fields (for documentation)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Check if request has JSON data
                if not request.is_json:
                    return jsonify({
                        'error': 'Content-Type must be application/json'
                    }), 400
                
                data = request.get_json()
                
                if not data:
                    return jsonify({
                        'error': 'No JSON data provided'
                    }), 400
                
                # Check required fields
                if required_fields:
                    missing_fields = []
                    for field in required_fields:
                        if field not in data or not data[field]:
                            missing_fields.append(field)
                    
                    if missing_fields:
                        return jsonify({
                            'error': f'Missing required fields: {", ".join(missing_fields)}'
                        }), 400
                
                # Log the request
                logger.info(f"Valid request received for {func.__name__}")
                
                return func(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"Request validation error: {str(e)}")
                return jsonify({
                    'error': 'Invalid request format'
                }), 400
        
        return wrapper
    return decorator

# Additional utility functions you might need

def format_response(data=None, message=None, status='success'):
    """
    Format consistent API responses
    """
    response = {
        'status': status,
        'timestamp': None  # You can add timestamp if needed
    }
    
    if message:
        response['message'] = message
    
    if data is not None:
        response['data'] = data
    
    return response

def handle_error(error, status_code=500):
    """
    Handle and format error responses
    """
    logger.error(f"Error occurred: {str(error)}")
    return jsonify({
        'status': 'error',
        'error': str(error)
    }), status_code

def sanitize_input(text):
    """
    Basic input sanitization
    """
    if not text:
        return ""
    
    # Remove potential harmful characters
    import re
    sanitized = re.sub(r'[<>"\']', '', str(text))
    return sanitized.strip()

def validate_medical_query(query):
    """
    Validate medical queries
    """
    if not query or len(query.strip()) < 3:
        return False, "Query must be at least 3 characters long"
    
    if len(query) > 1000:
        return False, "Query is too long (max 1000 characters)"
    
    return True, "Valid query"