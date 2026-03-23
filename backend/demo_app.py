"""
Simplified Backend for FractureDetect AI Demo
Basic Flask server for demonstration without deep learning dependencies
"""

import os
import sys
import time
import uuid
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DemoModelManager:
    """Demo model manager for testing without actual AI models"""
    
    def __init__(self):
        self.device = "cpu"
        self.fracture_types = [
            "Compression-Crush fracture",
            "Fracture Dislocation", 
            "Hairline Fracture",
            "Impacted fracture",
            "Longitudinal fracture",
            "Oblique fracture",
            "Spiral Fracture"
        ]
        logger.info("✅ Demo Model Manager initialized")
    
    def predict(self, image_array: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Generate demo prediction results"""
        time.sleep(1)  # Simulate processing time
        
        # Generate random but realistic predictions
        np.random.seed(int(time.time()) % 1000)
        
        # Binary classification (fracture detected or not)
        fracture_probability = np.random.random()
        fracture_detected = fracture_probability > 0.5
        
        result = {
            'fracture_detected': fracture_detected,
            'confidence': fracture_probability if fracture_detected else (1 - fracture_probability),
            'probabilities': [1 - fracture_probability, fracture_probability]
        }
        
        # If fracture detected, add fracture type
        if fracture_detected:
            type_idx = np.random.randint(0, len(self.fracture_types))
            type_confidence = 0.7 + np.random.random() * 0.25
            
            result.update({
                'fracture_type': self.fracture_types[type_idx],
                'type_confidence': type_confidence,
                'binary_confidence': fracture_probability
            })
        
        return result

def create_app() -> Flask:
    """Create and configure Flask application"""
    app = Flask(__name__)
    
    # Configuration
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
    
    # Enable CORS
    CORS(app, origins=["*"])
    
    # Initialize demo model manager
    model_manager = DemoModelManager()
    
    def validate_image(file) -> Tuple[bool, str]:
        """Validate uploaded image"""
        if not file:
            return False, "No file provided"
        
        if file.filename == '':
            return False, "No file selected"
        
        # Check file extension
        allowed_extensions = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp']
        filename = secure_filename(file.filename.lower())
        if not any(filename.endswith(ext) for ext in allowed_extensions):
            return False, f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        
        return True, "Valid"
    
    def load_and_validate_image(file) -> Tuple[Optional[np.ndarray], str]:
        """Load and validate image from file"""
        try:
            # Load image
            image = Image.open(file.stream).convert('RGB')
            image_array = np.array(image)
            
            # Basic validation
            if image_array.size == 0:
                return None, "Empty image"
            
            if len(image_array.shape) != 3 or image_array.shape[2] != 3:
                return None, "Invalid image format"
            
            # Check dimensions
            h, w = image_array.shape[:2]
            if h < 32 or w < 32:
                return None, "Image too small (minimum 32x32)"
            
            if h > 4096 or w > 4096:
                return None, "Image too large (maximum 4096x4096)"
            
            return image_array, "Valid"
            
        except Exception as e:
            return None, f"Invalid image: {str(e)}"
    
    # Routes
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '2.0.0-demo',
            'models_loaded': ['demo_binary', 'demo_multiclass'],
            'device': model_manager.device,
            'mode': 'demo'
        })
    
    @app.route('/info', methods=['GET'])
    def get_info():
        """Get API information"""
        return jsonify({
            'api_version': '2.0.0-demo',
            'model_info': {
                'backbone': 'demo-efficientnet-b4',
                'use_ensemble': True,
                'image_size': 384,
                'mode': 'demonstration'
            },
            'supported_formats': ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'],
            'max_file_size_mb': 16,
            'fracture_types': model_manager.fracture_types,
            'explainability_available': False,
            'demo_mode': True,
            'note': 'This is a demonstration version. Predictions are simulated for testing purposes.'
        })
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """Main prediction endpoint"""
        prediction_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Validate request
            if 'image' not in request.files:
                return jsonify({'error': 'No image file provided'}), 400
            
            file = request.files['image']
            
            # Validate file
            is_valid, message = validate_image(file)
            if not is_valid:
                return jsonify({'error': message}), 400
            
            # Load and validate image
            image_array, message = load_and_validate_image(file)
            if image_array is None:
                return jsonify({'error': message}), 400
            
            # Get options
            generate_explanation = request.form.get('generate_explanation', 'false').lower() == 'true'
            
            # Make prediction
            prediction_result = model_manager.predict(image_array)
            
            processing_time = time.time() - start_time
            
            # Prepare response
            response_data = {
                'prediction_id': prediction_id,
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round(processing_time * 1000, 2),
                'demo_mode': True,
                'note': 'This is a simulated prediction for demonstration purposes',
                **prediction_result
            }
            
            logger.info(f"Demo prediction completed: {prediction_id}")
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"Prediction error for {prediction_id}: {e}")
            
            return jsonify({
                'error': 'Internal server error',
                'prediction_id': prediction_id,
                'message': str(e),
                'demo_mode': True
            }), 500
    
    @app.route('/', methods=['GET'])
    def home():
        """Home endpoint"""
        return jsonify({
            'message': '🩻 FractureDetect AI Demo Server',
            'version': '2.0.0-demo',
            'status': 'running',
            'endpoints': {
                'health': '/health',
                'info': '/info', 
                'predict': '/predict',
                'frontend': 'Open frontend/enhanced_index.html in browser'
            }
        })
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({'error': 'Method not allowed'}), 405
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({'error': 'File too large'}), 413
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500
    
    return app

def main():
    """Main function to run the demo API server"""
    logger.info("Starting FractureDetect AI Demo Server...")
    
    # Create Flask app
    app = create_app()
    
    # Show startup info
    print("\n" + "="*60)
    print("🩻 FractureDetect AI - Demo Server Starting")
    print("="*60)
    print(f"🌐 Server: http://localhost:5000")
    print(f"📊 Health Check: http://localhost:5000/health")
    print(f"ℹ️  API Info: http://localhost:5000/info")
    print(f"🎨 Frontend: Open frontend/enhanced_index.html in browser")
    print("\n⚠️  NOTE: This is a DEMO version with simulated predictions")
    print("   Real AI models require PyTorch installation")
    print("="*60)
    
    # Run server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )

if __name__ == '__main__':
    main()