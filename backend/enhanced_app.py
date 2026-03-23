"""
Professional Flask Backend API for Bone Fracture Detection
Enhanced with proper error handling, validation, caching, and monitoring
"""

import os
import sys
import time
import uuid
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
import cv2

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.architectures import create_model, create_ensemble_model
from explainability.gradcam import AdvancedExplainer
from data.advanced_dataset import get_advanced_transforms
from configs.config import get_config
from evaluation.metrics import MetricsCalculator


class ModelManager:
    """Manages model loading and caching"""
    
    def __init__(self, config):
        self.config = config
        self.device = self._setup_device()
        self.models = {}
        self.explainers = {}
        self.transforms = self._setup_transforms()
        
        # Load models
        self._load_models()
        
    def _setup_device(self) -> torch.device:
        """Setup computation device"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logging.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            logging.info("Using CPU")
        return device
    
    def _setup_transforms(self):
        """Setup image transforms"""
        return get_advanced_transforms(
            img_size=self.config.model.img_size,
            is_training=False,
            use_albumentations=False  # Use torchvision for inference
        )
    
    def _load_models(self):
        """Load trained models"""
        try:
            # Load binary model
            if os.path.exists(self.config.api.binary_model_path):
                self.models['binary'] = self._load_single_model(
                    self.config.api.binary_model_path,
                    num_classes=2,
                    model_type='binary'
                )
                logging.info("✅ Binary model loaded successfully")
            else:
                logging.warning(f"Binary model not found: {self.config.api.binary_model_path}")
            
            # Load multiclass model
            if os.path.exists(self.config.api.multiclass_model_path):
                self.models['multiclass'] = self._load_single_model(
                    self.config.api.multiclass_model_path,
                    num_classes=len(self.config.model.fracture_types),
                    model_type='multiclass'
                )
                logging.info("✅ Multiclass model loaded successfully")
            else:
                logging.warning(f"Multiclass model not found: {self.config.api.multiclass_model_path}")
            
            # Setup explainers
            if self.config.api.generate_gradcam:
                self._setup_explainers()
                
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            raise
    
    def _load_single_model(self, model_path: str, num_classes: int, model_type: str):
        """Load a single model"""
        if self.config.model.use_ensemble:
            model = create_ensemble_model(
                self.config.model.ensemble_models,
                num_classes=num_classes,
                pretrained=False,
                freeze_backbone=self.config.model.freeze_backbone,
                dropout_rate=self.config.model.dropout_rate
            )
        else:
            model = create_model(
                self.config.model.backbone,
                num_classes=num_classes,
                pretrained=False,
                freeze_backbone=self.config.model.freeze_backbone,
                dropout_rate=self.config.model.dropout_rate
            )
        
        # Load state dict
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _setup_explainers(self):
        """Setup model explainers"""
        try:
            if 'binary' in self.models:
                self.explainers['binary'] = AdvancedExplainer(
                    model=self.models['binary'],
                    class_names=self.config.model.binary_classes,
                    device=self.device
                )
            
            if 'multiclass' in self.models:
                self.explainers['multiclass'] = AdvancedExplainer(
                    model=self.models['multiclass'],
                    class_names=self.config.model.fracture_types,
                    device=self.device
                )
            
            logging.info("✅ Model explainers initialized")
        except Exception as e:
            logging.warning(f"Could not initialize explainers: {e}")
    
    def predict(
        self,
        image: np.ndarray,
        return_probabilities: bool = True,
        generate_explanation: bool = False
    ) -> Dict[str, Any]:
        """Make prediction on image"""
        try:
            # Preprocess image
            input_tensor = self._preprocess_image(image)
            
            # Stage 1: Binary classification
            binary_result = self._predict_binary(input_tensor, image if generate_explanation else None)
            
            if not binary_result['fracture_detected']:
                return {
                    'fracture_detected': False,
                    'confidence': binary_result['confidence'],
                    'probabilities': binary_result.get('probabilities'),
                    'explanation': binary_result.get('explanation')
                }
            
            # Stage 2: Multiclass classification
            multiclass_result = self._predict_multiclass(input_tensor, image if generate_explanation else None)
            
            return {
                'fracture_detected': True,
                'fracture_type': multiclass_result['fracture_type'],
                'confidence': multiclass_result['confidence'],
                'probabilities': multiclass_result.get('probabilities'),
                'binary_confidence': binary_result['confidence'],
                'explanation': multiclass_result.get('explanation')
            }
            
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            raise
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        # Convert to PIL Image
        if image.max() > 1:
            image_pil = Image.fromarray(image.astype(np.uint8))
        else:
            image_pil = Image.fromarray((image * 255).astype(np.uint8))
        
        # Apply transforms
        input_tensor = self.transforms(image_pil).unsqueeze(0)
        return input_tensor.to(self.device)
    
    def _predict_binary(self, input_tensor: torch.Tensor, original_image: Optional[np.ndarray] = None) -> Dict:
        """Binary fracture detection"""
        if 'binary' not in self.models:
            raise ValueError("Binary model not loaded")
        
        with torch.no_grad():
            output = self.models['binary'](input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            result = {
                'fracture_detected': predicted.item() == 1,
                'confidence': confidence.item(),
                'probabilities': probabilities.cpu().numpy().tolist()[0]
            }
            
            # Generate explanation if requested
            if original_image is not None and 'binary' in self.explainers:
                try:
                    explanation = self.explainers['binary'].generate_gradcam_explanations(
                        input_tensor, original_image
                    )
                    result['explanation'] = self._process_explanation(explanation)
                except Exception as e:
                    logging.warning(f"Could not generate binary explanation: {e}")
            
            return result
    
    def _predict_multiclass(self, input_tensor: torch.Tensor, original_image: Optional[np.ndarray] = None) -> Dict:
        """Multiclass fracture type classification"""
        if 'multiclass' not in self.models:
            raise ValueError("Multiclass model not loaded")
        
        with torch.no_grad():
            output = self.models['multiclass'](input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            result = {
                'fracture_type': self.config.model.fracture_types[predicted.item()],
                'confidence': confidence.item(),
                'probabilities': probabilities.cpu().numpy().tolist()[0]
            }
            
            # Generate explanation if requested
            if original_image is not None and 'multiclass' in self.explainers:
                try:
                    explanation = self.explainers['multiclass'].generate_gradcam_explanations(
                        input_tensor, original_image
                    )
                    result['explanation'] = self._process_explanation(explanation)
                except Exception as e:
                    logging.warning(f"Could not generate multiclass explanation: {e}")
            
            return result
    
    def _process_explanation(self, explanation: Dict) -> Dict:
        """Process explanation results for API response"""
        if not explanation:
            return {}
        
        processed = {}
        for method_name, result in explanation.items():
            # Convert numpy arrays to base64 encoded images or URLs
            processed[method_name] = {
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence'],
                'heatmap_available': True
            }
        
        return processed


class PredictionLogger:
    """Logs predictions for monitoring and analysis"""
    
    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_prediction(
        self,
        prediction_id: str,
        image_info: Dict,
        prediction_result: Dict,
        processing_time: float,
        client_info: Dict
    ):
        """Log prediction details"""
        log_entry = {
            'prediction_id': prediction_id,
            'timestamp': datetime.now().isoformat(),
            'image_info': image_info,
            'prediction_result': prediction_result,
            'processing_time_ms': processing_time * 1000,
            'client_info': client_info
        }
        
        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\\n')


def create_app(config) -> Flask:
    """Create and configure Flask application"""
    app = Flask(__name__)
    
    # Configuration
    app.config['MAX_CONTENT_LENGTH'] = config.api.max_content_length
    app.config['UPLOAD_TIMEOUT'] = config.api.upload_timeout
    
    # Enable CORS
    CORS(app, origins=config.api.cors_origins)
    
    # Initialize components
    model_manager = ModelManager(config)
    prediction_logger = PredictionLogger(config.api.prediction_log_path)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/api.log'),
            logging.StreamHandler()
        ]
    )
    
    def validate_image(file) -> Tuple[bool, str]:
        """Validate uploaded image"""
        if not file:
            return False, "No file provided"
        
        if file.filename == '':
            return False, "No file selected"
        
        # Check file extension
        filename = secure_filename(file.filename.lower())
        if not any(filename.endswith(ext) for ext in config.api.allowed_extensions):
            return False, f"Invalid file type. Allowed: {', '.join(config.api.allowed_extensions)}"
        
        # Check file size
        file.seek(0, 2)  # Seek to end
        size = file.tell()
        file.seek(0)  # Reset
        
        if size > config.api.max_file_size:
            return False, f"File too large. Max size: {config.api.max_file_size // (1024*1024)}MB"
        
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
            'models_loaded': list(model_manager.models.keys()),
            'device': str(model_manager.device)
        })
    
    @app.route('/info', methods=['GET'])
    def get_info():
        """Get API information"""
        return jsonify({
            'api_version': config.version,
            'model_info': {
                'backbone': config.model.backbone,
                'use_ensemble': config.model.use_ensemble,
                'image_size': config.model.img_size
            },
            'supported_formats': config.api.allowed_extensions,
            'max_file_size_mb': config.api.max_file_size // (1024 * 1024),
            'fracture_types': config.model.fracture_types,
            'explainability_available': config.api.generate_gradcam
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
            prediction_result = model_manager.predict(
                image_array,
                return_probabilities=True,
                generate_explanation=generate_explanation
            )
            
            processing_time = time.time() - start_time
            
            # Prepare response
            response_data = {
                'prediction_id': prediction_id,
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round(processing_time * 1000, 2),
                **prediction_result
            }
            
            # Log prediction
            try:
                image_info = {
                    'filename': secure_filename(file.filename),
                    'size': image_array.shape,
                    'file_size_bytes': len(file.read())
                }
                file.seek(0)  # Reset file pointer
                
                client_info = {
                    'user_agent': request.headers.get('User-Agent'),
                    'ip_address': request.remote_addr
                }
                
                prediction_logger.log_prediction(
                    prediction_id, image_info, prediction_result,
                    processing_time, client_info
                )
            except Exception as e:
                logging.warning(f"Could not log prediction: {e}")
            
            return jsonify(response_data)
            
        except RequestEntityTooLarge:
            return jsonify({'error': 'File too large'}), 413
        
        except Exception as e:
            logging.error(f"Prediction error for {prediction_id}: {e}")
            logging.error(traceback.format_exc())
            
            return jsonify({
                'error': 'Internal server error',
                'prediction_id': prediction_id,
                'message': str(e) if config.api.debug else 'An error occurred during prediction'
            }), 500
    
    @app.route('/batch_predict', methods=['POST'])
    def batch_predict():
        """Batch prediction endpoint"""
        start_time = time.time()
        
        try:
            files = request.files.getlist('images')
            
            if not files or len(files) == 0:
                return jsonify({'error': 'No image files provided'}), 400
            
            if len(files) > 10:  # Limit batch size
                return jsonify({'error': 'Too many files. Maximum 10 images per batch'}), 400
            
            results = []
            
            for i, file in enumerate(files):
                try:
                    # Validate and process each file
                    is_valid, message = validate_image(file)
                    if not is_valid:
                        results.append({
                            'index': i,
                            'filename': file.filename,
                            'error': message,
                            'success': False
                        })
                        continue
                    
                    image_array, message = load_and_validate_image(file)
                    if image_array is None:
                        results.append({
                            'index': i,
                            'filename': file.filename,
                            'error': message,
                            'success': False
                        })
                        continue
                    
                    # Make prediction
                    prediction_result = model_manager.predict(image_array)
                    
                    results.append({
                        'index': i,
                        'filename': secure_filename(file.filename),
                        'success': True,
                        **prediction_result
                    })
                    
                except Exception as e:
                    results.append({
                        'index': i,
                        'filename': file.filename,
                        'error': str(e),
                        'success': False
                    })
            
            processing_time = time.time() - start_time
            
            return jsonify({
                'batch_id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round(processing_time * 1000, 2),
                'total_images': len(files),
                'successful_predictions': sum(1 for r in results if r.get('success')),
                'results': results
            })
            
        except Exception as e:
            logging.error(f"Batch prediction error: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
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
    """Main function to run the API server"""
    # Load configuration
    config = get_config()
    
    # Create directories
    Path("logs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    
    # Create Flask app
    app = create_app(config)
    
    # Run server
    logging.info(f"Starting Bone Fracture Detection API...")
    logging.info(f"Server: http://{config.api.host}:{config.api.port}")
    logging.info(f"Debug mode: {config.api.debug}")
    
    app.run(
        host=config.api.host,
        port=config.api.port,
        debug=config.api.debug,
        threaded=True
    )


if __name__ == '__main__':
    main()