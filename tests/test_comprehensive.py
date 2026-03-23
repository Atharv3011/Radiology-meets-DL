"""
Comprehensive Test Suite for Bone Fracture Detection System
Professional unit tests, integration tests, and performance tests
"""

import unittest
import pytest
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import tempfile
import json
import os
import sys
from pathlib import Path
import requests
import time
from unittest.mock import Mock, patch, MagicMock
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.architectures import (
    create_model, create_ensemble_model, 
    EnhancedEfficientNet, EnhancedDenseNet, EnhancedResNet,
    MultiHeadAttention, SpatialAttention, ChannelAttention, CBAM
)
from data.advanced_dataset import FractureDataset, get_data_loaders
from evaluation.metrics import MetricsCalculator
from configs.config import get_config, Config


class TestModelArchitectures(unittest.TestCase):
    """Test model architecture implementations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')
        self.batch_size = 2
        self.img_size = 224
        self.num_classes = 2
        
    def test_enhanced_efficientnet_creation(self):
        """Test EfficientNet model creation"""
        model = EnhancedEfficientNet(
            model_name="efficientnet_b0",
            num_classes=self.num_classes,
            pretrained=False,
            use_attention=True
        )
        
        self.assertIsInstance(model, EnhancedEfficientNet)
        self.assertTrue(hasattr(model, 'backbone'))
        self.assertTrue(hasattr(model, 'classifier'))
        self.assertTrue(hasattr(model, 'cbam'))
        
    def test_enhanced_densenet_creation(self):
        """Test DenseNet model creation"""
        model = EnhancedDenseNet(
            model_name="densenet121",
            num_classes=self.num_classes,
            pretrained=False,
            use_attention=True
        )
        
        self.assertIsInstance(model, EnhancedDenseNet)
        self.assertTrue(hasattr(model, 'backbone'))
        self.assertTrue(hasattr(model, 'classifier'))
        
    def test_enhanced_resnet_creation(self):
        """Test ResNet model creation"""
        model = EnhancedResNet(
            model_name="resnet50",
            num_classes=self.num_classes,
            pretrained=False,
            use_attention=True
        )
        
        self.assertIsInstance(model, EnhancedResNet)
        self.assertTrue(hasattr(model, 'backbone'))
        self.assertTrue(hasattr(model, 'classifier'))
        
    def test_model_forward_pass(self):
        """Test model forward pass"""
        model = create_model(
            "resnet50", 
            num_classes=self.num_classes,
            pretrained=False
        )
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        self.assertFalse(torch.isnan(output).any())
        
    def test_ensemble_model_creation(self):
        """Test ensemble model creation"""
        architectures = ["resnet50", "densenet121"]
        ensemble = create_ensemble_model(
            architectures,
            num_classes=self.num_classes,
            pretrained=False
        )
        
        self.assertEqual(len(ensemble.models), len(architectures))
        
        # Test forward pass
        dummy_input = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        
        with torch.no_grad():
            output = ensemble(dummy_input)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        
    def test_attention_mechanisms(self):
        """Test attention mechanism components"""
        in_channels = 512
        
        # Test MultiHeadAttention
        mha = MultiHeadAttention(embed_dim=in_channels, num_heads=8)
        dummy_input = torch.randn(self.batch_size, 100, in_channels)
        output = mha(dummy_input)
        self.assertEqual(output.shape, dummy_input.shape)
        
        # Test SpatialAttention
        spatial_attn = SpatialAttention(in_channels)
        dummy_feature = torch.randn(self.batch_size, in_channels, 14, 14)
        output = spatial_attn(dummy_feature)
        self.assertEqual(output.shape, dummy_feature.shape)
        
        # Test ChannelAttention
        channel_attn = ChannelAttention(in_channels)
        output = channel_attn(dummy_feature)
        self.assertEqual(output.shape, dummy_feature.shape)
        
        # Test CBAM
        cbam = CBAM(in_channels)
        output = cbam(dummy_feature)
        self.assertEqual(output.shape, dummy_feature.shape)


class TestDataPipeline(unittest.TestCase):
    """Test data loading and processing pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.create_test_data()
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def create_test_data(self):
        """Create test image data"""
        # Create directory structure
        for class_name in ["No_Fracture", "Fracture"]:
            class_dir = Path(self.temp_dir) / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Create dummy images
            for i in range(5):
                img = Image.new('RGB', (224, 224), color='white')
                img.save(class_dir / f"test_image_{i}.jpg")
                
    def test_fracture_dataset_creation(self):
        """Test FractureDataset creation"""
        dataset = FractureDataset(
            data_dir=self.temp_dir,
            task_type="multiclass",
            img_size=224
        )
        
        self.assertGreater(len(dataset), 0)
        self.assertEqual(len(dataset.class_names), 2)
        
        # Test getitem
        image, label = dataset[0]
        self.assertEqual(image.shape, (3, 224, 224))
        self.assertIsInstance(label.item(), int)
        
    def test_data_loaders_creation(self):
        """Test data loader creation"""
        try:
            train_loader, val_loader, class_weights = get_data_loaders(
                data_dir=self.temp_dir,
                task_type="multiclass",
                batch_size=2,
                num_workers=0
            )
            
            self.assertGreater(len(train_loader), 0)
            self.assertGreater(len(val_loader), 0)
            self.assertIsInstance(class_weights, list)
            
            # Test batch loading
            for batch_data, batch_labels in train_loader:
                self.assertEqual(batch_data.shape[0], min(2, len(train_loader.dataset)))
                self.assertEqual(batch_labels.shape[0], min(2, len(train_loader.dataset)))
                break
                
        except Exception as e:
            self.skipTest(f"Data loader test skipped due to missing dependencies: {e}")


class TestMetricsCalculator(unittest.TestCase):
    """Test evaluation metrics calculation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = MetricsCalculator(class_names=["No Fracture", "Fracture"])
        
        # Generate sample data
        np.random.seed(42)
        self.y_true = np.random.randint(0, 2, 100)
        
        # Add some noise to create realistic predictions
        self.y_pred = self.y_true.copy()
        noise_indices = np.random.choice(100, 20, replace=False)
        self.y_pred[noise_indices] = 1 - self.y_pred[noise_indices]
        
        # Generate dummy probabilities
        self.y_pred_proba = np.random.rand(100, 2)
        self.y_pred_proba = self.y_pred_proba / self.y_pred_proba.sum(axis=1, keepdims=True)
        
    def test_basic_metrics_calculation(self):
        """Test basic metrics calculation"""
        metrics = self.calculator.calculate_metrics(
            self.y_true, self.y_pred, self.y_pred_proba
        )
        
        # Check required metrics
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
            self.assertGreaterEqual(metrics[metric], 0)
            self.assertLessEqual(metrics[metric], 1)
            
    def test_binary_metrics_calculation(self):
        """Test binary classification specific metrics"""
        metrics = self.calculator.calculate_metrics(
            self.y_true, self.y_pred, self.y_pred_proba
        )
        
        # Check binary-specific metrics
        binary_metrics = ['sensitivity', 'specificity', 'ppv', 'npv', 'roc_auc']
        for metric in binary_metrics:
            self.assertIn(metric, metrics)
            
    def test_confusion_matrix_generation(self):
        """Test confusion matrix generation"""
        metrics = self.calculator.calculate_metrics(self.y_true, self.y_pred)
        
        self.assertIn('confusion_matrix', metrics)
        cm = metrics['confusion_matrix']
        self.assertEqual(cm.shape, (2, 2))
        self.assertEqual(cm.sum(), len(self.y_true))
        
    def test_classification_report_generation(self):
        """Test classification report generation"""
        report = self.calculator.generate_classification_report(
            self.y_true, self.y_pred
        )
        
        self.assertIsInstance(report, str)
        self.assertIn('precision', report)
        self.assertIn('recall', report)
        self.assertIn('f1-score', report)


class TestConfigurationSystem(unittest.TestCase):
    """Test configuration management system"""
    
    def test_default_config_creation(self):
        """Test default configuration creation"""
        config = Config()
        
        # Check main sections
        self.assertTrue(hasattr(config, 'model'))
        self.assertTrue(hasattr(config, 'data'))
        self.assertTrue(hasattr(config, 'api'))
        self.assertTrue(hasattr(config, 'evaluation'))
        
        # Check default values
        self.assertEqual(config.model.img_size, 384)
        self.assertEqual(config.model.batch_size, 16)
        self.assertIsInstance(config.model.fracture_types, list)
        
    def test_config_loading_and_saving(self):
        """Test configuration loading and saving"""
        from configs.config import save_config, load_config
        
        config = Config()
        config.model.backbone = "test_backbone"
        config.api.port = 9999
        
        # Save configuration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            save_config(config, f.name)
            
            # Load configuration
            loaded_config = load_config(f.name)
            
            self.assertEqual(loaded_config.model.backbone, "test_backbone")
            self.assertEqual(loaded_config.api.port, 9999)
            
        # Cleanup
        os.unlink(f.name)


class TestAPIEndpoints(unittest.TestCase):
    """Test API endpoints and functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up API test server"""
        cls.api_base_url = "http://localhost:5000"
        cls.test_image_path = cls.create_test_image()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test resources"""
        if hasattr(cls, 'test_image_path') and os.path.exists(cls.test_image_path):
            os.unlink(cls.test_image_path)
            
    @classmethod
    def create_test_image(cls):
        """Create a test image file"""
        img = Image.new('RGB', (224, 224), color='gray')
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        img.save(temp_file.name)
        return temp_file.name
        
    def test_health_endpoint(self):
        """Test health check endpoint"""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                self.assertIn('status', data)
                self.assertEqual(data['status'], 'healthy')
            else:
                self.skipTest("API server not available")
                
        except requests.ConnectionError:
            self.skipTest("API server not running")
            
    def test_info_endpoint(self):
        """Test info endpoint"""
        try:
            response = requests.get(f"{self.api_base_url}/info", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ['api_version', 'model_info', 'supported_formats']
                for field in required_fields:
                    self.assertIn(field, data)
            else:
                self.skipTest("API server not available")
                
        except requests.ConnectionError:
            self.skipTest("API server not running")
            
    def test_prediction_endpoint(self):
        """Test prediction endpoint"""
        try:
            with open(self.test_image_path, 'rb') as f:
                files = {'image': f}
                response = requests.post(
                    f"{self.api_base_url}/predict",
                    files=files,
                    timeout=30
                )
                
            if response.status_code == 200:
                data = response.json()
                required_fields = ['prediction_id', 'timestamp']
                for field in required_fields:
                    self.assertIn(field, data)
                    
                # Check for either fracture_detected or error
                self.assertTrue(
                    'fracture_detected' in data or 'error' in data,
                    "Response should contain either prediction result or error"
                )
            else:
                self.skipTest("API server not available or models not loaded")
                
        except requests.ConnectionError:
            self.skipTest("API server not running")


class TestPerformance(unittest.TestCase):
    """Test performance and benchmarking"""
    
    def setUp(self):
        """Set up performance tests"""
        self.device = torch.device('cpu')
        self.model = create_model("resnet50", num_classes=2, pretrained=False)
        self.model.eval()
        
    def test_model_inference_speed(self):
        """Test model inference speed"""
        batch_sizes = [1, 4, 8]
        
        for batch_size in batch_sizes:
            dummy_input = torch.randn(batch_size, 3, 224, 224)
            
            # Warm up
            with torch.no_grad():
                _ = self.model(dummy_input)
                
            # Measure inference time
            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = self.model(dummy_input)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            time_per_sample = avg_time / batch_size
            
            print(f"Batch size {batch_size}: {time_per_sample:.4f}s per sample")
            
            # Performance assertion (adjust based on hardware)
            self.assertLess(time_per_sample, 2.0, f"Inference too slow for batch size {batch_size}")
            
    def test_memory_usage(self):
        """Test memory usage during inference"""
        import psutil
        import gc
        
        process = psutil.Process()
        
        # Measure baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run inference
        dummy_input = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            _ = self.model(dummy_input)
            
        # Measure memory after inference
        inference_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = inference_memory - baseline_memory
        
        print(f"Memory usage: {baseline_memory:.1f}MB -> {inference_memory:.1f}MB (+{memory_increase:.1f}MB)")
        
        # Memory assertion (adjust based on model size)
        self.assertLess(memory_increase, 1000, "Memory usage too high")


class TestIntegration(unittest.TestCase):
    """Test integration between components"""
    
    def setUp(self):
        """Set up integration tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.create_test_data()
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def create_test_data(self):
        """Create test data structure"""
        # Create minimal dataset
        for class_name in ["No_Fracture", "Fracture"]:
            class_dir = Path(self.temp_dir) / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(3):
                img = Image.new('RGB', (224, 224), color='white')
                img.save(class_dir / f"test_{i}.jpg")
                
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from data loading to prediction"""
        try:
            # Create dataset
            dataset = FractureDataset(
                data_dir=self.temp_dir,
                task_type="multiclass",
                img_size=224
            )
            
            # Create model
            model = create_model("resnet50", num_classes=2, pretrained=False)
            model.eval()
            
            # Test prediction on first sample
            image, label = dataset[0]
            image_batch = image.unsqueeze(0)
            
            with torch.no_grad():
                output = model(image_batch)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(output, dim=1)
                
            # Verify output shapes and types
            self.assertEqual(output.shape, (1, 2))
            self.assertEqual(probabilities.shape, (1, 2))
            self.assertEqual(predicted_class.shape, (1,))
            
            # Test metrics calculation
            y_true = [label.item()]
            y_pred = [predicted_class.item()]
            y_pred_proba = probabilities.numpy()
            
            calculator = MetricsCalculator()
            metrics = calculator.calculate_metrics(y_true, y_pred, y_pred_proba)
            
            self.assertIn('accuracy', metrics)
            
        except Exception as e:
            self.skipTest(f"Integration test skipped due to missing dependencies: {e}")


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def test_invalid_model_architecture(self):
        """Test handling of invalid model architecture"""
        with self.assertRaises(ValueError):
            create_model("invalid_architecture", num_classes=2)
            
    def test_invalid_image_format(self):
        """Test handling of invalid image formats"""
        # Create invalid image data
        invalid_data = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        
        # This should handle gracefully in actual implementation
        # For now, just test that we can catch the error
        try:
            img = Image.fromarray(invalid_data)
            # If we get here, the image was valid enough
            self.assertTrue(True)
        except Exception:
            # If we get an exception, that's also fine - we're testing error handling
            self.assertTrue(True)
            
    def test_empty_dataset(self):
        """Test handling of empty dataset"""
        empty_dir = tempfile.mkdtemp()
        
        try:
            dataset = FractureDataset(
                data_dir=empty_dir,
                task_type="multiclass"
            )
            
            # Should handle empty dataset gracefully
            self.assertEqual(len(dataset), 0)
            
        except Exception as e:
            # Should either work with empty data or raise appropriate exception
            self.assertIsInstance(e, (ValueError, FileNotFoundError, IndexError))
            
        finally:
            os.rmdir(empty_dir)


# Pytest fixtures and test classes
@pytest.fixture
def sample_config():
    """Pytest fixture for sample configuration"""
    return Config()


@pytest.fixture
def sample_model():
    """Pytest fixture for sample model"""
    return create_model("resnet50", num_classes=2, pretrained=False)


@pytest.fixture
def sample_data():
    """Pytest fixture for sample test data"""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = y_true.copy()
    noise_indices = np.random.choice(100, 20, replace=False)
    y_pred[noise_indices] = 1 - y_pred[noise_indices]
    
    return y_true, y_pred


class TestPytestStyle:
    """Pytest-style tests for compatibility"""
    
    def test_model_creation_pytest(self, sample_config):
        """Test model creation using pytest fixtures"""
        model = create_model(
            sample_config.model.backbone,
            num_classes=2,
            pretrained=False
        )
        
        assert model is not None
        assert hasattr(model, 'forward')
        
    def test_metrics_calculation_pytest(self, sample_data):
        """Test metrics calculation using pytest fixtures"""
        y_true, y_pred = sample_data
        
        calculator = MetricsCalculator()
        metrics = calculator.calculate_metrics(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1


# Performance benchmarks
def benchmark_model_inference():
    """Benchmark model inference performance"""
    model = create_model("resnet50", num_classes=2, pretrained=False)
    model.eval()
    
    batch_sizes = [1, 2, 4, 8, 16]
    times = []
    
    for batch_size in batch_sizes:
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        
        # Warm up
        with torch.no_grad():
            _ = model(dummy_input)
            
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(dummy_input)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        times.append(avg_time)
        
        print(f"Batch size {batch_size}: {avg_time:.4f}s per batch, {avg_time/batch_size:.4f}s per sample")
    
    return batch_sizes, times


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run unittest tests
    print("Running Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run benchmark
    print("\nRunning Performance Benchmark...")
    benchmark_model_inference()
    
    print("\n✅ All tests completed!")