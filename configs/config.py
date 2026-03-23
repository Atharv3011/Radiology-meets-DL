"""
Professional Configuration Management for Bone Fracture Detection System
Centralized configuration with environment support and validation
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model architecture and training configuration"""
    # Architecture
    backbone: str = "efficientnet-b4"  # efficientnet-b4, densenet121, resnet50
    pretrained: bool = True
    freeze_backbone: bool = True
    dropout_rate: float = 0.4
    
    # Training
    img_size: int = 384  # Increased for better performance
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    
    # Binary classification
    binary_classes: List[str] = field(default_factory=lambda: ["No Fracture", "Fracture"])
    
    # Multi-class classification  
    fracture_types: List[str] = field(default_factory=lambda: [
        "Compression-Crush fracture",
        "Fracture Dislocation", 
        "Hairline Fracture",
        "Impacted fracture",
        "Longitudinal fracture",
        "Oblique fracture",
        "Spiral Fracture"
    ])
    
    # Advanced training settings
    use_mixed_precision: bool = True
    gradient_clip_norm: float = 1.0
    scheduler: str = "cosine"  # cosine, plateau, step
    warmup_epochs: int = 5
    
    # Model ensemble
    use_ensemble: bool = True
    ensemble_models: List[str] = field(default_factory=lambda: [
        "efficientnet-b4", "densenet121", "resnet50"
    ])


@dataclass 
class DataConfig:
    """Data processing and augmentation configuration"""
    # Paths
    data_dir: str = "Dataset"
    train_dir: str = "Dataset/train_valid"
    test_dir: str = "Dataset/test"
    classification_dir: str = "Dataset/classification"
    
    # Data split
    train_split: float = 0.8
    val_split: float = 0.2
    test_split: float = 0.0  # Use separate test set
    
    # Image preprocessing
    img_size: int = 384
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Data augmentation
    use_advanced_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    rotation_degrees: int = 20
    brightness_factor: float = 0.3
    contrast_factor: float = 0.3
    blur_prob: float = 0.1
    noise_prob: float = 0.1
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass
class APIConfig:
    """Backend API configuration"""
    # Server settings
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    
    # Performance
    max_content_length: int = 16 * 1024 * 1024  # 16MB
    upload_timeout: int = 30
    prediction_timeout: int = 10
    
    # Security
    allowed_extensions: List[str] = field(default_factory=lambda: [
        'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'
    ])
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    
    # CORS
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # Caching
    use_redis: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    cache_ttl: int = 3600  # 1 hour
    
    # Model paths
    binary_model_path: str = "models/binary_fracture_ensemble.pth"
    multiclass_model_path: str = "models/multiclass_fracture_ensemble.pth"
    
    # Explainability
    generate_gradcam: bool = True
    save_predictions: bool = True
    prediction_log_path: str = "logs/predictions.json"


@dataclass
class ExplainabilityConfig:
    """Model explainability configuration"""
    # Grad-CAM settings
    use_gradcam: bool = True
    gradcam_layer: str = "layer4"  # For ResNet
    gradcam_alpha: float = 0.4
    
    # Attention maps
    use_attention: bool = True
    attention_layers: List[str] = field(default_factory=lambda: ["layer3", "layer4"])
    
    # Output settings
    save_heatmaps: bool = True
    heatmap_dir: str = "outputs/heatmaps"
    overlay_original: bool = True


@dataclass
class EvaluationConfig:
    """Model evaluation configuration"""
    # Metrics
    compute_detailed_metrics: bool = True
    save_confusion_matrix: bool = True
    save_roc_curves: bool = True
    save_pr_curves: bool = True
    
    # Cross-validation
    use_cross_validation: bool = True
    cv_folds: int = 5
    stratified: bool = True
    
    # Bootstrap evaluation
    use_bootstrap: bool = True
    bootstrap_samples: int = 1000
    confidence_interval: float = 0.95
    
    # Output paths
    results_dir: str = "evaluation/results"
    plots_dir: str = "evaluation/plots"
    reports_dir: str = "evaluation/reports"


@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    # Docker
    docker_image: str = "fracture-detection:latest"
    container_port: int = 5000
    
    # Scaling
    min_replicas: int = 1
    max_replicas: int = 5
    cpu_limit: str = "2"
    memory_limit: str = "4Gi"
    
    # Health checks
    health_check_path: str = "/health"
    readiness_check_path: str = "/ready"
    
    # Monitoring
    enable_prometheus: bool = True
    metrics_port: int = 8080
    log_level: str = "INFO"


@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    api: APIConfig = field(default_factory=APIConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    
    # Global settings
    project_name: str = "Bone Fracture Detection System"
    version: str = "2.0.0"
    author: str = "Your Name"
    description: str = "AI-Powered Medical Image Analysis for Fracture Classification"
    
    # Device configuration
    device: str = "auto"  # auto, cpu, cuda, mps
    use_gpu: bool = True
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    
    # Reproducibility
    random_seed: int = 42
    deterministic: bool = True
    benchmark: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/fracture_detection.log"
    
    # Experiment tracking
    use_wandb: bool = True
    wandb_project: str = "fracture-detection"
    wandb_entity: Optional[str] = None


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file or environment variables"""
    config = Config()
    
    # Load from YAML if provided
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # Check if yaml_config is a dictionary
            if isinstance(yaml_config, dict):
                # Update config with YAML values
                for section, values in yaml_config.items():
                    if hasattr(config, section):
                        section_config = getattr(config, section)
                        if isinstance(values, dict):
                            for key, value in values.items():
                                if hasattr(section_config, key):
                                    setattr(section_config, key, value)
                        else:
                            # Handle direct values (not nested dictionaries)
                            setattr(config, section, values)
            else:
                print(f"Warning: YAML config file {config_path} is not a valid dictionary")
        except Exception as e:
            print(f"Warning: Could not load YAML config from {config_path}: {e}")
    
    # Override with environment variables
    _load_env_overrides(config)
    
    return config


def _load_env_overrides(config: Config) -> None:
    """Load configuration overrides from environment variables"""
    # Model settings
    if os.getenv("MODEL_BACKBONE"):
        config.model.backbone = os.getenv("MODEL_BACKBONE")
    if os.getenv("BATCH_SIZE"):
        config.model.batch_size = int(os.getenv("BATCH_SIZE"))
    if os.getenv("EPOCHS"):
        config.model.epochs = int(os.getenv("EPOCHS"))
    
    # API settings  
    if os.getenv("API_PORT"):
        config.api.port = int(os.getenv("API_PORT"))
    if os.getenv("API_HOST"):
        config.api.host = os.getenv("API_HOST")
    if os.getenv("DEBUG"):
        config.api.debug = os.getenv("DEBUG").lower() == "true"
    
    # Device settings
    if os.getenv("DEVICE"):
        config.device = os.getenv("DEVICE")
    
    # Paths
    if os.getenv("DATA_DIR"):
        config.data.data_dir = os.getenv("DATA_DIR")


def save_config(config: Config, output_path: str) -> None:
    """Save configuration to YAML file"""
    config_dict = {}
    
    # Convert dataclass to dict
    for field_name in config.__dataclass_fields__:
        field_value = getattr(config, field_name)
        if hasattr(field_value, '__dataclass_fields__'):
            # Nested dataclass
            nested_dict = {}
            for nested_field in field_value.__dataclass_fields__:
                nested_dict[nested_field] = getattr(field_value, nested_field)
            config_dict[field_name] = nested_dict
        else:
            config_dict[field_name] = field_value
    
    # Save to YAML
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _config
    if _config is None:
        config_path = os.getenv("CONFIG_PATH", "configs/config.yaml")
        _config = load_config(config_path)
    return _config


def set_config(config: Config) -> None:
    """Set global configuration instance"""
    global _config
    _config = config


if __name__ == "__main__":
    # Example usage
    config = get_config()
    print(f"Project: {config.project_name} v{config.version}")
    print(f"Model: {config.model.backbone}")
    print(f"API Port: {config.api.port}")
    
    # Save default configuration
    save_config(config, "configs/config.yaml")