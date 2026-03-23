#!/usr/bin/env python3
"""
FractureDetect AI - Project Setup Script
Comprehensive setup and installation script for the bone fracture detection system
"""

import os
import sys
import subprocess
import platform
import argparse
import logging
from pathlib import Path
import json
import urllib.request
import zipfile
import tarfile
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FractureDetectSetup:
    """Main setup class for FractureDetect AI"""
    
    def __init__(self, project_root: Optional[str] = None):
        """Initialize setup with project root directory"""
        self.project_root = Path(project_root or Path(__file__).parent.parent)
        self.platform = platform.system().lower()
        self.python_version = sys.version_info
        
        # Requirements
        self.min_python_version = (3, 8)
        self.required_packages = [
            'torch>=2.0.0',
            'torchvision>=0.15.0',
            'flask>=2.3.0',
            'pillow>=10.0.0',
            'numpy>=1.24.0',
            'opencv-python>=4.8.0',
            'scikit-learn>=1.3.0',
            'matplotlib>=3.7.0',
            'tqdm>=4.65.0'
        ]
        
        # Optional packages for enhanced features
        self.optional_packages = [
            'timm>=0.9.7',
            'albumentations>=1.3.0',
            'grad-cam>=1.4.8',
            'wandb>=0.15.0',
            'tensorboard>=2.13.0',
            'plotly>=5.15.0',
            'seaborn>=0.12.0'
        ]
        
    def check_python_version(self) -> bool:
        """Check if Python version meets requirements"""
        logger.info(f"Checking Python version: {self.python_version}")
        
        if self.python_version >= self.min_python_version:
            logger.info("✅ Python version check passed")
            return True
        else:
            logger.error(f"❌ Python {self.min_python_version[0]}.{self.min_python_version[1]}+ required")
            return False
    
    def check_system_requirements(self) -> Dict[str, bool]:
        """Check system requirements"""
        logger.info("Checking system requirements...")
        
        checks = {
            'python_version': self.check_python_version(),
            'pip_available': self._check_pip(),
            'git_available': self._check_git(),
            'sufficient_memory': self._check_memory(),
            'sufficient_disk': self._check_disk_space()
        }
        
        return checks
    
    def _check_pip(self) -> bool:
        """Check if pip is available"""
        try:
            subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                         check=True, capture_output=True)
            logger.info("✅ pip is available")
            return True
        except subprocess.CalledProcessError:
            logger.error("❌ pip is not available")
            return False
    
    def _check_git(self) -> bool:
        """Check if git is available"""
        try:
            subprocess.run(['git', '--version'], check=True, capture_output=True)
            logger.info("✅ git is available")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("⚠️ git is not available")
            return False
    
    def _check_memory(self) -> bool:
        """Check available memory"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb >= 4:
                logger.info(f"✅ Sufficient memory: {available_gb:.1f}GB available")
                return True
            else:
                logger.warning(f"⚠️ Low memory: {available_gb:.1f}GB available (4GB+ recommended)")
                return False
        except ImportError:
            logger.warning("⚠️ Cannot check memory (psutil not available)")
            return True
    
    def _check_disk_space(self) -> bool:
        """Check available disk space"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.project_root)
            free_gb = free / (1024**3)
            
            if free_gb >= 5:
                logger.info(f"✅ Sufficient disk space: {free_gb:.1f}GB available")
                return True
            else:
                logger.warning(f"⚠️ Low disk space: {free_gb:.1f}GB available (5GB+ recommended)")
                return False
        except:
            logger.warning("⚠️ Cannot check disk space")
            return True
    
    def create_virtual_environment(self, env_name: str = "venv") -> bool:
        """Create Python virtual environment"""
        logger.info(f"Creating virtual environment: {env_name}")
        
        env_path = self.project_root / env_name
        
        if env_path.exists():
            logger.info(f"Virtual environment already exists: {env_path}")
            return True
        
        try:
            subprocess.run([
                sys.executable, '-m', 'venv', str(env_path)
            ], check=True)
            
            logger.info(f"✅ Virtual environment created: {env_path}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to create virtual environment: {e}")
            return False
    
    def get_venv_python(self, env_name: str = "venv") -> str:
        """Get path to Python executable in virtual environment"""
        env_path = self.project_root / env_name
        
        if self.platform == "windows":
            python_path = env_path / "Scripts" / "python.exe"
        else:
            python_path = env_path / "bin" / "python"
        
        return str(python_path)
    
    def install_requirements(self, env_name: str = "venv", include_optional: bool = False) -> bool:
        """Install required packages"""
        logger.info("Installing required packages...")
        
        python_path = self.get_venv_python(env_name)
        
        # Install core requirements
        packages = self.required_packages.copy()
        if include_optional:
            packages.extend(self.optional_packages)
        
        for package in packages:
            try:
                logger.info(f"Installing {package}...")
                subprocess.run([
                    python_path, '-m', 'pip', 'install', package
                ], check=True, capture_output=True)
                logger.info(f"✅ Installed {package}")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install {package}: {e}")
                return False
        
        # Install from requirements.txt if available
        requirements_file = self.project_root / "requirements.txt"
        if requirements_file.exists():
            try:
                logger.info("Installing from requirements.txt...")
                subprocess.run([
                    python_path, '-m', 'pip', 'install', '-r', str(requirements_file)
                ], check=True)
                logger.info("✅ Installed packages from requirements.txt")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️ Some packages from requirements.txt failed to install: {e}")
        
        return True
    
    def create_directories(self) -> bool:
        """Create necessary project directories"""
        logger.info("Creating project directories...")
        
        directories = [
            "models/checkpoints",
            "models/pretrained", 
            "logs",
            "outputs/predictions",
            "outputs/explanations",
            "outputs/reports",
            "data/sample",
            "configs",
            "docs",
            "tests",
            "scripts",
            "backend",
            "frontend"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Created directory: {directory}")
        
        return True
    
    def create_config_files(self) -> bool:
        """Create default configuration files"""
        logger.info("Creating configuration files...")
        
        # Create default config.yaml
        config_content = """
# FractureDetect AI Configuration
project_name: "Bone Fracture Detection System"
version: "2.0.0"
author: "Your Name"

model:
  backbone: "efficientnet-b4"
  pretrained: true
  freeze_backbone: true
  dropout_rate: 0.4
  img_size: 384
  batch_size: 16
  epochs: 50
  learning_rate: 3e-4
  weight_decay: 1e-4
  use_ensemble: true

data:
  data_dir: "Dataset"
  train_split: 0.8
  val_split: 0.2
  num_workers: 4
  pin_memory: true

api:
  host: "0.0.0.0"
  port: 5000
  debug: false
  max_file_size: 10485760  # 10MB
  generate_gradcam: true

evaluation:
  compute_detailed_metrics: true
  save_confusion_matrix: true
  save_roc_curves: true

deployment:
  docker_image: "fracture-detection:latest"
  min_replicas: 1
  max_replicas: 5

# Device configuration
device: "auto"
use_gpu: true
random_seed: 42
deterministic: true

# Logging
log_level: "INFO"
log_file: "logs/fracture_detection.log"

# Experiment tracking
use_wandb: false
wandb_project: "fracture-detection"
"""
        
        config_file = self.project_root / "configs" / "config.yaml"
        with open(config_file, 'w') as f:
            f.write(config_content.strip())
        
        logger.info(f"✅ Created config file: {config_file}")
        
        # Create .env file
        env_content = """
# Environment Variables for FractureDetect AI
DEVICE=auto
DATA_DIR=Dataset
MODEL_DIR=models
API_PORT=5000
DEBUG=false
LOG_LEVEL=INFO
USE_WANDB=false
"""
        
        env_file = self.project_root / ".env"
        with open(env_file, 'w') as f:
            f.write(env_content.strip())
        
        logger.info(f"✅ Created environment file: {env_file}")
        
        return True
    
    def download_sample_data(self) -> bool:
        """Download sample data for testing"""
        logger.info("Setting up sample data...")
        
        # Create sample data directory structure
        sample_dir = self.project_root / "data" / "sample"
        
        # Create directories for binary classification
        for body_part in ["ELBOW", "HAND", "SHOULDER", "WRIST"]:
            for patient in ["patient001", "patient002"]:
                for study_type in ["positive", "negative"]:
                    study_dir = sample_dir / "train_valid" / body_part / patient / f"study_{study_type}"
                    study_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directories for multiclass classification
        fracture_types = [
            "Compression-Crush fracture",
            "Fracture Dislocation",
            "Hairline Fracture",
            "Impacted fracture",
            "Longitudinal fracture",
            "Oblique fracture",
            "Spiral Fracture"
        ]
        
        for fracture_type in fracture_types:
            fracture_dir = sample_dir / "classification" / fracture_type
            fracture_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ Sample data directory structure created")
        logger.info("📝 Note: Add your X-ray images to the data/sample directory")
        
        return True
    
    def create_startup_scripts(self) -> bool:
        """Create startup scripts for different platforms"""
        logger.info("Creating startup scripts...")
        
        # Windows batch script
        windows_script = f"""@echo off
echo Starting FractureDetect AI...

REM Activate virtual environment
call venv\\Scripts\\activate.bat

REM Start backend
echo Starting backend server...
start /B python backend\\enhanced_app.py

REM Wait a moment for backend to start
timeout /t 5 /nobreak >nul

REM Start frontend
echo Starting frontend server...
cd frontend
start /B python -m http.server 8080

echo ✅ FractureDetect AI is running!
echo 🌐 Frontend: http://localhost:8080/enhanced_index.html
echo 🔧 Backend API: http://localhost:5000
echo 📊 Health Check: http://localhost:5000/health

pause
"""
        
        windows_file = self.project_root / "start_windows.bat"
        with open(windows_file, 'w') as f:
            f.write(windows_script)
        
        # Unix shell script
        unix_script = f"""#!/bin/bash
echo "Starting FractureDetect AI..."

# Activate virtual environment
source venv/bin/activate

# Start backend
echo "Starting backend server..."
python backend/enhanced_app.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 5

# Start frontend
echo "Starting frontend server..."
cd frontend
python -m http.server 8080 &
FRONTEND_PID=$!

echo "✅ FractureDetect AI is running!"
echo "🌐 Frontend: http://localhost:8080/enhanced_index.html"
echo "🔧 Backend API: http://localhost:5000"
echo "📊 Health Check: http://localhost:5000/health"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap 'echo "Stopping services..."; kill $BACKEND_PID $FRONTEND_PID; exit' INT
wait
"""
        
        unix_file = self.project_root / "start_unix.sh"
        with open(unix_file, 'w') as f:
            f.write(unix_script)
        
        # Make Unix script executable
        try:
            os.chmod(unix_file, 0o755)
        except:
            pass
        
        logger.info(f"✅ Created startup scripts")
        
        return True
    
    def run_tests(self, env_name: str = "venv") -> bool:
        """Run basic tests to verify installation"""
        logger.info("Running installation tests...")
        
        python_path = self.get_venv_python(env_name)
        
        # Test Python imports
        test_imports = [
            'torch',
            'torchvision', 
            'flask',
            'PIL',
            'cv2',
            'numpy',
            'sklearn'
        ]
        
        for module in test_imports:
            try:
                subprocess.run([
                    python_path, '-c', f'import {module}; print("✅ {module} imported successfully")'
                ], check=True, capture_output=True, text=True)
                logger.info(f"✅ {module} test passed")
            except subprocess.CalledProcessError:
                logger.error(f"❌ {module} test failed")
                return False
        
        # Test CUDA availability (optional)
        try:
            result = subprocess.run([
                python_path, '-c', 
                'import torch; print(f"CUDA available: {torch.cuda.is_available()}")'
            ], capture_output=True, text=True)
            logger.info(f"GPU Info: {result.stdout.strip()}")
        except:
            logger.info("Could not check CUDA availability")
        
        return True
    
    def print_installation_summary(self):
        """Print installation summary and next steps"""
        logger.info("\n" + "="*60)
        logger.info("🎉 INSTALLATION COMPLETE!")
        logger.info("="*60)
        
        print("\n📋 Installation Summary:")
        print(f"   📁 Project Root: {self.project_root}")
        print(f"   🐍 Python Version: {self.python_version[0]}.{self.python_version[1]}.{self.python_version[2]}")
        print(f"   💻 Platform: {self.platform}")
        
        print("\n🚀 Quick Start:")
        if self.platform == "windows":
            print("   1. Run: start_windows.bat")
        else:
            print("   1. Run: ./start_unix.sh")
        print("   2. Open: http://localhost:8080/enhanced_index.html")
        print("   3. Upload an X-ray image and analyze!")
        
        print("\n📚 Documentation:")
        print("   📖 Installation Guide: docs/installation.md")
        print("   🚀 Deployment Guide: docs/deployment.md")
        print("   🧪 Run Tests: python -m pytest tests/")
        
        print("\n⚙️ Configuration:")
        print("   📝 Edit: configs/config.yaml")
        print("   🔧 Environment: .env")
        
        print("\n🤝 Support:")
        print("   💬 Issues: https://github.com/yourusername/fracture-detection/issues")
        print("   📧 Email: support@fracturedetect.ai")
        
        print("\n" + "="*60)
        logger.info("Ready to detect fractures with AI! 🩻✨")
        print("="*60)


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="FractureDetect AI Setup Script")
    parser.add_argument("--project-root", help="Project root directory")
    parser.add_argument("--env-name", default="venv", help="Virtual environment name")
    parser.add_argument("--include-optional", action="store_true", 
                       help="Install optional packages")
    parser.add_argument("--skip-tests", action="store_true", 
                       help="Skip installation tests")
    parser.add_argument("--quiet", action="store_true", 
                       help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Initialize setup
    setup = FractureDetectSetup(args.project_root)
    
    logger.info("🩻 FractureDetect AI Setup Starting...")
    logger.info(f"Project Root: {setup.project_root}")
    
    # Check system requirements
    checks = setup.check_system_requirements()
    if not all(checks.values()):
        logger.error("❌ System requirements check failed")
        critical_checks = ['python_version', 'pip_available']
        if not all(checks[check] for check in critical_checks):
            logger.error("Critical requirements not met. Please fix and try again.")
            sys.exit(1)
        else:
            logger.warning("Some optional requirements not met. Continuing...")
    
    # Setup steps
    steps = [
        ("Creating directories", lambda: setup.create_directories()),
        ("Creating virtual environment", lambda: setup.create_virtual_environment(args.env_name)),
        ("Installing packages", lambda: setup.install_requirements(args.env_name, args.include_optional)),
        ("Creating configuration files", lambda: setup.create_config_files()),
        ("Setting up sample data", lambda: setup.download_sample_data()),
        ("Creating startup scripts", lambda: setup.create_startup_scripts()),
    ]
    
    if not args.skip_tests:
        steps.append(("Running tests", lambda: setup.run_tests(args.env_name)))
    
    # Execute setup steps
    for step_name, step_func in steps:
        logger.info(f"📋 {step_name}...")
        try:
            if not step_func():
                logger.error(f"❌ {step_name} failed")
                sys.exit(1)
        except Exception as e:
            logger.error(f"❌ {step_name} failed: {e}")
            sys.exit(1)
    
    # Print summary
    setup.print_installation_summary()


if __name__ == "__main__":
    main()