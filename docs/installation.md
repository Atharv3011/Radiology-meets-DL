# 🚀 Installation Guide - FractureDetect AI

## System Requirements

### Hardware Requirements
- **CPU**: Intel i5-4590 / AMD FX 8350 or equivalent
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA GTX 1060 6GB or better (optional but recommended)
- **Storage**: 5GB free space
- **Internet**: Stable connection for downloading dependencies

### Software Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Ubuntu 18.04+
- **Python**: 3.8 or higher
- **Node.js**: 16.0+ (for development tools)
- **Git**: Latest version

## Quick Start (5 Minutes)

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/fracture-detection.git
cd fracture-detection
```

### 2. Install Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Download Pre-trained Models
```bash
# Download models (if available)
python scripts/download_models.py

# Or train your own models
python scripts/train_advanced.py
```

### 4. Start the Application
```bash
# Start backend
python backend/enhanced_app.py

# Open frontend (in another terminal)
# Navigate to frontend directory and open enhanced_index.html
```

## Detailed Installation

### Step 1: Environment Setup

#### Python Virtual Environment
```bash
# Create isolated environment
python -m venv fracture_env

# Activate environment
# Windows (Command Prompt)
fracture_env\Scripts\activate

# Windows (PowerShell)
fracture_env\Scripts\Activate.ps1

# macOS/Linux
source fracture_env/bin/activate
```

#### Verify Python Installation
```bash
python --version  # Should be 3.8+
pip --version     # Should be latest
```

### Step 2: GPU Setup (Optional but Recommended)

#### NVIDIA GPU Setup
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA toolkit (if not installed)
# Visit: https://developer.nvidia.com/cuda-toolkit
# Follow platform-specific instructions
```

#### AMD GPU Setup (ROCm)
```bash
# For AMD GPUs on Linux
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.4.2
```

### Step 3: Install Dependencies

#### Core Dependencies
```bash
# Install all requirements
pip install -r requirements.txt

# Verify critical packages
python -c "import torch, torchvision, flask, PIL, cv2; print('✅ Core packages installed')"
```

#### Optional Dependencies
```bash
# For enhanced visualization
pip install plotly seaborn

# For experiment tracking
pip install wandb tensorboard

# For deployment
pip install gunicorn docker

# For development
pip install jupyter pytest black flake8
```

### Step 4: Configuration

#### Create Configuration File
```bash
# Copy default configuration
cp configs/config.yaml.example configs/config.yaml

# Edit configuration
nano configs/config.yaml  # Linux/macOS
notepad configs/config.yaml  # Windows
```

#### Environment Variables
```bash
# Create .env file
echo "DEVICE=auto" > .env
echo "DATA_DIR=Dataset" >> .env
echo "MODEL_DIR=models" >> .env
echo "API_PORT=5000" >> .env
```

### Step 5: Data Setup

#### Download Sample Data
```bash
# Create data directories
mkdir -p Dataset/{train_valid,test,classification}

# Download MURA dataset (if available)
python scripts/download_data.py
```

#### Organize Your Data
```
Dataset/
├── train_valid/           # For binary classification
│   ├── ELBOW/
│   ├── FINGER/
│   ├── FOREARM/
│   ├── HAND/
│   ├── HUMERUS/
│   ├── SHOULDER/
│   └── WRIST/
├── test/                  # Test set
└── classification/        # For fracture type classification
    ├── Compression-Crush fracture/
    ├── Fracture Dislocation/
    ├── Hairline Fracture/
    ├── Impacted fracture/
    ├── Longitudinal fracture/
    ├── Oblique fracture/
    └── Spiral Fracture/
```

### Step 6: Model Training (Optional)

#### Quick Training
```bash
# Binary classification
python scripts/train_advanced.py --task binary --epochs 10

# Multiclass classification
python scripts/train_advanced.py --task multiclass --epochs 20
```

#### Full Training
```bash
# Complete training pipeline
python scripts/train_advanced.py --config configs/config.yaml
```

### Step 7: Start the Application

#### Development Mode
```bash
# Terminal 1: Start backend
cd backend
python enhanced_app.py

# Terminal 2: Start frontend (development server)
cd frontend
python -m http.server 8080
# Open http://localhost:8080/enhanced_index.html
```

#### Production Mode
```bash
# Using Gunicorn
gunicorn --bind 0.0.0.0:5000 backend.enhanced_app:app

# Using Docker
docker-compose up -d
```

## Verification

### Test Installation
```bash
# Run system check
python scripts/system_check.py

# Test API
curl http://localhost:5000/health

# Run unit tests
pytest tests/ -v
```

### Expected Output
```
✅ Python 3.9.12 detected
✅ PyTorch 2.1.0 with CUDA 11.8
✅ All dependencies installed
✅ Models loaded successfully
✅ API server running on port 5000
✅ Frontend accessible at http://localhost:8080
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
export BATCH_SIZE=8

# Use CPU instead
export DEVICE=cpu
```

#### 2. Port Already in Use
```bash
# Find process using port
lsof -i :5000  # macOS/Linux
netstat -ano | findstr :5000  # Windows

# Kill process or use different port
export API_PORT=5001
```

#### 3. Module Import Errors
```bash
# Reinstall package
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Check Python path
python -c "import sys; print(sys.path)"
```

#### 4. Model Loading Errors
```bash
# Clear model cache
rm -rf models/checkpoints/

# Re-download models
python scripts/download_models.py --force
```

### Platform-Specific Issues

#### Windows
```powershell
# Enable long paths
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# Install Visual C++ Redistributable
# Download from Microsoft website
```

#### macOS
```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew (if needed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Linux
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev python3-pip build-essential

# For Ubuntu 18.04
sudo apt-get install python3.8 python3.8-venv python3.8-dev
```

## Performance Optimization

### GPU Optimization
```bash
# Set memory growth
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Enable mixed precision
export USE_MIXED_PRECISION=true
```

### CPU Optimization
```bash
# Set number of threads
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

### Memory Optimization
```bash
# Reduce image cache
export USE_IMAGE_CACHE=false

# Reduce batch size
export BATCH_SIZE=4
```

## Development Setup

### IDE Configuration

#### Visual Studio Code
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black"
}
```

#### PyCharm
1. File → Settings → Project → Python Interpreter
2. Add Interpreter → Existing Environment
3. Select `venv/bin/python`

### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Next Steps

After successful installation:

1. **📖 Read the [User Guide](user_guide.md)** - Learn how to use the application
2. **🔧 Configure Models** - Customize settings in `configs/config.yaml`
3. **📊 Train Models** - Follow the [Training Guide](training_guide.md)
4. **🚀 Deploy** - Check the [Deployment Guide](deployment.md)
5. **🧪 Test** - Run the test suite with `pytest tests/`

## Support

### Getting Help
- **📧 Email**: support@fracturedetect.ai
- **💬 Discord**: [Join our community](https://discord.gg/fracturedetect)
- **📝 Issues**: [GitHub Issues](https://github.com/yourusername/fracture-detection/issues)
- **📚 Documentation**: [Full Documentation](https://docs.fracturedetect.ai)

### Reporting Bugs
Please include:
- Operating system and version
- Python version
- Error messages and stack traces
- Steps to reproduce
- System specifications

---

**🎉 Congratulations!** You've successfully installed FractureDetect AI. 
Ready to start detecting fractures with AI? Let's go! 🩻✨