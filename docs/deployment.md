# 🚀 Deployment Guide - FractureDetect AI

## Overview

This guide covers various deployment options for FractureDetect AI, from simple local deployment to enterprise-grade cloud solutions.

## Deployment Options

### 1. Local Development Server
### 2. Production Server with Gunicorn
### 3. Docker Containerization
### 4. Cloud Deployment (AWS, GCP, Azure)
### 5. Kubernetes Deployment
### 6. Edge Deployment

---

## 1. Local Development Server

### Quick Start
```bash
# Start backend
cd backend
python enhanced_app.py

# Start frontend
cd frontend
python -m http.server 8080
```

### Configuration
```python
# backend/enhanced_app.py
if __name__ == '__main__':
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True,
        threaded=True
    )
```

---

## 2. Production Server with Gunicorn

### Install Gunicorn
```bash
pip install gunicorn
```

### Basic Configuration
```bash
# Start with Gunicorn
gunicorn --bind 0.0.0.0:5000 --workers 4 backend.enhanced_app:app
```

### Advanced Configuration
```bash
# Create gunicorn.conf.py
cat > gunicorn.conf.py << EOF
bind = "0.0.0.0:5000"
workers = 4
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
timeout = 30
keepalive = 5
preload_app = True
user = "www-data"
group = "www-data"
tmp_upload_dir = None
EOF

# Start with configuration
gunicorn -c gunicorn.conf.py backend.enhanced_app:app
```

### Systemd Service
```bash
# Create service file
sudo nano /etc/systemd/system/fracturedetect.service
```

```ini
[Unit]
Description=FractureDetect AI Application
After=network.target

[Service]
Type=notify
User=www-data
Group=www-data
RuntimeDirectory=fracturedetect
WorkingDirectory=/opt/fracture-detection
Environment=PATH=/opt/fracture-detection/venv/bin
ExecStart=/opt/fracture-detection/venv/bin/gunicorn -c gunicorn.conf.py backend.enhanced_app:app
ExecReload=/bin/kill -s HUP $MAINPID
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable fracturedetect
sudo systemctl start fracturedetect
sudo systemctl status fracturedetect
```

### Nginx Reverse Proxy
```bash
# Install Nginx
sudo apt-get install nginx

# Create Nginx configuration
sudo nano /etc/nginx/sites-available/fracturedetect
```

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        root /opt/fracture-detection/frontend;
        index enhanced_index.html;
        try_files $uri $uri/ =404;
    }
    
    location /api/ {
        proxy_pass http://127.0.0.1:5000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Increase timeouts for large image uploads
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Increase max body size
        client_max_body_size 20M;
    }
    
    location /health {
        proxy_pass http://127.0.0.1:5000/health;
        access_log off;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/fracturedetect /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 3. Docker Containerization

### Create Dockerfile
```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "backend.enhanced_app:app"]
```

### Create .dockerignore
```
# .dockerignore
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis

.DS_Store
.vscode
.idea
*.swp
*.swo

models/*.pth
logs/
outputs/
data/
!data/sample/
```

### Build and Run
```bash
# Build image
docker build -t fracture-detection:latest .

# Run container
docker run -d \
    --name fracture-detection \
    -p 5000:5000 \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/logs:/app/logs \
    -e DEVICE=cpu \
    fracture-detection:latest
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  fracture-detection:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
      - ./configs:/app/configs
    environment:
      - DEVICE=auto
      - API_PORT=5000
      - DEBUG=false
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./frontend:/usr/share/nginx/html
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - fracture-detection
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

```bash
# Start with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Scale application
docker-compose up -d --scale fracture-detection=3
```

---

## 4. Cloud Deployment

### AWS EC2 Deployment

#### Launch EC2 Instance
```bash
# Using AWS CLI
aws ec2 run-instances \
    --image-id ami-0abcdef1234567890 \
    --count 1 \
    --instance-type t3.medium \
    --key-name my-key-pair \
    --security-group-ids sg-903004f8 \
    --subnet-id subnet-6e7f829e \
    --user-data file://user-data.sh
```

#### User Data Script
```bash
#!/bin/bash
# user-data.sh
yum update -y
yum install -y docker
service docker start
usermod -a -G docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Clone and start application
cd /home/ec2-user
git clone https://github.com/yourusername/fracture-detection.git
cd fracture-detection
docker-compose up -d
```

### AWS ECS Deployment

#### Task Definition
```json
{
  "family": "fracture-detection",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "fracture-detection",
      "image": "your-account.dkr.ecr.region.amazonaws.com/fracture-detection:latest",
      "portMappings": [
        {
          "containerPort": 5000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "DEVICE",
          "value": "cpu"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/fracture-detection",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Google Cloud Platform

#### Google Cloud Run
```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/fracture-detection

# Deploy to Cloud Run
gcloud run deploy fracture-detection \
    --image gcr.io/PROJECT_ID/fracture-detection \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10
```

### Azure Container Instances

#### Deploy to ACI
```bash
# Create resource group
az group create --name fracture-detection-rg --location eastus

# Deploy container
az container create \
    --resource-group fracture-detection-rg \
    --name fracture-detection \
    --image your-registry.azurecr.io/fracture-detection:latest \
    --cpu 2 \
    --memory 4 \
    --restart-policy Always \
    --ports 5000 \
    --environment-variables DEVICE=cpu API_PORT=5000
```

---

## 5. Kubernetes Deployment

### Kubernetes Manifests

#### Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fracture-detection
  labels:
    app: fracture-detection
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fracture-detection
  template:
    metadata:
      labels:
        app: fracture-detection
    spec:
      containers:
      - name: fracture-detection
        image: fracture-detection:latest
        ports:
        - containerPort: 5000
        env:
        - name: DEVICE
          value: "cpu"
        - name: API_PORT
          value: "5000"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: models
          mountPath: /app/models
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: logs
        persistentVolumeClaim:
          claimName: logs-pvc
```

#### Service
```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: fracture-detection-service
spec:
  selector:
    app: fracture-detection
  ports:
    - protocol: TCP
      port: 80
      targetPort: 5000
  type: ClusterIP
```

#### Ingress
```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: fracture-detection-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-body-size: "20m"
spec:
  tls:
  - hosts:
    - fracturedetect.ai
    secretName: fracture-detection-tls
  rules:
  - host: fracturedetect.ai
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: fracture-detection-service
            port:
              number: 80
```

#### PersistentVolume
```yaml
# k8s/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: models-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
  storageClassName: gp2
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: logs-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 5Gi
  storageClassName: gp2
```

### Deploy to Kubernetes
```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment
kubectl get pods
kubectl get services
kubectl get ingress

# View logs
kubectl logs -f deployment/fracture-detection

# Scale deployment
kubectl scale deployment fracture-detection --replicas=5
```

### Helm Chart
```yaml
# Chart.yaml
apiVersion: v2
name: fracture-detection
description: AI-powered bone fracture detection system
type: application
version: 1.0.0
appVersion: "1.0.0"
```

```yaml
# values.yaml
replicaCount: 3

image:
  repository: fracture-detection
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: fracturedetect.ai
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: fracture-detection-tls
      hosts:
        - fracturedetect.ai

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80
```

---

## 6. Edge Deployment

### NVIDIA Jetson
```bash
# Install JetPack SDK
sudo apt update
sudo apt install nvidia-jetpack

# Install PyTorch for Jetson
wget https://nvidia.box.com/shared/static/pytorch-wheel-for-jetson.whl
pip install pytorch-wheel-for-jetson.whl

# Optimize for inference
python scripts/optimize_for_edge.py --platform jetson
```

### Intel NUC
```bash
# Install Intel OpenVINO
wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.1/linux/l_openvino_toolkit_ubuntu20_2023.1.0.12185.47b736f63ed_x86_64.tgz
tar -xf l_openvino_toolkit_ubuntu20_2023.1.0.12185.47b736f63ed_x86_64.tgz
cd l_openvino_toolkit_ubuntu20_2023.1.0.12185.47b736f63ed
sudo ./install_openvino_dependencies.sh

# Convert model to OpenVINO format
python scripts/convert_to_openvino.py
```

### Raspberry Pi
```bash
# Install dependencies
sudo apt update
sudo apt install python3-opencv python3-numpy

# Install PyTorch CPU-only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Optimize model for ARM
python scripts/optimize_for_arm.py
```

---

## Monitoring and Observability

### Prometheus Metrics
```python
# Add to enhanced_app.py
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter('fracture_detection_requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('fracture_detection_request_duration_seconds', 'Request latency')

@app.route('/metrics')
def metrics():
    return generate_latest()
```

### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "FractureDetect AI Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(fracture_detection_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Response Time",
        "targets": [
          {
            "expr": "fracture_detection_request_duration_seconds",
            "legendFormat": "Response Time"
          }
        ]
      }
    ]
  }
}
```

### Logging Configuration
```yaml
# logging.yaml
version: 1
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  json:
    format: '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: default
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: json
    filename: logs/app.log
    maxBytes: 10485760
    backupCount: 5

  syslog:
    class: logging.handlers.SysLogHandler
    level: INFO
    formatter: json
    address: ['localhost', 514]

loggers:
  fracture_detection:
    level: INFO
    handlers: [console, file, syslog]
    propagate: no

root:
  level: INFO
  handlers: [console]
```

---

## Security Considerations

### SSL/TLS Configuration
```bash
# Generate SSL certificate with Let's Encrypt
sudo certbot --nginx -d your-domain.com

# Or use custom certificate
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/ssl/private/fracture-detection.key \
    -out /etc/ssl/certs/fracture-detection.crt
```

### Firewall Configuration
```bash
# UFW (Ubuntu)
sudo ufw allow 22
sudo ufw allow 80
sudo ufw allow 443
sudo ufw enable

# IPTables
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT
sudo iptables -A INPUT -j DROP
```

### Environment Variables Security
```bash
# Use secrets management
export DATABASE_URL=$(aws secretsmanager get-secret-value --secret-id db-credentials --query SecretString --output text)

# Or use HashiCorp Vault
vault kv get -field=api_key secret/fracture-detection
```

---

## Performance Tuning

### Gunicorn Optimization
```python
# gunicorn.conf.py
import multiprocessing

bind = "0.0.0.0:5000"
workers = min(multiprocessing.cpu_count() * 2 + 1, 8)
worker_class = "gevent"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
timeout = 300
keepalive = 5
preload_app = True
```

### Nginx Optimization
```nginx
# nginx.conf
worker_processes auto;
worker_rlimit_nofile 65535;

events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

http {
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css application/json application/javascript;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    server {
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://127.0.0.1:5000/;
        }
    }
}
```

---

## Backup and Recovery

### Database Backup
```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"

# Backup models
tar -czf "$BACKUP_DIR/models_$DATE.tar.gz" models/

# Backup configuration
tar -czf "$BACKUP_DIR/config_$DATE.tar.gz" configs/

# Upload to cloud storage
aws s3 cp "$BACKUP_DIR/models_$DATE.tar.gz" s3://fracture-detection-backups/
```

### Disaster Recovery
```bash
#!/bin/bash
# restore.sh
BACKUP_DATE=$1

# Download from cloud storage
aws s3 cp "s3://fracture-detection-backups/models_$BACKUP_DATE.tar.gz" /tmp/

# Restore models
tar -xzf "/tmp/models_$BACKUP_DATE.tar.gz" -C /

# Restart services
systemctl restart fracture-detection
```

---

## Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check memory usage
free -h
docker stats

# Reduce batch size
export BATCH_SIZE=1

# Use model quantization
python scripts/quantize_model.py
```

#### Slow Response Times
```bash
# Check CPU usage
top
htop

# Profile application
python -m cProfile -o profile.stats backend/enhanced_app.py

# Use model optimization
python scripts/optimize_model.py --backend tensorrt
```

#### SSL Certificate Issues
```bash
# Check certificate validity
openssl x509 -in certificate.crt -text -noout

# Renew Let's Encrypt certificate
sudo certbot renew

# Test SSL configuration
curl -I https://your-domain.com
```

---

## Scaling Strategies

### Horizontal Scaling
```bash
# Add more instances
docker-compose up -d --scale fracture-detection=5

# Use load balancer
kubectl apply -f load-balancer.yaml
```

### Vertical Scaling
```bash
# Increase container resources
docker update --memory=4g --cpus=2 fracture-detection

# Update Kubernetes resources
kubectl patch deployment fracture-detection -p '{"spec":{"template":{"spec":{"containers":[{"name":"fracture-detection","resources":{"limits":{"memory":"4Gi","cpu":"2"}}}]}}}}'
```

### Auto-scaling
```yaml
# HPA for Kubernetes
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fracture-detection-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fracture-detection
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

**🎉 Deployment Complete!** Your FractureDetect AI is now running in production. Monitor performance and scale as needed! 🚀