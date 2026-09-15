# Docker Deployment - Spezifikation
## Neuromorphe Traum-Engine v2.0

### Container-Architektur

Die Anwendung wird als Multi-Container-Setup mit Docker Compose bereitgestellt:

```
Docker Environment
├── backend/                    # FastAPI Backend Container
│   ├── Dockerfile
│   ├── requirements.txt
│   └── entrypoint.sh
├── frontend/                   # Streamlit Frontend Container
│   ├── Dockerfile
│   ├── requirements.txt
│   └── entrypoint.sh
├── nginx/                      # Reverse Proxy Container
│   ├── Dockerfile
│   ├── nginx.conf
│   └── ssl/
├── redis/                      # Caching & Session Storage
├── postgres/                   # Production Database (optional)
│   └── init.sql
├── docker-compose.yml          # Development Setup
├── docker-compose.prod.yml     # Production Setup
├── .env.example               # Environment Template
└── deploy/
    ├── docker-stack.yml       # Docker Swarm Stack
    ├── k8s/                   # Kubernetes Manifests
    └── scripts/               # Deployment Scripts
```

## Backend Container

### Dockerfile.backend
```dockerfile
# Multi-stage build für optimierte Image-Größe
FROM python:3.11-slim as builder

# Build-Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python Dependencies
COPY backend/requirements.txt /tmp/
RUN pip install --no-cache-dir --user -r /tmp/requirements.txt

# Production Stage
FROM python:3.11-slim as production

# System Dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Set environment variables
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create directories
RUN mkdir -p /app/data /app/uploads /app/models /app/logs \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser
WORKDIR /app

# Download and cache CLAP model
RUN python -c "\
import os; \
os.environ['TORCH_HOME'] = '/app/models'; \
os.environ['HF_HOME'] = '/app/models'; \
try: \
    import laion_clap; \
    model = laion_clap.CLAP_Module(enable_fusion=False); \
    model.load_ckpt(); \
    print('CLAP model successfully cached'); \
except Exception as e: \
    print(f'Warning: Could not cache CLAP model: {e}'); \
"

# Copy application code
COPY --chown=appuser:appuser backend/ /app/

# Copy entrypoint script
COPY --chown=appuser:appuser backend/entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Expose port
EXPOSE 8000

# Start application
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### backend/entrypoint.sh
```bash
#!/bin/bash
set -e

# Wait for database to be ready
if [ "$DATABASE_URL" ]; then
    echo "Waiting for database..."
    python -c "\
import time
import sys
from sqlalchemy import create_engine
from core.config import settings

for i in range(30):
    try:
        engine = create_engine(settings.database_url)
        engine.connect()
        print('Database is ready!')
        break
    except Exception as e:
        print(f'Database not ready yet: {e}')
        time.sleep(2)
else:
    print('Could not connect to database')
    sys.exit(1)
"
fi

# Run database migrations
echo "Running database migrations..."
python -c "\
from core.database import engine, Base
from models.database_models import *
Base.metadata.create_all(bind=engine)
print('Database tables created successfully')
"

# Initialize CLAP model
echo "Initializing CLAP model..."
python -c "\
from services.embedding_service import EmbeddingService
import asyncio

async def init_model():
    service = EmbeddingService()
    await service.initialize_model()
    print('CLAP model initialized successfully')

asyncio.run(init_model())
"

# Start the application
echo "Starting FastAPI application..."
exec "$@"
```

### backend/requirements.txt
```txt
# FastAPI & ASGI
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6

# Database
sqlalchemy==2.0.23
alembic==1.12.1
psycopg2-binary==2.9.9  # PostgreSQL
aiosqlite==0.19.0       # SQLite async

# Audio Processing
librosa==0.10.1
soundfile==0.12.1
numpy==1.24.4
scipy==1.11.4

# Machine Learning
laion-clap==1.1.4
torch==2.1.1
torchaudio==2.1.1
transformers==4.35.2
scikit-learn==1.3.2

# Caching & Background Tasks
redis==5.0.1
celery==5.3.4

# Utilities
pydantic==2.5.0
pydantic-settings==2.0.3
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-dotenv==1.0.0
requests==2.31.0
aiofiles==23.2.1

# Monitoring & Logging
prometheus-client==0.19.0
structlog==23.2.0

# Development
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2
black==23.11.0
isort==5.12.0
flake8==6.1.0
```

## Frontend Container

### Dockerfile.frontend
```dockerfile
FROM python:3.11-slim as production

# System dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r streamlit && useradd -r -g streamlit streamlit

# Install Python dependencies
COPY frontend/requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create directories
RUN mkdir -p /app/logs \
    && chown -R streamlit:streamlit /app

# Switch to non-root user
USER streamlit
WORKDIR /app

# Copy application code
COPY --chown=streamlit:streamlit frontend/ /app/

# Copy entrypoint script
COPY --chown=streamlit:streamlit frontend/entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose port
EXPOSE 8501

# Start application
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
```

### frontend/entrypoint.sh
```bash
#!/bin/bash
set -e

# Wait for backend to be ready
echo "Waiting for backend..."
python -c "\
import time
import requests
import sys
from config.settings import settings

for i in range(30):
    try:
        response = requests.get(f'{settings.backend_url}/api/v1/health', timeout=5)
        if response.status_code == 200:
            print('Backend is ready!')
            break
    except Exception as e:
        print(f'Backend not ready yet: {e}')
        time.sleep(2)
else:
    print('Could not connect to backend')
    sys.exit(1)
"

# Start Streamlit
echo "Starting Streamlit application..."
exec "$@"
```

### frontend/requirements.txt
```txt
# Streamlit
streamlit==1.28.1
streamlit-option-menu==0.3.6
streamlit-aggrid==0.3.4.post3
streamlit-audio-recorder==0.0.8
streamlit-dropzone==1.0.0

# Data & Visualization
pandas==2.1.3
plotly==5.17.0
altair==5.1.2
numpy==1.24.4

# HTTP & API
requests==2.31.0
httpx==0.25.2

# Utilities
pydantic==2.5.0
pydantic-settings==2.0.3
python-dotenv==1.0.0
pillow==10.1.0

# Audio Processing (for client-side preview)
librosa==0.10.1
soundfile==0.12.1

# Development
pytest==7.4.3
```

## Nginx Reverse Proxy

### nginx/Dockerfile
```dockerfile
FROM nginx:1.25-alpine

# Install certbot for SSL
RUN apk add --no-cache certbot certbot-nginx

# Copy configuration
COPY nginx/nginx.conf /etc/nginx/nginx.conf
COPY nginx/default.conf /etc/nginx/conf.d/default.conf

# Create directories
RUN mkdir -p /var/www/certbot /etc/nginx/ssl

# Copy SSL setup script
COPY nginx/setup-ssl.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/setup-ssl.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

EXPOSE 80 443

CMD ["nginx", "-g", "daemon off;"]
```

### nginx/nginx.conf
```nginx
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';
    
    access_log /var/log/nginx/access.log main;
    
    # Performance
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    client_max_body_size 100M;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript
               application/javascript application/xml+rss
               application/json application/xml;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=upload:10m rate=2r/s;
    
    # Include server configurations
    include /etc/nginx/conf.d/*.conf;
}
```

### nginx/default.conf
```nginx
# Upstream servers
upstream backend {
    server backend:8000;
    keepalive 32;
}

upstream frontend {
    server frontend:8501;
    keepalive 32;
}

# HTTP to HTTPS redirect
server {
    listen 80;
    server_name _;
    
    # Let's Encrypt challenge
    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
    
    # Redirect to HTTPS
    location / {
        return 301 https://$host$request_uri;
    }
}

# HTTPS server
server {
    listen 443 ssl http2;
    server_name _;
    
    # SSL configuration
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:50m;
    ssl_session_tickets off;
    
    # Modern SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Referrer-Policy "strict-origin-when-cross-origin";
    
    # Frontend (Streamlit)
    location / {
        proxy_pass http://frontend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support for Streamlit
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
    
    # Backend API
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts for long-running requests
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;
    }
    
    # File upload endpoint with special limits
    location /api/v1/audio/upload {
        limit_req zone=upload burst=5 nodelay;
        
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Extended timeouts for file uploads
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        
        # Large file support
        client_max_body_size 100M;
    }
    
    # Static files caching
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

## Docker Compose Configurations

### docker-compose.yml (Development)
```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app
      - ./data:/app/data
      - ./uploads:/app/uploads
      - ./models:/app/models
    environment:
      - DEBUG=true
      - DATABASE_URL=sqlite:///data/traum_engine.db
      - REDIS_URL=redis://redis:6379
      - UPLOAD_DIR=/app/uploads
      - MODEL_CACHE_DIR=/app/models
    depends_on:
      redis:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "8501:8501"
    volumes:
      - ./frontend:/app
    environment:
      - BACKEND_URL=http://backend:8000
      - DEBUG=true
    depends_on:
      backend:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  redis_data:

networks:
  default:
    name: neuromorphe-traum-engine
```

### docker-compose.prod.yml (Production)
```yaml
version: '3.8'

services:
  nginx:
    build:
      context: .
      dockerfile: nginx/Dockerfile
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - ./certbot/www:/var/www/certbot:ro
      - ./certbot/conf:/etc/letsencrypt:ro
    depends_on:
      - backend
      - frontend
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    expose:
      - "8000"
    volumes:
      - ./data:/app/data
      - ./uploads:/app/uploads
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - DEBUG=false
      - DATABASE_URL=${DATABASE_URL:-sqlite:///data/traum_engine.db}
      - REDIS_URL=redis://redis:6379
      - SECRET_KEY=${SECRET_KEY}
      - UPLOAD_DIR=/app/uploads
      - MODEL_CACHE_DIR=/app/models
    env_file:
      - .env
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    expose:
      - "8501"
    environment:
      - BACKEND_URL=http://backend:8000
      - DEBUG=false
    env_file:
      - .env
    depends_on:
      backend:
        condition: service_healthy
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '1.0'
        reservations:
          memory: 512M
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=${POSTGRES_DB:-traum_engine}
      - POSTGRES_USER=${POSTGRES_USER:-traum_user}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./postgres/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-traum_user}"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Monitoring (optional)
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    name: neuromorphe-traum-engine-prod
```

## Environment Configuration

### .env.example
```bash
# Application
APP_NAME="Neuromorphe Traum-Engine"
APP_VERSION="2.0.0"
DEBUG=false
SECRET_KEY=your-super-secret-key-here

# Database
DATABASE_URL=postgresql://traum_user:your_password@postgres:5432/traum_engine
POSTGRES_DB=traum_engine
POSTGRES_USER=traum_user
POSTGRES_PASSWORD=your_secure_password

# Redis
REDIS_URL=redis://redis:6379
CACHE_TTL=3600

# File Storage
UPLOAD_DIR=/app/uploads
MAX_FILE_SIZE=104857600  # 100MB
MODEL_CACHE_DIR=/app/models

# CLAP Model
CLAP_MODEL_NAME=laion/larger_clap_music_and_speech

# Security
ACCESS_TOKEN_EXPIRE_MINUTES=30
ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com
CORS_ORIGINS=https://your-domain.com,http://localhost:8501

# Monitoring
GRAFANA_PASSWORD=your_grafana_password
PROMETHEUS_RETENTION=15d

# SSL (for production)
SSL_CERT_PATH=/etc/nginx/ssl/cert.pem
SSL_KEY_PATH=/etc/nginx/ssl/key.pem
DOMAIN=your-domain.com
EMAIL=your-email@domain.com

# Performance
MAX_CONCURRENT_JOBS=3
WORKER_PROCESSES=2
WORKER_CONNECTIONS=1000
```

## Deployment Scripts

### deploy/deploy.sh
```bash
#!/bin/bash
set -e

# Deployment script for Neuromorphe Traum-Engine

ENVIRONMENT=${1:-development}
DOMAIN=${2:-localhost}

echo "🚀 Deploying Neuromorphe Traum-Engine ($ENVIRONMENT)"

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required but not installed."; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "❌ Docker Compose is required but not installed."; exit 1; }

# Create necessary directories
mkdir -p data uploads models logs

# Set permissions
chmod 755 data uploads models logs

if [ "$ENVIRONMENT" = "production" ]; then
    echo "📋 Production deployment"
    
    # Check if .env exists
    if [ ! -f .env ]; then
        echo "❌ .env file not found. Please copy .env.example to .env and configure it."
        exit 1
    fi
    
    # Setup SSL certificates
    if [ "$DOMAIN" != "localhost" ]; then
        echo "🔒 Setting up SSL certificates for $DOMAIN"
        ./deploy/setup-ssl.sh "$DOMAIN"
    fi
    
    # Deploy with production configuration
    docker-compose -f docker-compose.prod.yml up -d --build
    
    echo "✅ Production deployment completed!"
    echo "🌐 Application available at: https://$DOMAIN"
    echo "📊 Grafana dashboard: https://$DOMAIN:3000"
    echo "📈 Prometheus metrics: https://$DOMAIN:9090"
    
else
    echo "🛠️ Development deployment"
    
    # Deploy with development configuration
    docker-compose up -d --build
    
    echo "✅ Development deployment completed!"
    echo "🌐 Frontend: http://localhost:8501"
    echo "🔧 Backend API: http://localhost:8000"
    echo "📚 API Docs: http://localhost:8000/docs"
fi

# Show container status
echo "\n📊 Container Status:"
docker-compose ps

# Show logs
echo "\n📋 Recent logs:"
docker-compose logs --tail=20

echo "\n🎉 Deployment completed successfully!"
```

### deploy/setup-ssl.sh
```bash
#!/bin/bash
set -e

DOMAIN=$1
EMAIL=${2:-admin@$DOMAIN}

if [ -z "$DOMAIN" ]; then
    echo "Usage: $0 <domain> [email]"
    exit 1
fi

echo "🔒 Setting up SSL certificates for $DOMAIN"

# Create directories
mkdir -p nginx/ssl certbot/www certbot/conf

# Generate temporary self-signed certificate for initial setup
if [ ! -f nginx/ssl/cert.pem ]; then
    echo "📜 Generating temporary self-signed certificate"
    openssl req -x509 -nodes -days 1 -newkey rsa:2048 \
        -keyout nginx/ssl/key.pem \
        -out nginx/ssl/cert.pem \
        -subj "/CN=$DOMAIN"
fi

# Start nginx temporarily
docker-compose -f docker-compose.prod.yml up -d nginx

# Wait for nginx to be ready
sleep 10

# Obtain Let's Encrypt certificate
echo "🔐 Obtaining Let's Encrypt certificate"
docker run --rm \
    -v $(pwd)/certbot/www:/var/www/certbot \
    -v $(pwd)/certbot/conf:/etc/letsencrypt \
    certbot/certbot certonly \
    --webroot \
    --webroot-path=/var/www/certbot \
    --email $EMAIL \
    --agree-tos \
    --no-eff-email \
    -d $DOMAIN

# Copy certificates to nginx directory
cp certbot/conf/live/$DOMAIN/fullchain.pem nginx/ssl/cert.pem
cp certbot/conf/live/$DOMAIN/privkey.pem nginx/ssl/key.pem

# Restart nginx with real certificates
docker-compose -f docker-compose.prod.yml restart nginx

echo "✅ SSL certificates configured successfully!"

# Setup certificate renewal
echo "⏰ Setting up automatic certificate renewal"
(crontab -l 2>/dev/null; echo "0 12 * * * $(pwd)/deploy/renew-ssl.sh $DOMAIN") | crontab -

echo "🎉 SSL setup completed!"
```

Diese Docker-Deployment-Spezifikation bietet eine vollständige, produktionsreife Container-Architektur mit Sicherheit, Monitoring und automatisierter SSL-Konfiguration für die Neuromorphe Traum-Engine.