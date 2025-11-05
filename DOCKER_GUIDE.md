# Docker Deployment Guide

## 📦 Files Created

1. ✅ `Dockerfile` - Multi-stage production build
2. ✅ `docker-compose.yml` - Production deployment
3. ✅ `docker-compose.dev.yml` - Development with hot reload
4. ✅ `.dockerignore` - Optimized Docker builds
5. ✅ `.github/workflows/ci-cd.yml` - Complete CI/CD pipeline

---

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Production
docker-compose up -d

# Development (with hot reload)
docker-compose -f docker-compose.dev.yml up
```

### Option 2: Docker Build & Run

```bash
# Build image
docker build -t healthcare-rag-assistant .

# Run container
docker run -d \
  -p 8000:8000 \
  -e API_KEYS=your-api-key-here \
  --name healthcare-rag \
  healthcare-rag-assistant
```

---

## 📋 Detailed Instructions

### Build Docker Image

```bash
cd /home/morshed/own-folder/Documents/jap_llm

# Build the image
docker build -t healthcare-rag-assistant:latest .

# This will:
# - Install all dependencies
# - Download NLTK data
# - Create optimized production image
# - Time: ~5-10 minutes first build
```

### Run with Docker Compose (Production)

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Run with Docker Compose (Development)

```bash
# Start with hot reload
docker-compose -f docker-compose.dev.yml up

# Code changes in app/ will auto-reload!
# Ctrl+C to stop
```

### Access the Application

```bash
# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs

# Test endpoint
curl -H "X-API-Key: dev-key-12345" http://localhost:8000/admin/stats
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file or export variables:

```bash
# Required
API_KEYS=your-secret-key-1,your-secret-key-2

# Optional (has defaults)
ENV=production
LOG_LEVEL=INFO
PORT=8000
MAX_FILE_SIZE_MB=10
```

### Docker Compose Override

Create `docker-compose.override.yml` for local customization:

```yaml
version: '3.8'
services:
  healthcare-rag-api:
    environment:
      - API_KEYS=my-local-key
    ports:
      - "8080:8000"  # Use different port locally
```

---

## 📊 Docker Features

### Multi-Stage Build
- **Stage 1**: Builder (installs dependencies)
- **Stage 2**: Runtime (minimal image)
- **Result**: Smaller final image (~2GB vs ~4GB)

### Security Features
- ✅ Non-root user (appuser:1000)
- ✅ Minimal base image (python:3.10-slim)
- ✅ No unnecessary packages
- ✅ .env.example used (no secrets in image)

### Health Checks
- ✅ Built-in health check endpoint
- ✅ Automatic container restart if unhealthy
- ✅ 60 second startup grace period

### Data Persistence
- ✅ Named volumes for FAISS index
- ✅ Named volumes for logs
- ✅ Data survives container restarts

---

## 🧪 Testing Docker Build

### Test Locally

```bash
# Build
docker build -t healthcare-rag-test .

# Run
docker run -d -p 8001:8000 \
  -e API_KEYS=test-key \
  --name rag-test \
  healthcare-rag-test

# Wait for startup
sleep 30

# Test health
curl http://localhost:8001/health

# Should return:
# {"status":"healthy","faiss_index_size":0,...}

# Cleanup
docker stop rag-test
docker rm rag-test
```

### Test with Docker Compose

```bash
# Start
docker-compose up -d

# Check status
docker-compose ps

# Should show:
# NAME                    STATUS          PORTS
# healthcare-rag-assistant   Up (healthy)    0.0.0.0:8000->8000/tcp

# Test
curl http://localhost:8000/health

# Stop
docker-compose down
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

The `.github/workflows/ci-cd.yml` includes:

**7 Jobs**:
1. **Code Quality** - Black, Flake8, isort
2. **Tests** - pytest with coverage
3. **Docker Build** - Build and test image
4. **Security Scan** - Trivy vulnerability scanning
5. **Deploy** - Push to GitHub Container Registry
6. **Integration Tests** - End-to-end testing
7. **Release** - Create GitHub releases

### Triggers

- **Push to main/develop** → Full pipeline
- **Pull requests** → Tests + build only
- **Tags (v*.*)** → Full pipeline + release

### Setup GitHub Actions

1. **Push code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/healthcare-rag-assistant.git
   git push -u origin main
   ```

2. **GitHub Actions runs automatically** - no setup needed!

3. **View results**: Go to Actions tab in your GitHub repo

---

## 🐳 Docker Commands Reference

### Building

```bash
# Build with tag
docker build -t healthcare-rag:v1.0 .

# Build with no cache
docker build --no-cache -t healthcare-rag:latest .

# Build for specific platform
docker build --platform linux/amd64 -t healthcare-rag:latest .
```

### Running

```bash
# Run in foreground
docker run -p 8000:8000 -e API_KEYS=key healthcare-rag

# Run in background
docker run -d -p 8000:8000 -e API_KEYS=key healthcare-rag

# Run with custom environment file
docker run -p 8000:8000 --env-file .env healthcare-rag

# Run with volume mounts
docker run -d -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -e API_KEYS=key \
  healthcare-rag
```

### Debugging

```bash
# View logs
docker logs healthcare-rag-assistant

# Follow logs
docker logs -f healthcare-rag-assistant

# Execute command in container
docker exec -it healthcare-rag-assistant bash

# Check health
docker exec healthcare-rag-assistant curl http://localhost:8000/health
```

### Cleanup

```bash
# Stop container
docker stop healthcare-rag-assistant

# Remove container
docker rm healthcare-rag-assistant

# Remove image
docker rmi healthcare-rag-assistant

# Remove everything (including volumes)
docker-compose down -v
docker system prune -a
```

---

## 📊 Resource Requirements

### Minimum

- **CPU**: 1 core
- **RAM**: 2GB
- **Disk**: 5GB (models + data)

### Recommended

- **CPU**: 2 cores
- **RAM**: 4GB
- **Disk**: 10GB

### Configured in docker-compose.yml

```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
    reservations:
      cpus: '1.0'
      memory: 2G
```

---

## 🔒 Production Deployment Checklist

- [ ] Change default API keys in .env
- [ ] Set ENV=production
- [ ] Use strong, random API keys (32+ chars)
- [ ] Configure LOG_LEVEL=WARNING or ERROR
- [ ] Set up volume backups for data/
- [ ] Configure reverse proxy (nginx)
- [ ] Enable HTTPS/TLS
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure log aggregation
- [ ] Set up alerts for errors
- [ ] Test health check endpoint
- [ ] Test backup/restore procedure

---

## 🎯 Usage Examples

### Start Production Server

```bash
# 1. Set environment variables
export API_KEYS="prod-key-$(openssl rand -hex 32)"
echo "Your API key: $API_KEYS"

# 2. Start with docker-compose
docker-compose up -d

# 3. Verify
curl http://localhost:8000/health
```

### Development Workflow

```bash
# 1. Start dev server with hot reload
docker-compose -f docker-compose.dev.yml up

# 2. Edit code in app/ directory
# Changes auto-reload!

# 3. Test endpoints
curl -H "X-API-Key: dev-key-12345" \
  http://localhost:8000/admin/stats
```

### Run Tests in Docker

```bash
# Build test image
docker build -t healthcare-rag:test .

# Run tests
docker run --rm \
  -e API_KEYS=test-key \
  healthcare-rag:test \
  pytest tests/ -v
```

---

## 🎉 Summary

**Created Files**:
- ✅ `Dockerfile` (Multi-stage, optimized)
- ✅ `docker-compose.yml` (Production)
- ✅ `docker-compose.dev.yml` (Development)
- ✅ `.dockerignore` (Build optimization)
- ✅ `.github/workflows/ci-cd.yml` (Complete CI/CD)

**Ready to**:
- ✅ Build Docker image
- ✅ Deploy with docker-compose
- ✅ Run CI/CD on GitHub
- ✅ Deploy to production

**Your deployment is now production-ready!** 🚀

