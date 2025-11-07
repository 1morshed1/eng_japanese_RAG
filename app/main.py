# app/main.py
import time
import uuid
import logging
import nltk
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.middleware.auth import api_key_auth
from app.routes import ingest, retrieve, generate
from app.utils.logger import setup_logging
from app.services.faiss_service import faiss_service
from app.services.embedding_service import embedding_service

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Track startup time
start_time = time.time()

# Create FastAPI app
app = FastAPI(
    title="Healthcare RAG Assistant",
    description="Bilingual medical knowledge assistant for clinicians",
    version="1.0.0"
)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    start = time.time()
    
    logger.info(
        "Request started",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else "unknown"
        }
    )
    
    response = await call_next(request)
    
    duration = time.time() - start
    
    logger.info(
        "Request completed",
        extra={
            "request_id": request_id,
            "status_code": response.status_code,
            "duration": duration
        }
    )
    
    response.headers["X-Request-ID"] = request_id
    return response

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(",") if hasattr(settings, 'CORS_ORIGINS') 
                  else ["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Healthcare RAG Assistant...")
    
    # Download NLTK data if not present
    try:
        nltk.data.find('tokenizers/punkt_tab')
        logger.info("NLTK punkt_tab already available")
    except LookupError:
        try:
            logger.info("Downloading NLTK punkt_tab tokenizer...")
            nltk.download('punkt_tab', quiet=True)
            logger.info("NLTK punkt_tab downloaded successfully")
        except Exception as e:
            logger.warning(f"Failed to download NLTK punkt_tab (will use fallback): {e}")
    
    logger.info("FAISS index loaded automatically")
    logger.info("Services ready")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down...")
    faiss_service._save_index()
    logger.info("Shutdown complete")

# Include routes
app.include_router(ingest.router, prefix="/api/v1", tags=["ingest"])
app.include_router(retrieve.router, prefix="/api/v1", tags=["retrieve"])
app.include_router(generate.router, prefix="/api/v1", tags=["generate"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Healthcare RAG Assistant API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """health check endpoint"""
    try:
        return {
            "status": "healthy",
            "faiss_index_size": faiss_service.index.ntotal if faiss_service.index else 0,
            "total_documents": faiss_service.doc_counter,
            "model_loaded": embedding_service.model is not None,
            "uptime_seconds": int(time.time() - start_time)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/admin/stats", dependencies=[Depends(api_key_auth)])
async def get_admin_stats():
    """System statistics (requires authentication)"""
    import os
    
    index_size_mb = 0
    if faiss_service.index_path.exists():
        index_size_mb = os.path.getsize(faiss_service.index_path) / (1024 * 1024)
    
    return {
        "faiss": {
            "total_vectors": faiss_service.index.ntotal if faiss_service.index else 0,
            "total_documents": faiss_service.doc_counter,
            "embedding_dimension": faiss_service.embedding_dim,
            "index_size_mb": round(index_size_mb, 2)
        },
        "uptime_seconds": int(time.time() - start_time)
    }