# app/routes/ingest.py
"""
Document ingestion endpoint for uploading and processing text documents.

Handles file validation, text chunking, embedding generation, and storage
in the vector database with support for English and Japanese documents.
"""

import time
import uuid
import re
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any
from asyncio import timeout as async_timeout

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
import nltk
from nltk.tokenize import sent_tokenize

from app.middleware.auth import api_key_auth
from app.models.schemas import IngestResponse
from app.services.embedding_service import embedding_service
from app.services.faiss_service import faiss_service
from app.utils.language_detector import detect_language
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
        logger.info("Downloaded NLTK punkt tokenizer")
    except Exception as e:
        logger.warning(f"Failed to download NLTK data: {e}")

# Configuration
MAX_FILE_SIZE_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024
MAX_CHUNK_SIZE = getattr(settings, 'MAX_CHUNK_SIZE', 500)
ALLOWED_EXTENSIONS = {'.txt'}
INGEST_TIMEOUT = 60  # seconds


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent security issues.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Get just the filename, no path components
    filename = Path(filename).name
    
    # Remove any dangerous characters
    filename = re.sub(r'[^\w\s.-]', '', filename)
    
    # Limit length
    if len(filename) > 100:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:95] + '.' + ext if ext else name[:100]
    
    return filename or "unnamed.txt"


def calculate_content_hash(content: str) -> str:
    """
    Calculate SHA-256 hash of content for duplicate detection.
    
    Args:
        content: Text content
        
    Returns:
        Hex digest of content hash
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def chunk_text_by_sentences(
    text: str,
    language: str = "en",
    max_chunk_size: int = 500,
    overlap_sentences: int = 1
) -> List[str]:
    """
    Split text into chunks by sentences, maintaining semantic boundaries.
    
    Uses language-aware sentence splitting for English and Japanese.
    
    Args:
        text: Input text to chunk
        language: Language of text ('en' or 'ja')
        max_chunk_size: Target maximum characters per chunk
        overlap_sentences: Number of sentences to overlap between chunks
        
    Returns:
        List of text chunks
        
    Example:
        >>> chunks = chunk_text_by_sentences("Text here.", "en", 500)
        >>> len(chunks)
        1
    """
    # Split into sentences based on language
    if language == "ja":
        # Japanese sentence endings: 。！？
        sentences = re.split(r'(?<=[。！？])\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]
    else:
        # English - use NLTK
        try:
            sentences = sent_tokenize(text)
        except Exception as e:
            logger.warning(f"NLTK tokenization failed, using fallback: {e}")
            # Fallback to simple splitting
            sentences = re.split(r'(?<=[.!?])\s+', text)
    
    if not sentences:
        return []
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        # If adding this sentence exceeds limit and we have content
        if current_size + sentence_size > max_chunk_size and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap
            if overlap_sentences > 0 and len(current_chunk) > overlap_sentences:
                overlap = current_chunk[-overlap_sentences:]
            else:
                overlap = []
            
            current_chunk = overlap + [sentence]
            current_size = sum(len(s) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
    
    # Add final chunk if not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


async def validate_upload_file(file: UploadFile) -> tuple[str, str]:
    """
    Validate uploaded file and return its content and sanitized filename.
    
    Args:
        file: Uploaded file
        
    Returns:
        Tuple of (file content as string, sanitized filename)
        
    Raises:
        HTTPException: If validation fails
    """
    # Sanitize filename
    sanitized_name = sanitize_filename(file.filename or "unnamed.txt")
    
    # Check file extension
    file_ext = Path(sanitized_name).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Only {', '.join(ALLOWED_EXTENSIONS)} files are allowed."
        )
    
    # Read file content
    try:
        content = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file."
        )
    
    # Check file size
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty."
        )
    
    if len(content) > MAX_FILE_SIZE_BYTES:
        max_size_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {max_size_mb:.1f}MB."
        )
    
    # Decode and validate encoding
    try:
        text = content.decode('utf-8')
    except UnicodeDecodeError:
        try:
            # Try other common encodings
            text = content.decode('latin-1')
            logger.warning(f"File decoded as latin-1: {sanitized_name}")
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be UTF-8 or Latin-1 encoded text."
            )
    
    # Validate content
    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File contains only whitespace."
        )
    
    # Reset file pointer for potential reuse
    await file.seek(0)
    
    return text, sanitized_name


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest Document",
    description=(
        "Upload and process a text document for storage in the vector database. "
        "Supports English and Japanese documents. The document is automatically "
        "chunked, embedded, and indexed for retrieval."
    ),
    responses={
        201: {
            "description": "Document ingested successfully",
            "model": IngestResponse
        },
        400: {
            "description": "Invalid file or request"
        },
        401: {
            "description": "Invalid API key"
        },
        413: {
            "description": "File too large"
        },
        429: {
            "description": "Too many requests"
        },
        500: {
            "description": "Internal server error"
        }
    }
)
async def ingest_document(
    file: UploadFile = File(..., description="Text file to ingest (.txt only)"),
    api_key: str = Depends(api_key_auth)
) -> IngestResponse:
    """
    Ingest a document by processing, chunking, and storing it in the vector database.
    
    Process:
    1. Validates file type, size, and encoding
    2. Detects document language (English or Japanese)
    3. Chunks text into semantic segments
    4. Generates embeddings for each chunk
    5. Stores chunks and embeddings in FAISS index
    
    Args:
        file: Text file to ingest (.txt format)
        api_key: API key for authentication (injected by dependency)
        
    Returns:
        IngestResponse with processing results including document ID
        
    Raises:
        HTTPException: Various status codes for different error conditions
        
    Example:
```bash
        curl -X POST "http://localhost:8000/ingest" \
          -H "X-API-Key: your-api-key" \
          -F "file=@document.txt"
```
    """
    # Generate request ID for tracking
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    logger.info(
        "Document ingestion request received",
        extra={
            "request_id": request_id,
            "file_name": file.filename,
            "content_type": file.content_type,
            "api_key": api_key[:8] + "..." if api_key else None
        }
    )
    
    try:
        # Apply timeout to entire operation
        async with async_timeout(INGEST_TIMEOUT):
            return await _process_ingestion(file, request_id, start_time)
            
    except TimeoutError:
        logger.error(
            "Document ingestion timed out",
            extra={
                "request_id": request_id,
                "timeout": INGEST_TIMEOUT
            }
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Document processing timed out after {INGEST_TIMEOUT} seconds"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(
            "Unexpected error in document ingestion",
            extra={
                "request_id": request_id,
                "file_name": file.filename,
                "error": str(e),
                "error_type": type(e).__name__
            },
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the document."
        )


async def _process_ingestion(
    file: UploadFile,
    request_id: str,
    start_time: float
) -> IngestResponse:
    """
    Process document ingestion with detailed error handling.
    
    Args:
        file: Uploaded file
        request_id: Unique request identifier
        start_time: Request start timestamp
        
    Returns:
        IngestResponse with processing results
    """
    # Step 1: Validate and read file
    try:
        text_content, sanitized_filename = await validate_upload_file(file)
        file_size_kb = len(text_content.encode('utf-8')) / 1024
        
        logger.debug(
            "File validated successfully",
            extra={
                "request_id": request_id,
                "file_name": sanitized_filename,
                "size_kb": round(file_size_kb, 2)
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "File validation failed",
            extra={"request_id": request_id, "error": str(e)},
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File validation failed. Please check file format and try again."
        )
    
    # Step 2: Detect language
    try:
        language = detect_language(text_content)
        logger.debug(
            "Language detected",
            extra={
                "request_id": request_id,
                "language": language
            }
        )
    except Exception as e:
        logger.warning(
            f"Language detection failed, defaulting to English: {e}",
            extra={"request_id": request_id}
        )
        language = "en"
    
    # Step 3: Calculate content hash (for duplicate detection)
    content_hash = calculate_content_hash(text_content)
    
    # Step 4: Chunk text
    try:
        chunks = chunk_text_by_sentences(
            text_content,
            language=language,
            max_chunk_size=MAX_CHUNK_SIZE
        )
        
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No meaningful chunks could be extracted from the document."
            )
        
        # Validate chunks
        valid_chunks = [chunk for chunk in chunks if 10 <= len(chunk) <= 2000]
        
        if not valid_chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document contains no valid text chunks."
            )
        
        if len(valid_chunks) < len(chunks):
            logger.warning(
                f"Filtered out {len(chunks) - len(valid_chunks)} invalid chunks",
                extra={"request_id": request_id}
            )
        
        chunks = valid_chunks
        
        logger.debug(
            "Text chunked successfully",
            extra={
                "request_id": request_id,
                "chunk_count": len(chunks),
                "avg_chunk_size": sum(len(c) for c in chunks) / len(chunks)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Text chunking failed",
            extra={"request_id": request_id, "error": str(e)},
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process document text."
        )
    
    # Step 5: Generate embeddings
    try:
        embeddings = embedding_service.encode(
            chunks,
            batch_size=32,
            show_progress_bar=False
        )
        
        logger.debug(
            "Embeddings generated",
            extra={
                "request_id": request_id,
                "embedding_count": len(embeddings),
                "embedding_shape": embeddings.shape
            }
        )
        
    except Exception as e:
        logger.error(
            "Embedding generation failed",
            extra={"request_id": request_id, "error": str(e)},
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate document embeddings."
        )
    
    # Step 6: Generate document ID
    document_id = faiss_service.generate_document_id()
    
    # Step 7: Add to FAISS index
    try:
        chunks_created = faiss_service.add_document(
            document_id=document_id,
            chunks=chunks,
            embeddings=embeddings,
            language=language,
            metadata={
                "file_name": sanitized_filename,
                "original_size_kb": round(file_size_kb, 2),
                "chunk_count": len(chunks),
                "content_hash": content_hash,
                "ingested_at": time.time()
            }
        )
        
        logger.debug(
            "Document added to FAISS index",
            extra={
                "request_id": request_id,
                "document_id": document_id,
                "chunks_created": chunks_created
            }
        )
        
    except Exception as e:
        logger.error(
            "Failed to add document to FAISS index",
            extra={
                "request_id": request_id,
                "document_id": document_id,
                "error": str(e)
            },
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store document in database."
        )
    
    # Step 8: Prepare response
    processing_time = time.time() - start_time
    
    response = IngestResponse(
        status="success",
        document_id=document_id,
        language=language,
        chunks_created=chunks_created,
        file_size_kb=round(file_size_kb, 2),
        processing_time_seconds=round(processing_time, 3)
    )
    
    logger.info(
        "Document ingestion completed successfully",
        extra={
            "request_id": request_id,
            "document_id": document_id,
            "language": language,
            "chunks_created": chunks_created,
            "processing_time": round(processing_time, 3),
            "file_name": sanitized_filename
        }
    )
    
    return response


@router.get(
    "/ingest/health",
    summary="Health Check",
    description="Check if the ingest endpoint and its dependencies are healthy"
)
async def health_check() -> Dict[str, Any]:
    """
    Health check for ingest endpoint.
    
    Verifies that all required services are operational.
    
    Returns:
        Dictionary with health status of all services
    """
    health_status = {
        "endpoint": "ingest",
        "status": "healthy",
        "services": {}
    }
    
    # Check embedding service
    try:
        embedding_health = embedding_service.health_check()
        health_status["services"]["embedding"] = embedding_health
    except Exception as e:
        health_status["services"]["embedding"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check FAISS service
    try:
        faiss_health = faiss_service.health_check()
        health_status["services"]["faiss"] = faiss_health
    except Exception as e:
        health_status["services"]["faiss"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    return health_status


@router.get(
    "/ingest/stats",
    summary="Ingestion Statistics",
    description="Get statistics about ingested documents"
)
async def get_stats(
    api_key: str = Depends(api_key_auth)
) -> Dict[str, Any]:
    """
    Get statistics about ingested documents.
    
    Returns:
        Dictionary with ingestion statistics
    """
    try:
        stats = faiss_service.get_statistics()
        return {
            "status": "success",
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )