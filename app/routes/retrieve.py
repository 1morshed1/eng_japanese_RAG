# app/routes/retrieve.py

"""
Document retrieval endpoint for semantic search.

Searches the vector database for documents semantically similar to queries
with support for English and Japanese.
"""

import time
import uuid
import logging
from typing import Dict, Any
from asyncio import timeout as async_timeout  

from fastapi import APIRouter, Depends, HTTPException, status

from app.middleware.auth import api_key_auth
from app.models.schemas import RetrieveRequest, RetrieveResponse, SearchResult
from app.services.embedding_service import embedding_service
from app.services.faiss_service import faiss_service
from app.utils.language_detector import detect_language
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Configuration
RETRIEVE_TIMEOUT = getattr(settings, 'RETRIEVE_TIMEOUT', 10)
DEFAULT_TOP_K = getattr(settings, 'DEFAULT_TOP_K', 3)
MAX_TOP_K = getattr(settings, 'MAX_TOP_K', 100)


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Documents",
    description=(
        "Search for relevant documents based on semantic similarity. "
        "Supports bilingual queries (English and Japanese) with "
        "configurable result count and similarity threshold."
    ),
    responses={
        200: {
            "description": "Documents retrieved successfully",
            "model": RetrieveResponse
        },
        400: {
            "description": "Invalid request parameters"
        },
        401: {
            "description": "Invalid API key"
        },
        429: {
            "description": "Too many requests"
        },
        500: {
            "description": "Internal server error"
        }
    }
)
async def retrieve_documents(
    request: RetrieveRequest,
    api_key: str = Depends(api_key_auth)
) -> RetrieveResponse:
    """
    Retrieve relevant document chunks based on semantic similarity.
    
    Process:
    1. Validates query parameters
    2. Detects query language
    3. Generates query embedding
    4. Searches FAISS index for similar documents
    5. Filters results by similarity threshold
    6. Returns ranked results with metadata
    
    Args:
        request: RetrieveRequest containing query and search parameters
        api_key: API key for authentication (injected)
        
    Returns:
        RetrieveResponse with ranked search results
        
    Example:
```bash
        curl -X POST "http://localhost:8000/retrieve" \
          -H "X-API-Key: your-api-key" \
          -H "Content-Type: application/json" \
          -d '{"query": "diabetes management", "top_k": 3}'
```
    """
    # Generate request ID for tracking
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    logger.info(
        "Document retrieval request received",
        extra={
            "request_id": request_id,
            "query_preview": request.query[:100],
            "top_k": request.top_k,
            "min_score": request.min_score,
            "api_key": api_key[:8] + "..."
        }
    )
    
    try:
        # Apply timeout to entire operation
        async with async_timeout(RETRIEVE_TIMEOUT):
            return await _process_retrieval(request, request_id, start_time)
            
    except TimeoutError:
        logger.error(
            "Document retrieval timed out",
            extra={
                "request_id": request_id,
                "timeout": RETRIEVE_TIMEOUT
            }
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Request timed out after {RETRIEVE_TIMEOUT} seconds"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(
            "Unexpected error in document retrieval",
            extra={
                "request_id": request_id,
                "error": str(e),
                "error_type": type(e).__name__
            },
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again."
        )


async def _process_retrieval(
    request: RetrieveRequest,
    request_id: str,
    start_time: float
) -> RetrieveResponse:
    """
    Process retrieval request with detailed error handling.
    
    Args:
        request: The retrieve request
        request_id: Unique request identifier
        start_time: Request start timestamp
        
    Returns:
        RetrieveResponse with search results
    """
    # Step 1: Validate query
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty or whitespace"
        )
    
    # Validate top_k
    if request.top_k < 1 or request.top_k > MAX_TOP_K:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"top_k must be between 1 and {MAX_TOP_K}"
        )
    
    # Step 2: Detect query language
    try:
        query_language = detect_language(request.query)
        logger.debug(
            "Query language detected",
            extra={
                "request_id": request_id,
                "language": query_language
            }
        )
    except Exception as e:
        logger.warning(
            f"Language detection failed, defaulting to English: {e}",
            extra={"request_id": request_id}
        )
        query_language = "en"
    
    # Step 3: Generate query embedding
    try:
        # FIX: Extract first element from array
        query_embedding = embedding_service.encode([request.query])[0]
        
        logger.debug(
            "Query embedding generated",
            extra={
                "request_id": request_id,
                "embedding_shape": query_embedding.shape
            }
        )
        
    except Exception as e:
        logger.error(
            "Embedding generation failed",
            extra={
                "request_id": request_id,
                "error": str(e)
            },
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query. Please try again."
        )
    
    # Step 4: Search FAISS index
    try:
        search_results = faiss_service.search(
            query_embedding=query_embedding,
            top_k=request.top_k,
            min_score=request.min_score
        )
        
        logger.debug(
            "Search completed",
            extra={
                "request_id": request_id,
                "results_found": len(search_results),
                "top_score": search_results[0].get('similarity_score', 0) if search_results else 0
            }
        )
        
    except Exception as e:
        logger.error(
            "FAISS search failed",
            extra={
                "request_id": request_id,
                "error": str(e)
            },
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search service temporarily unavailable."
        )
    
    # Step 5: Check for empty results
    if not search_results:
        logger.info(
            "No documents found matching query",
            extra={
                "request_id": request_id,
                "query": request.query,
                "min_score": request.min_score
            }
        )
        # Return empty results (not an error)
        return RetrieveResponse(
            results=[],
            query_language=query_language,
            results_found=0
        )
    
    # Step 6: Convert to response format
    results = []
    for result in search_results:
        try:
            search_result = SearchResult(
                text=result.get('text', ''),
                similarity_score=result.get('similarity_score', 0.0),
                document_id=result.get('doc_id', 'unknown'),
                language=result.get('language', 'en'),
                chunk_id=result.get('chunk_id', 0)
            )
            results.append(search_result)
        except Exception as e:
            logger.warning(
                f"Failed to parse search result: {e}",
                extra={"request_id": request_id}
            )
            continue
    
    # Step 7: Prepare response
    retrieval_time = time.time() - start_time
    
    response = RetrieveResponse(
        results=results,
        query_language=query_language,
        results_found=len(results)
    )
    
    logger.info(
        "Document retrieval completed successfully",
        extra={
            "request_id": request_id,
            "query_language": query_language,
            "results_found": len(results),
            "retrieval_time": round(retrieval_time, 3)
        }
    )
    
    return response


@router.get(
    "/retrieve/health",
    summary="Health Check",
    description="Check if the retrieve endpoint and its dependencies are healthy"
)
async def health_check() -> Dict[str, Any]:
    """
    Health check for retrieve endpoint.
    
    Returns:
        Dictionary with health status
    """
    health_status = {
        "endpoint": "retrieve",
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