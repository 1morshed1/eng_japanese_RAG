# app/routes/generate.py
"""
Generate endpoint for creating contextual responses using RAG.

Provides natural language responses based on retrieved medical documents
with support for bilingual queries and responses.
"""

import logging
import time
import uuid
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from asyncio import timeout as async_timeout

from app.middleware.auth import api_key_auth
from app.models.schemas import GenerateRequest, GenerateResponse
from app.services.embedding_service import embedding_service
from app.services.faiss_service import faiss_service
from app.services.llm_service import llm_service
from app.services.translation_service import translation_service
from app.utils.language_detector import detect_language
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Configuration
GENERATION_TOP_K = getattr(settings, 'GENERATION_TOP_K', 5)
GENERATION_MIN_SCORE = getattr(settings, 'GENERATION_MIN_SCORE', 0.3)
GENERATION_TIMEOUT = getattr(settings, 'GENERATION_TIMEOUT', 30)


@router.post(
    "/generate",
    response_model=GenerateResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Response",
    description=(
        "Generate contextual medical responses using retrieved documents. "
        "Supports both English and Japanese queries with optional output "
        "language specification. Uses RAG (Retrieval-Augmented Generation) "
        "to provide accurate, source-backed responses."
    ),
    responses={
        200: {
            "description": "Response generated successfully",
            "model": GenerateResponse
        },
        400: {
            "description": "Invalid request"
        },
        401: {
            "description": "Invalid API key"
        },
        429: {
            "description": "Too many requests"
        },
        500: {
            "description": "Internal server error"
        },
        503: {
            "description": "Service temporarily unavailable"
        }
    }
)
async def generate_response(
    request: GenerateRequest,
    api_key: str = Depends(api_key_auth)
) -> GenerateResponse:
    """
    Generate a contextual response using retrieved documents.
    
    This endpoint:
    1. Detects the query language
    2. Translates query to English if needed (for better retrieval)
    3. Generates embeddings and retrieves relevant documents
    4. Generates a structured response using retrieved context
    5. Translates response to requested output language if needed
    
    Args:
        request: GenerateRequest containing query and optional output language
        api_key: API key for authentication (injected by dependency)
        
    Returns:
        GenerateResponse with generated response and source information
        
    Raises:
        HTTPException: Various status codes for different error conditions
        
    Example:
```python
        POST /generate
        {
            "query": "What are the diabetes management guidelines?",
            "output_language": "ja"
        }
```
    """
    # Generate request ID for tracking
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    logger.info(
        "Response generation request received",
        extra={
            "request_id": request_id,
            "query_preview": request.query[:100],
            "output_language": request.output_language,
            "api_key": api_key[:8] + "..."
        }
    )
    
    try:
        # Apply timeout to entire operation
        async with async_timeout(GENERATION_TIMEOUT):
            return await _process_generation_request(
                request, request_id, start_time
            )
            
    except TimeoutError:
        logger.error(
            "Response generation timed out",
            extra={
                "request_id": request_id,
                "timeout": GENERATION_TIMEOUT
            }
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Request timed out after {GENERATION_TIMEOUT} seconds"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(
            "Unexpected error in response generation",
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


async def _process_generation_request(
    request: GenerateRequest,
    request_id: str,
    start_time: float
) -> GenerateResponse:
    """
    Process the generation request with detailed error handling.
    
    Args:
        request: The generate request
        request_id: Unique request identifier
        start_time: Request start timestamp
        
    Returns:
        GenerateResponse with generated content
    """
    # Step 1: Detect query language
    try:
        query_language = detect_language(request.query)
        logger.debug(
            "Query language detected",
            extra={
                "request_id": request_id,
                "detected_language": query_language
            }
        )
    except Exception as e:
        logger.warning(
            f"Language detection failed, defaulting to English: {e}",
            extra={"request_id": request_id}
        )
        query_language = "en"
    
    # Step 2: Determine output language
    output_language = request.output_language or query_language
    
    # Validate output language
    if output_language not in ["en", "ja"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported output language: {output_language}"
        )
    
    # Step 3: Use original query for retrieval
    # The multilingual embedding model handles both EN and JA queries directly
    # No translation needed - it can match JA queries to JA documents and EN queries to EN documents
    query_for_retrieval = request.query
    
    # Note: Translation removed because:
    # 1. Multilingual embeddings work across languages
    # 2. Direct query matching works better for same-language documents
    # 3. Translation can introduce errors that hurt retrieval quality
    logger.debug(
        "Using original query for retrieval",
        extra={
            "request_id": request_id,
            "query_language": query_language,
            "query_preview": request.query[:50]
        }
    )
    
    # Step 4: Generate query embedding
    try:
        query_embedding = embedding_service.encode_single(query_for_retrieval)
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
    
    # Step 5: Retrieve relevant documents
    try:
        retrieved_docs = faiss_service.search(
            query_embedding=query_embedding,
            top_k=GENERATION_TOP_K,
            min_score=GENERATION_MIN_SCORE
        )
        
        logger.debug(
            "Documents retrieved",
            extra={
                "request_id": request_id,
                "documents_found": len(retrieved_docs),
                "top_score": retrieved_docs[0].get('similarity_score', 0) if retrieved_docs else 0
            }
        )
        
    except Exception as e:
        logger.error(
            "Document retrieval failed",
            extra={
                "request_id": request_id,
                "error": str(e)
            },
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document retrieval service temporarily unavailable."
        )
    
    # Step 6: Validate retrieved documents
    if not retrieved_docs:
        logger.warning(
            "No documents retrieved for query",
            extra={"request_id": request_id}
        )
        # LLM service will handle no results with appropriate message
    else:
        # Check quality of results
        high_quality_count = sum(
            1 for doc in retrieved_docs 
            if doc.get('similarity_score', 0) >= 0.5
        )
        
        if high_quality_count == 0:
            logger.warning(
                "No high-quality documents found",
                extra={
                    "request_id": request_id,
                    "best_score": retrieved_docs[0].get('similarity_score', 0)
                }
            )
    
    # Step 7: Generate response using LLM service
    try:
        llm_response = llm_service.generate_response(
            query=request.query,  # Use original query for context
            retrieved_docs=retrieved_docs,
            language=output_language
        )
        
        logger.debug(
            "LLM response generated",
            extra={
                "request_id": request_id,
                "response_length": len(llm_response["response"]),
                "sources_count": len(llm_response["sources"])
            }
        )
        
    except Exception as e:
        logger.error(
            "LLM response generation failed",
            extra={
                "request_id": request_id,
                "error": str(e)
            },
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate response. Please try again."
        )
    
    # Step 8: Prepare final response
    generation_time = time.time() - start_time
    
    response = GenerateResponse(
        query=request.query,
        response=llm_response["response"],
        language=output_language,
        sources=llm_response["sources"],
        generation_time_seconds=round(generation_time, 3)
    )
    
    logger.info(
        "Response generation completed successfully",
        extra={
            "request_id": request_id,
            "query_language": query_language,
            "output_language": output_language,
            "documents_used": len(retrieved_docs),
            "generation_time": round(generation_time, 3),
            "response_length": len(response.response)
        }
    )
    
    return response


@router.get(
    "/generate/health",
    summary="Health Check",
    description="Check if the generate endpoint and its dependencies are healthy"
)
async def health_check() -> Dict[str, Any]:
    """
    Health check for generate endpoint.
    
    Verifies that all required services (embedding, FAISS, LLM) are operational.
    
    Returns:
        Dictionary with health status of all services
    """
    health_status = {
        "endpoint": "generate",
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
    
    # Check LLM service
    try:
        llm_health = llm_service.health_check()
        health_status["services"]["llm"] = llm_health
    except Exception as e:
        health_status["services"]["llm"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check translation service
    try:
        translation_health = translation_service.health_check()
        health_status["services"]["translation"] = translation_health
    except Exception as e:
        health_status["services"]["translation"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    return health_status