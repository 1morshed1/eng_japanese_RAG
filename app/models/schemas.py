# app/models/schemas.py
"""
Pydantic models for request/response validation.

All models are automatically validated by FastAPI and generate
OpenAPI documentation with examples.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List, Literal
from datetime import datetime



# Shared Validators

def validate_query_field(v: str) -> str:
    """
    Shared validator for query fields.
    
    Ensures query is not empty or just whitespace.
    """
    if not v or not v.strip():
        raise ValueError('Query cannot be empty or whitespace')
    return v.strip()



# Ingest Models

class IngestResponse(BaseModel):
    """
    Response model for document ingestion.
    
    Returned after successfully processing and storing a document.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "document_id": "doc_a1b2c3d4",
                "language": "en",
                "chunks_created": 15,
                "file_size_kb": 42.3,
                "processing_time_seconds": 2.4
            }
        }
    )
    
    status: str = Field(
        ...,
        description="Status of the ingestion operation",
        examples=["success"]
    )
    document_id: str = Field(
        ...,
        description="Unique identifier for the ingested document",
        examples=["doc_a1b2c3d4"]
    )
    language: str = Field(
        ...,
        description="Detected language of the document (en or ja)",
        examples=["en", "ja"]
    )
    chunks_created: int = Field(
        ...,
        description="Number of text chunks created",
        ge=0,
        examples=[15]
    )
    file_size_kb: float = Field(
        ...,
        description="Size of the uploaded file in kilobytes",
        ge=0,
        examples=[42.3]
    )
    processing_time_seconds: float = Field(
        ...,
        description="Time taken to process the document",
        ge=0,
        examples=[2.4]
    )



# Retrieve Models

class RetrieveRequest(BaseModel):
    """
    Request model for document retrieval.
    
    Searches the vector database for documents semantically similar
    to the provided query.
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "query": "What are the latest recommendations for Type 2 diabetes management?",
                "top_k": 3,
                "min_score": 0.5
            }
        }
    )
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Search query in English or Japanese",
        examples=["What are the diabetes management guidelines?"]
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of results to return",
        examples=[3]
    )
    min_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold (0.0-1.0)",
        examples=[0.5]
    )
    
    @field_validator('query')
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        """Validate query is not empty"""
        return validate_query_field(v)


class SearchResult(BaseModel):
    """
    Individual search result from vector database.
    
    Represents a single document chunk with similarity score.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Type 2 diabetes management requires...",
                "similarity_score": 0.87,
                "document_id": "doc_a1b2c3d4",
                "language": "en",
                "chunk_id": 5
            }
        }
    )
    
    text: str = Field(
        ...,
        description="Content of the document chunk"
    )
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score (0.0-1.0)"
    )
    document_id: str = Field(
        ...,
        description="ID of the source document"
    )
    language: str = Field(
        ...,
        description="Language of the document (en or ja)"
    )
    chunk_id: int = Field(
        ...,
        ge=0,
        description="Chunk index within the document"
    )


class RetrieveResponse(BaseModel):
    """
    Response model for document retrieval.
    
    Contains search results and metadata about the query.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "text": "Type 2 diabetes management requires...",
                        "similarity_score": 0.87,
                        "document_id": "doc_a1b2c3d4",
                        "language": "en",
                        "chunk_id": 5
                    }
                ],
                "query_language": "en",
                "results_found": 3
            }
        }
    )
    
    results: List[SearchResult] = Field(
        ...,
        description="List of search results ordered by similarity"
    )
    query_language: str = Field(
        ...,
        description="Detected language of the query (en or ja)"
    )
    results_found: int = Field(
        ...,
        ge=0,
        description="Total number of results found"
    )



# Generate Models

class GenerateRequest(BaseModel):
    """
    Request model for response generation.
    
    Generates a contextual response based on retrieved documents.
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "query": "What are the dietary recommendations for diabetes?",
                "output_language": "ja"
            }
        }
    )
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Question to answer based on retrieved documents",
        examples=["What are the dietary recommendations?"]
    )
    output_language: Optional[Literal["en", "ja"]] = Field(
        default=None,
        description="Desired output language (en or ja). If not specified, uses query language."
    )
    
    @field_validator('query')
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        """Validate query is not empty"""
        return validate_query_field(v)


class Source(BaseModel):
    """
    Source document reference in generated response.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "doc_a1b2c3d4",
                "chunk_id": 5,
                "similarity_score": 0.87,
                "language": "en"
            }
        }
    )
    
    document_id: str = Field(..., description="Source document ID")
    chunk_id: int = Field(..., ge=0, description="Chunk index")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    language: str = Field(..., description="Language of source (en or ja)")


class GenerateResponse(BaseModel):
    """
    Response model for generated answers.
    
    Contains the AI-generated response with source citations.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What are the dietary recommendations?",
                "response": "Based on the retrieved medical guidelines regarding diabetes management...",
                "language": "en",
                "sources": [
                    {
                        "document_id": "doc_a1b2c3d4",
                        "chunk_id": 5,
                        "similarity_score": 0.87
                    }
                ],
                "generation_time_seconds": 1.2
            }
        }
    )
    
    query: str = Field(
        ...,
        description="Original query that was answered"
    )
    response: str = Field(
        ...,
        description="Generated response based on retrieved documents"
    )
    language: str = Field(
        ...,
        description="Language of the response (en or ja)"
    )
    sources: List[Source] = Field(
        ...,
        description="Source documents used to generate the response"
    )
    generation_time_seconds: float = Field(
        ...,
        ge=0,
        description="Time taken to generate the response"
    )



# Common Response Models

class HealthResponse(BaseModel):
    """Health check response model."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "faiss_index_size": 1523,
                "total_documents": 12,
                "model_loaded": True,
                "uptime_seconds": 3600
            }
        }
    )
    
    status: str = Field(..., description="Health status")
    faiss_index_size: int = Field(..., ge=0, description="Number of vectors in FAISS")
    total_documents: int = Field(..., ge=0, description="Number of documents ingested")
    model_loaded: bool = Field(..., description="Whether embedding model is loaded")
    uptime_seconds: int = Field(..., ge=0, description="Server uptime in seconds")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "Invalid file format",
                "detail": "Only .txt files are supported",
                "status_code": 400
            }
        }
    )
    
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Detailed error message")
    status_code: int = Field(..., description="HTTP status code")


class StatsResponse(BaseModel):
    """Admin statistics response model."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "faiss": {
                    "total_vectors": 1523,
                    "total_documents": 12,
                    "embedding_dimension": 384,
                    "index_size_mb": 8.4
                },
                "uptime_seconds": 3600
            }
        }
    )
    
    class FAISSStats(BaseModel):
        """FAISS index statistics."""
        total_vectors: int
        total_documents: int
        embedding_dimension: int
        index_size_mb: float
    
    faiss: FAISSStats
    uptime_seconds: int