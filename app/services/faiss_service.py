# app/services/faiss_service.py
"""
FAISS service for vector similarity search with persistent storage.

Provides robust vector index management with automatic persistence,
metadata tracking, and support for document chunking.
"""

import faiss
import json
import numpy as np
import time
import uuid
import hashlib
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from app.config import settings

logger = logging.getLogger(__name__)


class FAISSService:
    """
    Service for managing FAISS vector index with persistent storage.
    """
    
    def __init__(self, index_dir: str = None):
        """
        Initialize FAISS service with persistent storage.
        
        Args:
            index_dir: Directory for storing FAISS index and metadata
        """
        self.index_dir = Path(index_dir or settings.FAISS_INDEX_DIR)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.index_path = self.index_dir / "index.faiss"
        self.metadata_path = self.index_dir / "metadata.json"
        self.config_path = self.index_dir / "config.json"
        
        # FAISS index and metadata
        self.index = None
        self.metadata = {}
        self.doc_counter = 0
        self.embedding_dim = settings.EMBEDDING_DIM
        
        # Statistics
        self.total_vectors = 0
        self.total_documents = 0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Load or initialize index
        self._load_or_initialize()
        
        logger.info(
            "FAISSService initialized",
            extra={
                "total_vectors": self.total_vectors,
                "total_documents": self.total_documents,
                "index_dir": str(self.index_dir)
            }
        )

    def _load_or_initialize(self):
        """Load existing FAISS index or create new one."""
        try:
            if self.index_path.exists() and self.metadata_path.exists():
                logger.info("Loading existing FAISS index from disk")
                
                # Load FAISS index
                self.index = faiss.read_index(str(self.index_path))
                
                # Verify index type and dimension
                if self.index.d != self.embedding_dim:
                    logger.error(
                        f"Index dimension mismatch: expected {self.embedding_dim}, "
                        f"got {self.index.d}"
                    )
                    raise ValueError("Index dimension mismatch")
                
                # Load metadata
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                
                # Load configuration
                if self.config_path.exists():
                    with open(self.config_path, 'r') as f:
                        config = json.load(f)
                    self.doc_counter = config.get('doc_counter', 0)
                    self.total_vectors = config.get('total_vectors', self.index.ntotal)
                    self.total_documents = config.get('total_documents', 0)
                else:
                    # Backward compatibility
                    self.total_vectors = self.index.ntotal
                    self._recalculate_document_count()
                
                logger.info(
                    "FAISS index loaded successfully",
                    extra={
                        "vectors": self.total_vectors,
                        "documents": self.total_documents
                    }
                )
                
            else:
                logger.info("No existing index found, creating new FAISS index")
                self._create_new_index()
                
        except Exception as e:
            logger.error(
                "Failed to load FAISS index",
                extra={"error": str(e)},
                exc_info=True
            )
            logger.info("Creating new index due to loading failure")
            self._create_new_index()

    def _create_new_index(self):
        """
        Create a new FAISS index.
        
        Uses IndexFlatIP (inner product) which computes dot product.
        After L2 normalization, dot product equals cosine similarity.
        """
        try:
            # Use IndexFlatIP for cosine similarity (after normalization)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.metadata = {}
            self.doc_counter = 0
            self.total_vectors = 0
            self.total_documents = 0
            
            # Save initial state
            self._save_index()
            logger.info("Created new FAISS index (IndexFlatIP)")
            
        except Exception as e:
            logger.error(
                "Failed to create new FAISS index",
                extra={"error": str(e)},
                exc_info=True
            )
            raise RuntimeError(f"Could not create FAISS index: {str(e)}")

    def generate_document_id(self) -> str:
        """
        Generate a unique document ID.
        
        Returns:
            String document ID in format "doc_XXXXXXXX"
        """
        data = f"{self.doc_counter}_{time.time()}_{uuid.uuid4()}"
        hash_part = hashlib.md5(data.encode()).hexdigest()[:8]
        return f"doc_{hash_part}"

    def add_document(
        self, 
        document_id: str, 
        chunks: List[str], 
        embeddings: np.ndarray, 
        language: str,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Add a document to the FAISS index with immediate persistence.
        
        Args:
            document_id: Unique identifier for the document
            chunks: List of text chunks from the document
            embeddings: numpy array of embeddings for each chunk
            language: Language of the document ('en' or 'ja')
            metadata: Additional metadata for the document
            
        Returns:
            Number of chunks added
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If FAISS operation fails
        """
        # Input validation
        if not document_id or not chunks or len(embeddings) == 0:
            raise ValueError("Document ID, chunks, and embeddings cannot be empty")
        
        # Ensure embeddings is 2D
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        
        if len(embeddings.shape) != 2:
            raise ValueError(
                f"Embeddings must be 1D or 2D array, got shape {embeddings.shape}"
            )
            
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match number of embeddings "
                f"({embeddings.shape[0]})"
            )
            
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match "
                f"index dimension {self.embedding_dim}"
            )

        with self._lock:
            try:
                start_time = time.time()
                
                # Check if document already exists
                existing_chunks = self.get_document_chunks(document_id)
                is_new_document = len(existing_chunks) == 0
                
                if not is_new_document:
                    logger.warning(
                        f"Document {document_id} already exists, adding more chunks",
                        extra={"document_id": document_id}
                    )
                
                start_idx = self.index.ntotal
                
                # Normalize embeddings for cosine similarity
                embeddings = embeddings.astype('float32')
                faiss.normalize_L2(embeddings)
                
                # Add to FAISS index
                self.index.add(embeddings)
                
                # Store metadata for each chunk
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    vector_idx = start_idx + i
                    self.metadata[str(vector_idx)] = {
                        "doc_id": document_id,
                        "chunk_id": i,
                        "text": chunk,
                        "language": language,
                        "vector_index": vector_idx,
                        "timestamp": time.time(),
                        "embedding_norm": float(np.linalg.norm(embedding)),
                        "deleted": False,
                        **({} if metadata is None else metadata)
                    }
                
                chunks_added = len(chunks)
                self.total_vectors += chunks_added
                
                if is_new_document:
                    self.total_documents += 1
                    self.doc_counter += 1
                
                # Persist immediately to prevent data loss
                self._save_index()
                
                processing_time = time.time() - start_time
                logger.info(
                    "Document added to FAISS index",
                    extra={
                        "document_id": document_id,
                        "chunks_added": chunks_added,
                        "language": language,
                        "processing_time": round(processing_time, 3),
                        "total_vectors": self.total_vectors,
                        "total_documents": self.total_documents,
                        "is_new": is_new_document
                    }
                )
                
                return chunks_added
                
            except Exception as e:
                logger.error(
                    "Failed to add document to FAISS index",
                    extra={
                        "document_id": document_id,
                        "error": str(e)
                    },
                    exc_info=True
                )
                raise RuntimeError(f"Failed to add document to index: {str(e)}")

    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 3, 
        min_score: float = 0.5,
        language_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for similar documents in the FAISS index.
        
        Args:
            query_embedding: Query embedding vector (1D or 2D)
            top_k: Number of top results to return
            min_score: Minimum similarity score (0-1)
            language_filter: Filter results by language ('en' or 'ja')
            
        Returns:
            List of search results with metadata and similarity scores
            
        Example:
            >>> results = service.search(embedding, top_k=3, min_score=0.7)
            >>> for result in results:
            >>>     print(f"{result['text']}: {result['similarity_score']:.3f}")
        """
        if self.index.ntotal == 0:
            logger.debug("Index is empty, returning no results")
            return []
            
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        
        # Ensure query_embedding is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        if query_embedding.shape[0] != 1:
            raise ValueError(
                f"Only single query supported, got {query_embedding.shape[0]} queries. "
                "Use search_batch() for multiple queries."
            )
        
        if query_embedding.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension {query_embedding.shape[1]} does not match "
                f"index dimension {self.embedding_dim}"
            )

        with self._lock:
            try:
                # Normalize query embedding
                query_embedding = query_embedding.astype('float32')
                faiss.normalize_L2(query_embedding)
                
                # Search FAISS index (search more than needed for filtering)
                search_k = min(top_k * 3, self.index.ntotal)
                scores, indices = self.index.search(query_embedding, search_k)
                
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx == -1:  # No result
                        continue
                    
                    metadata = self.metadata.get(str(idx))
                    if not metadata:
                        continue
                    
                    # Skip deleted chunks
                    if metadata.get('deleted', False):
                        continue
                    
                    # Apply language filter if specified
                    if language_filter and metadata.get('language') != language_filter:
                        continue
                    
                    # Score is already cosine similarity (using IndexFlatIP)
                    similarity_score = float(score)
                    
                    # Clamp to [0, 1] due to numerical precision
                    similarity_score = max(0.0, min(1.0, similarity_score))
                    
                    if similarity_score >= min_score:
                        results.append({
                            **metadata,
                            "similarity_score": similarity_score
                        })
                
                # Sort by similarity score and return top_k
                results.sort(key=lambda x: x['similarity_score'], reverse=True)
                final_results = results[:top_k]
                
                logger.debug(
                    "FAISS search completed",
                    extra={
                        "results_found": len(final_results),
                        "min_score": min_score,
                        "top_k": top_k,
                        "language_filter": language_filter
                    }
                )
                
                return final_results
                
            except Exception as e:
                logger.error(
                    "FAISS search failed",
                    extra={"error": str(e)},
                    exc_info=True
                )
                raise RuntimeError(f"Search failed: {str(e)}")

    def get_document_chunks(self, document_id: str) -> List[Dict]:
        """
        Retrieve all chunks for a specific document.
        
        Args:
            document_id: Document ID to retrieve
            
        Returns:
            List of chunks with metadata, sorted by chunk_id
        """
        chunks = []
        for vector_idx, metadata in self.metadata.items():
            if (metadata.get('doc_id') == document_id and 
                not metadata.get('deleted', False)):
                chunks.append(metadata)
        
        # Sort by chunk_id
        chunks.sort(key=lambda x: x.get('chunk_id', 0))
        return chunks

    def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks of a document from the index.
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            True if document was found and deleted, False otherwise
        """
        with self._lock:
            chunks_to_delete = []
            
            # Find all chunks for this document
            for vector_idx, metadata in list(self.metadata.items()):
                if (metadata.get('doc_id') == document_id and 
                    not metadata.get('deleted', False)):
                    chunks_to_delete.append(vector_idx)
            
            if not chunks_to_delete:
                logger.warning(
                    f"Document {document_id} not found for deletion",
                    extra={"document_id": document_id}
                )
                return False
            
            # Mark chunks as deleted in metadata
            for vector_idx in chunks_to_delete:
                if vector_idx in self.metadata:
                    self.metadata[vector_idx]['deleted'] = True
                    self.metadata[vector_idx]['deleted_at'] = time.time()
            
            # Update document count
            self._recalculate_document_count()
            
            self._save_index()
            
            logger.info(
                "Document marked as deleted",
                extra={
                    "document_id": document_id,
                    "chunks_deleted": len(chunks_to_delete),
                    "total_documents": self.total_documents
                }
            )
            
            return True

    def compact_index(self):
        """
        Rebuild index without deleted documents.
        
        This removes deleted vectors from memory and rebuilds the index.
        This is an expensive operation and should be done periodically.
        
        Returns:
            Dictionary with compaction statistics
        """
        with self._lock:
            logger.info("Starting index compaction...")
            start_time = time.time()
            
            # Collect non-deleted entries
            valid_vectors = []
            valid_metadata = {}
            
            for idx in range(self.index.ntotal):
                metadata = self.metadata.get(str(idx))
                if metadata and not metadata.get('deleted', False):
                    # Get vector from index
                    vector = self.index.reconstruct(int(idx))
                    valid_vectors.append(vector)
                    
                    # Store with new index
                    new_idx = len(valid_vectors) - 1
                    valid_metadata[str(new_idx)] = {
                        **metadata,
                        "vector_index": new_idx
                    }
            
            # Create new index
            new_index = faiss.IndexFlatIP(self.embedding_dim)
            if valid_vectors:
                vectors_array = np.array(valid_vectors).astype('float32')
                new_index.add(vectors_array)
            
            # Store old stats
            old_vector_count = self.index.ntotal
            old_doc_count = self.total_documents
            
            # Replace old index
            self.index = new_index
            self.metadata = valid_metadata
            self.total_vectors = len(valid_vectors)
            self._recalculate_document_count()
            
            self._save_index()
            
            duration = time.time() - start_time
            
            stats = {
                "vectors_before": old_vector_count,
                "vectors_after": self.total_vectors,
                "vectors_removed": old_vector_count - self.total_vectors,
                "documents_before": old_doc_count,
                "documents_after": self.total_documents,
                "duration_seconds": round(duration, 2)
            }
            
            logger.info(
                "Index compaction completed",
                extra=stats
            )
            
            return stats

    def _recalculate_document_count(self):
        """Recalculate total_documents from metadata."""
        doc_ids = set()
        for metadata in self.metadata.values():
            if not metadata.get('deleted', False):
                doc_ids.add(metadata['doc_id'])
        self.total_documents = len(doc_ids)

    def _save_index(self):
        """Persist FAISS index, metadata, and configuration to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            # Save configuration
            config = {
                "doc_counter": self.doc_counter,
                "total_vectors": self.total_vectors,
                "total_documents": self.total_documents,
                "embedding_dim": self.embedding_dim,
                "last_updated": time.time(),
                "index_type": "IndexFlatIP"
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            logger.debug("FAISS index persisted to disk")
            
        except Exception as e:
            logger.error(
                "Failed to persist FAISS index",
                extra={"error": str(e)},
                exc_info=True
            )
            raise RuntimeError(f"Index persistence failed: {str(e)}")

    def get_statistics(self) -> Dict:
        """Get statistics about the FAISS index."""
        with self._lock:
            deleted_count = sum(
                1 for m in self.metadata.values() 
                if m.get('deleted', False)
            )
            
            return {
                "total_vectors": self.total_vectors,
                "total_documents": self.total_documents,
                "deleted_vectors": deleted_count,
                "active_vectors": self.total_vectors - deleted_count,
                "embedding_dimension": self.embedding_dim,
                "index_size": self.index.ntotal,
                "doc_counter": self.doc_counter,
                "index_type": "IndexFlatIP"
            }

    def health_check(self) -> Dict:
        """Perform health check on the FAISS service."""
        try:
            # Test search with random vector
            test_vector = np.random.rand(1, self.embedding_dim).astype('float32')
            results = self.search(test_vector, top_k=1, min_score=0)
            
            return {
                "status": "healthy",
                "index_loaded": True,
                "total_vectors": self.total_vectors,
                "total_documents": self.total_documents,
                "embedding_dimension": self.embedding_dim,
                "test_search_works": True
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "index_loaded": False,
                "error": str(e)
            }

    def clear_index(self):
        """Clear the entire FAISS index (for testing purposes)."""
        with self._lock:
            logger.warning("Clearing entire FAISS index")
            self._create_new_index()


# Global instance for dependency injection
faiss_service = FAISSService()