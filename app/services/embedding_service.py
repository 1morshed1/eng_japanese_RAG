# app/services/embedding_service.py

import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
import logging
from app.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Service for generating multilingual embeddings using sentence-transformers.
    Handles embedding generation for both English and Japanese text.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                      Defaults to settings.EMBEDDING_MODEL
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.model = None
        self.embedding_dim = settings.EMBEDDING_DIM
        self._model_loaded = False
        
        logger.info(f"EmbeddingService initialized with model: {self.model_name}")

    def _load_model(self):
        """
        Lazy load the embedding model to avoid slow startup.
        """
        if self._model_loaded:
            return
            
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self._model_loaded = True
            logger.info("Embedding model loaded successfully")
            
            # Verify embedding dimension
            test_embedding = self.model.encode(["test"])
            actual_dim = test_embedding.shape[1]
            if actual_dim != self.embedding_dim:
                logger.warning(
                    f"Model embedding dimension ({actual_dim}) doesn't match "
                    f"configured dimension ({self.embedding_dim}). Updating config."
                )
                self.embedding_dim = actual_dim
                
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise RuntimeError(f"Could not load embedding model: {str(e)}")

    def encode(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for input text(s).
        
        Args:
            texts: Single text string or list of text strings
            batch_size: Number of texts to process in parallel
            normalize_embeddings: Whether to L2 normalize embeddings
            show_progress_bar: Whether to show progress bar for large batches
            
        Returns:
            numpy array of shape (len(texts), embedding_dim) containing embeddings
            
        Raises:
            RuntimeError: If model fails to load or encode
            ValueError: If input texts are invalid
        """
        if not texts:
            raise ValueError("Texts cannot be empty")
            
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
            
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All texts must be strings")
            
        # Filter out empty strings
        valid_texts = [text for text in texts if text.strip()]
        if not valid_texts:
            raise ValueError("No valid non-empty texts provided")
            
        if len(valid_texts) != len(texts):
            logger.warning(f"Filtered out {len(texts) - len(valid_texts)} empty texts")

        try:
            self._load_model()
            
            logger.debug(
                f"Generating embeddings for {len(valid_texts)} texts, "
                f"batch_size={batch_size}"
            )
            
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True
            )
            
            logger.debug(
                f"Generated embeddings shape: {embeddings.shape}, "
                f"dtype: {embeddings.dtype}"
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(
                f"Embedding generation failed for {len(valid_texts)} texts: {str(e)}",
                extra={"error": str(e), "text_count": len(valid_texts)}
            )
            raise RuntimeError(f"Embedding generation failed: {str(e)}")

    def encode_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text string
            
        Returns:
            numpy array of shape (embedding_dim,) containing the embedding
        """
        embeddings = self.encode([text])
        return embeddings[0]

    def get_embedding_dimension(self) -> int:
        """
        Get the embedding dimension of the model.
        
        Returns:
            Integer representing embedding dimension
        """
        self._load_model()
        return self.embedding_dim

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        self._load_model()
        
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "model_loaded": self._model_loaded,
            "max_seq_length": self.model.max_seq_length if self.model else None
        }

    def health_check(self) -> dict:
        """
        Perform health check on the embedding service.
        
        Returns:
            Dictionary with health status and metrics
        """
        try:
            self._load_model()
            
            # Test with a simple embedding
            test_text = "Health check"
            embedding = self.encode_single(test_text)
            
            return {
                "status": "healthy",
                "model_loaded": self._model_loaded,
                "embedding_dimension": self.embedding_dim,
                "test_embedding_shape": embedding.shape,
                "test_embedding_norm": float(np.linalg.norm(embedding))
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "model_loaded": self._model_loaded,
                "error": str(e)
            }


# Global instance for easy dependency injection
embedding_service = EmbeddingService()


# For testing purposes
if __name__ == "__main__":
    # Quick test
    service = EmbeddingService()
    
    # Test single text
    embedding = service.encode_single("Hello, world!")
    print(f"Single embedding shape: {embedding.shape}")
    
    # Test multiple texts
    texts = ["Hello world", "こんにちは世界", "Medical diagnosis and treatment"]
    embeddings = service.encode(texts)
    print(f"Batch embeddings shape: {embeddings.shape}")
    
    # Test model info
    info = service.get_model_info()
    print(f"Model info: {info}")
    
    # Test health check
    health = service.health_check()
    print(f"Health check: {health}")