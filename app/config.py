# app/config.py
"""
Application configuration management using Pydantic Settings.

Environment variables are loaded from .env file or system environment.
All configuration is validated at startup.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import Optional
import os


class Settings(BaseSettings):
    """
    Application settings with validation.
    
    Environment Variables:
        API_KEYS: Comma-separated API keys (required)
        LOG_LEVEL: Logging level (default: INFO)
        MAX_FILE_SIZE_MB: Maximum file upload size (default: 10)
        EMBEDDING_MODEL: Sentence transformer model name
        ENV: Environment (development/production/testing)
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore" 
    )
    

    # API Configuration

    API_KEYS: str = Field(
        default="",
        description="Comma-separated list of valid API keys"
    )
    HOST: str = Field(default="0.0.0.0", description="API host")
    PORT: int = Field(default=8000, ge=1024, le=65535, description="API port")
    ENV: str = Field(default="development", description="Environment")
    CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:8000",
        description="Comma-separated CORS origins"
    )
    

    # Logging Configuration

    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    

    # Model Configuration

    EMBEDDING_MODEL: str = Field(
        default="paraphrase-multilingual-MiniLM-L12-v2",
        description="Sentence transformer model for embeddings"
    )
    TRANSLATION_MODEL_EN_JA: str = Field(
        default="Helsinki-NLP/opus-mt-en-ja",
        description="English to Japanese translation model"
    )
    TRANSLATION_MODEL_JA_EN: str = Field(
        default="Helsinki-NLP/opus-mt-ja-en",
        description="Japanese to English translation model"
    )
    

    # FAISS Configuration

    FAISS_INDEX_DIR: str = Field(
        default="data/faiss_index",
        description="Directory for FAISS index storage"
    )
    EMBEDDING_DIM: int = Field(
        default=384,
        description="Embedding dimension (must match model)"
    )
    

    # File Upload Limits

    MAX_FILE_SIZE_MB: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum file upload size in MB"
    )
    MAX_CHUNK_SIZE: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Target chunk size in characters"
    )
    MAX_CHUNK_OVERLAP: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Number of sentences to overlap between chunks"
    )
    ALLOWED_EXTENSIONS: str = Field(
        default=".txt",
        description="Comma-separated allowed file extensions"
    )
    

    # Validators
    
    @field_validator('API_KEYS', mode='before')
    @classmethod
    def validate_api_keys(cls, v):
        """Ensure API keys are provided"""
        if not v or v.strip() == "":
            raise ValueError(
                "API_KEYS must be set in environment variables. "
                "Example: API_KEYS=key1,key2,key3"
            )
        return v
    
    @field_validator('LOG_LEVEL')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()
    
    @field_validator('ENV')
    @classmethod
    def validate_env(cls, v):
        """Validate environment"""
        valid_envs = ["development", "production", "testing"]
        if v.lower() not in valid_envs:
            raise ValueError(f"ENV must be one of {valid_envs}")
        return v.lower()
    
    @field_validator('EMBEDDING_DIM')
    @classmethod
    def validate_embedding_dim(cls, v):
        """Validate embedding dimension"""
        valid_dims = [128, 256, 384, 512, 768, 1024]
        if v not in valid_dims:
            raise ValueError(
                f"EMBEDDING_DIM must be one of {valid_dims}. "
                f"Must match your embedding model's output dimension."
            )
        return v
    
    @field_validator('FAISS_INDEX_DIR', mode='after')
    @classmethod
    def create_index_dir(cls, v):
        """Ensure FAISS index directory exists"""
        os.makedirs(v, exist_ok=True)
        return v
    

    # Helper Methods
    
    def get_api_keys_list(self) -> list[str]:
        """
        Parse API_KEYS string into list.
        
        Returns:
            List of API key strings
        """
        return [key.strip() for key in self.API_KEYS.split(",") if key.strip()]
    
    def get_cors_origins_list(self) -> list[str]:
        """
        Parse CORS_ORIGINS string into list.
        
        Returns:
            List of allowed CORS origin URLs
        """
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]
    
    def get_allowed_extensions_set(self) -> set[str]:
        """
        Parse ALLOWED_EXTENSIONS string into set.
        
        Returns:
            Set of allowed file extensions
        """
        return {ext.strip() for ext in self.ALLOWED_EXTENSIONS.split(",") if ext.strip()}
    
    @property
    def max_file_size_bytes(self) -> int:
        """
        Convert MAX_FILE_SIZE_MB to bytes.
        
        Returns:
            Maximum file size in bytes
        """
        return self.MAX_FILE_SIZE_MB * 1024 * 1024
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.ENV == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.ENV == "development"


# Create global settings instance
settings = Settings()


# Print configuration summary (only in development)
if settings.is_development:
    print("\n" + "="*50)
    print("Healthcare RAG Assistant - Configuration Loaded")
    print("="*50)
    print(f"Environment: {settings.ENV}")
    print(f"Host: {settings.HOST}:{settings.PORT}")
    print(f"Log Level: {settings.LOG_LEVEL}")
    print(f"API Keys Configured: {len(settings.get_api_keys_list())}")
    print(f"Max File Size: {settings.MAX_FILE_SIZE_MB}MB")
    print(f"FAISS Index Dir: {settings.FAISS_INDEX_DIR}")
    print(f"Embedding Model: {settings.EMBEDDING_MODEL}")
    print(f"Embedding Dimension: {settings.EMBEDDING_DIM}")
    print("="*50 + "\n")