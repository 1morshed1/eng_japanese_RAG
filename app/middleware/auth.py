# app/middleware/auth.py
"""
API Key authentication middleware for FastAPI.

Provides secure API key validation using constant-time comparison
and comprehensive logging for security monitoring.
"""

import logging
import secrets
from typing import Optional

from fastapi import Header, HTTPException, Request
from app.config import settings

logger = logging.getLogger(__name__)


async def api_key_auth(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> str:
    """
    API Key authentication dependency.
    
    Validates the X-API-Key header against configured API keys using
    constant-time comparison to prevent timing attacks.
    
    Args:
        request: FastAPI request object for logging and tracking
        x_api_key: API key from X-API-Key header
        
    Returns:
        str: The validated API key
        
    Raises:
        HTTPException: 401 if API key is missing or invalid
        
    Example:
        >>> @app.get("/protected", dependencies=[Depends(api_key_auth)])
        >>> async def protected_endpoint():
        >>>     return {"message": "Success"}
        
    Security:
        - Uses secrets.compare_digest() for constant-time comparison
        - Logs authentication attempts without exposing keys
        - Includes request tracking via request_id
    """
    # Get request ID for tracking
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    # Get client IP for logging
    client_ip = request.client.host if request.client else "unknown"
    
    # Log authentication attempt
    logger.info(
        "Authentication attempt",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "client_ip": client_ip,
            "has_api_key": x_api_key is not None
        }
    )
    
    # Check if API key is provided
    if not x_api_key:
        logger.warning(
            "Missing API key",
            extra={
                "request_id": request_id,
                "client_ip": client_ip,
                "path": request.url.path
            }
        )
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include 'X-API-Key' header in your request.",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    # Get valid API keys from settings
    valid_keys = settings.get_api_keys_list()
    
    # Validate API key using constant-time comparison
    # This prevents timing attacks that could leak valid key information
    is_valid = any(
        secrets.compare_digest(x_api_key, valid_key)
        for valid_key in valid_keys
    )
    
    if not is_valid:
        # Log failed authentication (but don't log the actual key)
        logger.warning(
            "Invalid API key",
            extra={
                "request_id": request_id,
                "client_ip": client_ip,
                "path": request.url.path,
                "api_key_prefix": x_api_key[:8] + "..." if len(x_api_key) > 8 else "***"
            }
        )
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Please check your credentials.",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    # Log successful authentication
    logger.info(
        "Authentication successful",
        extra={
            "request_id": request_id,
            "client_ip": client_ip,
            "path": request.url.path
        }
    )
    
    return x_api_key


class APIKeyAuth:
    """
    Alternative class-based authentication dependency.
    
    Useful if you need to maintain state or customize behavior per instance.
    
    Example:
        >>> auth = APIKeyAuth()
        >>> @app.get("/protected", dependencies=[Depends(auth)])
        >>> async def protected_endpoint():
        >>>     return {"message": "Success"}
    """
    
    def __init__(self):
        """Initialize the API key authenticator."""
        self.valid_keys = set(settings.get_api_keys_list())
        logger.info(f"APIKeyAuth initialized with {len(self.valid_keys)} valid keys")
    
    async def __call__(
        self,
        request: Request,
        x_api_key: Optional[str] = Header(None, alias="X-API-Key")
    ) -> str:
        """
        Validate API key.
        
        This is a wrapper around the api_key_auth function for class-based usage.
        """
        # Use the same logic as the function
        return await api_key_auth(request, x_api_key)


# Create a global instance for easy importing
api_key_auth_instance = APIKeyAuth()