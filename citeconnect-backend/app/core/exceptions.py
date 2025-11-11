# app/core/exceptions.py

"""
Custom Exception Classes Module

This module defines the exception hierarchy for the CiteConnect application.
All custom exceptions inherit from CiteConnectException base class.

Exception Hierarchy:
- CiteConnectException (base)
  - AuthenticationError (401)
  - AuthorizationError (403)
  - ResourceNotFoundError (404)
  - ValidationError (400)
  - RateLimitError (429)
  - DatabaseError (500)
  - ExternalServiceError (503)

Usage:
    from app.core.exceptions import ResourceNotFoundError
    
    raise ResourceNotFoundError("Paper", "arxiv:2401.12345")
"""

import logging
from typing import Optional, Dict, Any

# Initialize logger for this module
logger = logging.getLogger(__name__)


class CiteConnectException(Exception):
    """
    Base exception class for CiteConnect application.
    
    All custom exceptions should inherit from this class to maintain
    a consistent exception hierarchy and handling mechanism.
    
    Attributes:
        message: Human-readable error message
        status_code: HTTP status code associated with this error
        details: Additional error details for debugging
    """
    
    def __init__(
        self, 
        message: str, 
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize CiteConnectException.
        
        Args:
            message: Error message describing what went wrong
            status_code: HTTP status code (default: 500)
            details: Optional dictionary with additional error context
        """
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)
        
        # Log exception creation
        logger.debug(
            f"Exception created: {self.__class__.__name__}",
            extra={
                "message": message,
                "status_code": status_code,
                "details": details
            }
        )


class AuthenticationError(CiteConnectException):
    """
    Exception raised when authentication fails.
    
    This includes invalid credentials, expired tokens, or missing
    authentication information.
    
    HTTP Status Code: 401 Unauthorized
    
    Examples:
        - Invalid email/password combination
        - Expired JWT token
        - Missing Authorization header
    """
    
    def __init__(self, message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None):
        """
        Initialize AuthenticationError.
        
        Args:
            message: Error message (default: "Authentication failed")
            details: Additional error context
        """
        logger.warning(f"Authentication error: {message}")
        super().__init__(message, status_code=401, details=details)


class AuthorizationError(CiteConnectException):
    """
    Exception raised when user is authenticated but not authorized.
    
    This occurs when a user tries to access resources or perform actions
    they don't have permission for.
    
    HTTP Status Code: 403 Forbidden
    
    Examples:
        - Accessing another user's private data
        - Performing admin actions without admin role
    """
    
    def __init__(self, message: str = "Not authorized", details: Optional[Dict[str, Any]] = None):
        """
        Initialize AuthorizationError.
        
        Args:
            message: Error message (default: "Not authorized")
            details: Additional error context
        """
        logger.warning(f"Authorization error: {message}")
        super().__init__(message, status_code=403, details=details)


class ResourceNotFoundError(CiteConnectException):
    """
    Exception raised when a requested resource doesn't exist.
    
    HTTP Status Code: 404 Not Found
    
    Examples:
        - Paper ID not found in database
        - User profile doesn't exist
        - Cluster ID is invalid
    """
    
    def __init__(self, resource: str, identifier: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize ResourceNotFoundError.
        
        Args:
            resource: Type of resource (e.g., "Paper", "User", "Cluster")
            identifier: Identifier that was not found
            details: Additional error context
        """
        message = f"{resource} with identifier '{identifier}' not found"
        logger.info(f"Resource not found: {message}")
        super().__init__(message, status_code=404, details=details)


class ValidationError(CiteConnectException):
    """
    Exception raised when input validation fails.
    
    HTTP Status Code: 400 Bad Request
    
    Examples:
        - Invalid email format
        - Missing required fields
        - Invalid parameter values
        - Schema validation failures
    """
    
    def __init__(self, message: str, field: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize ValidationError.
        
        Args:
            message: Validation error message
            field: Name of the field that failed validation (optional)
            details: Additional validation error details
        """
        if field:
            full_message = f"Validation error for field '{field}': {message}"
        else:
            full_message = f"Validation error: {message}"
        
        logger.info(f"Validation error: {full_message}")
        
        error_details = details or {}
        if field:
            error_details['field'] = field
        
        super().__init__(full_message, status_code=400, details=error_details)


class RateLimitError(CiteConnectException):
    """
    Exception raised when rate limit is exceeded.
    
    HTTP Status Code: 429 Too Many Requests
    
    Examples:
        - Too many API requests in short time
        - Exceeding search query limits
        - Too many login attempts
    """
    
    def __init__(
        self, 
        message: str = "Rate limit exceeded", 
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize RateLimitError.
        
        Args:
            message: Error message (default: "Rate limit exceeded")
            retry_after: Seconds until rate limit resets (optional)
            details: Additional error context
        """
        logger.warning(f"Rate limit exceeded: {message}")
        
        error_details = details or {}
        if retry_after:
            error_details['retry_after'] = retry_after
        
        super().__init__(message, status_code=429, details=error_details)


class DatabaseError(CiteConnectException):
    """
    Exception raised when database operations fail.
    
    HTTP Status Code: 500 Internal Server Error
    
    Examples:
        - Connection to database failed
        - Query execution error
        - Transaction rollback
        - Data integrity violations
    """
    
    def __init__(self, message: str, operation: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize DatabaseError.
        
        Args:
            message: Error message describing the database issue
            operation: Database operation that failed (e.g., "INSERT", "SELECT")
            details: Additional error context
        """
        if operation:
            full_message = f"Database error during {operation}: {message}"
        else:
            full_message = f"Database error: {message}"
        
        logger.error(f"Database error: {full_message}")
        
        error_details = details or {}
        if operation:
            error_details['operation'] = operation
        
        super().__init__(full_message, status_code=500, details=error_details)


class ExternalServiceError(CiteConnectException):
    """
    Exception raised when external service calls fail.
    
    HTTP Status Code: 503 Service Unavailable
    
    Examples:
        - Semantic Scholar API timeout
        - OpenAI API rate limit
        - Google Scholar scraping failure
        - Weaviate connection error
    """
    
    def __init__(
        self, 
        service: str, 
        message: str,
        is_temporary: bool = True,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ExternalServiceError.
        
        Args:
            service: Name of the external service (e.g., "Semantic Scholar", "OpenAI")
            message: Error message describing what failed
            is_temporary: Whether the error is likely temporary (default: True)
            details: Additional error context
        """
        full_message = f"{service} service error: {message}"
        logger.error(f"External service error: {full_message}")
        
        error_details = details or {}
        error_details['service'] = service
        error_details['is_temporary'] = is_temporary
        
        super().__init__(full_message, status_code=503, details=error_details)


class EmbeddingError(CiteConnectException):
    """
    Exception raised when embedding generation fails.
    
    HTTP Status Code: 500 Internal Server Error
    
    Examples:
        - SPECTER model loading failure
        - Embedding generation timeout
        - Invalid input text for embedding
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize EmbeddingError.
        
        Args:
            message: Error message describing the embedding failure
            details: Additional error context
        """
        full_message = f"Embedding generation error: {message}"
        logger.error(full_message)
        super().__init__(full_message, status_code=500, details=details)


class CachingError(CiteConnectException):
    """
    Exception raised when caching operations fail.
    
    HTTP Status Code: 500 Internal Server Error
    
    Note: These errors are typically logged but don't stop execution,
    as the system can continue without caching.
    
    Examples:
        - Redis connection lost
        - Cache serialization failure
        - Cache key not found
    """
    
    def __init__(self, message: str, operation: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize CachingError.
        
        Args:
            message: Error message describing the caching issue
            operation: Cache operation that failed (e.g., "get", "set", "delete")
            details: Additional error context
        """
        if operation:
            full_message = f"Cache {operation} error: {message}"
        else:
            full_message = f"Caching error: {message}"
        
        logger.warning(f"Caching error: {full_message}")
        
        error_details = details or {}
        if operation:
            error_details['operation'] = operation
        
        super().__init__(full_message, status_code=500, details=error_details)