# app/core/security.py

"""
Security Module

This module handles authentication security including:
- JWT token generation and verification
- Password hashing and verification
- Token payload extraction

Uses:
- python-jose for JWT operations
- passlib with bcrypt for password hashing

Usage:
    from app.core.security import create_access_token, verify_password, hash_password
    
    # Hash password
    hashed = hash_password("mypassword")
    
    # Verify password
    is_valid = verify_password("mypassword", hashed)
    
    # Create JWT token
    token = create_access_token(data={"sub": "user@example.com"})
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import get_settings
from app.core.exceptions import AuthenticationError

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Password hashing context using bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Get settings
settings = get_settings()


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    Bcrypt automatically handles salt generation and uses adaptive hashing
    to remain secure against brute-force attacks.
    
    Args:
        password: Plain text password to hash
    
    Returns:
        Hashed password string safe for storage
    
    Raises:
        ValueError: If password is empty or None
    
    Example:
        >>> hashed = hash_password("MySecurePassword123")
        >>> print(hashed)
        $2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW
    """
    logger.info("Hashing password")
    
    if not password:
        logger.error("Attempted to hash empty password")
        raise ValueError("Password cannot be empty")
    
    try:
        hashed_password = pwd_context.hash(password)
        logger.debug("Password hashed successfully")
        return hashed_password
    except Exception as e:
        logger.error(f"Failed to hash password: {str(e)}", exc_info=True)
        raise


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    
    Compares the plain text password with the stored hash using
    constant-time comparison to prevent timing attacks.
    
    Args:
        plain_password: Plain text password to verify
        hashed_password: Stored password hash
    
    Returns:
        True if password matches, False otherwise
    
    Example:
        >>> hashed = hash_password("MyPassword")
        >>> verify_password("MyPassword", hashed)
        True
        >>> verify_password("WrongPassword", hashed)
        False
    """
    logger.debug("Verifying password")
    
    if not plain_password or not hashed_password:
        logger.warning("Password verification called with empty password or hash")
        return False
    
    try:
        is_valid = pwd_context.verify(plain_password, hashed_password)
        
        if is_valid:
            logger.debug("Password verification successful")
        else:
            logger.debug("Password verification failed - incorrect password")
        
        return is_valid
        
    except Exception as e:
        logger.error(f"Error during password verification: {str(e)}", exc_info=True)
        return False


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.
    
    Generates a signed JWT token containing the provided data plus
    expiration time. The token is signed using the SECRET_KEY from settings.
    
    Args:
        data: Dictionary of data to encode in the token (e.g., {"sub": "user@example.com"})
        expires_delta: Optional custom expiration time. If not provided, uses default from settings
    
    Returns:
        Encoded JWT token string
    
    Raises:
        Exception: If token creation fails
    
    Example:
        >>> token = create_access_token(
        ...     data={"sub": "user@example.com", "user_id": 123}
        ... )
        >>> print(token)
        eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
    """
    logger.info(
        "Creating access token",
        extra={"data_keys": list(data.keys())}
    )
    
    try:
        # Copy data to avoid modifying original
        to_encode = data.copy()
        
        # Calculate expiration time
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                hours=settings.ACCESS_TOKEN_EXPIRE_HOURS
            )
        
        # Add standard JWT claims
        to_encode.update({
            "exp": expire,  # Expiration time
            "iat": datetime.utcnow(),  # Issued at time
            "type": "access"  # Token type
        })
        
        # Encode token
        encoded_jwt = jwt.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM
        )
        
        logger.info(
            "Access token created successfully",
            extra={
                "expires_at": expire.isoformat(),
                "token_type": "access"
            }
        )
        
        return encoded_jwt
        
    except Exception as e:
        logger.error(f"Failed to create access token: {str(e)}", exc_info=True)
        raise


def create_refresh_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT refresh token.
    
    Refresh tokens have longer expiration times and are used to obtain
    new access tokens without requiring the user to log in again.
    
    Args:
        data: Dictionary of data to encode in the token
        expires_delta: Optional custom expiration time. If not provided, uses default from settings
    
    Returns:
        Encoded JWT refresh token string
    
    Example:
        >>> token = create_refresh_token(data={"sub": "user@example.com"})
    """
    logger.info(
        "Creating refresh token",
        extra={"data_keys": list(data.keys())}
    )
    
    try:
        # Copy data to avoid modifying original
        to_encode = data.copy()
        
        # Calculate expiration time (longer than access token)
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                days=settings.REFRESH_TOKEN_EXPIRE_DAYS
            )
        
        # Add standard JWT claims
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"  # Token type
        })
        
        # Encode token
        encoded_jwt = jwt.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM
        )
        
        logger.info(
            "Refresh token created successfully",
            extra={
                "expires_at": expire.isoformat(),
                "token_type": "refresh"
            }
        )
        
        return encoded_jwt
        
    except Exception as e:
        logger.error(f"Failed to create refresh token: {str(e)}", exc_info=True)
        raise


def decode_token(token: str) -> Dict[str, Any]:
    """
    Decode and verify a JWT token.
    
    Verifies the token signature and expiration, then returns the payload.
    
    Args:
        token: JWT token string to decode
    
    Returns:
        Dictionary containing the token payload
    
    Raises:
        AuthenticationError: If token is invalid, expired, or tampered with
    
    Example:
        >>> token = create_access_token(data={"sub": "user@example.com"})
        >>> payload = decode_token(token)
        >>> print(payload["sub"])
        user@example.com
    """
    logger.debug("Decoding JWT token")
    
    try:
        # Decode and verify token
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        
        logger.debug(
            "Token decoded successfully",
            extra={"payload_keys": list(payload.keys())}
        )
        
        return payload
        
    except jwt.ExpiredSignatureError:
        logger.warning("Token has expired")
        raise AuthenticationError(
            message="Token has expired",
            details={"error": "expired_token"}
        )
        
    except jwt.JWTClaimsError as e:
        logger.warning(f"Invalid token claims: {str(e)}")
        raise AuthenticationError(
            message="Invalid token claims",
            details={"error": "invalid_claims"}
        )
        
    except JWTError as e:
        logger.warning(f"Invalid token: {str(e)}")
        raise AuthenticationError(
            message="Could not validate credentials",
            details={"error": "invalid_token"}
        )
        
    except Exception as e:
        logger.error(f"Unexpected error decoding token: {str(e)}", exc_info=True)
        raise AuthenticationError(
            message="Could not validate credentials",
            details={"error": "token_decode_error"}
        )


def get_token_payload(token: str, verify_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Get token payload with optional type verification.
    
    Decodes token and optionally verifies it's the expected type
    (access or refresh).
    
    Args:
        token: JWT token string
        verify_type: Expected token type ("access" or "refresh"). If None, type is not verified
    
    Returns:
        Token payload dictionary
    
    Raises:
        AuthenticationError: If token is invalid or wrong type
    
    Example:
        >>> token = create_access_token(data={"sub": "user@example.com"})
        >>> payload = get_token_payload(token, verify_type="access")
    """
    logger.info(
        "Getting token payload",
        extra={"verify_type": verify_type}
    )
    
    # Decode token
    payload = decode_token(token)
    
    # Verify token type if requested
    if verify_type:
        token_type = payload.get("type")
        
        if token_type != verify_type:
            logger.warning(
                f"Token type mismatch: expected {verify_type}, got {token_type}"
            )
            raise AuthenticationError(
                message=f"Invalid token type. Expected {verify_type} token",
                details={
                    "error": "invalid_token_type",
                    "expected": verify_type,
                    "received": token_type
                }
            )
    
    return payload


def extract_user_id_from_token(token: str) -> int:
    """
    Extract user ID from JWT token.
    
    Convenience function to decode token and extract the user_id claim.
    The token must contain a "sub" claim with the user ID.
    
    Args:
        token: JWT token string
    
    Returns:
        User ID as integer
    
    Raises:
        AuthenticationError: If token is invalid or doesn't contain user_id
    
    Example:
        >>> token = create_access_token(data={"sub": "123"})
        >>> user_id = extract_user_id_from_token(token)
        >>> print(user_id)
        123
    """
    logger.debug("Extracting user ID from token")
    
    payload = decode_token(token)
    
    # Get user identifier from "sub" claim
    user_id_str = payload.get("sub")
    
    if not user_id_str:
        logger.error("Token does not contain 'sub' claim")
        raise AuthenticationError(
            message="Invalid token: missing user identifier",
            details={"error": "missing_sub_claim"}
        )
    
    try:
        user_id = int(user_id_str)
        logger.debug(f"Extracted user_id: {user_id}")
        return user_id
        
    except ValueError:
        logger.error(f"Invalid user_id format in token: {user_id_str}")
        raise AuthenticationError(
            message="Invalid token: malformed user identifier",
            details={"error": "invalid_user_id_format"}
        )


def extract_email_from_token(token: str) -> str:
    """
    Extract email from JWT token.
    
    Convenience function to decode token and extract the email claim.
    
    Args:
        token: JWT token string
    
    Returns:
        Email address as string
    
    Raises:
        AuthenticationError: If token is invalid or doesn't contain email
    
    Example:
        >>> token = create_access_token(data={"sub": "123", "email": "user@example.com"})
        >>> email = extract_email_from_token(token)
        >>> print(email)
        user@example.com
    """
    logger.debug("Extracting email from token")
    
    payload = decode_token(token)
    
    email = payload.get("email")
    
    if not email:
        logger.error("Token does not contain 'email' claim")
        raise AuthenticationError(
            message="Invalid token: missing email",
            details={"error": "missing_email_claim"}
        )
    
    logger.debug(f"Extracted email: {email}")
    return email


def verify_token_not_expired(token: str) -> bool:
    """
    Check if token is expired without raising an exception.
    
    Args:
        token: JWT token string
    
    Returns:
        True if token is valid and not expired, False otherwise
    
    Example:
        >>> token = create_access_token(data={"sub": "123"})
        >>> is_valid = verify_token_not_expired(token)
        >>> print(is_valid)
        True
    """
    logger.debug("Checking token expiration")
    
    try:
        decode_token(token)
        logger.debug("Token is valid and not expired")
        return True
        
    except AuthenticationError as e:
        if e.details.get("error") == "expired_token":
            logger.debug("Token is expired")
        else:
            logger.debug(f"Token is invalid: {e.message}")
        return False


def create_token_pair(user_id: int, email: str, additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    Create both access and refresh tokens for a user.
    
    Convenience function to generate both tokens at once during login.
    
    Args:
        user_id: User's ID
        email: User's email address
        additional_data: Optional additional data to include in tokens
    
    Returns:
        Dictionary with access_token and refresh_token
    
    Example:
        >>> tokens = create_token_pair(user_id=123, email="user@example.com")
        >>> print(tokens["access_token"])
        >>> print(tokens["refresh_token"])
    """
    logger.info(
        "Creating token pair",
        extra={"user_id": user_id, "email": email}
    )
    
    # Prepare token data
    token_data = {
        "sub": str(user_id),
        "email": email
    }
    
    # Add any additional data
    if additional_data:
        token_data.update(additional_data)
    
    # Create both tokens
    access_token = create_access_token(data=token_data)
    refresh_token = create_refresh_token(data=token_data)
    
    logger.info(
        "Token pair created successfully",
        extra={"user_id": user_id}
    )
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


# Initialize module logger
logger.info("Security module loaded successfully")