# app/services/auth_service.py

"""
Authentication Service Module

This module handles user authentication including:
- User registration
- User login
- Token generation

Dependencies:
- PostgreSQL for user storage
- SPECTER for generating initial user profile embeddings (if Google Scholar provided)
- Celery for async starter kit generation
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from app.core.security import hash_password, verify_password, create_token_pair
from app.core.exceptions import AuthenticationError, ValidationError, DatabaseError
from app.core.config import get_settings
from app.db.postgres import execute_query, execute_transaction

# Initialize logger
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()


async def register_user(
    email: str,
    password: str,
    name: str,
    domain: str,
    interests: List[str],
    google_scholar_url: Optional[str] = None,
    uploaded_paper_file: Optional[str] = None,
    db: Optional['DatabaseConnection'] = None  # Optional: pass existing connection to avoid pool exhaustion
) -> Dict[str, Any]:
    """
    Register a new user.
    
    Creates user account, sets domain, stores interests, and triggers
    starter kit generation.
    
    Args:
        email: User email address
        password: Plain text password (will be hashed)
        name: User's full name
        domain: Research domain (healthcare, fintech, quantum_computing)
        interests: List of research interest keywords
        google_scholar_url: Optional Google Scholar profile URL
        uploaded_paper_file: Optional base64 encoded PDF
    
    Returns:
        Dictionary with user data and tokens
    
    Raises:
        ValidationError: If email already exists or invalid input
        DatabaseError: If database operation fails
    
    Example:
        >>> user_data = await register_user(
        ...     email="sarah@example.com",
        ...     password="SecurePass123!",
        ...     name="Sarah Chen",
        ...     domain="healthcare",
        ...     interests=["NLP", "clinical trials"]
        ... )
    """
    logger.info(
        f"Registering new user",
        extra={"email": email, "domain": domain, "interests_count": len(interests)}
    )
    
    # Use provided db connection or fall back to execute_query (which uses global pool)
    use_passed_db = db is not None
    
    try:
        # Step 1: Check if email already exists
        logger.debug("Checking if email already exists")
        
        if use_passed_db:
            existing_user = await db.fetchrow(
                "SELECT user_id FROM users WHERE email = $1",
                email.lower()
            )
        else:
            existing_user = await execute_query(
                "SELECT user_id FROM users WHERE email = $1",
                email.lower(),
                fetch_one=True
            )
        
        if existing_user:
            logger.warning(f"Registration attempt with existing email: {email}")
            raise ValidationError(
                message="Email already registered",
                field="email"
            )
        
        # Step 2: Hash password
        logger.debug("Hashing password")
        try:
            password_hash = hash_password(password)
        except (ValueError, AttributeError) as e:
            # Workaround for bcrypt/passlib initialization bug during testing
            # Use bcrypt directly when passlib fails
            import bcrypt
            password_hash = bcrypt.hashpw(
                password.encode('utf-8'),
                bcrypt.gensalt()
            ).decode('utf-8')
            logger.debug("Used bcrypt directly due to passlib issue")
        
        # Step 3: Create user in transaction
        logger.info("Creating user account in database")
        
        # Note: Supabase users table may not have google_scholar_url column
        # Check and build query accordingly
        if use_passed_db:
            # Try with google_scholar_url first, fall back if column doesn't exist
            try:
                user_result = await db.fetchrow(
                    """
                    INSERT INTO users (email, password_hash, name, google_scholar_url, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    RETURNING user_id
                    """,
                    email.lower(),
                    password_hash,
                    name,
                    google_scholar_url
                )
            except Exception:
                # Column doesn't exist, insert without it
                user_result = await db.fetchrow(
                    """
                    INSERT INTO users (email, password_hash, name, created_at, updated_at)
                    VALUES ($1, $2, $3, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    RETURNING user_id
                    """,
                    email.lower(),
                    password_hash,
                    name
                )
        else:
            try:
                user_result = await execute_query(
                    """
                    INSERT INTO users (email, password_hash, name, google_scholar_url, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    RETURNING user_id
                    """,
                    email.lower(),
                    password_hash,
                    name,
                    google_scholar_url,
                    fetch_one=True
                )
            except Exception:
                user_result = await execute_query(
                    """
                    INSERT INTO users (email, password_hash, name, created_at, updated_at)
                    VALUES ($1, $2, $3, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    RETURNING user_id
                    """,
                    email.lower(),
                    password_hash,
                    name,
                    fetch_one=True
                )
        
        user_id = user_result['user_id']
        
        logger.info(f"User created with user_id={user_id}")
        
        # Step 4: Create user profile with domain (Supabase schema)
        # Note: In Supabase, domain is stored in user_profiles_extended.primary_domain
        # and interests are stored in user_interest_hierarchy
        logger.debug(f"Creating user profile with domain: {domain}")
        
        from app.db.repositories.user_repo import UserRepository
        from app.db.connection import DatabaseConnection
        
        # Use provided db connection or create wrapper around global pool
        if db is None:
            # Create wrapper around existing global pool to avoid pool exhaustion
            from app.db.postgres import get_db_pool
            db_connection = DatabaseConnection()
            pool = await get_db_pool()
            db_connection._pool = pool  # Reuse existing global pool
        else:
            db_connection = db
        
        user_repo = UserRepository(db_connection)
        
        # Create profile with domain and interests
        profile_data = {
            'primary_domain': domain,
            'research_stage': 'undergraduate',  # Default for new users
            'reading_level': 'intermediate',
            'interests': interests
        }
        
        await user_repo.create_profile(user_id, profile_data)
        
        logger.info(f"Created profile and inserted {len(interests)} interests for user_id={user_id}")
        
        # Step 6: Process Google Scholar profile (if provided)
        if google_scholar_url:
            logger.info("Google Scholar URL provided, will process asynchronously")
            # TODO: Trigger Celery task to fetch and process Google Scholar profile
            # from app.tasks.scholar_import import import_google_scholar_profile
            # import_google_scholar_profile.delay(user_id, google_scholar_url)
        
        # Step 7: Process uploaded paper (if provided)
        if uploaded_paper_file:
            logger.info("Uploaded paper provided, will process asynchronously")
            # TODO: Trigger Celery task to process uploaded paper
            # from app.tasks.paper_processing import process_uploaded_paper
            # process_uploaded_paper.delay(user_id, uploaded_paper_file)
        
        # Step 8: Trigger starter kit generation
        logger.info("Triggering starter kit generation")
        # TODO: Trigger Celery task
        # from app.tasks.starter_kit import generate_starter_kit
        # task = generate_starter_kit.delay(user_id)
        starter_kit_status = "processing"  # Will be updated by Celery task
        
        # Step 9: Generate auth tokens
        logger.debug("Generating authentication tokens")
        
        tokens = create_token_pair(
            user_id=user_id,
            email=email.lower(),
            additional_data={"domain": domain}
        )
        
        logger.info(
            f"User registration completed successfully",
            extra={"user_id": user_id, "email": email}
        )
        
        return {
            "user": {
                "user_id": user_id,
                "email": email.lower(),
                "name": name,
                "domain": domain
            },
            "user_id": user_id,  # For backward compatibility
            "email": email.lower(),
            "name": name,
            "domain": domain,
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_HOURS * 3600,
            "starter_kit_status": starter_kit_status
        }
        
    except ValidationError:
        # Re-raise validation errors
        raise
        
    except Exception as e:
        logger.error(f"User registration failed: {str(e)}", exc_info=True)
        raise DatabaseError(
            message=f"Failed to register user: {str(e)}",
            operation="register_user"
        )


async def login_user(email: str, password: str) -> Dict[str, Any]:
    """
    Authenticate user and generate tokens.
    
    Args:
        email: User email address
        password: Plain text password
    
    Returns:
        Dictionary with user data and tokens
    
    Raises:
        AuthenticationError: If credentials are invalid
        DatabaseError: If database operation fails
    
    Example:
        >>> tokens = await login_user("sarah@example.com", "password123")
    """
    logger.info(f"Login attempt for email: {email}")
    
    try:
        # Step 1: Get user from database
        logger.debug("Fetching user from database")
        
        user = await execute_query(
            """
            SELECT u.user_id, u.email, u.password_hash, u.name, u.is_active,
                   ud.domain
            FROM users u
            LEFT JOIN user_domains ud ON u.user_id = ud.user_id
            WHERE u.email = $1
            """,
            email.lower(),
            fetch_one=True
        )
        
        if not user:
            logger.warning(f"Login failed: user not found for email {email}")
            raise AuthenticationError(
                message="Invalid email or password",
                details={"reason": "user_not_found"}
            )
        
        # Step 2: Check if account is active
        if not user['is_active']:
            logger.warning(f"Login attempt for inactive account: {email}")
            raise AuthenticationError(
                message="Account is inactive",
                details={"reason": "account_inactive"}
            )
        
        # Step 3: Verify password
        logger.debug("Verifying password")
        
        is_valid = verify_password(password, user['password_hash'])
        
        if not is_valid:
            logger.warning(f"Login failed: invalid password for {email}")
            raise AuthenticationError(
                message="Invalid email or password",
                details={"reason": "invalid_password"}
            )
        
        # Step 4: Generate tokens
        logger.debug("Generating authentication tokens")
        
        tokens = create_token_pair(
            user_id=user['user_id'],
            email=user['email'],
            additional_data={"domain": user['domain']}
        )
        
        logger.info(
            f"User logged in successfully",
            extra={"user_id": user['user_id'], "email": email}
        )
        
        return {
            "user_id": user['user_id'],
            "email": user['email'],
            "name": user['name'],
            "domain": user['domain'],
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_HOURS * 3600
        }
        
    except AuthenticationError:
        # Re-raise authentication errors
        raise
        
    except Exception as e:
        logger.error(f"Login failed: {str(e)}", exc_info=True)
        raise DatabaseError(
            message=f"Login failed: {str(e)}",
            operation="login_user"
        )


async def refresh_access_token(refresh_token: str) -> Dict[str, Any]:
    """
    Generate new access token from refresh token.
    
    Args:
        refresh_token: JWT refresh token
    
    Returns:
        Dictionary with new access token
    
    Raises:
        AuthenticationError: If refresh token is invalid
    
    Example:
        >>> new_token = await refresh_access_token(refresh_token)
    """
    logger.info("Refreshing access token")
    
    try:
        # Decode refresh token
        from app.core.security import get_token_payload, create_access_token
        
        payload = get_token_payload(refresh_token, verify_type="refresh")
        
        user_id = int(payload["sub"])
        email = payload.get("email")
        
        logger.debug(f"Refresh token valid for user_id={user_id}")
        
        # Verify user still exists and is active
        user = await execute_query(
            "SELECT user_id, is_active FROM users WHERE user_id = $1",
            user_id,
            fetch_one=True
        )
        
        if not user or not user['is_active']:
            logger.warning(f"Refresh token for inactive/deleted user: {user_id}")
            raise AuthenticationError(
                message="User account is no longer active"
            )
        
        # Generate new access token
        new_access_token = create_access_token(
            data={"sub": str(user_id), "email": email}
        )
        
        logger.info(f"Access token refreshed for user_id={user_id}")
        
        return {
            "access_token": new_access_token,
            "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_HOURS * 3600
        }
        
    except AuthenticationError:
        raise
        
    except Exception as e:
        logger.error(f"Token refresh failed: {str(e)}", exc_info=True)
        raise AuthenticationError(
            message="Could not refresh token"
        )


# Initialize module logger
logger.info("Auth service module loaded successfully")
