"""
User management API endpoints.
Handles user registration, profile management, and authentication.
UPDATED: Added JWT authentication to protected endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, validator
import bcrypt
from datetime import datetime, timedelta
from jose import JWTError, jwt

from app.config import settings
from app.utils.logger import get_logger
from app.db.connection import get_db, DatabaseConnection
from app.db.repositories.user_repo import UserRepository
from app.services.user_embedding_service import UserEmbeddingService
from app.api.v1.auth import get_current_user  # NEW
import re
logger = get_logger(__name__)

router = APIRouter()

# Password hashing
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# Pydantic models
class UserCreate(BaseModel):
    """User registration request."""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=72)
    full_name: Optional[str] = Field(None, max_length=100)
    
    @validator('password')
    def validate_password(cls, v):
        """Ensure password has complexity."""
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain digit")
        return v


class UserProfileCreate(BaseModel):
    """User profile creation request."""
    primary_domain: str = Field(..., description="Primary research domain")
    reading_level: str = Field(..., description="Reading level")
    interests: list[str] = Field(
        ..., 
        min_items=3, 
        max_items=10, 
        description="Research interests"
    )
    
    research_stage: Optional[str] = None
    sub_domains: Optional[list[str]] = Field(None, max_items=5)
    research_methods: Optional[list[str]] = None
    research_goals: Optional[list[str]] = Field(None, max_items=5)
    time_availability: Optional[str] = None
    years_experience: Optional[int] = Field(None, ge=0, le=50)
    h_index: Optional[int] = Field(None, ge=0)
    
    prefers_recent_papers: bool = True
    prefers_high_impact: bool = False
    prefers_open_access: bool = True
    
    preferred_venues: Optional[list[str]] = None
    institution: Optional[str] = Field(None, max_length=200)
    department: Optional[str] = Field(None, max_length=200)
    looking_for_collaborators: bool = False
    google_scholar_url: Optional[str] = Field(None, max_length=500)
    semantic_scholar_author_id: Optional[str] = Field(None, max_length=100)
    
    @validator('research_stage')
    def validate_research_stage(cls, v):
        if v and v not in settings.ALLOWED_RESEARCH_STAGES:
            raise ValueError(
                f"Research stage must be one of {settings.ALLOWED_RESEARCH_STAGES}"
            )
        return v
    
    @validator('primary_domain')
    def validate_domain(cls, v):
        if v not in settings.ALLOWED_DOMAINS:
            raise ValueError(
                f"Domain must be one of {settings.ALLOWED_DOMAINS}"
            )
        return v
    
    @validator('reading_level')
    def validate_reading_level(cls, v):
        if v not in settings.ALLOWED_READING_LEVELS:
            raise ValueError(
                f"Reading level must be one of {settings.ALLOWED_READING_LEVELS}"
            )
        return v
    
    @validator('time_availability')
    def validate_time_availability(cls, v):
        if v and v not in settings.ALLOWED_TIME_AVAILABILITY:
            raise ValueError(
                f"Time availability must be one of {settings.ALLOWED_TIME_AVAILABILITY}"
            )
        return v


class UserProfileUpdate(BaseModel):
    """User profile update request."""
    research_stage: Optional[str] = None
    primary_domain: Optional[str] = None
    interests: Optional[list[str]] = None
    sub_domains: Optional[list[str]] = None
    research_methods: Optional[list[str]] = None
    research_goals: Optional[list[str]] = None
    reading_level: Optional[str] = None
    time_availability: Optional[str] = None
    years_experience: Optional[int] = None
    h_index: Optional[int] = None
    prefers_recent_papers: Optional[bool] = None
    prefers_high_impact: Optional[bool] = None
    prefers_open_access: Optional[bool] = None
    preferred_venues: Optional[list[str]] = None
    institution: Optional[str] = None
    department: Optional[str] = None
    looking_for_collaborators: Optional[bool] = None
    google_scholar_url: Optional[str] = None
    semantic_scholar_author_id: Optional[str] = None


class UserLogin(BaseModel):
    """User login request."""
    email: EmailStr
    password: str


class Token(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """JWT token payload."""
    user_id: Optional[int] = None
    email: Optional[str] = None


def get_user_repo(db: DatabaseConnection = Depends(get_db)) -> UserRepository:
    """Dependency to get user repository."""
    return UserRepository(db)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash using bcrypt directly."""
    return bcrypt.checkpw(
        plain_password.encode('utf-8'),
        hashed_password.encode('utf-8')
    )


def get_password_hash(password: str) -> str:
    """Hash password using bcrypt directly."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )
    
    return encoded_jwt


@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    summary="Register new user",
    description="Create a new user account"
)
async def register_user(
    user_data: UserCreate,
    user_repo: UserRepository = Depends(get_user_repo)
):
    """
    Register a new user.
    
    Args:
        user_data: User registration data
        user_repo: User repository
        
    Returns:
        Created user info and token
    """
    logger.info(
        "User registration request",
        email=user_data.email
    )
    
    try:
        # --- INPUT VALIDATION START ---
        
        # 1. Email Validation
        # Basic regex pattern for email validation
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_regex, user_data.email):
             raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid email address format"
            )

        # 2. Password Validation
        password = user_data.password
        
        # Check length (at least 8 characters)
        if len(password) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters long"
            )

        # Check for at least 1 uppercase letter
        if not any(char.isupper() for char in password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must contain at least one uppercase letter"
            )
        # --- INPUT VALIDATION END ---

        # Check if user exists
        existing_user = await user_repo.find_by_email(user_data.email)
        
        if existing_user:
            logger.warning(
                "Registration attempt for existing email",
                email=user_data.email
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Hash password
        hashed_password = get_password_hash(user_data.password)
        
        # Create user
        user = await user_repo.create({
            'email': user_data.email,
            'password_hash': hashed_password,
            'name': user_data.full_name or user_data.email.split('@')[0],
            'is_active': True
        })
        
        # Initialize recommendation state
        await user_repo.initialize_recommendation_state(
            user['user_id'],
            initial_stage='cold_start'
        )
        
        # Create access token
        access_token = create_access_token(
            data={
                "user_id": user['user_id'],
                "email": user['email']
            }
        )
        
        logger.info(
            "User registered successfully",
            user_id=user['user_id'],
            email=user_data.email
        )
        
        return {
            "user_id": user['user_id'],
            "email": user['email'],
            "access_token": access_token,
            "token_type": "bearer",
            "message": "User registered successfully. Please create your profile next."
        }
        
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(
            "User registration failed",
            email=user_data.email,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post(
    "/login",
    summary="User login",
    description="Authenticate user and get access token"
)
async def login(
    login_data: UserLogin,
    user_repo: UserRepository = Depends(get_user_repo)
):
    """
    Authenticate user.
    
    Args:
        login_data: Login credentials
        user_repo: User repository
        
    Returns:
        Access token
        user_id
    """
    logger.info("Login attempt", email=login_data.email)
    
    try:
        # Find user
        user = await user_repo.find_by_email(login_data.email)
        
        if not user:
            logger.warning("Login failed: user not found", email=login_data.email)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        # Verify password
        if not verify_password(login_data.password, user['password_hash']):
            logger.warning("Login failed: invalid password", email=login_data.email)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        # Check if active
        if not user['is_active']:
            logger.warning("Login failed: inactive account", email=login_data.email)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is inactive"
            )
        
        # Create access token
        access_token = create_access_token(
            data={
                "user_id": user['user_id'],
                "email": user['email']
            }
        )
        
        logger.info(
            "Login successful",
            user_id=user['user_id'],
            email=login_data.email
        )
        
        return {
            "access_token": access_token,
            "user_id": user['user_id']
        }
        
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(
            "Login failed",
            email=login_data.email,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post(
    "/logout",
    status_code=status.HTTP_200_OK,
    summary="User logout",
    description="Logout user (client-side token invalidation)"
)
async def logout(
    current_user: dict = Depends(get_current_user)
):
    """
    Logout user.
    
    Note: With JWT, logout is primarily client-side (delete token).
    This endpoint validates the token and logs the logout event.
    For true server-side invalidation, implement token blacklisting.
    
    Args:
        current_user: Authenticated user from token
        
    Returns:
        Logout confirmation
    """
    logger.info(
        "User logout",
        user_id=current_user['user_id'],
        email=current_user['email']
    )
    
    return {
        "message": "Logged out successfully",
        "user_id": current_user['user_id']
    }


@router.post(
    "/{user_id}/profile",
    status_code=status.HTTP_201_CREATED,
    summary="Create user profile",
    description="Create extended user profile. **Requires authentication.**"
)
async def create_profile(
    user_id: int,
    profile_data: UserProfileCreate,
    current_user: dict = Depends(get_current_user),  # AUTHENTICATION REQUIRED
    user_repo: UserRepository = Depends(get_user_repo),
    db: DatabaseConnection = Depends(get_db)
):
    """Create user profile."""
    
    # Verify user is creating their own profile
    if current_user['user_id'] != user_id:
        logger.warning(
            "Unauthorized profile creation attempt",
            requesting_user=current_user['user_id'],
            target_user=user_id
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only create your own profile"
        )
    
    logger.info(
        "Profile creation request",
        user_id=user_id,
        research_stage=profile_data.research_stage,
        domain=profile_data.primary_domain,
        interest_count=len(profile_data.interests)
    )
    
    try:
        existing_profile = await user_repo.get_profile(user_id)
        
        if existing_profile:
            logger.warning("Profile already exists", user_id=user_id)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Profile already exists. Use PUT to update."
            )
        
        profile = await user_repo.create_profile(
            user_id,
            profile_data.dict()
        )
        
        logger.info(
            "Profile created successfully",
            user_id=user_id,
            completeness=profile.get('profile_completeness'),
            interest_count=len(profile.get('interests', {}).get('all', []))
        )
        
        try:
            user_embedding_service = UserEmbeddingService(db)
            embeddings = await user_embedding_service.get_or_generate_user_embeddings(user_id)
            
            logger.info(
                "User embeddings generated",
                user_id=user_id,
                minilm_dim=len(embeddings['minilm']),
                specter_dim=len(embeddings['specter'])
            )
        except Exception as emb_error:
            logger.error(
                "Embedding generation failed, but profile created",
                user_id=user_id,
                error=str(emb_error),
                exc_info=True
            )

        return {
            "user_id": user_id,
            "profile": profile,
            "message": "Profile created successfully. Interests saved to hierarchy table."
        }
        
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(
            "Profile creation failed",
            user_id=user_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Profile creation failed: {str(e)}" if settings.DEBUG else "Profile creation failed"
        )


@router.get(
    "/{user_id}/profile",
    summary="Get user profile",
    description="Retrieve user's extended profile. **Requires authentication.**"
)
async def get_profile(
    user_id: int,
    current_user: dict = Depends(get_current_user),  # AUTHENTICATION REQUIRED
    user_repo: UserRepository = Depends(get_user_repo)
):
    """Get user profile."""
    
    # Verify user is accessing their own profile
    if current_user['user_id'] != user_id:
        logger.warning(
            "Unauthorized profile access attempt",
            requesting_user=current_user['user_id'],
            target_user=user_id
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view your own profile"
        )
    
    logger.debug("Profile retrieval request", user_id=user_id)
    
    try:
        profile = await user_repo.get_profile(user_id)
        
        if not profile:
            logger.warning("Profile not found", user_id=user_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Profile not found"
            )
        
        logger.debug(
            "Profile retrieved",
            user_id=user_id,
            completeness=profile.get('profile_completeness')
        )
        
        return {
            "user_id": user_id,
            "profile": profile
        }
        
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(
            "Profile retrieval failed",
            user_id=user_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Profile retrieval failed"
        )


@router.put(
    "/{user_id}/profile",
    summary="Update user profile",
    description="Update user's extended profile. **Requires authentication.**"
)
async def update_profile(
    user_id: int,
    updates: UserProfileUpdate,
    current_user: dict = Depends(get_current_user),  # AUTHENTICATION REQUIRED
    user_repo: UserRepository = Depends(get_user_repo)
):
    """Update user profile."""
    
    # Verify user is updating their own profile
    if current_user['user_id'] != user_id:
        logger.warning(
            "Unauthorized profile update attempt",
            requesting_user=current_user['user_id'],
            target_user=user_id
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only update your own profile"
        )
    
    logger.info(
        "Profile update request",
        user_id=user_id,
        fields=list(updates.dict(exclude_unset=True).keys())
    )
    
    try:
        update_data = updates.dict(exclude_unset=True)
        
        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields to update"
            )
        
        profile = await user_repo.update_profile(user_id, update_data)
        
        if not profile:
            logger.warning("Profile not found for update", user_id=user_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Profile not found"
            )
        
        logger.info(
            "Profile updated successfully",
            user_id=user_id,
            completeness=profile.get('profile_completeness')
        )
        
        return {
            "user_id": user_id,
            "profile": profile,
            "message": "Profile updated successfully"
        }
        
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(
            "Profile update failed",
            user_id=user_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Profile update failed"
        )


@router.get(
    "/{user_id}/interests",
    summary="Get user interests",
    description="Get user's interest hierarchy. **Requires authentication.**"
)
async def get_user_interests(
    user_id: int,
    current_user: dict = Depends(get_current_user),  # AUTHENTICATION REQUIRED
    user_repo: UserRepository = Depends(get_user_repo)
):
    """Get user's interest hierarchy."""
    
    # Verify user is accessing their own interests
    if current_user['user_id'] != user_id:
        logger.warning(
            "Unauthorized interests access attempt",
            requesting_user=current_user['user_id'],
            target_user=user_id
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view your own interests"
        )
    
    logger.debug("Interests retrieval request", user_id=user_id)
    
    try:
        interests = await user_repo.get_user_interests(user_id)
        
        by_level = {
            'level_1': [],
            'level_2': [],
            'level_3': []
        }
        
        for interest in interests:
            level_key = f"level_{interest['interest_level']}"
            by_level[level_key].append({
                'term': interest['interest_term'],
                'confidence': float(interest['confidence_score']),
                'source': interest['source']
            })
        
        logger.debug(
            "Interests retrieved",
            user_id=user_id,
            total=len(interests)
        )
        
        return {
            "user_id": user_id,
            "interests": by_level,
            "total_count": len(interests)
        }
        
    except Exception as e:
        logger.error(
            "Interests retrieval failed",
            user_id=user_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Interests retrieval failed"
        )


@router.get(
    "/{user_id}/state",
    summary="Get user recommendation state",
    description="Get user's current recommendation stage. **Requires authentication.**"
)
async def get_user_state(
    user_id: int,
    current_user: dict = Depends(get_current_user),  # AUTHENTICATION REQUIRED
    user_repo: UserRepository = Depends(get_user_repo)
):
    """Get user recommendation state."""
    
    # Verify user is accessing their own state
    if current_user['user_id'] != user_id:
        logger.warning(
            "Unauthorized state access attempt",
            requesting_user=current_user['user_id'],
            target_user=user_id
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view your own state"
        )
    
    logger.debug("State retrieval request", user_id=user_id)
    
    try:
        state = await user_repo.get_recommendation_state(user_id)
        
        if not state:
            logger.warning("State not found", user_id=user_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User state not found"
            )
        
        logger.debug(
            "State retrieved",
            user_id=user_id,
            stage=state['recommendation_stage']
        )
        
        return {
            "user_id": user_id,
            "state": dict(state)
        }
        
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(
            "State retrieval failed",
            user_id=user_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="State retrieval failed"
        )