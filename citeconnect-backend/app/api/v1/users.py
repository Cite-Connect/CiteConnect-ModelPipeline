"""
User management API endpoints.
Handles user registration, profile management, and authentication.
UPDATED: Matches Supabase schema with interests in separate table.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, validator
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import JWTError, jwt

from app.config import settings
from app.utils.logger import get_logger
from app.db.connection import get_db, DatabaseConnection
from app.db.repositories.user_repo import UserRepository

logger = get_logger(__name__)

router = APIRouter()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Allowed values from Supabase schema
ALLOWED_DOMAINS = ['healthcare', 'fintech', 'quantum_computing']
ALLOWED_RESEARCH_STAGES = [
    'undergraduate', 'masters', 'phd', 'postdoc', 
    'professor', 'industry', 'independent'
]
ALLOWED_READING_LEVELS = ['introductory', 'intermediate', 'advanced', 'expert']
ALLOWED_TIME_AVAILABILITY = ['casual_reader', 'part_time_researcher', 'full_time_researcher']


# Pydantic models
class UserCreate(BaseModel):
    """User registration request."""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=72)  # Max 72 for bcrypt
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
    """
    User profile creation request.
    NOTE: Interests are stored in separate user_interest_hierarchy table.
    """
    # Required fields
    primary_domain: str = Field(..., description="Primary research domain")
    reading_level: str = Field(..., description="Reading level")
    interests: list[str] = Field(
        ..., 
        min_items=3, 
        max_items=10, 
        description="Research interests (stored in user_interest_hierarchy table)"
    )
    
    # Optional profile fields
    research_stage: Optional[str] = None
    sub_domains: Optional[list[str]] = Field(None, max_items=5)
    research_methods: Optional[list[str]] = None
    research_goals: Optional[list[str]] = Field(None, max_items=5)
    time_availability: Optional[str] = None
    years_experience: Optional[int] = Field(None, ge=0, le=50)
    h_index: Optional[int] = Field(None, ge=0)
    
    # Preference flags
    prefers_recent_papers: bool = True
    prefers_high_impact: bool = False
    prefers_open_access: bool = True
    
    # Optional metadata
    preferred_venues: Optional[list[str]] = None
    institution: Optional[str] = Field(None, max_length=200)
    department: Optional[str] = Field(None, max_length=200)
    looking_for_collaborators: bool = False
    google_scholar_url: Optional[str] = Field(None, max_length=500)
    semantic_scholar_author_id: Optional[str] = Field(None, max_length=100)
    
    @validator('research_stage')
    def validate_research_stage(cls, v):
        if v and v not in ALLOWED_RESEARCH_STAGES:
            raise ValueError(
                f"Research stage must be one of {ALLOWED_RESEARCH_STAGES}"
            )
        return v
    
    @validator('primary_domain')
    def validate_domain(cls, v):
        if v not in ALLOWED_DOMAINS:
            raise ValueError(
                f"Domain must be one of {ALLOWED_DOMAINS}. "
                f"Currently supported: healthcare, fintech, quantum_computing"
            )
        return v
    
    @validator('reading_level')
    def validate_reading_level(cls, v):
        if v not in ALLOWED_READING_LEVELS:
            raise ValueError(
                f"Reading level must be one of {ALLOWED_READING_LEVELS}"
            )
        return v
    
    @validator('time_availability')
    def validate_time_availability(cls, v):
        if v and v not in ALLOWED_TIME_AVAILABILITY:
            raise ValueError(
                f"Time availability must be one of {ALLOWED_TIME_AVAILABILITY}"
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


class Token(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """JWT token payload."""
    user_id: Optional[int] = None
    email: Optional[str] = None


def get_user_repo(db: DatabaseConnection = Depends(get_db)) -> UserRepository:
    """
    Dependency to get user repository.
    
    Args:
        db: Database connection
        
    Returns:
        UserRepository: User repository instance
    """
    return UserRepository(db)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash password."""
    return pwd_context.hash(password)


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
    response_model=Token,
    summary="User login",
    description="Authenticate user and get access token"
)
async def login(
    email: EmailStr,
    password: str,
    user_repo: UserRepository = Depends(get_user_repo)
):
    """
    Authenticate user.
    
    Args:
        email: User email
        password: User password
        user_repo: User repository
        
    Returns:
        Access token
    """
    logger.info("Login attempt", email=email)
    
    try:
        # Find user
        user = await user_repo.find_by_email(email)
        
        if not user:
            logger.warning("Login failed: user not found", email=email)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        # Verify password
        if not verify_password(password, user['password_hash']):
            logger.warning("Login failed: invalid password", email=email)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        # Check if active
        if not user['is_active']:
            logger.warning("Login failed: inactive account", email=email)
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
            email=email
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer"
        }
        
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(
            "Login failed",
            email=email,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post(
    "/{user_id}/profile",
    status_code=status.HTTP_201_CREATED,
    summary="Create user profile",
    description="""
    Create extended user profile for personalized recommendations.
    
    **Note:** Interests are stored in a separate hierarchical table (user_interest_hierarchy).
    The API accepts interests as a simple array, but they're stored with:
    - interest_level: 1 (broad), 2 (specific), 3 (narrow)
    - confidence_score: 1.0 for explicit user input
    - source: 'explicit' for user-provided interests
    """
)
async def create_profile(
    user_id: int,
    profile_data: UserProfileCreate,
    user_repo: UserRepository = Depends(get_user_repo)
):
    """
    Create user profile.
    
    Args:
        user_id: User identifier
        profile_data: Profile data
        user_repo: User repository
        
    Returns:
        Created profile with interests included
    """
    logger.info(
        "Profile creation request",
        user_id=user_id,
        research_stage=profile_data.research_stage,
        domain=profile_data.primary_domain,
        interest_count=len(profile_data.interests)
    )
    
    try:
        # Check if profile exists
        existing_profile = await user_repo.get_profile(user_id)
        
        if existing_profile:
            logger.warning(
                "Profile already exists",
                user_id=user_id
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Profile already exists. Use PUT to update."
            )
        
        # Create profile (interests will be saved to hierarchy table automatically)
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
    description="Retrieve user's extended profile with interests from hierarchy table"
)
async def get_profile(
    user_id: int,
    user_repo: UserRepository = Depends(get_user_repo)
):
    """
    Get user profile.
    
    Args:
        user_id: User identifier
        user_repo: User repository
        
    Returns:
        User profile with interests structured by level
    """
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
    description="""
    Update user's extended profile.
    
    **Note:** Updating interests will replace all existing explicit interests.
    Inferred interests (if any) are preserved.
    """
)
async def update_profile(
    user_id: int,
    updates: UserProfileUpdate,
    user_repo: UserRepository = Depends(get_user_repo)
):
    """
    Update user profile.
    
    Args:
        user_id: User identifier
        updates: Profile updates
        user_repo: User repository
        
    Returns:
        Updated profile with interests
    """
    logger.info(
        "Profile update request",
        user_id=user_id,
        fields=list(updates.dict(exclude_unset=True).keys())
    )
    
    try:
        # Only update provided fields
        update_data = updates.dict(exclude_unset=True)
        
        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields to update"
            )
        
        # Update profile (handles interests separately)
        profile = await user_repo.update_profile(
            user_id,
            update_data
        )
        
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
    description="Get user's interest hierarchy with levels and confidence scores"
)
async def get_user_interests(
    user_id: int,
    user_repo: UserRepository = Depends(get_user_repo)
):
    """
    Get user's interest hierarchy.
    
    Args:
        user_id: User identifier
        user_repo: User repository
        
    Returns:
        Interest hierarchy
    """
    logger.debug("Interests retrieval request", user_id=user_id)
    
    try:
        interests = await user_repo.get_user_interests(user_id)
        
        # Group by level
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
    description="Get user's current recommendation stage and statistics"
)
async def get_user_state(
    user_id: int,
    user_repo: UserRepository = Depends(get_user_repo)
):
    """
    Get user recommendation state.
    
    Args:
        user_id: User identifier
        user_repo: User repository
        
    Returns:
        User state information
    """
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