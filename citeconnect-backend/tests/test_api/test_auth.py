"""
Tests for authentication API endpoints and dependencies.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import HTTPException, status
from jose import jwt
from datetime import datetime, timedelta

from app.api.v1.auth import get_current_user, get_current_user_optional
from app.config import settings


class TestGetCurrentUser:
    """Tests for get_current_user dependency."""
    
    @pytest.mark.asyncio
    async def test_get_current_user_valid_token(self, mock_user_repo, sample_user):
        """Test successful authentication with valid token."""
        # Create valid JWT token
        token_data = {
            "user_id": 1,
            "email": "test@example.com",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        token = jwt.encode(token_data, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        
        # Mock credentials
        credentials = MagicMock()
        credentials.credentials = token
        
        # Mock user repository
        with patch('app.api.v1.auth.UserRepository', return_value=mock_user_repo):
            with patch('app.api.v1.auth.get_db', return_value=MagicMock()):
                user = await get_current_user(credentials, MagicMock())
                
                assert user['user_id'] == 1
                assert user['email'] == 'test@example.com'
    
    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self):
        """Test authentication fails with invalid token."""
        credentials = MagicMock()
        credentials.credentials = "invalid_token"
        
        with patch('app.api.v1.auth.get_db', return_value=MagicMock()):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(credentials, MagicMock())
            
            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    
    @pytest.mark.asyncio
    async def test_get_current_user_missing_user_id(self):
        """Test authentication fails when token missing user_id."""
        token_data = {
            "email": "test@example.com",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        token = jwt.encode(token_data, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        
        credentials = MagicMock()
        credentials.credentials = token
        
        with patch('app.api.v1.auth.get_db', return_value=MagicMock()):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(credentials, MagicMock())
            
            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    
    @pytest.mark.asyncio
    async def test_get_current_user_inactive_account(self, mock_user_repo):
        """Test authentication fails for inactive user."""
        # Create user with inactive status
        mock_user_repo.find_by_id = AsyncMock(return_value={
            'user_id': 1,
            'email': 'test@example.com',
            'is_active': False
        })
        
        token_data = {
            "user_id": 1,
            "email": "test@example.com",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        token = jwt.encode(token_data, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        
        credentials = MagicMock()
        credentials.credentials = token
        
        with patch('app.api.v1.auth.UserRepository', return_value=mock_user_repo):
            with patch('app.api.v1.auth.get_db', return_value=MagicMock()):
                with pytest.raises(HTTPException) as exc_info:
                    await get_current_user(credentials, MagicMock())
                
                assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
    
    @pytest.mark.asyncio
    async def test_get_current_user_optional_no_token(self):
        """Test optional auth returns None when no token provided."""
        with patch('app.api.v1.auth.get_db', return_value=MagicMock()):
            user = await get_current_user_optional(None, MagicMock())
            assert user is None
    
    @pytest.mark.asyncio
    async def test_get_current_user_optional_valid_token(self, mock_user_repo, sample_user):
        """Test optional auth returns user when valid token provided."""
        token_data = {
            "user_id": 1,
            "email": "test@example.com",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        token = jwt.encode(token_data, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        
        credentials = MagicMock()
        credentials.credentials = token
        
        with patch('app.api.v1.auth.UserRepository', return_value=mock_user_repo):
            with patch('app.api.v1.auth.get_db', return_value=MagicMock()):
                user = await get_current_user_optional(credentials, MagicMock())
                
                assert user is not None
                assert user['user_id'] == 1

