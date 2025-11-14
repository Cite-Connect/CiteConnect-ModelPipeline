# app/api/v1/__init__.py

"""
API Version 1 Package

This package contains all v1 API endpoint modules.
"""

# Import routers for easy access
from app.api.v1 import auth, users

__all__ = ["auth", "users"]
