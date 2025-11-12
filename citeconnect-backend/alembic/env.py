# alembic/env.py

"""
Alembic Environment Configuration

This script is run whenever the alembic migration tool is invoked.
It sets up the database connection and migration context.
"""

import logging
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool
from alembic import context

# Add parent directory to Python path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now we can import app settings
from app.core.config import get_settings

# Get settings
settings = get_settings()

# Alembic Config object
config = context.config

# Override sqlalchemy.url with our settings
# Use the regular DATABASE_URL (not async version)
database_url = settings.DATABASE_URL
if database_url.startswith("postgresql+asyncpg://"):
    # Convert async URL to sync URL for Alembic
    database_url = database_url.replace("postgresql+asyncpg://", "postgresql://")

config.set_main_option("sqlalchemy.url", database_url)

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Add your model's MetaData object here for 'autogenerate' support
# For now, we're writing migrations manually, so set to None
target_metadata = None

# Initialize logger
logger = logging.getLogger('alembic.env')


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.
    
    This configures the context with just a URL and not an Engine,
    though an Engine is acceptable here as well. By skipping the Engine
    creation we don't even need a DBAPI to be available.
    
    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    
    logger.info(f"Running offline migrations with URL: {url}")
    
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.
    
    In this scenario we need to create an Engine and associate
    a connection with the context.
    """
    # Get the database URL
    url = config.get_main_option("sqlalchemy.url")
    
    logger.info(f"Running online migrations with URL: {url}")
    
    # Create engine configuration
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = url
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


# Run migrations based on mode
if context.is_offline_mode():
    logger.info("Running migrations in offline mode")
    run_migrations_offline()
else:
    logger.info("Running migrations in online mode")
    run_migrations_online()
