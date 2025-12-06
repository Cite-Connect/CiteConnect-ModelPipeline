"""
Base repository pattern for database operations.
Provides common CRUD operations with logging.
"""
from typing import TypeVar, Generic, Optional, List, Any
from abc import ABC, abstractmethod
import asyncpg
from app.db.connection import DatabaseConnection
from app.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class BaseRepository(ABC, Generic[T]):
    """
    Abstract base repository for database operations.
    Provides common patterns for CRUD operations.
    """
    
    def __init__(self, db: DatabaseConnection):
        """
        Initialize repository with database connection.
        
        Args:
            db: Database connection instance
        """
        self.db = db
        logger.debug(
            "Repository initialized",
            repository=self.__class__.__name__
        )
    
    @property
    @abstractmethod
    def table_name(self) -> str:
        """Table name for this repository."""
        pass
    
    async def find_by_id(
        self,
        id_value: Any,
        id_column: str = "id"
    ) -> Optional[asyncpg.Record]:
        """
        Find record by ID.
        
        Args:
            id_value: ID value to search for
            id_column: Name of ID column
            
        Returns:
            Optional[Record]: Found record or None
        """
        logger.debug(
            "Finding record by ID",
            table=self.table_name,
            id_column=id_column,
            id_value=id_value
        )
        
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE {id_column} = $1
        """
        
        try:
            result = await self.db.fetchrow(query, id_value)
            logger.debug(
                "Find by ID complete",
                found=result is not None,
                table=self.table_name
            )
            return result
        except Exception as e:
            logger.error(
                "Find by ID failed",
                table=self.table_name,
                id_value=id_value,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def find_all(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[asyncpg.Record]:
        """
        Find all records with pagination.
        
        Args:
            limit: Maximum number of records
            offset: Number of records to skip
            
        Returns:
            List[Record]: Found records
        """
        logger.debug(
            "Finding all records",
            table=self.table_name,
            limit=limit,
            offset=offset
        )
        
        query = f"""
            SELECT * FROM {self.table_name}
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
        """
        
        try:
            results = await self.db.fetch(
                query,
                limit or 1000,  # Default limit
                offset
            )
            logger.debug(
                "Find all complete",
                table=self.table_name,
                count=len(results)
            )
            return results
        except Exception as e:
            logger.error(
                "Find all failed",
                table=self.table_name,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def create(self, data: dict[str, Any]) -> asyncpg.Record:
        """
        Create new record.
        
        Args:
            data: Field values for new record
            
        Returns:
            Record: Created record with generated fields
        """
        logger.debug(
            "Creating record",
            table=self.table_name,
            fields=list(data.keys())
        )
        
        columns = ", ".join(data.keys())
        placeholders = ", ".join(f"${i+1}" for i in range(len(data)))
        values = list(data.values())
        
        query = f"""
            INSERT INTO {self.table_name} ({columns})
            VALUES ({placeholders})
            RETURNING *
        """
        
        try:
            result = await self.db.fetchrow(query, *values)
            logger.info(
                "Record created",
                table=self.table_name,
                id=result.get('id') if result else None
            )
            return result
        except Exception as e:
            logger.error(
                "Create failed",
                table=self.table_name,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def update(
        self,
        id_value: Any,
        data: dict[str, Any],
        id_column: str = "id"
    ) -> Optional[asyncpg.Record]:
        """
        Update existing record.
        
        Args:
            id_value: ID of record to update
            data: Fields to update
            id_column: Name of ID column
            
        Returns:
            Optional[Record]: Updated record or None
        """
        logger.debug(
            "Updating record",
            table=self.table_name,
            id_column=id_column,
            id_value=id_value,
            fields=list(data.keys())
        )
        
        set_clause = ", ".join(
            f"{key} = ${i+2}" for i, key in enumerate(data.keys())
        )
        values = [id_value] + list(data.values())
        
        query = f"""
            UPDATE {self.table_name}
            SET {set_clause}, updated_at = NOW()
            WHERE {id_column} = $1
            RETURNING *
        """
        
        try:
            result = await self.db.fetchrow(query, *values)
            logger.info(
                "Record updated",
                table=self.table_name,
                id_value=id_value,
                found=result is not None
            )
            return result
        except Exception as e:
            logger.error(
                "Update failed",
                table=self.table_name,
                id_value=id_value,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def delete(
        self,
        id_value: Any,
        id_column: str = "id"
    ) -> bool:
        """
        Delete record by ID.
        
        Args:
            id_value: ID of record to delete
            id_column: Name of ID column
            
        Returns:
            bool: True if record was deleted
        """
        logger.debug(
            "Deleting record",
            table=self.table_name,
            id_column=id_column,
            id_value=id_value
        )
        
        query = f"""
            DELETE FROM {self.table_name}
            WHERE {id_column} = $1
        """
        
        try:
            result = await self.db.execute(query, id_value)
            deleted = result == "DELETE 1"
            logger.info(
                "Delete complete",
                table=self.table_name,
                id_value=id_value,
                deleted=deleted
            )
            return deleted
        except Exception as e:
            logger.error(
                "Delete failed",
                table=self.table_name,
                id_value=id_value,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def exists(
        self,
        id_value: Any,
        id_column: str = "id"
    ) -> bool:
        """
        Check if record exists.
        
        Args:
            id_value: ID to check
            id_column: Name of ID column
            
        Returns:
            bool: True if record exists
        """
        logger.debug(
            "Checking record existence",
            table=self.table_name,
            id_column=id_column,
            id_value=id_value
        )
        
        query = f"""
            SELECT EXISTS(
                SELECT 1 FROM {self.table_name}
                WHERE {id_column} = $1
            )
        """
        
        try:
            result = await self.db.fetchval(query, id_value)
            logger.debug(
                "Existence check complete",
                table=self.table_name,
                exists=result
            )
            return result
        except Exception as e:
            logger.error(
                "Existence check failed",
                table=self.table_name,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def count(self, where_clause: Optional[str] = None) -> int:
        """
        Count records matching criteria.
        
        Args:
            where_clause: Optional WHERE clause
            
        Returns:
            int: Number of matching records
        """
        logger.debug(
            "Counting records",
            table=self.table_name,
            where_clause=where_clause
        )
        
        query = f"SELECT COUNT(*) FROM {self.table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"
        
        try:
            result = await self.db.fetchval(query)
            logger.debug(
                "Count complete",
                table=self.table_name,
                count=result
            )
            return result
        except Exception as e:
            logger.error(
                "Count failed",
                table=self.table_name,
                error=str(e),
                exc_info=True
            )
            raise