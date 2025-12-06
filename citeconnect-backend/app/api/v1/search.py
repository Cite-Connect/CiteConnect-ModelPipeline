"""
Search API endpoints.
Provides standalone paper search functionality.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from app.utils.logger import get_logger
from app.db.connection import get_db, DatabaseConnection
from app.db.repositories.paper_repo import PaperRepository
from app.models.paper import PaperSearchRequest, PaperResponse
from app.services.bootstrap.embedding_service import get_embedding_service

logger = get_logger(__name__)

router = APIRouter()


def get_paper_repo(db: DatabaseConnection = Depends(get_db)) -> PaperRepository:
    """Dependency to get paper repository."""
    return PaperRepository(db)


@router.post(
    "/",
    response_model=List[PaperResponse],
    summary="Search papers",
    description="Hybrid search: keyword + semantic similarity"
)
async def search_papers(
    search_request: PaperSearchRequest,
    paper_repo: PaperRepository = Depends(get_paper_repo)
):
    """
    Hybrid paper search combining keyword and semantic methods.
    
    Args:
        search_request: Search parameters
        paper_repo: Paper repository
        
    Returns:
        List of matching papers ranked by relevance
    """
    logger.info(
        "Search request received",
        query=search_request.query,
        limit=search_request.limit,
        domain=search_request.domain_filter
    )
    
    try:
        # ────────────────────────────────────────────────
        # Phase 1: Keyword Search
        # ────────────────────────────────────────────────
        keyword_results = await paper_repo.search_by_text(
            search_text=search_request.query,
            limit=search_request.limit
        )
        
        for paper in keyword_results:
            paper = dict(paper)
            paper['match_type'] = 'keyword'
        
        # Apply filters
        if search_request.domain_filter:
            keyword_results = [
                p for p in keyword_results
                if dict(p).get('domain') == search_request.domain_filter
            ]
        
        if search_request.min_year:
            keyword_results = [
                p for p in keyword_results
                if dict(p).get('year', 0) >= search_request.min_year
            ]
        
        # ────────────────────────────────────────────────
        # Phase 2: Semantic Search (Optional Enhancement)
        # ────────────────────────────────────────────────
        # Uncomment to enable semantic search
        '''
        embedding_service = get_embedding_service()
        query_embedding = embedding_service.encode_text(
            text=search_request.query,
            model='minilm'
        )
        
        semantic_results = await paper_repo.semantic_search(
            embedding=query_embedding,
            model='minilm',
            domain=search_request.domain_filter,
            limit=search_request.limit // 2
        )
        
        # Merge results (keyword + semantic)
        all_results = _merge_search_results(keyword_results, semantic_results)
        '''
        
        # For now, return keyword-only results
        all_results = [dict(p) for p in keyword_results]
        
        logger.info(
            "Search complete",
            query=search_request.query,
            results=len(all_results)
        )
        
        return all_results
        
    except Exception as e:
        logger.error(
            "Search failed",
            query=search_request.query,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed"
        )


@router.get(
    "/autocomplete",
    summary="Search autocomplete",
    description="Get search suggestions based on partial query"
)
async def autocomplete(
    query: str,
    limit: int = 5,
    paper_repo: PaperRepository = Depends(get_paper_repo)
):
    """
    Get search suggestions for autocomplete.
    
    Args:
        query: Partial search query
        limit: Maximum suggestions
        paper_repo: Paper repository
        
    Returns:
        List of suggestions
    """
    if len(query) < 2:
        return []
    
    try:
        # Search for papers matching prefix
        papers = await paper_repo.search_by_text(
            search_text=query,
            limit=limit
        )
        
        # Extract unique titles/topics
        suggestions = list(set([
            dict(p)['title'][:100] for p in papers
        ]))[:limit]
        
        return suggestions
        
    except Exception as e:
        logger.error(f"Autocomplete failed: {e}")
        return []