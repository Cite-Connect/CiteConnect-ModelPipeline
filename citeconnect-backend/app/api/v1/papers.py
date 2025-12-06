"""
Paper management API endpoints.
Handles paper search, retrieval, and metadata operations.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Optional, List
from app.utils.logger import get_logger
from app.db.connection import get_db, DatabaseConnection
from app.db.repositories.paper_repo import PaperRepository
from app.models.paper import PaperResponse, PaperSearchRequest

logger = get_logger(__name__)

router = APIRouter()


def get_paper_repo(db: DatabaseConnection = Depends(get_db)) -> PaperRepository:
    """
    Dependency to get paper repository.
    
    Args:
        db: Database connection
        
    Returns:
        PaperRepository: Paper repository instance
    """
    return PaperRepository(db)


@router.get(
    "/{paper_id}",
    response_model=PaperResponse,
    summary="Get paper by ID",
    description="Retrieve paper details by paper ID"
)
async def get_paper(
    paper_id: str,
    paper_repo: PaperRepository = Depends(get_paper_repo)
):
    """
    Get paper by ID.
    
    Args:
        paper_id: Paper identifier (SHA-1 hash)
        paper_repo: Paper repository
        
    Returns:
        Paper details
    """
    logger.debug("Paper retrieval request", paper_id=paper_id)
    
    try:
        paper = await paper_repo.find_by_paper_id(paper_id)
        
        if not paper:
            logger.warning("Paper not found", paper_id=paper_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Paper {paper_id} not found"
            )
        
        logger.debug("Paper retrieved", paper_id=paper_id)
        
        return dict(paper)
        
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(
            "Paper retrieval failed",
            paper_id=paper_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Paper retrieval failed"
        )


@router.post(
    "/search",
    response_model=List[PaperResponse],
    summary="Search papers",
    description="Full-text search on paper titles and abstracts"
)
async def search_papers(
    search_request: PaperSearchRequest,
    paper_repo: PaperRepository = Depends(get_paper_repo)
):
    """
    Search papers by text query.
    
    Args:
        search_request: Search parameters
        paper_repo: Paper repository
        
    Returns:
        List of matching papers
    """
    logger.info(
        "Paper search request",
        query=search_request.query,
        limit=search_request.limit
    )
    
    try:
        papers = await paper_repo.search_by_text(
            search_text=search_request.query,
            limit=search_request.limit
        )
        
        # Apply filters if provided
        if search_request.domain_filter:
            papers = [
                p for p in papers
                if p.get('domain') == search_request.domain_filter
            ]
        
        if search_request.min_year:
            papers = [
                p for p in papers
                if p.get('year', 0) >= search_request.min_year
            ]
        
        logger.info(
            "Paper search complete",
            query=search_request.query,
            results=len(papers)
        )
        
        return [dict(p) for p in papers]
        
    except Exception as e:
        logger.error(
            "Paper search failed",
            query=search_request.query,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Paper search failed"
        )


@router.get(
    "/{paper_id}/citations",
    summary="Get paper citations",
    description="Get papers that cite this paper"
)
async def get_citations(
    paper_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    paper_repo: PaperRepository = Depends(get_paper_repo)
):
    """
    Get papers citing this paper.
    
    Args:
        paper_id: Paper identifier
        limit: Maximum results
        paper_repo: Paper repository
        
    Returns:
        List of citing papers
    """
    logger.debug("Citations request", paper_id=paper_id)
    
    try:
        # Get citing paper IDs
        citation_ids = await paper_repo.get_paper_citations(paper_id)
        
        if not citation_ids:
            logger.debug("No citations found", paper_id=paper_id)
            return {
                "paper_id": paper_id,
                "citation_count": 0,
                "citations": []
            }
        
        # Fetch paper details
        citing_papers = await paper_repo.find_by_ids(
            citation_ids[:limit]
        )
        
        logger.info(
            "Citations retrieved",
            paper_id=paper_id,
            count=len(citing_papers)
        )
        
        return {
            "paper_id": paper_id,
            "citation_count": len(citation_ids),
            "citations": [dict(p) for p in citing_papers]
        }
        
    except Exception as e:
        logger.error(
            "Citations retrieval failed",
            paper_id=paper_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Citations retrieval failed"
        )


@router.get(
    "/{paper_id}/references",
    summary="Get paper references",
    description="Get papers referenced by this paper"
)
async def get_references(
    paper_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    paper_repo: PaperRepository = Depends(get_paper_repo)
):
    """
    Get papers referenced by this paper.
    
    Args:
        paper_id: Paper identifier
        limit: Maximum results
        paper_repo: Paper repository
        
    Returns:
        List of referenced papers
    """
    logger.debug("References request", paper_id=paper_id)
    
    try:
        # Get reference IDs
        reference_ids = await paper_repo.get_paper_references(paper_id)
        
        if not reference_ids:
            logger.debug("No references found", paper_id=paper_id)
            return {
                "paper_id": paper_id,
                "reference_count": 0,
                "references": []
            }
        
        # Fetch paper details
        referenced_papers = await paper_repo.find_by_ids(
            reference_ids[:limit]
        )
        
        logger.info(
            "References retrieved",
            paper_id=paper_id,
            count=len(referenced_papers)
        )
        
        return {
            "paper_id": paper_id,
            "reference_count": len(reference_ids),
            "references": [dict(p) for p in referenced_papers]
        }
        
    except Exception as e:
        logger.error(
            "References retrieval failed",
            paper_id=paper_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="References retrieval failed"
        )


@router.get(
    "/domain/{domain}",
    response_model=List[PaperResponse],
    summary="Get papers by domain",
    description="Get papers in a specific research domain"
)
async def get_papers_by_domain(
    domain: str,
    limit: int = Query(default=20, ge=1, le=100),
    min_year: Optional[int] = Query(default=None, ge=1900),
    paper_repo: PaperRepository = Depends(get_paper_repo)
):
    """
    Get papers by domain.
    
    Args:
        domain: Research domain
        limit: Maximum results
        min_year: Optional minimum year
        paper_repo: Paper repository
        
    Returns:
        List of papers in domain
    """
    logger.info(
        "Domain papers request",
        domain=domain,
        limit=limit,
        min_year=min_year
    )
    
    try:
        papers = await paper_repo.find_by_domain(
            domain=domain,
            limit=limit,
            min_year=min_year
        )
        
        logger.info(
            "Domain papers retrieved",
            domain=domain,
            count=len(papers)
        )
        
        return [dict(p) for p in papers]
        
    except Exception as e:
        logger.error(
            "Domain papers retrieval failed",
            domain=domain,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Domain papers retrieval failed"
        )


@router.get(
    "/trending",
    response_model=List[PaperResponse],
    summary="Get trending papers",
    description="Get recently published papers with high citation velocity"
)
async def get_trending(
    domain: Optional[str] = Query(default=None),
    days: int = Query(default=30, ge=7, le=90),
    limit: int = Query(default=20, ge=1, le=50),
    paper_repo: PaperRepository = Depends(get_paper_repo)
):
    """
    Get trending papers.
    
    Args:
        domain: Optional domain filter
        days: Consider papers from last N days
        limit: Maximum results
        paper_repo: Paper repository
        
    Returns:
        List of trending papers
    """
    logger.info(
        "Trending papers request",
        domain=domain,
        days=days,
        limit=limit
    )
    
    try:
        papers = await paper_repo.get_trending_papers(
            domain=domain,
            days=days,
            limit=limit
        )
        
        logger.info(
            "Trending papers retrieved",
            domain=domain,
            count=len(papers)
        )
        
        return [dict(p) for p in papers]
        
    except Exception as e:
        logger.error(
            "Trending papers retrieval failed",
            domain=domain,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Trending papers retrieval failed"
        )


@router.get(
    "/batch",
    response_model=List[PaperResponse],
    summary="Get multiple papers by IDs",
    description="Retrieve multiple papers in one request"
)
async def get_papers_batch(
    paper_ids: List[str] = Query(..., description="List of paper IDs"),
    paper_repo: PaperRepository = Depends(get_paper_repo)
):
    """
    Get multiple papers by IDs.
    
    Args:
        paper_ids: List of paper identifiers
        paper_repo: Paper repository
        
    Returns:
        List of papers
    """
    logger.debug(
        "Batch paper retrieval",
        count=len(paper_ids)
    )
    
    try:
        if len(paper_ids) > 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 50 papers per request"
            )
        
        papers = await paper_repo.find_by_ids(paper_ids)
        
        logger.debug(
            "Batch retrieval complete",
            requested=len(paper_ids),
            found=len(papers)
        )
        
        return [dict(p) for p in papers]
        
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(
            "Batch retrieval failed",
            count=len(paper_ids),
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch retrieval failed"
        )