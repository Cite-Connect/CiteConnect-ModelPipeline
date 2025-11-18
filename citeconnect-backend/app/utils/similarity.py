"""
Similarity Computation Utilities

Provides functions for computing similarity between embeddings
and ranking/scoring papers.
"""

import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Similarity score between 0 and 1 (1 = identical)
    """
    # Handle edge cases
    if len(vec1) == 0 or len(vec2) == 0:
        return 0.0
    
    # Normalize vectors
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)
    
    # Compute dot product
    similarity = np.dot(vec1_norm, vec2_norm)
    
    # Clip to [0, 1] range
    similarity = np.clip(similarity, 0.0, 1.0)
    
    return float(similarity)


def cosine_similarity_batch(
    query_vec: np.ndarray,
    database_vecs: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between query and multiple vectors
    
    Args:
        query_vec: Query vector (768,)
        database_vecs: Database vectors (n, 768)
    
    Returns:
        Array of similarity scores (n,)
    """
    # Normalize query
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    
    # Normalize database vectors
    db_norms = np.linalg.norm(database_vecs, axis=1, keepdims=True) + 1e-10
    db_normalized = database_vecs / db_norms
    
    # Compute similarities (dot product)
    similarities = np.dot(db_normalized, query_norm)
    
    # Clip to [0, 1]
    similarities = np.clip(similarities, 0.0, 1.0)
    
    return similarities


def rank_by_similarity(
    query_embedding: np.ndarray,
    papers_with_embeddings: List[dict],
    top_k: int = 20
) -> List[Tuple[dict, float]]:
    """
    Rank papers by cosine similarity to query
    
    Args:
        query_embedding: Query embedding (768,)
        papers_with_embeddings: List of dicts with 'embedding' key
        top_k: Number of top results to return
    
    Returns:
        List of (paper, similarity_score) tuples, sorted by score descending
    """
    # Extract embeddings
    embeddings = np.array([p['embedding'] for p in papers_with_embeddings])
    
    # Compute similarities
    similarities = cosine_similarity_batch(query_embedding, embeddings)
    
    # Create (paper, score) tuples
    ranked = list(zip(papers_with_embeddings, similarities))
    
    # Sort by similarity descending
    ranked.sort(key=lambda x: x[1], reverse=True)
    
    # Return top K
    return ranked[:top_k]


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Normalize scores to [0, 1] range using min-max scaling
    
    Args:
        scores: Array of scores
    
    Returns:
        Normalized scores
    """
    if len(scores) == 0:
        return scores
    
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    if max_score == min_score:
        return np.ones_like(scores)
    
    return (scores - min_score) / (max_score - min_score)


def calculate_composite_score(
    semantic_similarity: float,
    citation_count: int,
    year: int,
    max_citations: int,
    current_year: int = 2025,
    weights: dict = None
) -> float:
    """
    Calculate composite recommendation score
    
    Default weights from scoping document:
    - Semantic similarity: 0.35
    - Popularity (citations): 0.15
    - Recency: 0.10
    - [Citation relevance, keyword match, diversity to be added]
    
    Args:
        semantic_similarity: Cosine similarity (0-1)
        citation_count: Number of citations
        year: Publication year
        max_citations: Max citations in result set (for normalization)
        current_year: Current year for recency calculation
        weights: Optional custom weights dict
    
    Returns:
        Composite score (0-1)
    """
    # Default weights
    if weights is None:
        weights = {
            'semantic': 0.35,
            'citations': 0.15,
            'recency': 0.10
        }
    
    # Normalize citation count (0-1)
    popularity_score = citation_count / max_citations if max_citations > 0 else 0.0
    
    # Normalize recency (0-1, recent = higher)
    # Papers from current year get 1.0, older papers get lower scores
    max_age = 10  # Consider papers up to 10 years old
    age = current_year - year
    recency_score = max(0, 1 - (age / max_age))
    
    # Composite score
    score = (
        weights['semantic'] * semantic_similarity +
        weights['citations'] * popularity_score +
        weights['recency'] * recency_score
    )
    
    return float(score)