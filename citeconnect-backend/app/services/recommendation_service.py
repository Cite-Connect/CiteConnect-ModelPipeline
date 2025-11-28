"""
Recommendation Service

Generates personalized paper recommendations using:
- User profile embeddings
- Pickle file from DataPipeline (embeddings_db.pkl)
- Composite scoring (semantic + citations + recency)
- Fairness-aware re-ranking based on model bias analysis
"""

import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

from app.services.embedding_service import embedding_service
from app.services.fairness_service import fairness_aware_rerank
from app.utils.similarity import (
    cosine_similarity_batch,
    calculate_composite_score,
    rank_by_similarity,  # may be unused but kept for compatibility
)
from app.db.postgres import execute_query  # currently unused, kept for future extensions

logger = logging.getLogger(__name__)


class RecommendationService:
    """Service for generating personalized recommendations"""

    def __init__(self, pickle_path: Optional[str] = None):
        """
        Initialize recommendation service

        Args:
            pickle_path: Path to embeddings_db.pkl file
                         Default: ./working_data/embeddings_db.pkl
        """
        if pickle_path is None:
            # Default path (assumes FastAPI runs from citeconnect-backend/)
            base_dir = Path(__file__).parent.parent.parent
            pickle_path = base_dir / "working_data" / "embeddings_db.pkl"

        self.pickle_path = Path(pickle_path)
        self._paper_chunks: Optional[List[dict]] = None

        logger.info("Recommendation service initialized")
        logger.info(f"  Pickle path: {self.pickle_path}")

    def _load_papers(self) -> List[dict]:
        """
        Load papers from pickle file (lazy loading)

        Pickle structure:
        {
            'chunks': [
                {
                    'chunk_id': str,
                    'paper_id': str,
                    'paper_title': str,
                    'paper_year': int,
                    'citation_count': int,
                    'fieldsOfStudy': [...],
                    'text': str,
                    ...
                },
                ...
            ],
            'embeddings': numpy array of shape (n_chunks, embedding_dim)
        }

        Returns:
            List of paper chunks with embeddings combined.
        """
        if self._paper_chunks is not None:
            return self._paper_chunks

        if not self.pickle_path.exists():
            raise FileNotFoundError(
                f"Embeddings pickle not found: {self.pickle_path}\n"
                f"Run DataPipeline first to generate embeddings_db.pkl"
            )

        logger.info(f"Loading papers from pickle: {self.pickle_path}")

        with open(self.pickle_path, "rb") as f:
            data = pickle.load(f)

        # Extract chunks and embeddings
        chunks = data["chunks"]
        embeddings = data["embeddings"]

        logger.info(f"✓ Loaded {len(chunks)} chunks with embeddings {embeddings.shape}")

        # Verify lengths match
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
            )

        # Combine chunks with their embeddings
        paper_chunks: List[dict] = []
        for chunk, embedding in zip(chunks, embeddings):
            combined = {
                **chunk,  # all chunk metadata
                "embedding": embedding,  # numpy vector
            }
            paper_chunks.append(combined)

        self._paper_chunks = paper_chunks
        logger.info("✓ Combined chunks with embeddings")

        return self._paper_chunks

    async def generate_recommendations(
        self,
        user_id: int,
        top_k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Generate personalized recommendations for a user.

        Args:
            user_id: User ID
            top_k: Number of recommendations to return
            filters: Optional filters (min_year, min_citations, domain, etc.)

        Returns:
            List of recommended papers with scores.
        """
        logger.info(f"Generating {top_k} recommendations for user {user_id}")

        # Step 1: Get user profile embedding
        user_embedding = await embedding_service.get_user_profile_embedding(user_id)

        # Step 2: Load papers from pickle
        all_chunks = self._load_papers()

        # Step 3: Apply filters (if any)
        filtered_chunks = self._apply_filters(all_chunks, filters)
        logger.info(f"  After filtering: {len(filtered_chunks)} chunks")

        if not filtered_chunks:
            logger.warning(
                "No chunks available after filtering – returning empty recommendations."
            )
            return []

        # Step 4: Compute semantic similarities
        embeddings_matrix = np.array([c["embedding"] for c in filtered_chunks])
        similarities = cosine_similarity_batch(user_embedding, embeddings_matrix)

        # Step 5: Calculate composite scores
        scored_papers: List[Dict] = []

        max_citations = max(c.get("citation_count", 0) for c in filtered_chunks) or 0

        for chunk, semantic_sim in zip(filtered_chunks, similarities):
            year = chunk.get("paper_year", 2020)
            citation_count = chunk.get("citation_count", 0)

            composite_score = calculate_composite_score(
                semantic_similarity=float(semantic_sim),
                citation_count=citation_count,
                year=year,
                max_citations=max_citations,
            )

            scored_papers.append(
                {
                    "paper_id": chunk["paper_id"],
                    "chunk_id": chunk["chunk_id"],
                    "title": chunk["paper_title"],
                    "year": year,
                    "citation_count": citation_count,
                    "text": chunk["text"][:500],  # Preview only
                    "composite_score": float(composite_score),
                    "score_components": {
                        "semantic_similarity": float(semantic_sim),
                        "normalized_citations": (
                            citation_count / max_citations if max_citations > 0 else 0.0
                        ),
                        # Simple recency score: 1 when current year, decays over ~10 years
                        "recency_score": max(0.0, 1.0 - ((2025 - year) / 10.0)),
                    },
                }
            )

        # Step 6: Deduplicate by paper_id (keep highest scoring chunk per paper)
        unique_papers: Dict[str, Dict] = {}
        for paper in scored_papers:
            pid = paper["paper_id"]
            if pid not in unique_papers or paper["composite_score"] > unique_papers[pid][
                "composite_score"
            ]:
                unique_papers[pid] = paper

        # Step 7: Sort by composite score (baseline ranking)
        recommendations = list(unique_papers.values())
        recommendations.sort(key=lambda x: x["composite_score"], reverse=True)

        # Step 7b: Fairness-aware re-ranking (boost under-served fields)
        # Uses fairness_config.json generated by scripts/model_bias_slicing.py
        recommendations = fairness_aware_rerank(
            recommendations,
            score_key="composite_score",
            boost=1.05,  # small, gentle boost to under-served fields
        )

        # Step 8: Return top K
        final_recs = recommendations[:top_k]

        if final_recs:
            logger.info(f"✓ Generated {len(final_recs)} recommendations")
            logger.info(
                "  Top score: %.3f", float(final_recs[0]["composite_score"])
            )
            logger.info(
                "  Top paper: %s...",
                final_recs[0]["title"][:60],
            )
        else:
            logger.info("✓ Generated 0 recommendations (no matching papers)")

        return final_recs

    def _apply_filters(
        self,
        chunks: List[dict],
        filters: Optional[Dict] = None,
    ) -> List[dict]:
        """
        Apply filters to paper chunks.

        Args:
            chunks: List of paper chunks
            filters: Dict with filter criteria

        Returns:
            Filtered list of chunks.
        """
        if not filters:
            return chunks

        filtered = chunks

        # Filter by minimum year
        if "min_year" in filters:
            min_year = filters["min_year"]
            filtered = [c for c in filtered if c.get("paper_year", 0) >= min_year]

        # Filter by minimum citations
        if "min_citations" in filters:
            min_cites = filters["min_citations"]
            filtered = [
                c for c in filtered if c.get("citation_count", 0) >= min_cites
            ]

        # Filter by domain (if stored in chunks)
        if "domain" in filters:
            domain = filters["domain"]
            filtered = [
                c
                for c in filtered
                if c.get("domain", "").lower() == str(domain).lower()
            ]

        return filtered

    async def generate_starter_kit(self, user_id: int) -> Dict:
        """
        Generate initial "starter kit" of recommendations.

        Called during user registration (background task).

        Args:
            user_id: New user ID

        Returns:
            Dict with cluster information and top papers.
        """
        logger.info(f"Generating starter kit for user {user_id}")

        # Generate recommendations (20 papers for clustering)
        recommendations = await self.generate_recommendations(
            user_id=user_id,
            top_k=20,
        )

        # TODO: Implement K-means clustering (Phase 2).
        # For now, return as a single cluster.
        starter_kit = {
            "user_id": user_id,
            "total_papers": len(recommendations),
            "clusters": [
                {
                    "cluster_id": 1,
                    "cluster_name": "Recommended Papers",
                    "theme": "Based on your interests",
                    "papers": recommendations,
                }
            ],
            "generated_at": "now",
        }

        logger.info(
            "✓ Generated starter kit with %d papers",
            len(recommendations),
        )

        return starter_kit


# Create singleton instance
recommendation_service = RecommendationService()
