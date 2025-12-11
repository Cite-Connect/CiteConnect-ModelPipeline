# tests/integration/test_service_integration.py
"""
Service integration tests - critical business logic only
Tests recommendation generation and evaluation with real database
Uses EXISTING papers from database (no test paper creation)
"""

import pytest
import os
import numpy as np
from typing import Dict, List

from app.db.repositories.user_repo import UserRepository
from app.db.repositories.paper_repo import PaperRepository
from app.db.repositories.interaction_repo import InteractionRepository
from app.db.repositories.ground_truth_repo import GroundTruthRepository
from app.services.recommendation_service import RecommendationService
from app.services.evaluation_service import EvaluationService
from app.core.security import hash_password


# ============================================================================
# Helper Functions
# ============================================================================

async def create_test_user_with_profile(test_db, email=None, domain="healthcare"):
    """Create user with complete profile for testing"""
    if email is None:
        email = f"test_user_{os.getpid()}@test.integration.com"
    
    # Create user
    try:
        password_hash = hash_password("TestPass123!")
    except Exception as e:
        print(f"Password hashing failed: {e}")
        import bcrypt
        password_hash = bcrypt.hashpw(
            b"TestPass123!",
            bcrypt.gensalt()
        ).decode('utf-8')
    
    user = await test_db.fetchrow(
        """
        INSERT INTO users (email, password_hash, name, is_active)
        VALUES ($1, $2, $3, $4)
        RETURNING user_id, email, name
        """,
        email, password_hash, "Test User", True
    )
    
    user_id = user['user_id']
    
    # Create profile
    await test_db.execute(
        """
        INSERT INTO user_profiles_extended (
            user_id, primary_domain, research_stage, reading_level, 
            years_experience, prefers_recent_papers
        )
        VALUES ($1, $2, $3, $4, $5, $6)
        """,
        user_id, domain, 'phd', 'intermediate', 3, True
    )
    
    # Create interests
    interests = ['machine learning', 'disease prediction', 'clinical trials']
    for interest in interests:
        await test_db.execute(
            """
            INSERT INTO user_interest_hierarchy (
                user_id, interest_term, interest_level, source, confidence_score
            )
            VALUES ($1, $2, $3, $4, $5)
            """,
            user_id, interest, 1, 'explicit', 1.0
        )
    
    # Create recommendation state
    await test_db.execute(
        """
        INSERT INTO user_recommendation_state (
            user_id, recommendation_stage, interaction_count
        )
        VALUES ($1, $2, $3)
        """,
        user_id, 'cold_start', 0
    )
    
    return dict(user)


async def create_user_embedding(test_db, user_id: int, model='minilm'):
    """Create user embedding"""
    table_name = f"user_embeddings_{model}"
    
    # Generate random embedding
    dim = 384 if model == 'minilm' else 768
    embedding = np.random.rand(dim).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    
    # Convert to string format for pgvector
    embedding_str = '[' + ','.join(str(x) for x in embedding.tolist()) + ']'
    
    await test_db.execute(
        f"""
        INSERT INTO {table_name} (user_id, embedding, generation_method)
        VALUES ($1, $2, $3)
        ON CONFLICT (user_id) DO UPDATE SET embedding = $2, last_updated = NOW()
        """,
        user_id,
        embedding_str,
        'profile_based'
    )


async def get_existing_papers(test_db, domain="healthcare", limit=30):
    """
    Get existing papers from database.
    
    Args:
        test_db: Database connection
        domain: Domain to filter by
        limit: Maximum number of papers
        
    Returns:
        List of papers
        
    Raises:
        pytest.skip if not enough papers found
    """
    papers = await test_db.fetch(
        """
        SELECT paper_id, title, abstract, domain, year, citation_count, authors, venue
        FROM papers
        WHERE domain = $1
        ORDER BY citation_count DESC
        LIMIT $2
        """,
        domain, limit
    )
    
    if not papers or len(papers) < 10:
        pytest.skip(
            f"Not enough {domain} papers in database (found {len(papers) if papers else 0}, need at least 10). "
            f"Run data ingestion first."
        )
    
    return [dict(p) for p in papers]


async def verify_embeddings_exist(test_db, domain="healthcare", min_count=10):
    """
    Verify paper embeddings exist in database.
    
    Raises:
        pytest.skip if not enough embeddings found
    """
    count = await test_db.fetchval(
        """
        SELECT COUNT(*)
        FROM paper_embeddings_minilm pe
        JOIN papers p ON pe.paper_id = p.paper_id
        WHERE p.domain = $1
        """,
        domain
    )
    
    if count < min_count:
        pytest.skip(
            f"Not enough paper embeddings in database (found {count}, need at least {min_count}). "
            f"Run embedding generation first."
        )
    
    return count


# ============================================================================
# Test Class
# ============================================================================

@pytest.mark.integration
class TestServiceIntegration:
    """Test critical service workflows with real database and existing papers"""
    
    @pytest.mark.asyncio
    async def test_cold_start_recommendation_flow(self, test_db):
        """
        Test: End-to-end cold-start recommendation generation
        Uses EXISTING papers from database
        
        CRITICAL PATH:
        1. Create user with profile and interests
        2. Use existing papers with embeddings
        3. Generate recommendations (should return 10 papers)
        4. Verify recommendations match user domain and interests
        """
        print("\n" + "="*60)
        print("TEST: Cold-Start Recommendation Flow")
        print("="*60)
        
        # [1/4] Verify database has required data
        print("\n[1/4] Verifying database has papers and embeddings...")
        embedding_count = await verify_embeddings_exist(test_db, domain="healthcare", min_count=10)
        print(f"✓ Found {embedding_count} paper embeddings in database")
        
        # [2/4] Create test user with profile
        print("\n[2/4] Creating test user with profile...")
        user = await create_test_user_with_profile(test_db, domain="healthcare")
        user_id = user['user_id']
        print(f"✓ User created: {user['email']} (ID: {user_id})")
        
        # [3/4] Load existing papers
        print("\n[3/4] Loading existing papers from database...")
        papers = await get_existing_papers(test_db, domain="healthcare", limit=30)
        print(f"✓ Loaded {len(papers)} existing papers")
        
        # [4/4] Create user embeddings
        print("\n[4/4] Creating user embeddings...")
        await create_user_embedding(test_db, user_id, model='minilm')
        await create_user_embedding(test_db, user_id, model='specter')
        print(f"✓ User embeddings created (MiniLM + SPECTER)")
        
        # Generate recommendations
        print("\n" + "-"*60)
        print("Generating recommendations...")
        print("-"*60)
        
        rec_service = RecommendationService(test_db)
        
        try:
            recommendations = await rec_service.generate_recommendations(
                user_id=user_id,
                count=10,
                model='minilm'
            )
            
            # Verify results
            assert 'papers' in recommendations, "Should return papers"
            assert len(recommendations['papers']) > 0, "Should return at least some papers"
            assert recommendations['method'] == 'cold_start', "Should be cold-start method"
            assert recommendations['model_used'] == 'minilm', "Should use specified model"
            
            # Verify papers have required fields
            first_paper = recommendations['papers'][0]
            assert 'paper_id' in first_paper, "Paper should have paper_id"
            assert 'title' in first_paper, "Paper should have title"
            assert 'final_score' in first_paper or 'relevance_score' in first_paper, "Paper should have score"
            assert 'relevance_explanation' in first_paper, "Paper should have explanation"
            
            # Verify domain relevance
            healthcare_count = sum(
                1 for p in recommendations['papers'] 
                if p.get('domain') == 'healthcare'
            )
            assert healthcare_count > 0, "Should have papers in user's domain"
            
            # Print results
            print(f"\n✅ TEST PASSED")
            print(f"✓ Generated {len(recommendations['papers'])} recommendations")
            print(f"✓ Healthcare papers: {healthcare_count}/{len(recommendations['papers'])}")
            
            scores = [p.get('final_score') or p.get('relevance_score', 0) for p in recommendations['papers']]
            avg_score = sum(scores) / len(scores) if scores else 0
            print(f"✓ Average score: {avg_score:.3f}")
            print(f"✓ Total candidates: {recommendations.get('total_candidates', 'N/A')}")
            
            # Show top 3 recommendations
            print(f"\nTop 3 Recommendations:")
            for i, paper in enumerate(recommendations['papers'][:3], 1):
                score = paper.get('final_score') or paper.get('relevance_score', 0)
                print(f"{i}. {paper['title'][:60]}...")
                print(f"   Score: {score:.3f}")
                print(f"   Explanation: {paper.get('relevance_explanation', 'N/A')[:80]}...")
            
        except Exception as e:
            print(f"\n❌ TEST FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            pytest.fail(f"Recommendation generation failed: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_evaluation_metrics_calculation(self, test_db):
        """
        Test: Recommendation evaluation with real data
        Uses EXISTING papers from database
        """
        print("\n" + "="*60)
        print("TEST: Evaluation Metrics Calculation")
        print("="*60)
        
        # Setup
        print("\n[1/3] Verifying database has papers...")
        await verify_embeddings_exist(test_db, domain="healthcare", min_count=10)
        
        print("\n[2/3] Creating test user...")
        user = await create_test_user_with_profile(test_db, domain="healthcare")
        user_id = user['user_id']
        
        # Load existing papers
        papers = await get_existing_papers(test_db, domain="healthcare", limit=30)
        print(f"✓ Setup complete: {len(papers)} papers loaded")
        
        # Create user embeddings
        print("\n[3/3] Creating user embeddings...")
        await create_user_embedding(test_db, user_id, model='minilm')
        await create_user_embedding(test_db, user_id, model='specter')
        print(f"✓ User embeddings created")
        
        # Generate recommendations
        print("\n" + "-"*60)
        print("Generating recommendations...")
        print("-"*60)
        
        rec_service = RecommendationService(test_db)
        recommendations = await rec_service.generate_recommendations(
            user_id=user_id,
            count=10,
            model='minilm'
        )
        print(f"✓ Generated {len(recommendations['papers'])} recommendations")
        
        # Evaluate
        print("\n" + "-"*60)
        print("Evaluating recommendations...")
        print("-"*60)
        
        eval_service = EvaluationService(test_db)
        
        try:
            evaluation = await eval_service.evaluate_cold_start_recommendations(
                user_id=user_id,
                recommendations=recommendations['papers'],
                model='minilm',
                store_result=True
            )
            
            # Verify evaluation structure
            assert 'profile_alignment' in evaluation, "Should have profile_alignment"
            assert 'ground_truth_quality' in evaluation, "Should have ground_truth_quality"
            assert 'combined_score' in evaluation, "Should have combined_score"
            assert 'passes_threshold' in evaluation, "Should have passes_threshold"
            
            # Verify metrics are in valid range
            assert 0 <= evaluation['profile_alignment'] <= 1, "Profile alignment should be 0-1"
            assert 0 <= evaluation['ground_truth_quality'] <= 1, "GT quality should be 0-1"
            assert 0 <= evaluation['combined_score'] <= 1, "Combined score should be 0-1"
            
            # Verify evaluation was stored
            stored = await test_db.fetchrow(
                "SELECT * FROM cold_start_evaluations WHERE user_id = $1 ORDER BY evaluation_timestamp DESC LIMIT 1",
                user_id
            )
            assert stored is not None, "Evaluation should be stored in database"
            assert stored['user_id'] == user_id
            
            # Print results
            print(f"\n✅ TEST PASSED")
            print(f"✓ Profile alignment: {evaluation['profile_alignment']:.3f}")
            print(f"✓ Ground truth quality: {evaluation['ground_truth_quality']:.3f}")
            print(f"✓ Combined score: {evaluation['combined_score']:.3f}")
            print(f"✓ Passes threshold: {evaluation['passes_threshold']}")
            print(f"✓ Evaluation stored: evaluation_id={stored.get('evaluation_id')}")
            
        except Exception as e:
            print(f"\n❌ TEST FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            pytest.fail(f"Evaluation failed: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_recommendation_with_interactions(self, test_db):
        """
        Test: User interactions tracking
        Uses EXISTING papers from database
        """
        print("\n" + "="*60)
        print("TEST: Recommendations with Interactions")
        print("="*60)
        
        # Setup
        print("\n[1/3] Verifying database...")
        await verify_embeddings_exist(test_db, domain="healthcare", min_count=10)
        
        print("\n[2/3] Creating test user...")
        user = await create_test_user_with_profile(test_db, domain="healthcare")
        user_id = user['user_id']
        
        # Load existing papers
        papers = await get_existing_papers(test_db, domain="healthcare", limit=30)
        print(f"✓ Setup complete: {len(papers)} papers loaded")
        
        # Create user embeddings
        print("\n[3/3] Creating user embeddings...")
        await create_user_embedding(test_db, user_id, model='minilm')
        await create_user_embedding(test_db, user_id, model='specter')
        print(f"✓ User embeddings created")
        
        # Generate initial recommendations
        print("\n" + "-"*60)
        print("Generating initial recommendations...")
        print("-"*60)
        
        rec_service = RecommendationService(test_db)
        initial_recs = await rec_service.generate_recommendations(
            user_id=user_id,
            count=10,
            model='minilm'
        )
        
        assert len(initial_recs['papers']) > 0, "Should generate recommendations"
        print(f"✓ Generated {len(initial_recs['papers'])} recommendations")
        
        # User interacts with papers
        print("\n" + "-"*60)
        print("Creating user interactions...")
        print("-"*60)
        
        interaction_repo = InteractionRepository(test_db)
        
        for i, paper in enumerate(initial_recs['papers'][:3]):
            interaction_type = 'like' if i % 2 == 0 else 'save'
            await interaction_repo.create_interaction(
                user_id=user_id,
                paper_id=paper['paper_id'],
                interaction_type=interaction_type,
                duration_seconds=60 if interaction_type == 'like' else None
            )
            print(f"  {i+1}. {interaction_type}: {paper['title'][:50]}...")
        
        # Verify interactions were created
        interactions = await interaction_repo.get_user_interactions(user_id, limit=10)
        assert len(interactions) >= 3, "Should have at least 3 interactions"
        
        # Verify recommendation state
        state = await test_db.fetchrow(
            "SELECT * FROM user_recommendation_state WHERE user_id = $1",
            user_id
        )
        assert state is not None, "Should have recommendation state"
        
        # Print results
        print(f"\n✅ TEST PASSED")
        print(f"✓ Created {len(interactions)} interactions")
        print(f"✓ Recommendation stage: {state['recommendation_stage']}")
        print(f"✓ Interaction count: {state['interaction_count']}")
        
        # Show interactions
        print(f"\nRecorded Interactions:")
        for i, interaction in enumerate(interactions[:3], 1):
            interaction_type = interaction.get('interaction_type', 'unknown')
            title = interaction.get('title', 'Unknown')[:50]
            print(f"{i}. {interaction_type}: {title}")