"""
Comprehensive verification script for CiteConnect backend setup.

This script tests:
1. All imports (no circular imports)
2. Configuration loading
3. Model instantiation
4. Security functions (password hashing, JWT)
5. Database client modules
"""

import sys
import traceback
from datetime import datetime


def print_section(title):
    """Print formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_imports():
    """Test all module imports."""
    print_section("Testing Module Imports")
    
    tests = [
        ("Core - Exceptions", "from app.core.exceptions import CiteConnectException, AuthenticationError, DatabaseError"),
        ("Core - Logging", "from app.core.logging import setup_logging, get_logger, JSONFormatter"),
        ("Core - Config", "from app.core.config import get_settings, Settings"),
        ("Core - Security", "from app.core.security import hash_password, verify_password, create_access_token"),
        ("DB - PostgreSQL", "from app.db.postgres import get_db_pool, execute_query"),
        ("DB - Redis", "from app.db.redis_client import get_redis_client, cache_set, cache_get"),
        ("DB - Weaviate", "from app.db.weaviate_client import get_weaviate_client, search_papers"),
        ("DB - Neo4j", "from app.db.neo4j_client import get_neo4j_driver, execute_query"),
        ("Models - User", "from app.models.user import User, UserInterest, UserDomain"),
        ("Models - Paper", "from app.models.paper import Paper, PaperMetadata, PaperWithScore"),
        ("Models - Cluster", "from app.models.cluster import Cluster, ClusterPaper"),
        ("Models - Interaction", "from app.models.interaction import Interaction, InteractionContext"),
        ("Models - Graph", "from app.models.graph import GraphNode, GraphEdge, CitationNetwork"),
    ]
    
    passed = 0
    failed = 0
    
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"✓ {name:30s} - OK")
            passed += 1
        except Exception as e:
            print(f"✗ {name:30s} - FAILED: {str(e)}")
            failed += 1
    
    print(f"\nImport Tests: {passed} passed, {failed} failed")
    return failed == 0


def test_configuration():
    """Test configuration loading."""
    print_section("Testing Configuration")
    
    try:
        from app.core.config import get_settings
        
        settings = get_settings()
        
        print(f"✓ Settings loaded successfully")
        print(f"  - Environment: {settings.ENVIRONMENT}")
        print(f"  - Debug: {settings.DEBUG}")
        print(f"  - Log Level: {settings.LOG_LEVEL}")
        print(f"  - Database Host: {settings.POSTGRES_HOST}")
        print(f"  - Redis Host: {settings.REDIS_HOST}")
        print(f"  - SPECTER Model: {settings.SPECTER_MODEL_NAME}")
        print(f"  - Embedding Dimension: {settings.EMBEDDING_DIMENSION}")
        print(f"  - CORS Origins: {settings.ALLOWED_ORIGINS}")
        
        # Test validators
        assert settings.ENVIRONMENT in ['development', 'staging', 'production']
        assert settings.LOG_LEVEL in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        assert settings.EMBEDDING_DIMENSION == 768
        
        print(f"\n✓ Configuration validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_security():
    """Test security functions."""
    print_section("Testing Security Functions")
    
    try:
        from app.core.security import (
            hash_password, 
            verify_password, 
            create_access_token, 
            decode_token,
            create_token_pair
        )
        
        # Test password hashing
        print("Testing password hashing...")
        password = "TestPassword123!"
        hashed = hash_password(password)
        print(f"  ✓ Password hashed: {len(hashed)} characters")
        
        # Test password verification
        print("Testing password verification...")
        is_valid = verify_password(password, hashed)
        is_invalid = verify_password("WrongPassword", hashed)
        assert is_valid == True
        assert is_invalid == False
        print(f"  ✓ Password verification working correctly")
        
        # Test JWT token creation
        print("Testing JWT token creation...")
        token_data = {"sub": "123", "email": "test@example.com"}
        access_token = create_access_token(data=token_data)
        print(f"  ✓ Access token created: {len(access_token)} characters")
        
        # Test JWT token decoding
        print("Testing JWT token decoding...")
        payload = decode_token(access_token)
        assert payload["sub"] == "123"
        assert payload["email"] == "test@example.com"
        assert "exp" in payload
        assert "iat" in payload
        print(f"  ✓ Token decoded successfully")
        
        # Test token pair creation
        print("Testing token pair creation...")
        tokens = create_token_pair(user_id=123, email="test@example.com")
        assert "access_token" in tokens
        assert "refresh_token" in tokens
        assert tokens["token_type"] == "bearer"
        print(f"  ✓ Token pair created successfully")
        
        print(f"\n✓ All security tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Security test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_models():
    """Test model instantiation."""
    print_section("Testing Model Instantiation")
    
    try:
        from app.models.user import User, UserInterest, UserDomain
        from app.models.paper import Paper, PaperMetadata
        from app.models.cluster import Cluster, ClusterPaper
        from app.models.interaction import Interaction
        from app.models.graph import GraphNode, GraphEdge
        
        # Test User model
        print("Testing User model...")
        user = User(
            user_id=123,
            email="test@example.com",
            password_hash="hashed_password",
            name="Test User",
            domain="healthcare"
        )
        assert user.email == "test@example.com"
        assert user.domain == "healthcare"
        print(f"  ✓ User model created")
        
        # Test UserInterest model
        print("Testing UserInterest model...")
        interest = UserInterest(
            user_id=123,
            interest_keyword="machine learning",
            source="manual"
        )
        assert interest.weight == 1.0
        print(f"  ✓ UserInterest model created")
        
        # Test Paper model
        print("Testing Paper model...")
        paper = Paper(
            paper_id="arxiv:2401.001",
            title="Test Paper",
            authors=["Author 1", "Author 2"],
            year=2024,
            abstract="This is a test abstract",
            domain="healthcare"
        )
        assert paper.year == 2024
        assert len(paper.authors) == 2
        print(f"  ✓ Paper model created")
        
        # Test ClusterPaper model
        print("Testing ClusterPaper model...")
        cluster_paper = ClusterPaper(
            paper_id="arxiv:2401.001",
            title="Test Paper",
            year=2024,
            is_reference_paper=True,
            similarity_to_reference=1.0,
            position_x=250.0,
            position_y=200.0
        )
        assert cluster_paper.is_reference_paper == True
        print(f"  ✓ ClusterPaper model created")
        
        # Test Interaction model
        print("Testing Interaction model...")
        interaction = Interaction(
            user_id=123,
            paper_id="arxiv:2401.001",
            interaction_type="view"
        )
        assert interaction.interaction_type == "view"
        print(f"  ✓ Interaction model created")
        
        # Test GraphNode model
        print("Testing GraphNode model...")
        node = GraphNode(
            paper_id="arxiv:2401.001",
            title="Test Paper",
            year=2024,
            domain="healthcare",
            similarity_score=0.95
        )
        assert node.similarity_score == 0.95
        print(f"  ✓ GraphNode model created")
        
        # Test GraphEdge model
        print("Testing GraphEdge model...")
        edge = GraphEdge(
            source="arxiv:2401.001",
            target="arxiv:2401.002",
            weight=0.85,
            edge_type="semantic"
        )
        assert edge.weight == 0.85
        print(f"  ✓ GraphEdge model created")
        
        print(f"\n✓ All model tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_validators():
    """Test model validators."""
    print_section("Testing Model Validators")
    
    try:
        from app.models.user import UserDomain
        from app.models.paper import Paper
        from pydantic import ValidationError
        
        # Test domain validator
        print("Testing domain validator...")
        try:
            invalid_domain = UserDomain(
                user_id=123,
                domain="invalid_domain"
            )
            print(f"  ✗ Domain validator failed to catch invalid domain")
            return False
        except ValidationError:
            print(f"  ✓ Domain validator correctly rejected invalid domain")
        
        # Test year validator
        print("Testing year validator...")
        try:
            invalid_year = Paper(
                paper_id="test:001",
                title="Test",
                abstract="Test abstract",
                year=1800,  # Too old
                domain="healthcare"
            )
            print(f"  ✗ Year validator failed to catch invalid year")
            return False
        except ValidationError:
            print(f"  ✓ Year validator correctly rejected invalid year")
        
        # Test email validator
        print("Testing email validator...")
        from app.models.user import User
        try:
            invalid_email = User(
                email="not-an-email",
                password_hash="hash",
                name="Test User",
                domain="healthcare"
            )
            print(f"  ✗ Email validator failed to catch invalid email")
            return False
        except ValidationError:
            print(f"  ✓ Email validator correctly rejected invalid email")
        
        print(f"\n✓ All validator tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Validator test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_exception_hierarchy():
    """Test custom exception hierarchy."""
    print_section("Testing Exception Hierarchy")
    
    try:
        from app.core.exceptions import (
            CiteConnectException,
            AuthenticationError,
            ValidationError,
            DatabaseError
        )
        
        # Test base exception
        print("Testing CiteConnectException...")
        base_exc = CiteConnectException("Test error", status_code=500)
        assert base_exc.message == "Test error"
        assert base_exc.status_code == 500
        print(f"  ✓ Base exception works")
        
        # Test AuthenticationError
        print("Testing AuthenticationError...")
        auth_exc = AuthenticationError("Invalid credentials")
        assert auth_exc.status_code == 401
        print(f"  ✓ AuthenticationError works")
        
        # Test ValidationError
        print("Testing ValidationError...")
        val_exc = ValidationError("Invalid input", field="email")
        assert val_exc.status_code == 400
        print(f"  ✓ ValidationError works")
        
        # Test DatabaseError
        print("Testing DatabaseError...")
        db_exc = DatabaseError("Connection failed", operation="SELECT")
        assert db_exc.status_code == 500
        print(f"  ✓ DatabaseError works")
        
        print(f"\n✓ All exception tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Exception test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_logging():
    """Test logging configuration."""
    print_section("Testing Logging Configuration")
    
    try:
        from app.core.logging import setup_logging, get_logger, JSONFormatter, ColoredFormatter
        
        # Test logger setup
        print("Testing logger setup...")
        setup_logging(log_level="INFO", environment="development")
        print(f"  ✓ Logging configured")
        
        # Test logger creation
        print("Testing logger creation...")
        logger = get_logger(__name__)
        logger.info("Test log message")
        print(f"  ✓ Logger created and working")
        
        # Test formatters
        print("Testing formatters...")
        json_formatter = JSONFormatter()
        colored_formatter = ColoredFormatter()
        print(f"  ✓ Formatters instantiated")
        
        print(f"\n✓ All logging tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Logging test failed: {str(e)}")
        traceback.print_exc()
        return False


def generate_report(results):
    """Generate final test report."""
    print_section("Test Summary")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r)
    failed_tests = total_tests - passed_tests
    
    print(f"Total Tests:  {total_tests}")
    print(f"Passed:       {passed_tests} ✓")
    print(f"Failed:       {failed_tests} ✗")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print("\n" + "="*60)
    print("Detailed Results:")
    print("="*60 + "\n")
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10s} - {test_name}")
    
    print("\n" + "="*60)
    
    if failed_tests == 0:
        print("\n🎉 All tests passed! Setup is working correctly.")
        print("\nYou can proceed with implementing the remaining modules.")
    else:
        print(f"\n⚠️  {failed_tests} test(s) failed. Please review errors above.")
    
    print("="*60 + "\n")
    
    return failed_tests == 0


def main():
    """Run all tests and generate report."""
    print("\n" + "="*60)
    print("  CiteConnect Backend Setup Verification")
    print("  " + str(datetime.now()))
    print("="*60)
    
    # Run all tests
    results = {
        "Module Imports": test_imports(),
        "Configuration": test_configuration(),
        "Security Functions": test_security(),
        "Models": test_models(),
        "Validators": test_validators(),
        "Exception Hierarchy": test_exception_hierarchy(),
        "Logging": test_logging()
    }
    
    # Generate report
    all_passed = generate_report(results)
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
