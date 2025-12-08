"""
LLM Service for Query Refinement

Uses OpenAI to refine and expand user search queries
based on their profile and interests for better semantic search results.
"""
from typing import Dict, List, Optional
import hashlib
import json

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Try to import openai
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai not installed. LLM query refinement will be disabled.")


class LLMService:
    """
    Service for LLM-based query refinement using OpenAI.
    
    Refines user search queries to better match academic paper embeddings
    by expanding terms, capturing intent, and incorporating user profile context.
    """
    
    def __init__(self):
        """Initialize LLM service with OpenAI."""
        self.api_key = settings.OPENAI_API_KEY
        self.model_name = settings.OPENAI_MODEL
        self.client = None
        self.enabled = False
        
        logger.info(
            "LLM service initialization",
            openai_available=OPENAI_AVAILABLE,
            api_key_set=bool(self.api_key),
            api_key_length=len(self.api_key) if self.api_key else 0,
            model=self.model_name
        )
        
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI library not available - LLM service disabled")
            return
        
        if not self.api_key:
            logger.warning(
                "OPENAI_API_KEY not set - LLM service disabled",
                settings_openai_key=settings.OPENAI_API_KEY,
                settings_type=type(settings.OPENAI_API_KEY).__name__
            )
            return
        
        try:
            # Initialize OpenAI client
            self.client = AsyncOpenAI(api_key=self.api_key)
            self.enabled = True
            logger.info(
                "LLM service initialized",
                model=self.model_name,
                provider="openai"
            )
        except Exception as e:
            logger.error(
                "Failed to initialize OpenAI client",
                error=str(e),
                exc_info=True
            )
            self.enabled = False
    
    async def refine_search_query(
        self,
        query: str,
        user_profile: Dict,
        user_interests: List[str],
        cache_key: Optional[str] = None
    ) -> str:
        """
        Refine search query using LLM based on user profile.
        
        Expands and refines the query to better match academic paper embeddings
        by incorporating user's domain, research stage, and interests.
        
        Args:
            query: Original user search query
            user_profile: User profile dict (domain, research_stage, reading_level)
            user_interests: List of user interest terms
            cache_key: Optional cache key for refined query (if caching externally)
            
        Returns:
            Refined/expanded query string
            
        Example:
            Original: "neural networks"
            Refined: "deep learning neural networks convolutional networks transformer models"
        """
        if not self.enabled:
            logger.debug("LLM service disabled - returning original query")
            return query
        
        if not query or not query.strip():
            return query
        
        logger.debug(
            "Refining search query with LLM",
            original_query=query[:50],
            model=self.model_name
        )
        
        try:
            # Build prompt
            prompt = self._build_refinement_prompt(query, user_profile, user_interests)
            
            # Generate refined query
            response = await self._generate_refined_query(prompt)
            
            # Clean and validate response
            refined_query = self._clean_refined_query(response, query)
            
            logger.info(
                "Query refined successfully",
                original_length=len(query),
                refined_length=len(refined_query),
                model=self.model_name
            )
            
            return refined_query
            
        except Exception as e:
            error_str = str(e)
            # Check if it's a quota/rate limit error
            if "429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower():
                # Check if limit is 0 (API not enabled or no free tier access)
                if "limit: 0" in error_str:
                    logger.warning(
                        "OpenAI API free tier not available (limit: 0) - using original query",
                        error="Free tier quota limit is 0",
                        suggestion="Check OpenAI API key and billing status. Visit: https://platform.openai.com/api-keys"
                    )
                else:
                    logger.warning(
                        "OpenAI API quota/rate limit exceeded - using original query",
                        error=error_str[:200],
                        suggestion="Wait for quota reset or check billing. Visit: https://platform.openai.com/usage"
                    )
            else:
                logger.error(
                    "Query refinement failed - using original query",
                    error=error_str[:200],
                    exc_info=True
                )
            # Fallback to original query on any error
            return query
    
    def _build_refinement_prompt(
        self,
        query: str,
        user_profile: Dict,
        user_interests: List[str]
    ) -> str:
        """
        Build prompt for query refinement with knowledge base alignment.
        
        Args:
            query: Original search query
            user_profile: User profile
            user_interests: User interests
            
        Returns:
            Formatted prompt string with knowledge base context
        """
        domain = user_profile.get('primary_domain', 'general')
        research_stage = user_profile.get('research_stage', 'intermediate')
        reading_level = user_profile.get('reading_level', 'intermediate')
        
        interests_text = ', '.join(user_interests[:5]) if user_interests else 'general research'
        
        # Knowledge base instruction set
        kb_instructions = f"""
# CiteConnect Knowledge Base Context

## System Overview
CiteConnect is an academic paper recommendation system that uses semantic search with embeddings. Papers are collected from Semantic Scholar API.

## Domain Taxonomy
- Current domain: {domain}
- Available domains: healthcare, fintech, quantum_computing
- Each paper is tagged with exactly one domain

## Paper Structure & Metadata
Papers in the database contain:
- **title**: Full paper title (searchable)
- **abstract**: Paper abstract/summary (searchable)
- **introduction**: Introduction section text (used for embeddings)
- **authors**: Array of author names (searchable)
- **year**: Publication year (filterable)
- **venue**: Journal or conference name (e.g., "Nature", "ICML")
- **citation_count**: Number of citations (filterable)
- **domain**: Research domain tag ({domain} domain)

## Embedding Generation Process
- **Model**: SPECTER2 and MiniLM (allenai/specter2)
- **Dimensions**: 768-dimensional vectors
- **Input Format**: Embeddings generated from: `title + abstract + introduction`
- **Query Embeddings**: User search queries are encoded using the same SPECTER2 model

## Search & Indexing Architecture
- **Distance Metric**: Cosine similarity (0.0 = identical, 1.0 = orthogonal)
- **Search Process**:
  1. User query is encoded to 768-dim embedding using SPECTER2
  2. Weaviate performs nearest-neighbor search (cosine similarity)
  3. Results filtered by domain, year, citation_count
  4. Top-k papers returned ranked by similarity score
- **Hybrid Search**: Combines semantic similarity (primary) + keyword matching (secondary)

## Query Refinement Guidelines
When refining queries, align with the knowledge base structure:

**DO:**
- Use {domain}-specific academic terminology
- Include technical terms that appear in Semantic Scholar papers
- Match the academic writing style found in paper titles/abstracts
- Keep queries concise (10-20 words) to match typical paper title/abstract length
- Use synonyms and related terms that researchers use in academic papers
- Match paper structure: queries should mirror how papers are structured (title-like phrases, abstract-like concepts)

**DON'T:**
- Over-expand queries (longer queries don't necessarily improve semantic matching)
- Use colloquial language (use academic terminology)
- Add unnecessary filler words
- Create queries that are too generic or too specific

**Semantic Similarity Optimization:**
- Match Paper Structure: Queries should mirror how papers are structured
- Use Academic Vocabulary: Prefer terms from academic literature over general terms
- Domain Alignment: Include domain-specific terminology from {domain} papers
- Concept Density: Pack multiple related concepts into the query (like paper titles do)
- Technical Precision: Use precise technical terms rather than generic descriptions
"""
        
        prompt = f"""{kb_instructions}

## User Context
- Domain: {domain}
- Research Stage: {research_stage}
- Reading Level: {reading_level}
- Interests: {interests_text}

## Original Query
"{query}"

## Task
Refine this query to better match academic paper titles and abstracts in the CiteConnect knowledge base. The refined query will be used for semantic search using SPECTER2 embeddings (768-dim, cosine similarity).

Refinement Goals:
1. Better match the structure and terminology of papers in the knowledge base
2. Align with the {domain} domain taxonomy and academic vocabulary
3. Optimize for semantic similarity search using SPECTER2 embeddings
4. Maintain conciseness (10-20 words) while capturing true academic intent
5. Include {domain}-specific technical terms that appear in Semantic Scholar papers

Guidelines:
- Use {domain}-specific academic terminology
- Match paper title/abstract structure and vocabulary
- Include technical terms from Semantic Scholar papers
- Optimize for SPECTER2 semantic similarity matching
- Use academic writing style, not colloquial language

Return ONLY the refined query, no explanations or additional text:

Refined Query:"""
        
        return prompt
    
    async def _generate_refined_query(self, prompt: str) -> str:
        """
        Call OpenAI API to generate refined query.
        
        Args:
            prompt: Full prompt for query refinement
            
        Returns:
            Refined query string
        """
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        try:
            # Call OpenAI API (async)
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an academic search query refinement assistant for CiteConnect, an academic paper recommendation system. 

Your task is to refine user queries to better match academic papers indexed in the knowledge base. Papers are collected from Semantic Scholar API and indexed using SPECTER2 embeddings (768-dim vectors generated from title + abstract + introduction).

Key principles:
- Use domain-specific academic terminology
- Match paper title/abstract structure and vocabulary
- Optimize for SPECTER2 semantic similarity (cosine similarity)
- Keep queries concise (10-20 words)
- Return ONLY the refined query, no explanations or additional text"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for more consistent results
                max_tokens=100,  # Limit output length
            )
            
            refined_query = response.choices[0].message.content.strip()
            
            logger.debug(
                "OpenAI response received",
                response_length=len(refined_query),
                tokens_used=response.usage.total_tokens if hasattr(response, 'usage') else None
            )
            
            return refined_query
            
        except Exception as e:
            logger.error(
                "OpenAI API call failed",
                error=str(e),
                exc_info=True
            )
            raise
    
    def _clean_refined_query(self, refined_query: str, original_query: str) -> str:
        """
        Clean and validate refined query.
        
        Args:
            refined_query: Raw LLM output
            original_query: Original query (fallback)
            
        Returns:
            Cleaned refined query
        """
        if not refined_query:
            return original_query
        
        # Remove quotes if present
        refined_query = refined_query.strip('"\'')
        
        # Remove common prefixes LLM might add
        prefixes_to_remove = [
            "Refined Query:",
            "Query:",
            "Refined:",
            "Here's the refined query:",
        ]
        
        for prefix in prefixes_to_remove:
            if refined_query.lower().startswith(prefix.lower()):
                refined_query = refined_query[len(prefix):].strip()
        
        # Limit length (prevent overly long queries)
        max_length = 200
        if len(refined_query) > max_length:
            refined_query = refined_query[:max_length].rsplit(' ', 1)[0]
            logger.warning(
                "Refined query truncated",
                original_length=len(refined_query),
                max_length=max_length
            )
        
        # If cleaned query is too short or seems invalid, use original
        if len(refined_query) < len(original_query) * 0.5:
            logger.warning(
                "Refined query too short - using original",
                refined_length=len(refined_query),
                original_length=len(original_query)
            )
            return original_query
        
        return refined_query
    
    def generate_cache_key(
        self,
        query: str,
        user_id: int,
        domain: str
    ) -> str:
        """
        Generate cache key for refined query.
        
        Args:
            query: Original query
            user_id: User ID
            domain: User domain
            
        Returns:
            Cache key string
        """
        # Create hash of query + user context
        cache_string = f"{query}:{user_id}:{domain}"
        cache_hash = hashlib.md5(cache_string.encode()).hexdigest()
        return f"llm:refined_query:{cache_hash}"


# Singleton instance
_llm_service_instance: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """
    Get or create LLM service singleton.
    
    Returns:
        LLMService instance
    """
    global _llm_service_instance
    
    if _llm_service_instance is None:
        _llm_service_instance = LLMService()
    
    return _llm_service_instance
