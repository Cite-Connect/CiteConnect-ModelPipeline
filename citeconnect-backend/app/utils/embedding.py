"""
SPECTER2 Embedding Utilities

Provides wrapper for SPECTER2 model to generate embeddings for:
- User interest profiles
- Paper queries
- Individual papers

Model: allenai/specter2_base (768-dimensional embeddings)
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


class SPECTEREmbedder:
    """
    Wrapper for sentence embedding model
    
    NOTE: Using all-MiniLM-L6-v2 (384-dim) to match DataPipeline embeddings
    TODO: Switch to allenai/specter2_base (768-dim) when DataPipeline updated
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding model
        
        Args:
            model_name: HuggingFace model identifier
                       Default: all-MiniLM-L6-v2 (384-dim, matches current pickle)
                       Future: allenai/specter2_base (768-dim)
        """
        logger.info(f"Loading embedding model: {model_name}")
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        
        logger.info(f"✓ Embedding model loaded successfully")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Embedding dim: 384 (all-MiniLM) or 768 (SPECTER2)")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for single text
        
        Args:
            text: Input text
        
        Returns:
            384-dimensional embedding vector (all-MiniLM-L6-v2)
            OR 768-dimensional (SPECTER2)
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling for sentence-transformers models
            # Use CLS token for SPECTER2
            if "specter" in self.model_name.lower():
                # SPECTER2: use CLS token
                embedding = outputs.last_hidden_state[:, 0, :].squeeze()
            else:
                # Sentence-transformers: use mean pooling
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embedding = (sum_embeddings / sum_mask).squeeze()
        
        return embedding.numpy()
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for batch of texts
        
        Args:
            texts: List of input texts
        
        Returns:
            Array of shape (n_texts, 768)
        """
        embeddings = []
        
        for text in texts:
            emb = self.embed_text(text)
            embeddings.append(emb)
        
        return np.array(embeddings)
    
    def embed_paper(self, title: str, abstract: str) -> np.ndarray:
        """
        Generate embedding for a paper
        
        Args:
            title: Paper title
            abstract: Paper abstract
        
        Returns:
            768-dimensional embedding
        """
        # Format as SPECTER2 expects
        text = f"[TITLE] {title} [ABSTRACT] {abstract}"
        return self.embed_text(text)
    
    def embed_user_interests(
        self,
        interests: List[str],
        domain: str,
        weights: List[float] = None
    ) -> np.ndarray:
        """
        Generate embedding for user's research interests
        
        Creates text representation of user profile
        
        Args:
            interests: List of interest keywords
            domain: Research domain
            weights: Optional weights for each interest (0-1)
        
        Returns:
            Embedding vector representing user profile
            (384-dim for all-MiniLM, 768-dim for SPECTER2)
        """
        if weights is None:
            weights = [1.0] * len(interests)
        
        # Create weighted keyword string
        weighted_keywords = []
        for keyword, weight in zip(interests, weights):
            # Repeat keyword based on weight (1.0 weight = 3 repetitions)
            repetitions = int(weight * 3)
            weighted_keywords.extend([keyword] * repetitions)
        
        # Create text representation
        # For all-MiniLM: just use keywords
        # For SPECTER2: use paper format
        if "specter" in self.model_name.lower():
            # SPECTER2 format
            title = f"Research in {domain}"
            abstract = " ".join(weighted_keywords)
            text = f"[TITLE] {title} [ABSTRACT] {abstract}"
        else:
            # Sentence-transformer format
            text = f"{domain} research: " + " ".join(weighted_keywords)
        
        return self.embed_text(text)


# Global instance (lazy loaded)
_embedder: SPECTEREmbedder = None


def get_embedder() -> SPECTEREmbedder:
    """Get or create global SPECTER embedder instance"""
    global _embedder
    
    if _embedder is None:
        _embedder = SPECTEREmbedder()
    
    return _embedder