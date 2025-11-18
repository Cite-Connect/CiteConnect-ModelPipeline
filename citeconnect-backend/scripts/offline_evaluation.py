#!/usr/bin/env python3

"""
Offline Evaluation Using Citation-Based Ground Truth

Implements the strategy from KT document:
1. Treat each paper as a "user"
2. Use paper's abstract as "query"  
3. Use paper's bibliography (references) as ground truth
4. Measure how many cited papers appear in recommendations

This is the standard evaluation method for paper recommendation systems
(used in SPECTER, CitationBERT, etc.)

Run: python scripts/offline_evaluation.py
"""

import asyncio
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import ast
from typing import List, Set, Dict
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.embedding import get_embedder
from app.utils.similarity import cosine_similarity_batch, calculate_composite_score


class OfflineEvaluator:
    """
    Offline evaluation using paper-to-paper citation prediction
    
    Ground Truth: Papers in the bibliography (references from pickle)
    Data Source: embeddings_db.pkl only (no CSV needed)
    """
    
    def __init__(self, pickle_path: str):
        """
        Initialize evaluator
        
        Args:
            pickle_path: Path to embeddings pickle
        """
        self.pickle_path = Path(pickle_path)
        self.embedder = get_embedder()
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load pickle data"""
        print("Loading data from pickle...")
        
        # Load pickle with embeddings
        with open(self.pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        self.chunks = data['chunks']
        self.embeddings = data['embeddings']
        
        print(f"✓ Loaded {len(self.chunks)} chunks with embeddings")
        
        # Create paper_id → data mappings
        self.paper_embeddings = {}
        self.paper_metadata = {}
        self.paper_references = {}
        
        for chunk, embedding in zip(self.chunks, self.embeddings):
            paper_id = chunk['paper_id']
            
            # Store embedding (use first chunk's embedding as paper embedding)
            if paper_id not in self.paper_embeddings:
                self.paper_embeddings[paper_id] = embedding
                
                # Store metadata
                self.paper_metadata[paper_id] = {
                    'title': chunk.get('paper_title', 'Unknown'),
                    'year': chunk.get('paper_year', 0),
                    'citation_count': chunk.get('citation_count', 0),
                    'text': chunk.get('text', '')
                }
                
                # Store references (bibliography - ground truth)
                refs = chunk.get('references', [])
                self.paper_references[paper_id] = set(refs) if isinstance(refs, list) else set()
        
        print(f"✓ Processed {len(self.paper_embeddings)} unique papers")
        print(f"✓ Papers with bibliographies: {sum(1 for refs in self.paper_references.values() if refs)}\n")
    
    def get_ground_truth(self, paper_id: str) -> Set[str]:
        """
        Get ground truth for a paper (its bibliography)
        
        Args:
            paper_id: Paper ID
        
        Returns:
            Set of paper IDs this paper cites (references)
        """
        return self.paper_references.get(paper_id, set())
    
    def generate_recommendations_for_paper(
        self,
        seed_paper_id: str,
        top_k: int = 10,
        use_abstract: bool = True
    ) -> List[str]:
        """
        Generate recommendations for a paper (treating it as user)
        
        Args:
            seed_paper_id: Paper to generate recommendations for
            top_k: Number of recommendations
            use_abstract: If True, use abstract text from chunk
        
        Returns:
            List of recommended paper IDs (ordered by score)
        """
        # Get seed paper metadata
        if seed_paper_id not in self.paper_metadata:
            raise ValueError(f"Paper not found: {seed_paper_id}")
        
        seed_metadata = self.paper_metadata[seed_paper_id]
        
        # Get query text (abstract from chunk)
        query_text = seed_metadata['text']
        
        if not query_text:
            raise ValueError(f"No text for paper: {seed_paper_id}")
        
        # Generate embedding for query
        query_embedding = self.embedder.embed_text(query_text)
        
        # Get all paper embeddings (except seed paper itself)
        candidate_papers = [
            (pid, emb) for pid, emb in self.paper_embeddings.items()
            if pid != seed_paper_id
        ]
        
        if not candidate_papers:
            return []
        
        paper_ids, embeddings_list = zip(*candidate_papers)
        embeddings_matrix = np.array(embeddings_list)
        
        # Compute similarities
        similarities = cosine_similarity_batch(query_embedding, embeddings_matrix)
        
        # Rank by similarity
        ranked_indices = np.argsort(similarities)[::-1]  # Descending
        
        # Return top K paper IDs
        recommended_ids = [paper_ids[i] for i in ranked_indices[:top_k]]
        
        return recommended_ids
    
    def evaluate_paper(
        self,
        seed_paper_id: str,
        top_k: int = 10,
        use_abstract: bool = True
    ) -> Dict:
        """
        Evaluate recommendations for a single paper
        
        Args:
            seed_paper_id: Paper to evaluate
            top_k: Number of recommendations to generate
            use_abstract: Use abstract (True) or intro (False)
        
        Returns:
            Evaluation metrics
        """
        # Generate recommendations
        recommendations = self.generate_recommendations_for_paper(
            seed_paper_id,
            top_k=top_k,
            use_abstract=use_abstract
        )
        
        # Get ground truth (bibliography)
        ground_truth = self.get_ground_truth(seed_paper_id)
        
        if not ground_truth:
            return {
                'paper_id': seed_paper_id,
                'precision_at_k': 0.0,
                'recall_at_k': 0.0,
                'mrr': 0.0,
                'ground_truth_size': 0,
                'note': 'No bibliography available'
            }
        
        # Calculate metrics
        recommended_set = set(recommendations)
        hits = len(recommended_set.intersection(ground_truth))
        
        precision = hits / top_k if top_k > 0 else 0.0
        recall = hits / len(ground_truth) if ground_truth else 0.0
        
        # MRR
        mrr = 0.0
        for i, paper_id in enumerate(recommendations):
            if paper_id in ground_truth:
                mrr = 1.0 / (i + 1)
                break
        
        return {
            'paper_id': seed_paper_id,
            'precision_at_k': precision,
            'recall_at_k': recall,
            'mrr': mrr,
            'hits': hits,
            'ground_truth_size': len(ground_truth),
            'recommendations': recommendations[:5]  # Top 5 for inspection
        }
    
    def evaluate_all_papers(self, sample_size: int = None) -> Dict:
        """
        Evaluate on all papers with bibliographies
        
        Args:
            sample_size: If set, evaluate random sample; else all papers
        
        Returns:
            Aggregated evaluation metrics
        """
        print("="*80)
        print("  Offline Evaluation: Paper-to-Paper Citation Prediction")
        print("="*80 + "\n")
        
        # Filter papers with references
        papers_with_refs = [
            paper_id for paper_id, refs in self.paper_references.items()
            if len(refs) > 0
        ]
        
        print(f"Papers with bibliographies: {len(papers_with_refs)}")
        print(f"Papers in database: {len(self.paper_embeddings)}")
        
        if not papers_with_refs:
            print("\n✗ No papers with bibliographies found!")
            print("  Cannot perform evaluation without ground truth.")
            return {}
        
        if sample_size and sample_size < len(papers_with_refs):
            import random
            papers_to_eval = random.sample(papers_with_refs, sample_size)
            print(f"Evaluating random sample: {sample_size} papers\n")
        else:
            papers_to_eval = papers_with_refs
            print(f"Evaluating all {len(papers_to_eval)} papers\n")
        
        # Evaluate each paper
        results = []
        
        for i, paper_id in enumerate(papers_to_eval, 1):
            metadata = self.paper_metadata[paper_id]
            title = metadata['title']
            
            if len(title) > 60:
                title = title[:60] + "..."
            
            print(f"[{i}/{len(papers_to_eval)}] {title}")
            
            try:
                metrics = self.evaluate_paper(paper_id, top_k=10)
                results.append(metrics)
                
                print(f"      Precision@10: {metrics['precision_at_k']:.3f}")
                print(f"      Recall@10: {metrics['recall_at_k']:.3f}")
                print(f"      MRR: {metrics['mrr']:.3f}")
                print(f"      Hits: {metrics['hits']}/{metrics['ground_truth_size']}\n")
                
            except Exception as e:
                print(f"      ✗ Error: {str(e)}\n")
        
        # Aggregate results
        if results:
            avg_precision = np.mean([r['precision_at_k'] for r in results])
            avg_recall = np.mean([r['recall_at_k'] for r in results])
            avg_mrr = np.mean([r['mrr'] for r in results])
            avg_hits = np.mean([r['hits'] for r in results])
            avg_bib_size = np.mean([r['ground_truth_size'] for r in results])
            
            print("="*80)
            print("  Aggregate Results")
            print("="*80 + "\n")
            
            print(f"Papers evaluated: {len(results)}")
            print(f"Average bibliography size: {avg_bib_size:.1f}")
            print(f"\nMetrics:")
            print(f"  Precision@10: {avg_precision:.3f} {'✓' if avg_precision >= 0.60 else '✗'} (target: ≥0.60)")
            print(f"  Recall@10: {avg_recall:.3f} {'✓' if avg_recall >= 0.75 else '✗'} (target: ≥0.75)")
            print(f"  MRR: {avg_mrr:.3f} {'✓' if avg_mrr >= 0.70 else '✗'} (target: ≥0.70)")
            print(f"  Average hits: {avg_hits:.1f}/10 recommendations\n")
            
            print("Note: Citation prediction metrics are typically lower than user-based metrics.")
            print("      Precision 0.10-0.25 is research-grade for citation prediction (SPECTER baseline).")
            print("      MRR > 0.40 indicates good ranking quality.\n")
            
            return {
                'num_papers_evaluated': len(results),
                'avg_bibliography_size': avg_bib_size,
                'avg_precision_at_10': avg_precision,
                'avg_recall_at_10': avg_recall,
                'avg_mrr': avg_mrr,
                'avg_hits': avg_hits,
                'all_results': results
            }
        else:
            print("✗ No papers evaluated successfully\n")
            return {}


async def main():
    """Run offline evaluation"""
    
    # Path to pickle (no CSV needed!)
    pickle_path = "working_data/embeddings_db.pkl"
    
    # Check pickle exists
    if not Path(pickle_path).exists():
        print(f"✗ Pickle not found: {pickle_path}")
        print(f"  Make sure embeddings_db.pkl is in working_data/")
        return
    
    # Create evaluator (only needs pickle now!)
    evaluator = OfflineEvaluator(pickle_path)
    
    # Run evaluation on all papers with bibliographies
    results = evaluator.evaluate_all_papers(sample_size=None)
    
    # Save results
    if results:
        import json
        output_path = "offline_evaluation_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✓ Results saved to: {output_path}")
        print(f"\nNext steps:")
        print(f"  1. Review results in offline_evaluation_results.json")
        print(f"  2. Import these metrics into MLflow: python scripts/run_experiment.py")
        print(f"  3. View in MLflow UI: mlflow ui\n")


if __name__ == "__main__":
    asyncio.run(main())