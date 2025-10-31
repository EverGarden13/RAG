"""
Evaluation metrics for RAG system.
Implements Exact Match (EM) and nDCG@10 metrics.
"""

import re
import string
from typing import List, Dict, Any, Tuple
import numpy as np

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ExactMatchEvaluator:
    """Evaluator for Exact Match (EM) metric."""
    
    @staticmethod
    def normalize_answer(text: str) -> str:
        """
        Normalize answer text for comparison.
        
        Args:
            text: Answer text
            
        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove articles
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    @classmethod
    def compute_em(cls, prediction: str, reference: str) -> float:
        """
        Compute Exact Match score.
        
        Args:
            prediction: Predicted answer
            reference: Reference answer
            
        Returns:
            EM score (1.0 or 0.0)
        """
        pred_norm = cls.normalize_answer(prediction)
        ref_norm = cls.normalize_answer(reference)
        
        return 1.0 if pred_norm == ref_norm else 0.0
    
    @classmethod
    def compute_em_batch(cls, predictions: List[str], 
                        references: List[str]) -> Tuple[float, List[float]]:
        """
        Compute EM for a batch of predictions.
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            
        Returns:
            Tuple of (average_em, individual_scores)
        """
        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must match")
        
        scores = []
        for pred, ref in zip(predictions, references):
            score = cls.compute_em(pred, ref)
            scores.append(score)
        
        avg_score = np.mean(scores) if scores else 0.0
        
        logger.info(f"EM Score: {avg_score:.4f} ({sum(scores)}/{len(scores)} exact matches)")
        return avg_score, scores


class NDCGEvaluator:
    """Evaluator for Normalized Discounted Cumulative Gain (nDCG@k)."""
    
    @staticmethod
    def compute_dcg(relevances: List[float], k: int = 10) -> float:
        """
        Compute Discounted Cumulative Gain.
        
        Args:
            relevances: List of relevance scores (binary or graded)
            k: Cutoff position
            
        Returns:
            DCG score
        """
        relevances = relevances[:k]
        dcg = 0.0
        
        for i, rel in enumerate(relevances, 1):
            dcg += rel / np.log2(i + 1)
        
        return dcg
    
    @classmethod
    def compute_ndcg(cls, retrieved_ids: List[str], 
                    relevant_ids: List[str],
                    k: int = 10) -> float:
        """
        Compute Normalized Discounted Cumulative Gain at k.
        
        Args:
            retrieved_ids: List of retrieved document IDs (in rank order)
            relevant_ids: List of relevant document IDs
            k: Cutoff position
            
        Returns:
            nDCG@k score
        """
        # Create relevance list (1 if relevant, 0 otherwise)
        relevances = [1.0 if doc_id in relevant_ids else 0.0 
                     for doc_id in retrieved_ids[:k]]
        
        # Compute DCG
        dcg = cls.compute_dcg(relevances, k)
        
        # Compute IDCG (ideal DCG with all relevant docs at top)
        ideal_relevances = [1.0] * min(len(relevant_ids), k)
        idcg = cls.compute_dcg(ideal_relevances, k)
        
        # Compute nDCG
        if idcg == 0.0:
            return 0.0
        
        ndcg = dcg / idcg
        return ndcg
    
    @classmethod
    def compute_ndcg_batch(cls, retrieved_docs_list: List[List[str]],
                          relevant_docs_list: List[List[str]],
                          k: int = 10) -> Tuple[float, List[float]]:
        """
        Compute nDCG@k for a batch of queries.
        
        Args:
            retrieved_docs_list: List of retrieved document ID lists
            relevant_docs_list: List of relevant document ID lists
            k: Cutoff position
            
        Returns:
            Tuple of (average_ndcg, individual_scores)
        """
        if len(retrieved_docs_list) != len(relevant_docs_list):
            raise ValueError("Number of retrieved and relevant lists must match")
        
        scores = []
        for retrieved, relevant in zip(retrieved_docs_list, relevant_docs_list):
            score = cls.compute_ndcg(retrieved, relevant, k)
            scores.append(score)
        
        avg_score = np.mean(scores) if scores else 0.0
        
        logger.info(f"nDCG@{k} Score: {avg_score:.4f}")
        return avg_score, scores


class RAGEvaluator:
    """Combined evaluator for RAG system."""
    
    def __init__(self, k: int = 10):
        """
        Initialize RAG evaluator.
        
        Args:
            k: Cutoff position for nDCG
        """
        self.k = k
        self.em_evaluator = ExactMatchEvaluator()
        self.ndcg_evaluator = NDCGEvaluator()
        
    def evaluate(self, predictions: List[Dict[str, Any]],
                references: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate RAG system predictions.
        
        Args:
            predictions: List of prediction dicts with 'id', 'answer', 'retrieved_docs'
            references: List of reference dicts with 'id', 'answer', 'supporting_ids'
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {len(predictions)} predictions...")
        
        # Match predictions to references by ID
        pred_dict = {p['id']: p for p in predictions}
        ref_dict = {r['id']: r for r in references}
        
        # Get common IDs
        common_ids = set(pred_dict.keys()) & set(ref_dict.keys())
        
        if not common_ids:
            logger.warning("No matching IDs between predictions and references")
            return {
                'em': 0.0,
                'ndcg@10': 0.0,
                'num_evaluated': 0
            }
        
        logger.info(f"Evaluating {len(common_ids)} matching predictions")
        
        # Prepare data for evaluation
        pred_answers = []
        ref_answers = []
        retrieved_docs_list = []
        relevant_docs_list = []
        
        for query_id in common_ids:
            pred = pred_dict[query_id]
            ref = ref_dict[query_id]
            
            pred_answers.append(pred.get('answer', ''))
            ref_answers.append(ref.get('answer', ''))
            
            # Extract retrieved document IDs
            retrieved_docs = pred.get('retrieved_docs', [])
            if retrieved_docs and isinstance(retrieved_docs[0], list):
                # Format: [[id, score], ...]
                retrieved_ids = [doc[0] for doc in retrieved_docs]
            else:
                # Format: [id, ...]
                retrieved_ids = retrieved_docs
            
            retrieved_docs_list.append(retrieved_ids)
            relevant_docs_list.append(ref.get('supporting_ids', []))
        
        # Compute EM
        em_score, em_scores = self.em_evaluator.compute_em_batch(
            pred_answers, ref_answers
        )
        
        # Compute nDCG@k
        ndcg_score, ndcg_scores = self.ndcg_evaluator.compute_ndcg_batch(
            retrieved_docs_list, relevant_docs_list, k=self.k
        )
        
        # Compute combined score (equal weight)
        combined_score = (em_score + ndcg_score) / 2.0
        
        results = {
            'em': em_score,
            f'ndcg@{self.k}': ndcg_score,
            'combined': combined_score,
            'num_evaluated': len(common_ids),
            'em_scores': em_scores,
            'ndcg_scores': ndcg_scores
        }
        
        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Exact Match (EM):     {em_score:.4f}")
        logger.info(f"nDCG@{self.k}:              {ndcg_score:.4f}")
        logger.info(f"Combined Score:       {combined_score:.4f}")
        logger.info(f"Queries Evaluated:    {len(common_ids)}")
        logger.info("=" * 60)
        
        return results
    
    def evaluate_from_files(self, prediction_file: str,
                           reference_file: str) -> Dict[str, Any]:
        """
        Evaluate from JSONL files.
        
        Args:
            prediction_file: Path to predictions JSONL file
            reference_file: Path to references JSONL file
            
        Returns:
            Dictionary of evaluation metrics
        """
        import json
        
        # Load predictions
        predictions = []
        with open(prediction_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    predictions.append(json.loads(line))
        
        # Load references
        references = []
        with open(reference_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    references.append(json.loads(line))
        
        logger.info(f"Loaded {len(predictions)} predictions and {len(references)} references")
        
        return self.evaluate(predictions, references)
