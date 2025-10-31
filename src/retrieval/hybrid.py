"""
Hybrid retrieval system for combining multiple retrieval methods.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from collections import defaultdict

from src.interfaces.base import BaseRetriever
from src.models.data_models import Query, Document, RetrievalResult
from src.utils.exceptions import RetrievalError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class HybridRetriever(BaseRetriever):
    """Hybrid retriever that combines multiple retrieval methods."""
    
    def __init__(self, retrievers: List[BaseRetriever], 
                 weights: Optional[List[float]] = None,
                 fusion_method: str = "rrf"):
        """
        Initialize hybrid retriever.
        
        Args:
            retrievers: List of retriever instances to combine
            weights: Weights for each retriever (if None, equal weights)
            fusion_method: Method for combining scores ("rrf", "combsum", "weighted")
        """
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)
        self.fusion_method = fusion_method
        
        if len(self.weights) != len(self.retrievers):
            raise ValueError("Number of weights must match number of retrievers")
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
    def index_documents(self, documents: List[Document]) -> None:
        """Index documents for all retrievers."""
        logger.info(f"Indexing documents for {len(self.retrievers)} retrievers")
        
        for i, retriever in enumerate(self.retrievers):
            try:
                logger.info(f"Indexing with {retriever.get_method_name()}")
                retriever.index_documents(documents)
            except Exception as e:
                logger.error(f"Failed to index with {retriever.get_method_name()}: {e}")
                raise RetrievalError(f"Hybrid indexing failed for {retriever.get_method_name()}: {e}")
    
    def retrieve(self, query: Query, k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve documents using hybrid fusion of multiple methods.
        
        Args:
            query: Query object
            k: Number of documents to retrieve
            
        Returns:
            List of RetrievalResult objects with fused scores
        """
        try:
            # Retrieve from all methods
            all_results = []
            for i, retriever in enumerate(self.retrievers):
                try:
                    # Retrieve more documents for better fusion
                    results = retriever.retrieve(query, k=k*2)
                    all_results.append((results, self.weights[i], retriever.get_method_name()))
                except Exception as e:
                    logger.warning(f"Retrieval failed for {retriever.get_method_name()}: {e}")
                    continue
            
            if not all_results:
                raise RetrievalError("All retrieval methods failed")
            
            # Fuse results
            if self.fusion_method == "rrf":
                fused_results = self._reciprocal_rank_fusion(all_results, k)
            elif self.fusion_method == "combsum":
                fused_results = self._combsum_fusion(all_results, k)
            elif self.fusion_method == "weighted":
                fused_results = self._weighted_fusion(all_results, k)
            else:
                raise ValueError(f"Unknown fusion method: {self.fusion_method}")
            
            logger.debug(f"Hybrid retrieval returned {len(fused_results)} documents")
            return fused_results
            
        except Exception as e:
            logger.error(f"Error during hybrid retrieval: {e}")
            raise RetrievalError(f"Hybrid retrieval failed: {e}")
    
    def get_method_name(self) -> str:
        """Return the name of the hybrid method."""
        method_names = [r.get_method_name() for r in self.retrievers]
        return f"hybrid_{self.fusion_method}_{'_'.join(method_names)}"
    
    def _reciprocal_rank_fusion(self, all_results: List, k: int) -> List[RetrievalResult]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        Args:
            all_results: List of (results, weight, method_name) tuples
            k: Number of final results to return
            
        Returns:
            List of fused RetrievalResult objects
        """
        rrf_scores = defaultdict(float)
        doc_methods = defaultdict(set)
        
        # RRF constant (typically 60)
        rrf_k = 60
        
        for results, weight, method_name in all_results:
            for result in results:
                doc_id = result.document_id
                rank = result.rank
                
                # RRF formula: 1 / (k + rank)
                rrf_score = weight / (rrf_k + rank)
                rrf_scores[doc_id] += rrf_score
                doc_methods[doc_id].add(method_name)
        
        # Sort by RRF score and create results
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        fused_results = []
        for rank, (doc_id, score) in enumerate(sorted_docs, 1):
            methods = list(doc_methods[doc_id])
            result = RetrievalResult(
                document_id=doc_id,
                score=score,
                method=f"rrf_{'+'.join(methods)}",
                rank=rank
            )
            fused_results.append(result)
        
        return fused_results
    
    def _combsum_fusion(self, all_results: List, k: int) -> List[RetrievalResult]:
        """
        Combine results using CombSUM (weighted sum of normalized scores).
        
        Args:
            all_results: List of (results, weight, method_name) tuples
            k: Number of final results to return
            
        Returns:
            List of fused RetrievalResult objects
        """
        combined_scores = defaultdict(float)
        doc_methods = defaultdict(set)
        
        for results, weight, method_name in all_results:
            if not results:
                continue
                
            # Normalize scores to [0, 1] range
            scores = [r.score for r in results]
            if max(scores) > min(scores):
                min_score, max_score = min(scores), max(scores)
                normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]
            else:
                normalized_scores = [1.0] * len(scores)
            
            for result, norm_score in zip(results, normalized_scores):
                doc_id = result.document_id
                combined_scores[doc_id] += weight * norm_score
                doc_methods[doc_id].add(method_name)
        
        # Sort by combined score and create results
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        fused_results = []
        for rank, (doc_id, score) in enumerate(sorted_docs, 1):
            methods = list(doc_methods[doc_id])
            result = RetrievalResult(
                document_id=doc_id,
                score=score,
                method=f"combsum_{'+'.join(methods)}",
                rank=rank
            )
            fused_results.append(result)
        
        return fused_results
    
    def _weighted_fusion(self, all_results: List, k: int) -> List[RetrievalResult]:
        """
        Simple weighted fusion of raw scores.
        
        Args:
            all_results: List of (results, weight, method_name) tuples
            k: Number of final results to return
            
        Returns:
            List of fused RetrievalResult objects
        """
        weighted_scores = defaultdict(float)
        doc_methods = defaultdict(set)
        
        for results, weight, method_name in all_results:
            for result in results:
                doc_id = result.document_id
                weighted_scores[doc_id] += weight * result.score
                doc_methods[doc_id].add(method_name)
        
        # Sort by weighted score and create results
        sorted_docs = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        fused_results = []
        for rank, (doc_id, score) in enumerate(sorted_docs, 1):
            methods = list(doc_methods[doc_id])
            result = RetrievalResult(
                document_id=doc_id,
                score=score,
                method=f"weighted_{'+'.join(methods)}",
                rank=rank
            )
            fused_results.append(result)
        
        return fused_results