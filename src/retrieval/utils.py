"""
Utilities for sparse retrieval methods including indexing and search helpers.
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

from src.models.data_models import Document, Query, RetrievalResult
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class SparseIndexManager:
    """Manager for sparse retrieval indices with efficient storage and loading."""
    
    def __init__(self, index_dir: str = "./data/indices"):
        """
        Initialize index manager.
        
        Args:
            index_dir: Directory to store indices
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
    def get_index_path(self, method: str, dataset_name: str = "hq_small") -> str:
        """
        Get the path for a specific index.
        
        Args:
            method: Retrieval method name (e.g., 'bm25', 'tfidf')
            dataset_name: Name of the dataset
            
        Returns:
            Path to the index file
        """
        filename = f"{dataset_name}_{method}_index.pkl"
        return str(self.index_dir / filename)
    
    def index_exists(self, method: str, dataset_name: str = "hq_small") -> bool:
        """
        Check if an index exists for the given method and dataset.
        
        Args:
            method: Retrieval method name
            dataset_name: Name of the dataset
            
        Returns:
            True if index exists, False otherwise
        """
        index_path = self.get_index_path(method, dataset_name)
        return os.path.exists(index_path)
    
    def get_index_info(self, method: str, dataset_name: str = "hq_small") -> Optional[Dict[str, Any]]:
        """
        Get information about an existing index.
        
        Args:
            method: Retrieval method name
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with index information or None if not exists
        """
        index_path = self.get_index_path(method, dataset_name)
        info_path = index_path.replace('.pkl', '_info.json')
        
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load index info: {e}")
        
        return None
    
    def save_index_info(self, method: str, info: Dict[str, Any], 
                       dataset_name: str = "hq_small") -> None:
        """
        Save index information.
        
        Args:
            method: Retrieval method name
            info: Information dictionary to save
            dataset_name: Name of the dataset
        """
        index_path = self.get_index_path(method, dataset_name)
        info_path = index_path.replace('.pkl', '_info.json')
        
        try:
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save index info: {e}")


class SparseSearchUtils:
    """Utilities for sparse retrieval search operations."""
    
    @staticmethod
    def normalize_scores(results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Normalize retrieval scores to [0, 1] range.
        
        Args:
            results: List of retrieval results
            
        Returns:
            List of results with normalized scores
        """
        if not results:
            return results
        
        scores = [r.score for r in results]
        min_score, max_score = min(scores), max(scores)
        
        if max_score == min_score:
            # All scores are the same, set to 1.0
            for result in results:
                result.score = 1.0
        else:
            # Normalize to [0, 1]
            for result in results:
                result.score = (result.score - min_score) / (max_score - min_score)
        
        return results
    
    @staticmethod
    def filter_results_by_threshold(results: List[RetrievalResult], 
                                  threshold: float = 0.0) -> List[RetrievalResult]:
        """
        Filter results by minimum score threshold.
        
        Args:
            results: List of retrieval results
            threshold: Minimum score threshold
            
        Returns:
            Filtered list of results
        """
        return [r for r in results if r.score >= threshold]
    
    @staticmethod
    def deduplicate_results(results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Remove duplicate documents from results, keeping highest score.
        
        Args:
            results: List of retrieval results
            
        Returns:
            Deduplicated list of results
        """
        seen_docs = {}
        
        for result in results:
            doc_id = result.document_id
            if doc_id not in seen_docs or result.score > seen_docs[doc_id].score:
                seen_docs[doc_id] = result
        
        # Sort by score and reassign ranks
        deduplicated = sorted(seen_docs.values(), key=lambda x: x.score, reverse=True)
        for i, result in enumerate(deduplicated):
            result.rank = i + 1
        
        return deduplicated
    
    @staticmethod
    def merge_results(results_list: List[List[RetrievalResult]], 
                     method_name: str = "merged") -> List[RetrievalResult]:
        """
        Merge multiple result lists into a single ranked list.
        
        Args:
            results_list: List of result lists to merge
            method_name: Name for the merged method
            
        Returns:
            Merged and ranked list of results
        """
        all_results = []
        
        for results in results_list:
            all_results.extend(results)
        
        # Remove duplicates and sort by score
        merged = SparseSearchUtils.deduplicate_results(all_results)
        
        # Update method name
        for result in merged:
            result.method = method_name
        
        return merged
    
    @staticmethod
    def calculate_retrieval_metrics(results: List[RetrievalResult], 
                                  relevant_docs: List[str]) -> Dict[str, float]:
        """
        Calculate basic retrieval metrics.
        
        Args:
            results: List of retrieval results
            relevant_docs: List of relevant document IDs
            
        Returns:
            Dictionary with calculated metrics
        """
        if not results or not relevant_docs:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        retrieved_docs = [r.document_id for r in results]
        relevant_set = set(relevant_docs)
        retrieved_set = set(retrieved_docs)
        
        # Calculate metrics
        true_positives = len(relevant_set & retrieved_set)
        precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
        recall = true_positives / len(relevant_set) if relevant_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "retrieved_count": len(retrieved_set),
            "relevant_count": len(relevant_set),
            "true_positives": true_positives
        }


class TextPreprocessor:
    """Text preprocessing utilities for sparse retrieval."""
    
    def __init__(self, lowercase: bool = True, 
                 remove_punctuation: bool = True,
                 min_token_length: int = 2):
        """
        Initialize text preprocessor.
        
        Args:
            lowercase: Whether to convert text to lowercase
            remove_punctuation: Whether to remove punctuation
            min_token_length: Minimum token length to keep
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.min_token_length = min_token_length
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sparse retrieval.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation (simple approach)
        if self.remove_punctuation:
            import string
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Split into tokens and filter by length
        tokens = text.split()
        if self.min_token_length > 0:
            tokens = [t for t in tokens if len(t) >= self.min_token_length]
        
        return ' '.join(tokens)
    
    def preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """
        Preprocess a list of documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of documents with preprocessed text
        """
        processed_docs = []
        
        for doc in documents:
            processed_text = self.preprocess_text(doc.text)
            processed_doc = Document(
                id=doc.id,
                text=processed_text,
                embeddings=doc.embeddings.copy()
            )
            processed_docs.append(processed_doc)
        
        return processed_docs
    
    def preprocess_query(self, query: Query) -> Query:
        """
        Preprocess a query.
        
        Args:
            query: Query object
            
        Returns:
            Query with preprocessed text
        """
        processed_text = self.preprocess_text(query.text)
        return Query(
            id=query.id,
            text=processed_text,
            conversation_id=query.conversation_id,
            turn_number=query.turn_number
        )