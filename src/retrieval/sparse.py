"""
Sparse retrieval implementations using BM25 and TF-IDF.
"""

import os
import pickle
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import bm25s

from src.interfaces.base import BaseRetriever
from src.models.data_models import Query, Document, RetrievalResult
from src.utils.exceptions import RetrievalError, IndexError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class BM25Retriever(BaseRetriever):
    """BM25 retrieval implementation using bm25s library."""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75, index_path: Optional[str] = None):
        """
        Initialize BM25 retriever.
        
        Args:
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (length normalization)
            index_path: Path to save/load BM25 index
        """
        self.k1 = k1
        self.b = b
        self.index_path = index_path
        self.bm25_index = None
        self.documents = {}
        self.doc_ids = []
        
    def index_documents(self, documents: List[Document]) -> None:
        """
        Index documents for BM25 retrieval.
        
        Args:
            documents: List of Document objects to index
        """
        try:
            logger.info(f"Indexing {len(documents)} documents with BM25")
            
            # Store documents and extract texts
            self.documents = {doc.id: doc for doc in documents}
            self.doc_ids = [doc.id for doc in documents]
            doc_texts = [doc.text for doc in documents]
            
            # Create BM25 index
            self.bm25_index = bm25s.BM25(k1=self.k1, b=self.b)
            
            # Tokenize documents (simple whitespace tokenization)
            tokenized_docs = [text.lower().split() for text in doc_texts]
            
            # Index the documents
            self.bm25_index.index(tokenized_docs)
            
            # Save index if path provided
            if self.index_path:
                self._save_index()
                
            logger.info("BM25 indexing completed successfully")
            
        except Exception as e:
            logger.error(f"Error during BM25 indexing: {e}")
            raise IndexError(f"Failed to index documents with BM25: {e}")
    
    def retrieve(self, query: Query, k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve top-k documents for a query using BM25.
        
        Args:
            query: Query object containing the search text
            k: Number of documents to retrieve
            
        Returns:
            List of RetrievalResult objects sorted by relevance score
        """
        try:
            if self.bm25_index is None:
                raise RetrievalError("BM25 index not initialized. Call index_documents first.")
            
            # Tokenize query
            tokenized_query = query.text.lower().split()
            
            # Get scores for all documents
            scores = self.bm25_index.get_scores(tokenized_query)
            
            # Get top-k results
            top_k_indices = np.argsort(scores)[::-1][:k]
            
            results = []
            for rank, idx in enumerate(top_k_indices):
                doc_id = self.doc_ids[idx]
                score = float(scores[idx])
                
                result = RetrievalResult(
                    document_id=doc_id,
                    score=score,
                    method="bm25",
                    rank=rank + 1
                )
                results.append(result)
            
            logger.debug(f"BM25 retrieved {len(results)} documents for query: {query.text[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error during BM25 retrieval: {e}")
            raise RetrievalError(f"BM25 retrieval failed: {e}")
    
    def get_method_name(self) -> str:
        """Return the name of the retrieval method."""
        return "bm25"
    
    def _save_index(self) -> None:
        """Save BM25 index to disk."""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            index_data = {
                'bm25_index': self.bm25_index,
                'doc_ids': self.doc_ids,
                'k1': self.k1,
                'b': self.b
            }
            
            with open(self.index_path, 'wb') as f:
                pickle.dump(index_data, f)
                
            logger.info(f"BM25 index saved to {self.index_path}")
            
        except Exception as e:
            logger.error(f"Error saving BM25 index: {e}")
            raise IndexError(f"Failed to save BM25 index: {e}")
    
    def load_index(self) -> bool:
        """
        Load BM25 index from disk.
        
        Returns:
            True if index loaded successfully, False otherwise
        """
        try:
            if not self.index_path or not os.path.exists(self.index_path):
                return False
            
            with open(self.index_path, 'rb') as f:
                index_data = pickle.load(f)
            
            self.bm25_index = index_data['bm25_index']
            self.doc_ids = index_data['doc_ids']
            self.k1 = index_data['k1']
            self.b = index_data['b']
            
            logger.info(f"BM25 index loaded from {self.index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading BM25 index: {e}")
            return False


class TFIDFRetriever(BaseRetriever):
    """TF-IDF retrieval implementation using scikit-learn."""
    
    def __init__(self, max_features: Optional[int] = None, 
                 ngram_range: tuple = (1, 2),
                 index_path: Optional[str] = None):
        """
        Initialize TF-IDF retriever.
        
        Args:
            max_features: Maximum number of features to use
            ngram_range: Range of n-grams to extract
            index_path: Path to save/load TF-IDF index
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.index_path = index_path
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
        )
        
        self.doc_vectors = None
        self.documents = {}
        self.doc_ids = []
    
    def index_documents(self, documents: List[Document]) -> None:
        """
        Index documents for TF-IDF retrieval.
        
        Args:
            documents: List of Document objects to index
        """
        try:
            logger.info(f"Indexing {len(documents)} documents with TF-IDF")
            
            # Store documents and extract texts
            self.documents = {doc.id: doc for doc in documents}
            self.doc_ids = [doc.id for doc in documents]
            doc_texts = [doc.text for doc in documents]
            
            # Fit vectorizer and transform documents
            self.doc_vectors = self.vectorizer.fit_transform(doc_texts)
            
            # Save index if path provided
            if self.index_path:
                self._save_index()
                
            logger.info("TF-IDF indexing completed successfully")
            
        except Exception as e:
            logger.error(f"Error during TF-IDF indexing: {e}")
            raise IndexError(f"Failed to index documents with TF-IDF: {e}")
    
    def retrieve(self, query: Query, k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve top-k documents for a query using TF-IDF.
        
        Args:
            query: Query object containing the search text
            k: Number of documents to retrieve
            
        Returns:
            List of RetrievalResult objects sorted by relevance score
        """
        try:
            if self.doc_vectors is None:
                raise RetrievalError("TF-IDF index not initialized. Call index_documents first.")
            
            # Transform query to vector
            query_vector = self.vectorizer.transform([query.text])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
            
            # Get top-k results
            top_k_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for rank, idx in enumerate(top_k_indices):
                doc_id = self.doc_ids[idx]
                score = float(similarities[idx])
                
                result = RetrievalResult(
                    document_id=doc_id,
                    score=score,
                    method="tfidf",
                    rank=rank + 1
                )
                results.append(result)
            
            logger.debug(f"TF-IDF retrieved {len(results)} documents for query: {query.text[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error during TF-IDF retrieval: {e}")
            raise RetrievalError(f"TF-IDF retrieval failed: {e}")
    
    def get_method_name(self) -> str:
        """Return the name of the retrieval method."""
        return "tfidf"
    
    def _save_index(self) -> None:
        """Save TF-IDF index to disk."""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            index_data = {
                'vectorizer': self.vectorizer,
                'doc_vectors': self.doc_vectors,
                'doc_ids': self.doc_ids,
                'max_features': self.max_features,
                'ngram_range': self.ngram_range
            }
            
            with open(self.index_path, 'wb') as f:
                pickle.dump(index_data, f)
                
            logger.info(f"TF-IDF index saved to {self.index_path}")
            
        except Exception as e:
            logger.error(f"Error saving TF-IDF index: {e}")
            raise IndexError(f"Failed to save TF-IDF index: {e}")
    
    def load_index(self) -> bool:
        """
        Load TF-IDF index from disk.
        
        Returns:
            True if index loaded successfully, False otherwise
        """
        try:
            if not self.index_path or not os.path.exists(self.index_path):
                return False
            
            with open(self.index_path, 'rb') as f:
                index_data = pickle.load(f)
            
            self.vectorizer = index_data['vectorizer']
            self.doc_vectors = index_data['doc_vectors']
            self.doc_ids = index_data['doc_ids']
            self.max_features = index_data['max_features']
            self.ngram_range = index_data['ngram_range']
            
            logger.info(f"TF-IDF index loaded from {self.index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading TF-IDF index: {e}")
            return False