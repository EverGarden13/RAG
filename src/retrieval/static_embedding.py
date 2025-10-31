"""
Static embedding retrieval implementations using Word2Vec, GloVe, and model2vec.
"""

import os
import pickle
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
from gensim.models import KeyedVectors

from src.interfaces.base import BaseRetriever
from src.models.data_models import Query, Document, RetrievalResult
from src.utils.exceptions import RetrievalError, IndexError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class Word2VecRetriever(BaseRetriever):
    """Word2Vec retrieval implementation using gensim."""
    
    def __init__(self, model_name: str = "word2vec-google-news-300", 
                 index_path: Optional[str] = None):
        """
        Initialize Word2Vec retriever.
        
        Args:
            model_name: Name of the pre-trained Word2Vec model from gensim
            index_path: Path to save/load document embeddings
        """
        self.model_name = model_name
        self.index_path = index_path
        self.word2vec_model = None
        self.doc_embeddings = None
        self.documents = {}
        self.doc_ids = []
        
    def _load_model(self) -> None:
        """Load pre-trained Word2Vec model."""
        try:
            if self.word2vec_model is None:
                logger.info(f"Loading Word2Vec model: {self.model_name}")
                self.word2vec_model = api.load(self.model_name)
                logger.info("Word2Vec model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Word2Vec model: {e}")
            raise RetrievalError(f"Failed to load Word2Vec model: {e}")
    
    def _text_to_embedding(self, text: str) -> np.ndarray:
        """
        Convert text to embedding by averaging word vectors.
        
        Args:
            text: Input text to embed
            
        Returns:
            Document embedding as numpy array
        """
        words = text.lower().split()
        word_vectors = []
        
        for word in words:
            if word in self.word2vec_model:
                word_vectors.append(self.word2vec_model[word])
        
        if not word_vectors:
            # Return zero vector if no words found in vocabulary
            return np.zeros(self.word2vec_model.vector_size)
        
        # Average word vectors
        return np.mean(word_vectors, axis=0)
    
    def index_documents(self, documents: List[Document]) -> None:
        """
        Index documents for Word2Vec retrieval.
        
        Args:
            documents: List of Document objects to index
        """
        try:
            logger.info(f"Indexing {len(documents)} documents with Word2Vec")
            
            # Load Word2Vec model
            self._load_model()
            
            # Store documents
            self.documents = {doc.id: doc for doc in documents}
            self.doc_ids = [doc.id for doc in documents]
            
            # Generate embeddings for all documents
            embeddings = []
            for doc in documents:
                embedding = self._text_to_embedding(doc.text)
                embeddings.append(embedding)
            
            self.doc_embeddings = np.array(embeddings)
            
            # Save embeddings if path provided
            if self.index_path:
                self._save_index()
                
            logger.info("Word2Vec indexing completed successfully")
            
        except Exception as e:
            logger.error(f"Error during Word2Vec indexing: {e}")
            raise IndexError(f"Failed to index documents with Word2Vec: {e}")
    
    def retrieve(self, query: Query, k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve top-k documents for a query using Word2Vec.
        
        Args:
            query: Query object containing the search text
            k: Number of documents to retrieve
            
        Returns:
            List of RetrievalResult objects sorted by relevance score
        """
        try:
            if self.doc_embeddings is None:
                raise RetrievalError("Word2Vec index not initialized. Call index_documents first.")
            
            # Generate query embedding
            query_embedding = self._text_to_embedding(query.text).reshape(1, -1)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_embedding, self.doc_embeddings).flatten()
            
            # Get top-k results
            top_k_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for rank, idx in enumerate(top_k_indices):
                doc_id = self.doc_ids[idx]
                score = float(similarities[idx])
                
                result = RetrievalResult(
                    document_id=doc_id,
                    score=score,
                    method="word2vec",
                    rank=rank + 1
                )
                results.append(result)
            
            logger.debug(f"Word2Vec retrieved {len(results)} documents for query: {query.text[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error during Word2Vec retrieval: {e}")
            raise RetrievalError(f"Word2Vec retrieval failed: {e}")
    
    def get_method_name(self) -> str:
        """Return the name of the retrieval method."""
        return "word2vec"
    
    def _save_index(self) -> None:
        """Save document embeddings to disk."""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            index_data = {
                'doc_embeddings': self.doc_embeddings,
                'doc_ids': self.doc_ids,
                'model_name': self.model_name
            }
            
            with open(self.index_path, 'wb') as f:
                pickle.dump(index_data, f)
                
            logger.info(f"Word2Vec index saved to {self.index_path}")
            
        except Exception as e:
            logger.error(f"Error saving Word2Vec index: {e}")
            raise IndexError(f"Failed to save Word2Vec index: {e}")
    
    def load_index(self) -> bool:
        """
        Load document embeddings from disk.
        
        Returns:
            True if index loaded successfully, False otherwise
        """
        try:
            if not self.index_path or not os.path.exists(self.index_path):
                return False
            
            with open(self.index_path, 'rb') as f:
                index_data = pickle.load(f)
            
            self.doc_embeddings = index_data['doc_embeddings']
            self.doc_ids = index_data['doc_ids']
            self.model_name = index_data['model_name']
            
            # Load the Word2Vec model
            self._load_model()
            
            logger.info(f"Word2Vec index loaded from {self.index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Word2Vec index: {e}")
            return False


class GloVeRetriever(BaseRetriever):
    """GloVe retrieval implementation using gensim."""
    
    def __init__(self, model_name: str = "glove-wiki-gigaword-300", 
                 index_path: Optional[str] = None):
        """
        Initialize GloVe retriever.
        
        Args:
            model_name: Name of the pre-trained GloVe model from gensim
            index_path: Path to save/load document embeddings
        """
        self.model_name = model_name
        self.index_path = index_path
        self.glove_model = None
        self.doc_embeddings = None
        self.documents = {}
        self.doc_ids = []
        
    def _load_model(self) -> None:
        """Load pre-trained GloVe model."""
        try:
            if self.glove_model is None:
                logger.info(f"Loading GloVe model: {self.model_name}")
                self.glove_model = api.load(self.model_name)
                logger.info("GloVe model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading GloVe model: {e}")
            raise RetrievalError(f"Failed to load GloVe model: {e}")
    
    def _text_to_embedding(self, text: str) -> np.ndarray:
        """
        Convert text to embedding by averaging word vectors.
        
        Args:
            text: Input text to embed
            
        Returns:
            Document embedding as numpy array
        """
        words = text.lower().split()
        word_vectors = []
        
        for word in words:
            if word in self.glove_model:
                word_vectors.append(self.glove_model[word])
        
        if not word_vectors:
            # Return zero vector if no words found in vocabulary
            return np.zeros(self.glove_model.vector_size)
        
        # Average word vectors
        return np.mean(word_vectors, axis=0)
    
    def index_documents(self, documents: List[Document]) -> None:
        """
        Index documents for GloVe retrieval.
        
        Args:
            documents: List of Document objects to index
        """
        try:
            logger.info(f"Indexing {len(documents)} documents with GloVe")
            
            # Load GloVe model
            self._load_model()
            
            # Store documents
            self.documents = {doc.id: doc for doc in documents}
            self.doc_ids = [doc.id for doc in documents]
            
            # Generate embeddings for all documents
            embeddings = []
            for doc in documents:
                embedding = self._text_to_embedding(doc.text)
                embeddings.append(embedding)
            
            self.doc_embeddings = np.array(embeddings)
            
            # Save embeddings if path provided
            if self.index_path:
                self._save_index()
                
            logger.info("GloVe indexing completed successfully")
            
        except Exception as e:
            logger.error(f"Error during GloVe indexing: {e}")
            raise IndexError(f"Failed to index documents with GloVe: {e}")
    
    def retrieve(self, query: Query, k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve top-k documents for a query using GloVe.
        
        Args:
            query: Query object containing the search text
            k: Number of documents to retrieve
            
        Returns:
            List of RetrievalResult objects sorted by relevance score
        """
        try:
            if self.doc_embeddings is None:
                raise RetrievalError("GloVe index not initialized. Call index_documents first.")
            
            # Generate query embedding
            query_embedding = self._text_to_embedding(query.text).reshape(1, -1)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_embedding, self.doc_embeddings).flatten()
            
            # Get top-k results
            top_k_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for rank, idx in enumerate(top_k_indices):
                doc_id = self.doc_ids[idx]
                score = float(similarities[idx])
                
                result = RetrievalResult(
                    document_id=doc_id,
                    score=score,
                    method="glove",
                    rank=rank + 1
                )
                results.append(result)
            
            logger.debug(f"GloVe retrieved {len(results)} documents for query: {query.text[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error during GloVe retrieval: {e}")
            raise RetrievalError(f"GloVe retrieval failed: {e}")
    
    def get_method_name(self) -> str:
        """Return the name of the retrieval method."""
        return "glove"
    
    def _save_index(self) -> None:
        """Save document embeddings to disk."""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            index_data = {
                'doc_embeddings': self.doc_embeddings,
                'doc_ids': self.doc_ids,
                'model_name': self.model_name
            }
            
            with open(self.index_path, 'wb') as f:
                pickle.dump(index_data, f)
                
            logger.info(f"GloVe index saved to {self.index_path}")
            
        except Exception as e:
            logger.error(f"Error saving GloVe index: {e}")
            raise IndexError(f"Failed to save GloVe index: {e}")
    
    def load_index(self) -> bool:
        """
        Load document embeddings from disk.
        
        Returns:
            True if index loaded successfully, False otherwise
        """
        try:
            if not self.index_path or not os.path.exists(self.index_path):
                return False
            
            with open(self.index_path, 'rb') as f:
                index_data = pickle.load(f)
            
            self.doc_embeddings = index_data['doc_embeddings']
            self.doc_ids = index_data['doc_ids']
            self.model_name = index_data['model_name']
            
            # Load the GloVe model
            self._load_model()
            
            logger.info(f"GloVe index loaded from {self.index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading GloVe index: {e}")
            return False


class Model2VecRetriever(BaseRetriever):
    """Model2Vec retrieval implementation for efficient static embeddings."""
    
    def __init__(self, model_name: str = "minishlab/potion-base-8M", 
                 index_path: Optional[str] = None):
        """
        Initialize Model2Vec retriever.
        
        Args:
            model_name: Name of the pre-trained Model2Vec model
            index_path: Path to save/load document embeddings
        """
        self.model_name = model_name
        self.index_path = index_path
        self.model2vec_model = None
        self.doc_embeddings = None
        self.documents = {}
        self.doc_ids = []
        
    def _load_model(self) -> None:
        """Load pre-trained Model2Vec model."""
        try:
            if self.model2vec_model is None:
                logger.info(f"Loading Model2Vec model: {self.model_name}")
                try:
                    from model2vec import StaticModel
                    self.model2vec_model = StaticModel.from_pretrained(self.model_name)
                    logger.info("Model2Vec model loaded successfully")
                except ImportError:
                    logger.warning("model2vec library not available, falling back to Word2Vec")
                    # Fallback to Word2Vec if model2vec is not available
                    self.model2vec_model = api.load("word2vec-google-news-300")
                    logger.info("Fallback to Word2Vec model loaded")
        except Exception as e:
            logger.error(f"Error loading Model2Vec model: {e}")
            raise RetrievalError(f"Failed to load Model2Vec model: {e}")
    
    def _text_to_embedding(self, text: str) -> np.ndarray:
        """
        Convert text to embedding using Model2Vec.
        
        Args:
            text: Input text to embed
            
        Returns:
            Document embedding as numpy array
        """
        try:
            # Try Model2Vec encoding first
            if hasattr(self.model2vec_model, 'encode'):
                embedding = self.model2vec_model.encode([text])[0]
                return np.array(embedding)
            else:
                # Fallback to word averaging for Word2Vec
                words = text.lower().split()
                word_vectors = []
                
                for word in words:
                    if word in self.model2vec_model:
                        word_vectors.append(self.model2vec_model[word])
                
                if not word_vectors:
                    # Return zero vector if no words found
                    return np.zeros(self.model2vec_model.vector_size)
                
                # Average word vectors
                return np.mean(word_vectors, axis=0)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(300)  # Default dimension
    
    def index_documents(self, documents: List[Document]) -> None:
        """
        Index documents for Model2Vec retrieval.
        
        Args:
            documents: List of Document objects to index
        """
        try:
            logger.info(f"Indexing {len(documents)} documents with Model2Vec")
            
            # Load Model2Vec model
            self._load_model()
            
            # Store documents
            self.documents = {doc.id: doc for doc in documents}
            self.doc_ids = [doc.id for doc in documents]
            
            # Generate embeddings for all documents
            embeddings = []
            for doc in documents:
                embedding = self._text_to_embedding(doc.text)
                embeddings.append(embedding)
            
            self.doc_embeddings = np.array(embeddings)
            
            # Save embeddings if path provided
            if self.index_path:
                self._save_index()
                
            logger.info("Model2Vec indexing completed successfully")
            
        except Exception as e:
            logger.error(f"Error during Model2Vec indexing: {e}")
            raise IndexError(f"Failed to index documents with Model2Vec: {e}")
    
    def retrieve(self, query: Query, k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve top-k documents for a query using Model2Vec.
        
        Args:
            query: Query object containing the search text
            k: Number of documents to retrieve
            
        Returns:
            List of RetrievalResult objects sorted by relevance score
        """
        try:
            if self.doc_embeddings is None:
                raise RetrievalError("Model2Vec index not initialized. Call index_documents first.")
            
            # Generate query embedding
            query_embedding = self._text_to_embedding(query.text).reshape(1, -1)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_embedding, self.doc_embeddings).flatten()
            
            # Get top-k results
            top_k_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for rank, idx in enumerate(top_k_indices):
                doc_id = self.doc_ids[idx]
                score = float(similarities[idx])
                
                result = RetrievalResult(
                    document_id=doc_id,
                    score=score,
                    method="model2vec",
                    rank=rank + 1
                )
                results.append(result)
            
            logger.debug(f"Model2Vec retrieved {len(results)} documents for query: {query.text[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error during Model2Vec retrieval: {e}")
            raise RetrievalError(f"Model2Vec retrieval failed: {e}")
    
    def get_method_name(self) -> str:
        """Return the name of the retrieval method."""
        return "model2vec"
    
    def _save_index(self) -> None:
        """Save document embeddings to disk."""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            index_data = {
                'doc_embeddings': self.doc_embeddings,
                'doc_ids': self.doc_ids,
                'model_name': self.model_name
            }
            
            with open(self.index_path, 'wb') as f:
                pickle.dump(index_data, f)
                
            logger.info(f"Model2Vec index saved to {self.index_path}")
            
        except Exception as e:
            logger.error(f"Error saving Model2Vec index: {e}")
            raise IndexError(f"Failed to save Model2Vec index: {e}")
    
    def load_index(self) -> bool:
        """
        Load document embeddings from disk.
        
        Returns:
            True if index loaded successfully, False otherwise
        """
        try:
            if not self.index_path or not os.path.exists(self.index_path):
                return False
            
            with open(self.index_path, 'rb') as f:
                index_data = pickle.load(f)
            
            self.doc_embeddings = index_data['doc_embeddings']
            self.doc_ids = index_data['doc_ids']
            self.model_name = index_data['model_name']
            
            # Load the Model2Vec model
            self._load_model()
            
            logger.info(f"Model2Vec index loaded from {self.index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Model2Vec index: {e}")
            return False