"""
Dense retrieval implementations using HuggingFace models.
Includes E5, BGE, and GTE models with FAISS indexing for efficient search.
"""

import os
import pickle
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.interfaces.base import BaseRetriever
from src.models.data_models import Query, Document, RetrievalResult
from src.utils.exceptions import RetrievalError, IndexError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class E5Retriever(BaseRetriever):
    """E5 model retrieval implementation using sentence-transformers and FAISS."""
    
    def __init__(self, model_name: str = "intfloat/e5-base-v2", 
                 index_path: Optional[str] = None,
                 batch_size: int = 32):
        """
        Initialize E5 retriever.
        
        Args:
            model_name: HuggingFace model name (e5-base-v2 or e5-large-v2)
            index_path: Path to save/load FAISS index and embeddings
            batch_size: Batch size for embedding generation
        """
        self.model_name = model_name
        self.index_path = index_path
        self.batch_size = batch_size
        
        self.model = None
        self.faiss_index = None
        self.doc_embeddings = None
        self.documents = {}
        self.doc_ids = []
        
    def _load_model(self) -> None:
        """Load E5 model from HuggingFace."""
        try:
            if self.model is None:
                logger.info(f"Loading E5 model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                logger.info("E5 model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading E5 model: {e}")
            raise RetrievalError(f"Failed to load E5 model: {e}")
    
    def _preprocess_text(self, text: str, is_query: bool = False) -> str:
        """
        Preprocess text for E5 models with required prefixes.
        
        Args:
            text: Input text to preprocess
            is_query: Whether the text is a query (True) or document (False)
            
        Returns:
            Preprocessed text with appropriate prefix
        """
        # E5 models require specific prefixes for optimal performance
        if is_query:
            return f"query: {text}"
        else:
            return f"passage: {text}"
    
    def _generate_embeddings(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            is_query: Whether texts are queries or documents
            
        Returns:
            Numpy array of embeddings
        """
        try:
            # Preprocess texts with appropriate prefixes
            processed_texts = [self._preprocess_text(text, is_query) for text in texts]
            
            # Generate embeddings in batches
            embeddings = []
            for i in range(0, len(processed_texts), self.batch_size):
                batch = processed_texts[i:i + self.batch_size]
                batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
                embeddings.extend(batch_embeddings)
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise RetrievalError(f"Failed to generate embeddings: {e}")
    
    def index_documents(self, documents: List[Document]) -> None:
        """
        Index documents for E5 retrieval with FAISS.
        
        Args:
            documents: List of Document objects to index
        """
        try:
            logger.info(f"Indexing {len(documents)} documents with E5")
            
            # Load E5 model
            self._load_model()
            
            # Store documents
            self.documents = {doc.id: doc for doc in documents}
            self.doc_ids = [doc.id for doc in documents]
            doc_texts = [doc.text for doc in documents]
            
            # Generate embeddings for all documents
            logger.info("Generating document embeddings...")
            self.doc_embeddings = self._generate_embeddings(doc_texts, is_query=False)
            
            # Create FAISS index
            embedding_dim = self.doc_embeddings.shape[1]
            logger.info(f"Creating FAISS index with dimension {embedding_dim}")
            
            # Use IndexFlatIP for cosine similarity (inner product after normalization)
            self.faiss_index = faiss.IndexFlatIP(embedding_dim)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.doc_embeddings)
            
            # Add embeddings to FAISS index
            self.faiss_index.add(self.doc_embeddings)
            
            # Save index if path provided
            if self.index_path:
                self._save_index()
                
            logger.info("E5 indexing completed successfully")
            
        except Exception as e:
            logger.error(f"Error during E5 indexing: {e}")
            raise IndexError(f"Failed to index documents with E5: {e}")
    
    def retrieve(self, query: Query, k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve top-k documents for a query using E5.
        
        Args:
            query: Query object containing the search text
            k: Number of documents to retrieve
            
        Returns:
            List of RetrievalResult objects sorted by relevance score
        """
        try:
            if self.faiss_index is None:
                raise RetrievalError("E5 index not initialized. Call index_documents first.")
            
            # Generate query embedding
            query_embedding = self._generate_embeddings([query.text], is_query=True)
            
            # Normalize query embedding for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search FAISS index
            scores, indices = self.faiss_index.search(query_embedding, k)
            
            # Convert results to RetrievalResult objects
            results = []
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
                if idx < len(self.doc_ids):  # Valid index
                    doc_id = self.doc_ids[idx]
                    
                    result = RetrievalResult(
                        document_id=doc_id,
                        score=float(score),
                        method=self.get_method_name(),
                        rank=rank + 1
                    )
                    results.append(result)
            
            logger.debug(f"E5 retrieved {len(results)} documents for query: {query.text[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error during E5 retrieval: {e}")
            raise RetrievalError(f"E5 retrieval failed: {e}")
    
    def get_method_name(self) -> str:
        """Return the name of the retrieval method."""
        if "large" in self.model_name.lower():
            return "e5-large"
        else:
            return "e5-base"
    
    def _save_index(self) -> None:
        """Save FAISS index and embeddings to disk."""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Save FAISS index
            faiss_path = f"{self.index_path}.faiss"
            faiss.write_index(self.faiss_index, faiss_path)
            
            # Save metadata
            metadata = {
                'doc_embeddings': self.doc_embeddings,
                'doc_ids': self.doc_ids,
                'model_name': self.model_name,
                'batch_size': self.batch_size
            }
            
            with open(self.index_path, 'wb') as f:
                pickle.dump(metadata, f)
                
            logger.info(f"E5 index saved to {self.index_path}")
            
        except Exception as e:
            logger.error(f"Error saving E5 index: {e}")
            raise IndexError(f"Failed to save E5 index: {e}")
    
    def load_index(self) -> bool:
        """
        Load FAISS index and embeddings from disk.
        
        Returns:
            True if index loaded successfully, False otherwise
        """
        try:
            if not self.index_path or not os.path.exists(self.index_path):
                return False
            
            # Load metadata
            with open(self.index_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.doc_embeddings = metadata['doc_embeddings']
            self.doc_ids = metadata['doc_ids']
            self.model_name = metadata['model_name']
            self.batch_size = metadata['batch_size']
            
            # Load FAISS index
            faiss_path = f"{self.index_path}.faiss"
            if os.path.exists(faiss_path):
                self.faiss_index = faiss.read_index(faiss_path)
            
            # Load the E5 model
            self._load_model()
            
            logger.info(f"E5 index loaded from {self.index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading E5 index: {e}")
            return False


class BGERetriever(BaseRetriever):
    """BGE model retrieval implementation using sentence-transformers and FAISS."""
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", 
                 index_path: Optional[str] = None,
                 batch_size: int = 32):
        """
        Initialize BGE retriever.
        
        Args:
            model_name: HuggingFace model name (bge-base-en-v1.5 or bge-large-en-v1.5)
            index_path: Path to save/load FAISS index and embeddings
            batch_size: Batch size for embedding generation
        """
        self.model_name = model_name
        self.index_path = index_path
        self.batch_size = batch_size
        
        self.model = None
        self.faiss_index = None
        self.doc_embeddings = None
        self.documents = {}
        self.doc_ids = []
        
    def _load_model(self) -> None:
        """Load BGE model from HuggingFace."""
        try:
            if self.model is None:
                logger.info(f"Loading BGE model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                logger.info("BGE model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading BGE model: {e}")
            raise RetrievalError(f"Failed to load BGE model: {e}")
    
    def _preprocess_text(self, text: str, is_query: bool = False) -> str:
        """
        Preprocess text for BGE models.
        
        Args:
            text: Input text to preprocess
            is_query: Whether the text is a query (True) or document (False)
            
        Returns:
            Preprocessed text (BGE doesn't require special prefixes)
        """
        # BGE models work well without special prefixes
        return text
    
    def _generate_embeddings(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            is_query: Whether texts are queries or documents
            
        Returns:
            Numpy array of embeddings
        """
        try:
            # Preprocess texts
            processed_texts = [self._preprocess_text(text, is_query) for text in texts]
            
            # Generate embeddings in batches
            embeddings = []
            for i in range(0, len(processed_texts), self.batch_size):
                batch = processed_texts[i:i + self.batch_size]
                batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
                embeddings.extend(batch_embeddings)
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise RetrievalError(f"Failed to generate embeddings: {e}")
    
    def index_documents(self, documents: List[Document]) -> None:
        """
        Index documents for BGE retrieval with FAISS.
        
        Args:
            documents: List of Document objects to index
        """
        try:
            logger.info(f"Indexing {len(documents)} documents with BGE")
            
            # Load BGE model
            self._load_model()
            
            # Store documents
            self.documents = {doc.id: doc for doc in documents}
            self.doc_ids = [doc.id for doc in documents]
            doc_texts = [doc.text for doc in documents]
            
            # Generate embeddings for all documents
            logger.info("Generating document embeddings...")
            self.doc_embeddings = self._generate_embeddings(doc_texts, is_query=False)
            
            # Create FAISS index
            embedding_dim = self.doc_embeddings.shape[1]
            logger.info(f"Creating FAISS index with dimension {embedding_dim}")
            
            # Use IndexFlatIP for cosine similarity (inner product after normalization)
            self.faiss_index = faiss.IndexFlatIP(embedding_dim)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.doc_embeddings)
            
            # Add embeddings to FAISS index
            self.faiss_index.add(self.doc_embeddings)
            
            # Save index if path provided
            if self.index_path:
                self._save_index()
                
            logger.info("BGE indexing completed successfully")
            
        except Exception as e:
            logger.error(f"Error during BGE indexing: {e}")
            raise IndexError(f"Failed to index documents with BGE: {e}")
    
    def retrieve(self, query: Query, k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve top-k documents for a query using BGE.
        
        Args:
            query: Query object containing the search text
            k: Number of documents to retrieve
            
        Returns:
            List of RetrievalResult objects sorted by relevance score
        """
        try:
            if self.faiss_index is None:
                raise RetrievalError("BGE index not initialized. Call index_documents first.")
            
            # Generate query embedding
            query_embedding = self._generate_embeddings([query.text], is_query=True)
            
            # Normalize query embedding for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search FAISS index
            scores, indices = self.faiss_index.search(query_embedding, k)
            
            # Convert results to RetrievalResult objects
            results = []
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
                if idx < len(self.doc_ids):  # Valid index
                    doc_id = self.doc_ids[idx]
                    
                    result = RetrievalResult(
                        document_id=doc_id,
                        score=float(score),
                        method=self.get_method_name(),
                        rank=rank + 1
                    )
                    results.append(result)
            
            logger.debug(f"BGE retrieved {len(results)} documents for query: {query.text[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error during BGE retrieval: {e}")
            raise RetrievalError(f"BGE retrieval failed: {e}")
    
    def get_method_name(self) -> str:
        """Return the name of the retrieval method."""
        if "large" in self.model_name.lower():
            return "bge-large"
        else:
            return "bge-base"
    
    def _save_index(self) -> None:
        """Save FAISS index and embeddings to disk."""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Save FAISS index
            faiss_path = f"{self.index_path}.faiss"
            faiss.write_index(self.faiss_index, faiss_path)
            
            # Save metadata
            metadata = {
                'doc_embeddings': self.doc_embeddings,
                'doc_ids': self.doc_ids,
                'model_name': self.model_name,
                'batch_size': self.batch_size
            }
            
            with open(self.index_path, 'wb') as f:
                pickle.dump(metadata, f)
                
            logger.info(f"BGE index saved to {self.index_path}")
            
        except Exception as e:
            logger.error(f"Error saving BGE index: {e}")
            raise IndexError(f"Failed to save BGE index: {e}")
    
    def load_index(self) -> bool:
        """
        Load FAISS index and embeddings from disk.
        
        Returns:
            True if index loaded successfully, False otherwise
        """
        try:
            if not self.index_path or not os.path.exists(self.index_path):
                return False
            
            # Load metadata
            with open(self.index_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.doc_embeddings = metadata['doc_embeddings']
            self.doc_ids = metadata['doc_ids']
            self.model_name = metadata['model_name']
            self.batch_size = metadata['batch_size']
            
            # Load FAISS index
            faiss_path = f"{self.index_path}.faiss"
            if os.path.exists(faiss_path):
                self.faiss_index = faiss.read_index(faiss_path)
            
            # Load the BGE model
            self._load_model()
            
            logger.info(f"BGE index loaded from {self.index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading BGE index: {e}")
            return False


class GTERetriever(BaseRetriever):
    """GTE model retrieval implementation using sentence-transformers and FAISS."""
    
    def __init__(self, model_name: str = "thenlper/gte-base", 
                 index_path: Optional[str] = None,
                 batch_size: int = 32):
        """
        Initialize GTE retriever.
        
        Args:
            model_name: HuggingFace model name (thenlper/gte-base)
            index_path: Path to save/load FAISS index and embeddings
            batch_size: Batch size for embedding generation
        """
        self.model_name = model_name
        self.index_path = index_path
        self.batch_size = batch_size
        
        self.model = None
        self.faiss_index = None
        self.doc_embeddings = None
        self.documents = {}
        self.doc_ids = []
        
    def _load_model(self) -> None:
        """Load GTE model from HuggingFace."""
        try:
            if self.model is None:
                logger.info(f"Loading GTE model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                logger.info("GTE model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading GTE model: {e}")
            raise RetrievalError(f"Failed to load GTE model: {e}")
    
    def _preprocess_text(self, text: str, is_query: bool = False) -> str:
        """
        Preprocess text for GTE models.
        
        Args:
            text: Input text to preprocess
            is_query: Whether the text is a query (True) or document (False)
            
        Returns:
            Preprocessed text (GTE doesn't require special prefixes)
        """
        # GTE models work well without special prefixes
        return text
    
    def _generate_embeddings(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            is_query: Whether texts are queries or documents
            
        Returns:
            Numpy array of embeddings
        """
        try:
            # Preprocess texts
            processed_texts = [self._preprocess_text(text, is_query) for text in texts]
            
            # Generate embeddings in batches
            embeddings = []
            for i in range(0, len(processed_texts), self.batch_size):
                batch = processed_texts[i:i + self.batch_size]
                batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
                embeddings.extend(batch_embeddings)
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise RetrievalError(f"Failed to generate embeddings: {e}")
    
    def index_documents(self, documents: List[Document]) -> None:
        """
        Index documents for GTE retrieval with FAISS.
        
        Args:
            documents: List of Document objects to index
        """
        try:
            logger.info(f"Indexing {len(documents)} documents with GTE")
            
            # Load GTE model
            self._load_model()
            
            # Store documents
            self.documents = {doc.id: doc for doc in documents}
            self.doc_ids = [doc.id for doc in documents]
            doc_texts = [doc.text for doc in documents]
            
            # Generate embeddings for all documents
            logger.info("Generating document embeddings...")
            self.doc_embeddings = self._generate_embeddings(doc_texts, is_query=False)
            
            # Create FAISS index
            embedding_dim = self.doc_embeddings.shape[1]
            logger.info(f"Creating FAISS index with dimension {embedding_dim}")
            
            # Use IndexFlatIP for cosine similarity (inner product after normalization)
            self.faiss_index = faiss.IndexFlatIP(embedding_dim)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.doc_embeddings)
            
            # Add embeddings to FAISS index
            self.faiss_index.add(self.doc_embeddings)
            
            # Save index if path provided
            if self.index_path:
                self._save_index()
                
            logger.info("GTE indexing completed successfully")
            
        except Exception as e:
            logger.error(f"Error during GTE indexing: {e}")
            raise IndexError(f"Failed to index documents with GTE: {e}")
    
    def retrieve(self, query: Query, k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve top-k documents for a query using GTE.
        
        Args:
            query: Query object containing the search text
            k: Number of documents to retrieve
            
        Returns:
            List of RetrievalResult objects sorted by relevance score
        """
        try:
            if self.faiss_index is None:
                raise RetrievalError("GTE index not initialized. Call index_documents first.")
            
            # Generate query embedding
            query_embedding = self._generate_embeddings([query.text], is_query=True)
            
            # Normalize query embedding for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search FAISS index
            scores, indices = self.faiss_index.search(query_embedding, k)
            
            # Convert results to RetrievalResult objects
            results = []
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
                if idx < len(self.doc_ids):  # Valid index
                    doc_id = self.doc_ids[idx]
                    
                    result = RetrievalResult(
                        document_id=doc_id,
                        score=float(score),
                        method=self.get_method_name(),
                        rank=rank + 1
                    )
                    results.append(result)
            
            logger.debug(f"GTE retrieved {len(results)} documents for query: {query.text[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error during GTE retrieval: {e}")
            raise RetrievalError(f"GTE retrieval failed: {e}")
    
    def get_method_name(self) -> str:
        """Return the name of the retrieval method."""
        return "gte-base"
    
    def _save_index(self) -> None:
        """Save FAISS index and embeddings to disk."""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Save FAISS index
            faiss_path = f"{self.index_path}.faiss"
            faiss.write_index(self.faiss_index, faiss_path)
            
            # Save metadata
            metadata = {
                'doc_embeddings': self.doc_embeddings,
                'doc_ids': self.doc_ids,
                'model_name': self.model_name,
                'batch_size': self.batch_size
            }
            
            with open(self.index_path, 'wb') as f:
                pickle.dump(metadata, f)
                
            logger.info(f"GTE index saved to {self.index_path}")
            
        except Exception as e:
            logger.error(f"Error saving GTE index: {e}")
            raise IndexError(f"Failed to save GTE index: {e}")
    
    def load_index(self) -> bool:
        """
        Load FAISS index and embeddings from disk.
        
        Returns:
            True if index loaded successfully, False otherwise
        """
        try:
            if not self.index_path or not os.path.exists(self.index_path):
                return False
            
            # Load metadata
            with open(self.index_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.doc_embeddings = metadata['doc_embeddings']
            self.doc_ids = metadata['doc_ids']
            self.model_name = metadata['model_name']
            self.batch_size = metadata['batch_size']
            
            # Load FAISS index
            faiss_path = f"{self.index_path}.faiss"
            if os.path.exists(faiss_path):
                self.faiss_index = faiss.read_index(faiss_path)
            
            # Load the GTE model
            self._load_model()
            
            logger.info(f"GTE index loaded from {self.index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading GTE index: {e}")
            return False