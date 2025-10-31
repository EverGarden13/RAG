"""
Multi-vector retrieval implementations using ColBERT models.
Includes ColBERT and GTE-ColBERT with token-level interaction.
"""

import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.interfaces.base import BaseRetriever
from src.models.data_models import Query, Document, RetrievalResult
from src.utils.exceptions import RetrievalError, IndexError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ColBERTRetriever(BaseRetriever):
    """ColBERT multi-vector retrieval implementation with MaxSim scoring."""
    
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0", 
                 index_path: Optional[str] = None,
                 batch_size: int = 16,
                 max_length: int = 512,
                 doc_max_length: int = 180,
                 query_max_length: int = 32):
        """
        Initialize ColBERT retriever.
        
        Args:
            model_name: HuggingFace model name for ColBERT
            index_path: Path to save/load index and embeddings
            batch_size: Batch size for embedding generation
            max_length: Maximum sequence length
            doc_max_length: Maximum document length
            query_max_length: Maximum query length
        """
        self.model_name = model_name
        self.index_path = index_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.doc_max_length = doc_max_length
        self.query_max_length = query_max_length
        
        self.model = None
        self.tokenizer = None
        self.doc_embeddings = {}  # Dict of doc_id -> token embeddings
        self.documents = {}
        self.doc_ids = []
        self.device = None
        
    def _load_model(self) -> None:
        """Load ColBERT model from HuggingFace."""
        try:
            if self.model is None:
                logger.info(f"Loading ColBERT model: {self.model_name}")
                
                # Check if CUDA is available
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                logger.info(f"Using device: {self.device}")
                
                # Load tokenizer and model
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                ).to(self.device)
                
                self.model.eval()
                logger.info("ColBERT model loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading ColBERT model: {e}")
            logger.warning("Attempting to use alternative ColBERT-compatible model...")
            
            try:
                # Fallback to BERT-based model for ColBERT-style retrieval
                self.model_name = "bert-base-uncased"
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
                self.model.eval()
                logger.info("Using BERT-based model for ColBERT-style retrieval")
            except Exception as e2:
                logger.error(f"Error loading fallback model: {e2}")
                raise RetrievalError(f"Failed to load ColBERT model: {e}")
    
    def _encode_text(self, texts: List[str], max_length: int) -> torch.Tensor:
        """
        Encode texts to token-level embeddings.
        
        Args:
            texts: List of texts to encode
            max_length: Maximum sequence length
            
        Returns:
            Token embeddings tensor [batch_size, seq_len, hidden_dim]
        """
        try:
            # Tokenize
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(self.device)
            
            # Generate token embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use last hidden state as token embeddings
                token_embeddings = outputs.last_hidden_state
                
                # Normalize token embeddings
                token_embeddings = F.normalize(token_embeddings, p=2, dim=2)
            
            return token_embeddings
            
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            raise RetrievalError(f"Failed to encode text: {e}")
    
    def _maxsim_score(self, query_embeddings: torch.Tensor, 
                      doc_embeddings: torch.Tensor) -> float:
        """
        Compute MaxSim score between query and document embeddings.
        
        Args:
            query_embeddings: Query token embeddings [query_len, hidden_dim]
            doc_embeddings: Document token embeddings [doc_len, hidden_dim]
            
        Returns:
            MaxSim score
        """
        try:
            # Compute similarity matrix [query_len, doc_len]
            similarity_matrix = torch.matmul(query_embeddings, doc_embeddings.T)
            
            # MaxSim: for each query token, find max similarity with any doc token
            max_similarities = similarity_matrix.max(dim=1)[0]
            
            # Sum over query tokens
            maxsim_score = max_similarities.sum().item()
            
            return maxsim_score
            
        except Exception as e:
            logger.error(f"Error computing MaxSim score: {e}")
            raise RetrievalError(f"Failed to compute MaxSim score: {e}")
    
    def index_documents(self, documents: List[Document]) -> None:
        """
        Index documents for ColBERT retrieval.
        
        Args:
            documents: List of Document objects to index
        """
        try:
            logger.info(f"Indexing {len(documents)} documents with ColBERT")
            
            # Load ColBERT model
            self._load_model()
            
            # Store documents
            self.documents = {doc.id: doc for doc in documents}
            self.doc_ids = [doc.id for doc in documents]
            
            # Generate token embeddings for all documents
            logger.info("Generating document token embeddings...")
            
            for i in range(0, len(documents), self.batch_size):
                batch_docs = documents[i:i + self.batch_size]
                batch_texts = [doc.text for doc in batch_docs]
                batch_ids = [doc.id for doc in batch_docs]
                
                # Encode batch
                batch_embeddings = self._encode_text(batch_texts, self.doc_max_length)
                
                # Store embeddings for each document
                for doc_id, embeddings in zip(batch_ids, batch_embeddings):
                    # Store on CPU to save GPU memory
                    self.doc_embeddings[doc_id] = embeddings.cpu()
                
                if (i + self.batch_size) % 100 == 0:
                    logger.info(f"Processed {min(i + self.batch_size, len(documents))}/{len(documents)} documents")
            
            # Save index if path provided
            if self.index_path:
                self._save_index()
                
            logger.info("ColBERT indexing completed successfully")
            
        except Exception as e:
            logger.error(f"Error during ColBERT indexing: {e}")
            raise IndexError(f"Failed to index documents with ColBERT: {e}")
    
    def retrieve(self, query: Query, k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve top-k documents for a query using ColBERT MaxSim.
        
        Args:
            query: Query object containing the search text
            k: Number of documents to retrieve
            
        Returns:
            List of RetrievalResult objects sorted by relevance score
        """
        try:
            if not self.doc_embeddings:
                raise RetrievalError("ColBERT index not initialized. Call index_documents first.")
            
            # Encode query
            query_embeddings = self._encode_text([query.text], self.query_max_length)
            query_embeddings = query_embeddings[0]  # Remove batch dimension
            
            # Compute MaxSim scores for all documents
            scores = []
            for doc_id in self.doc_ids:
                doc_embeddings = self.doc_embeddings[doc_id].to(self.device)
                score = self._maxsim_score(query_embeddings, doc_embeddings)
                scores.append((doc_id, score))
            
            # Sort by score (descending)
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # Convert to RetrievalResult objects
            results = []
            for rank, (doc_id, score) in enumerate(scores[:k]):
                result = RetrievalResult(
                    document_id=doc_id,
                    score=float(score),
                    method=self.get_method_name(),
                    rank=rank + 1
                )
                results.append(result)
            
            logger.debug(f"ColBERT retrieved {len(results)} documents for query: {query.text[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error during ColBERT retrieval: {e}")
            raise RetrievalError(f"ColBERT retrieval failed: {e}")
    
    def get_method_name(self) -> str:
        """Return the name of the retrieval method."""
        return "colbert"
    
    def _save_index(self) -> None:
        """Save document embeddings to disk."""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Save metadata
            metadata = {
                'doc_embeddings': self.doc_embeddings,
                'doc_ids': self.doc_ids,
                'model_name': self.model_name,
                'batch_size': self.batch_size,
                'max_length': self.max_length,
                'doc_max_length': self.doc_max_length,
                'query_max_length': self.query_max_length
            }
            
            with open(self.index_path, 'wb') as f:
                pickle.dump(metadata, f)
                
            logger.info(f"ColBERT index saved to {self.index_path}")
            
        except Exception as e:
            logger.error(f"Error saving ColBERT index: {e}")
            raise IndexError(f"Failed to save ColBERT index: {e}")
    
    def load_index(self) -> bool:
        """
        Load document embeddings from disk.
        
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
            self.max_length = metadata.get('max_length', 512)
            self.doc_max_length = metadata.get('doc_max_length', 180)
            self.query_max_length = metadata.get('query_max_length', 32)
            
            # Load the ColBERT model
            self._load_model()
            
            logger.info(f"ColBERT index loaded from {self.index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ColBERT index: {e}")
            return False


class GTEColBERTRetriever(ColBERTRetriever):
    """GTE-ColBERT multi-vector retrieval implementation."""
    
    def __init__(self, model_name: str = "lightonai/GTE-ModernColBERT-v", 
                 index_path: Optional[str] = None,
                 batch_size: int = 16,
                 max_length: int = 512,
                 doc_max_length: int = 180,
                 query_max_length: int = 32):
        """
        Initialize GTE-ColBERT retriever.
        
        Args:
            model_name: HuggingFace model name for GTE-ColBERT
            index_path: Path to save/load index and embeddings
            batch_size: Batch size for embedding generation
            max_length: Maximum sequence length
            doc_max_length: Maximum document length
            query_max_length: Maximum query length
        """
        super().__init__(
            model_name=model_name,
            index_path=index_path,
            batch_size=batch_size,
            max_length=max_length,
            doc_max_length=doc_max_length,
            query_max_length=query_max_length
        )
    
    def get_method_name(self) -> str:
        """Return the name of the retrieval method."""
        return "gte-colbert"
