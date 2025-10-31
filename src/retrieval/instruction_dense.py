"""
Instruction-based dense retrieval implementations using HuggingFace models.
Includes E5-Mistral and Qwen3-Embedding models with instruction-following capabilities.
"""

import os
import pickle
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch

from src.interfaces.base import BaseRetriever
from src.models.data_models import Query, Document, RetrievalResult
from src.utils.exceptions import RetrievalError, IndexError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class E5MistralRetriever(BaseRetriever):
    """E5-Mistral 7B instruction-based retrieval implementation."""
    
    def __init__(self, model_name: str = "intfloat/e5-mistral-7b-instruct", 
                 index_path: Optional[str] = None,
                 batch_size: int = 8,
                 max_length: int = 512):
        """
        Initialize E5-Mistral retriever.
        
        Args:
            model_name: HuggingFace model name for E5-Mistral
            index_path: Path to save/load FAISS index and embeddings
            batch_size: Batch size for embedding generation (smaller for 7B model)
            max_length: Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.index_path = index_path
        self.batch_size = batch_size
        self.max_length = max_length
        
        self.model = None
        self.faiss_index = None
        self.doc_embeddings = None
        self.documents = {}
        self.doc_ids = []
        
    def _load_model(self) -> None:
        """Load E5-Mistral model from HuggingFace."""
        try:
            if self.model is None:
                logger.info(f"Loading E5-Mistral model: {self.model_name}")
                logger.info("Note: This is a 7B model and may require significant GPU memory")
                
                # Check if CUDA is available
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Using device: {device}")
                
                # Load model with appropriate settings
                self.model = SentenceTransformer(
                    self.model_name,
                    device=device,
                    trust_remote_code=True
                )
                
                # Set max sequence length
                self.model.max_seq_length = self.max_length
                
                logger.info("E5-Mistral model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading E5-Mistral model: {e}")
            raise RetrievalError(f"Failed to load E5-Mistral model: {e}")
    
    def _create_instruction_prompt(self, text: str, is_query: bool = False) -> str:
        """
        Create instruction-based prompt for E5-Mistral.
        
        Args:
            text: Input text
            is_query: Whether the text is a query (True) or document (False)
            
        Returns:
            Instruction-formatted prompt
        """
        if is_query:
            # For queries, use task-specific instruction
            instruction = "Given a question, retrieve passages that answer the question"
            return f"Instruct: {instruction}\nQuery: {text}"
        else:
            # For documents, no instruction prefix needed
            return text
    
    def _generate_embeddings(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Generate embeddings for a list of texts with instruction awareness.
        
        Args:
            texts: List of texts to embed
            is_query: Whether texts are queries or documents
            
        Returns:
            Numpy array of embeddings
        """
        try:
            # Create instruction-based prompts
            processed_texts = [self._create_instruction_prompt(text, is_query) for text in texts]
            
            # Generate embeddings in batches
            embeddings = []
            for i in range(0, len(processed_texts), self.batch_size):
                batch = processed_texts[i:i + self.batch_size]
                logger.debug(f"Processing batch {i//self.batch_size + 1}/{(len(processed_texts)-1)//self.batch_size + 1}")
                
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
                embeddings.extend(batch_embeddings)
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Error generating E5-Mistral embeddings: {e}")
            raise RetrievalError(f"Failed to generate embeddings: {e}")
    
    def index_documents(self, documents: List[Document]) -> None:
        """
        Index documents for E5-Mistral retrieval with FAISS.
        
        Args:
            documents: List of Document objects to index
        """
        try:
            logger.info(f"Indexing {len(documents)} documents with E5-Mistral")
            logger.info("This may take a while due to the large model size...")
            
            # Load E5-Mistral model
            self._load_model()
            
            # Store documents
            self.documents = {doc.id: doc for doc in documents}
            self.doc_ids = [doc.id for doc in documents]
            doc_texts = [doc.text for doc in documents]
            
            # Generate embeddings for all documents
            logger.info("Generating document embeddings with instruction awareness...")
            self.doc_embeddings = self._generate_embeddings(doc_texts, is_query=False)
            
            # Create FAISS index
            embedding_dim = self.doc_embeddings.shape[1]
            logger.info(f"Creating FAISS index with dimension {embedding_dim}")
            
            # Use IndexFlatIP for cosine similarity (embeddings already normalized)
            self.faiss_index = faiss.IndexFlatIP(embedding_dim)
            
            # Add embeddings to FAISS index
            self.faiss_index.add(self.doc_embeddings)
            
            # Save index if path provided
            if self.index_path:
                self._save_index()
                
            logger.info("E5-Mistral indexing completed successfully")
            
        except Exception as e:
            logger.error(f"Error during E5-Mistral indexing: {e}")
            raise IndexError(f"Failed to index documents with E5-Mistral: {e}")
    
    def retrieve(self, query: Query, k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve top-k documents for a query using E5-Mistral.
        
        Args:
            query: Query object containing the search text
            k: Number of documents to retrieve
            
        Returns:
            List of RetrievalResult objects sorted by relevance score
        """
        try:
            if self.faiss_index is None:
                raise RetrievalError("E5-Mistral index not initialized. Call index_documents first.")
            
            # Generate query embedding with instruction
            query_embedding = self._generate_embeddings([query.text], is_query=True)
            
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
            
            logger.debug(f"E5-Mistral retrieved {len(results)} documents for query: {query.text[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error during E5-Mistral retrieval: {e}")
            raise RetrievalError(f"E5-Mistral retrieval failed: {e}")
    
    def get_method_name(self) -> str:
        """Return the name of the retrieval method."""
        return "e5-mistral"
    
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
                'batch_size': self.batch_size,
                'max_length': self.max_length
            }
            
            with open(self.index_path, 'wb') as f:
                pickle.dump(metadata, f)
                
            logger.info(f"E5-Mistral index saved to {self.index_path}")
            
        except Exception as e:
            logger.error(f"Error saving E5-Mistral index: {e}")
            raise IndexError(f"Failed to save E5-Mistral index: {e}")
    
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
            self.max_length = metadata.get('max_length', 512)
            
            # Load FAISS index
            faiss_path = f"{self.index_path}.faiss"
            if os.path.exists(faiss_path):
                self.faiss_index = faiss.read_index(faiss_path)
            
            # Load the E5-Mistral model
            self._load_model()
            
            logger.info(f"E5-Mistral index loaded from {self.index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading E5-Mistral index: {e}")
            return False


class QwenEmbeddingRetriever(BaseRetriever):
    """Qwen3-Embedding instruction-based retrieval implementation."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-Embedder-0.5B-Instruct", 
                 index_path: Optional[str] = None,
                 batch_size: int = 16,
                 max_length: int = 512):
        """
        Initialize Qwen embedding retriever.
        
        Args:
            model_name: HuggingFace model name for Qwen embedding
            index_path: Path to save/load FAISS index and embeddings
            batch_size: Batch size for embedding generation
            max_length: Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.index_path = index_path
        self.batch_size = batch_size
        self.max_length = max_length
        
        self.model = None
        self.faiss_index = None
        self.doc_embeddings = None
        self.documents = {}
        self.doc_ids = []
        
    def _load_model(self) -> None:
        """Load Qwen embedding model from HuggingFace."""
        try:
            if self.model is None:
                logger.info(f"Loading Qwen embedding model: {self.model_name}")
                
                # Check if CUDA is available
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Using device: {device}")
                
                # Try to load with sentence-transformers
                try:
                    self.model = SentenceTransformer(
                        self.model_name,
                        device=device,
                        trust_remote_code=True
                    )
                    self.model.max_seq_length = self.max_length
                    logger.info("Qwen embedding model loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load with sentence-transformers: {e}")
                    logger.info("Attempting alternative loading method...")
                    
                    # Alternative: Use transformers directly
                    from transformers import AutoModel, AutoTokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        trust_remote_code=True
                    )
                    self.model = AutoModel.from_pretrained(
                        self.model_name,
                        trust_remote_code=True
                    ).to(device)
                    self.model.eval()
                    self.use_transformers_directly = True
                    logger.info("Qwen embedding model loaded with transformers")
                    
        except Exception as e:
            logger.error(f"Error loading Qwen embedding model: {e}")
            raise RetrievalError(f"Failed to load Qwen embedding model: {e}")
    
    def _create_instruction_prompt(self, text: str, is_query: bool = False) -> str:
        """
        Create instruction-based prompt for Qwen embedding.
        
        Args:
            text: Input text
            is_query: Whether the text is a query (True) or document (False)
            
        Returns:
            Instruction-formatted prompt
        """
        if is_query:
            # For queries, use retrieval instruction
            return f"Instruct: Retrieve relevant documents for this question\nQuery: {text}"
        else:
            # For documents, simple format
            return text
    
    def _generate_embeddings(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Generate embeddings for a list of texts with instruction awareness.
        
        Args:
            texts: List of texts to embed
            is_query: Whether texts are queries or documents
            
        Returns:
            Numpy array of embeddings
        """
        try:
            # Create instruction-based prompts
            processed_texts = [self._create_instruction_prompt(text, is_query) for text in texts]
            
            # Check if using sentence-transformers or transformers directly
            if hasattr(self, 'use_transformers_directly') and self.use_transformers_directly:
                # Use transformers directly
                embeddings = []
                device = next(self.model.parameters()).device
                
                for i in range(0, len(processed_texts), self.batch_size):
                    batch = processed_texts[i:i + self.batch_size]
                    
                    # Tokenize
                    inputs = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors='pt'
                    ).to(device)
                    
                    # Generate embeddings
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        # Use mean pooling
                        batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                        batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                        embeddings.extend(batch_embeddings.cpu().numpy())
                
                return np.array(embeddings)
            else:
                # Use sentence-transformers
                embeddings = []
                for i in range(0, len(processed_texts), self.batch_size):
                    batch = processed_texts[i:i + self.batch_size]
                    logger.debug(f"Processing batch {i//self.batch_size + 1}/{(len(processed_texts)-1)//self.batch_size + 1}")
                    
                    batch_embeddings = self.model.encode(
                        batch,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
                    embeddings.extend(batch_embeddings)
                
                return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Error generating Qwen embeddings: {e}")
            raise RetrievalError(f"Failed to generate embeddings: {e}")
    
    def index_documents(self, documents: List[Document]) -> None:
        """
        Index documents for Qwen embedding retrieval with FAISS.
        
        Args:
            documents: List of Document objects to index
        """
        try:
            logger.info(f"Indexing {len(documents)} documents with Qwen embedding")
            
            # Load Qwen embedding model
            self._load_model()
            
            # Store documents
            self.documents = {doc.id: doc for doc in documents}
            self.doc_ids = [doc.id for doc in documents]
            doc_texts = [doc.text for doc in documents]
            
            # Generate embeddings for all documents
            logger.info("Generating document embeddings with instruction awareness...")
            self.doc_embeddings = self._generate_embeddings(doc_texts, is_query=False)
            
            # Create FAISS index
            embedding_dim = self.doc_embeddings.shape[1]
            logger.info(f"Creating FAISS index with dimension {embedding_dim}")
            
            # Use IndexFlatIP for cosine similarity (embeddings already normalized)
            self.faiss_index = faiss.IndexFlatIP(embedding_dim)
            
            # Add embeddings to FAISS index
            self.faiss_index.add(self.doc_embeddings)
            
            # Save index if path provided
            if self.index_path:
                self._save_index()
                
            logger.info("Qwen embedding indexing completed successfully")
            
        except Exception as e:
            logger.error(f"Error during Qwen embedding indexing: {e}")
            raise IndexError(f"Failed to index documents with Qwen embedding: {e}")
    
    def retrieve(self, query: Query, k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve top-k documents for a query using Qwen embedding.
        
        Args:
            query: Query object containing the search text
            k: Number of documents to retrieve
            
        Returns:
            List of RetrievalResult objects sorted by relevance score
        """
        try:
            if self.faiss_index is None:
                raise RetrievalError("Qwen embedding index not initialized. Call index_documents first.")
            
            # Generate query embedding with instruction
            query_embedding = self._generate_embeddings([query.text], is_query=True)
            
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
            
            logger.debug(f"Qwen embedding retrieved {len(results)} documents for query: {query.text[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error during Qwen embedding retrieval: {e}")
            raise RetrievalError(f"Qwen embedding retrieval failed: {e}")
    
    def get_method_name(self) -> str:
        """Return the name of the retrieval method."""
        return "qwen-embedding"
    
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
                'batch_size': self.batch_size,
                'max_length': self.max_length
            }
            
            with open(self.index_path, 'wb') as f:
                pickle.dump(metadata, f)
                
            logger.info(f"Qwen embedding index saved to {self.index_path}")
            
        except Exception as e:
            logger.error(f"Error saving Qwen embedding index: {e}")
            raise IndexError(f"Failed to save Qwen embedding index: {e}")
    
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
            self.max_length = metadata.get('max_length', 512)
            
            # Load FAISS index
            faiss_path = f"{self.index_path}.faiss"
            if os.path.exists(faiss_path):
                self.faiss_index = faiss.read_index(faiss_path)
            
            # Load the Qwen embedding model
            self._load_model()
            
            logger.info(f"Qwen embedding index loaded from {self.index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Qwen embedding index: {e}")
            return False
