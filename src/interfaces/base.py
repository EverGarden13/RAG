"""
Base interfaces for retrieval and generation components.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from src.models.data_models import Query, Document, RetrievalResult, GenerationResult


class BaseRetriever(ABC):
    """Abstract base class for all retrieval methods."""
    
    @abstractmethod
    def index_documents(self, documents: List[Document]) -> None:
        """Index documents for retrieval."""
        pass
    
    @abstractmethod
    def retrieve(self, query: Query, k: int = 10) -> List[RetrievalResult]:
        """Retrieve top-k documents for a query."""
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Return the name of the retrieval method."""
        pass


class BaseGenerator(ABC):
    """Abstract base class for all generation methods."""
    
    @abstractmethod
    def generate(self, query: Query, retrieved_docs: List[Document]) -> GenerationResult:
        """Generate answer based on query and retrieved documents."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name of the generation model."""
        pass


class BaseEmbedder(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts into embeddings."""
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the dimension of embeddings."""
        pass


class BaseProcessor(ABC):
    """Abstract base class for document processors."""
    
    @abstractmethod
    def process_document(self, text: str) -> str:
        """Process and clean document text."""
        pass
    
    @abstractmethod
    def process_query(self, text: str) -> str:
        """Process and clean query text."""
        pass


class BaseEvaluator(ABC):
    """Abstract base class for evaluation metrics."""
    
    @abstractmethod
    def evaluate(self, predictions: List[Dict[str, Any]], 
                references: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate predictions against references."""
        pass