"""
Core data models for the RAG system.
Defines data classes for Query, Document, RetrievalResult, and SystemOutput.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np


@dataclass
class Query:
    """Represents a user query with optional conversation context."""
    id: str
    text: str
    conversation_id: Optional[str] = None
    turn_number: int = 1


@dataclass
class Document:
    """Represents a document in the collection with embeddings."""
    id: str
    text: str
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Represents a single retrieval result with score and metadata."""
    document_id: str
    score: float
    method: str
    rank: int


@dataclass
class GenerationResult:
    """Represents the result of answer generation."""
    answer: str
    retrieved_docs: List[Tuple[str, float]]
    reasoning_steps: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class SystemOutput:
    """Final system output in the required JSONL format."""
    id: str
    question: str
    answer: str
    retrieved_docs: List[List[Union[str, float]]]
    metadata: Dict[str, Any] = field(default_factory=dict)