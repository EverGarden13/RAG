"""
Configuration classes for system settings and model parameters.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class RetrievalConfig:
    """Configuration for retrieval methods and parameters."""
    methods: List[str] = field(default_factory=lambda: ["bm25"])
    weights: List[float] = field(default_factory=lambda: [1.0])
    k_retrieve: int = 100
    k_final: int = 10
    
    # Model-specific configurations
    embedding_models: Dict[str, str] = field(default_factory=dict)
    sparse_methods: List[str] = field(default_factory=lambda: ["bm25", "tfidf"])
    dense_methods: List[str] = field(default_factory=lambda: ["e5-base", "bge-base"])


@dataclass
class GenerationConfig:
    """Configuration for generation models and parameters."""
    model_name: str = "qwen/qwen-2.5-1.5b-instruct"
    max_tokens: int = 512
    temperature: float = 0.1
    enable_multi_turn: bool = True
    enable_agentic: bool = True
    
    # OpenRouter API configuration
    api_key: Optional[str] = None
    base_url: str = "https://openrouter.ai/api/v1"


@dataclass
class DataConfig:
    """Configuration for dataset and document processing."""
    dataset_name: str = "hotpot_qa"
    dataset_config: str = "distractor"
    cache_dir: str = "./data/cache"
    processed_dir: str = "./data/processed"
    
    # Document processing parameters
    max_doc_length: int = 512
    chunk_size: int = 256
    chunk_overlap: int = 50


@dataclass
class UIConfig:
    """Configuration for user interface components."""
    interface_type: str = "terminal"  # "terminal" or "web"
    web_host: str = "localhost"
    web_port: int = 8080
    show_intermediate_steps: bool = True
    show_retrieved_docs: bool = True


@dataclass
class SystemConfig:
    """Main system configuration combining all components."""
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # Evaluation settings
    evaluation: Dict[str, Any] = field(default_factory=lambda: {
        "metrics": ["exact_match", "ndcg_at_10"],
        "output_format": "jsonl"
    })
    
    # Logging and monitoring
    log_level: str = "INFO"
    log_file: Optional[str] = None
    enable_performance_monitoring: bool = True