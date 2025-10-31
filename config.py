"""
Main configuration file for the RAG system.
"""

import os
from src.models.config import SystemConfig, RetrievalConfig, GenerationConfig, DataConfig, UIConfig


def load_config() -> SystemConfig:
    """Load system configuration from environment variables and defaults."""
    
    # Retrieval configuration
    retrieval_config = RetrievalConfig(
        methods=["bm25", "e5-base"],
        weights=[0.6, 0.4],
        k_retrieve=100,
        k_final=10,
        embedding_models={
            "e5-base": "intfloat/e5-base-v2",
            "e5-large": "intfloat/e5-large-v2",
            "bge-base": "BAAI/bge-base-en-v1.5",
            "bge-large": "BAAI/bge-large-en-v1.5",
            "gte-base": "Alibaba-NLP/gte-modernbert-base"
        }
    )
    
    # Generation configuration
    generation_config = GenerationConfig(
        model_name=os.getenv("QWEN_MODEL", "qwen/qwen-2.5-1.5b-instruct"),
        max_tokens=int(os.getenv("MAX_TOKENS", "512")),
        temperature=float(os.getenv("TEMPERATURE", "0.1")),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        enable_multi_turn=True,
        enable_agentic=True
    )
    
    # Data configuration
    data_config = DataConfig(
        dataset_name="hotpot_qa",
        dataset_config="distractor",
        cache_dir="./data/cache",
        processed_dir="./data/processed"
    )
    
    # UI configuration
    ui_config = UIConfig(
        interface_type=os.getenv("INTERFACE_TYPE", "terminal"),
        show_intermediate_steps=True,
        show_retrieved_docs=True
    )
    
    # Main system configuration
    config = SystemConfig(
        retrieval=retrieval_config,
        generation=generation_config,
        data=data_config,
        ui=ui_config,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE"),
        enable_performance_monitoring=True
    )
    
    return config


# Global configuration instance
CONFIG = load_config()