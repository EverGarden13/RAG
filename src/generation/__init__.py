"""
Generation module for RAG system.
Includes OpenRouter API integration and generation utilities.
"""

from src.generation.openrouter_client import OpenRouterClient, GenerationConfig
from src.generation.prompt_templates import PromptTemplate, PromptManager
from src.generation.rag_pipeline import BasicRAGPipeline, MultiHopRAGPipeline
from src.generation.multi_turn import (
    ConversationStateManager,
    QueryReformulator,
    ContextPruner,
    EntityTracker
)

__all__ = [
    'OpenRouterClient',
    'GenerationConfig',
    'PromptTemplate',
    'PromptManager',
    'BasicRAGPipeline',
    'MultiHopRAGPipeline',
    'ConversationStateManager',
    'QueryReformulator',
    'ContextPruner',
    'EntityTracker'
]
