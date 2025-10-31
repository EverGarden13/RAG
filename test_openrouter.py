"""
Test script for OpenRouter API client.
Tests generation capabilities with Qwen2.5 models.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.generation.openrouter_client import OpenRouterClient, GenerationConfig
from src.generation.prompt_templates import PromptManager
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def test_openrouter_client():
    """Test OpenRouter client with sample prompts."""
    
    logger.info("=" * 80)
    logger.info("Testing OpenRouter API Client")
    logger.info("=" * 80)
    
    # Check if API key is set
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set in environment variables")
        logger.info("Please set your OpenRouter API key in .env file")
        return
    
    try:
        # Initialize client
        logger.info("Initializing OpenRouter client...")
        client = OpenRouterClient()
        
        # Show available models
        models = client.get_available_models()
        logger.info(f"Available models: {list(models.keys())}")
        
        # Initialize prompt manager
        prompt_manager = PromptManager()
        
        # Test 1: Simple generation
        logger.info("\n" + "=" * 80)
        logger.info("Test 1: Simple Generation")
        logger.info("=" * 80)
        
        simple_prompt = "What is the capital of France? Answer in one word."
        logger.info(f"Prompt: {simple_prompt}")
        
        response = client.generate(simple_prompt)
        logger.info(f"Response: {response}")
        
        # Test 2: RAG-style prompt
        logger.info("\n" + "=" * 80)
        logger.info("Test 2: RAG-Style Prompt")
        logger.info("=" * 80)
        
        documents = [
            {
                "id": "doc1",
                "text": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel."
            },
            {
                "id": "doc2",
                "text": "The tower is 330 metres tall and was completed in 1889. It was the world's tallest man-made structure until 1930."
            }
        ]
        
        question = "Who is the Eiffel Tower named after?"
        
        rag_prompt = prompt_manager.create_basic_rag_prompt(question, documents)
        logger.info(f"Question: {question}")
        logger.info("Generated RAG prompt (truncated):")
        logger.info(rag_prompt[:200] + "...")
        
        response = client.generate(rag_prompt)
        logger.info(f"Response: {response}")
        
        # Test 3: Multi-hop prompt
        logger.info("\n" + "=" * 80)
        logger.info("Test 3: Multi-Hop Reasoning Prompt")
        logger.info("=" * 80)
        
        multi_hop_docs = [
            {
                "id": "doc1",
                "text": "Barack Obama was born in Honolulu, Hawaii on August 4, 1961."
            },
            {
                "id": "doc2",
                "text": "Michelle Obama, born Michelle LaVaughn Robinson, is married to Barack Obama."
            },
            {
                "id": "doc3",
                "text": "Michelle LaVaughn Robinson was born on January 17, 1964, in Chicago, Illinois."
            }
        ]
        
        multi_hop_question = "Where was Barack Obama's wife born?"
        
        multi_hop_prompt = prompt_manager.create_multi_hop_prompt(multi_hop_question, multi_hop_docs)
        logger.info(f"Question: {multi_hop_question}")
        
        response = client.generate(multi_hop_prompt)
        logger.info(f"Response: {response}")
        
        # Test 4: Different model sizes
        logger.info("\n" + "=" * 80)
        logger.info("Test 4: Testing Different Model Sizes")
        logger.info("=" * 80)
        
        test_prompt = "Explain quantum computing in one sentence."
        
        for model_alias in ["qwen2.5-0.5b", "qwen2.5-1.5b"]:
            logger.info(f"\nTesting {model_alias}...")
            try:
                response = client.generate(test_prompt, model=models[model_alias])
                logger.info(f"Response: {response}")
            except Exception as e:
                logger.error(f"Error with {model_alias}: {e}")
        
        # Show stats
        logger.info("\n" + "=" * 80)
        logger.info("Client Statistics")
        logger.info("=" * 80)
        stats = client.get_stats()
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        
        logger.info("\n" + "=" * 80)
        logger.info("OpenRouter client testing completed successfully!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    test_openrouter_client()
