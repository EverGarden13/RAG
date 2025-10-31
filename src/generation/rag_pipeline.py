"""
Basic RAG pipeline implementation.
Combines retrieval and generation for question answering.
"""

from typing import List, Dict, Any, Optional
import json

from src.interfaces.base import BaseRetriever
from src.models.data_models import Query, Document, RetrievalResult, SystemOutput
from src.generation.openrouter_client import OpenRouterClient, GenerationConfig
from src.generation.prompt_templates import PromptManager
from src.utils.logging_config import get_logger
from src.utils.exceptions import GenerationError, RetrievalError

logger = get_logger(__name__)


class BasicRAGPipeline:
    """Basic single-turn RAG pipeline."""
    
    def __init__(self, retriever: BaseRetriever,
                 generator: OpenRouterClient,
                 prompt_manager: Optional[PromptManager] = None,
                 k_retrieve: int = 10,
                 max_context_length: int = 2000):
        """
        Initialize RAG pipeline.
        
        Args:
            retriever: Retriever instance
            generator: OpenRouter client for generation
            prompt_manager: Prompt manager (creates default if None)
            k_retrieve: Number of documents to retrieve
            max_context_length: Maximum context length for generation
        """
        self.retriever = retriever
        self.generator = generator
        self.prompt_manager = prompt_manager or PromptManager()
        self.k_retrieve = k_retrieve
        self.max_context_length = max_context_length
        
        logger.info(f"RAG pipeline initialized with retriever: {retriever.get_method_name()}")
    
    def process_query(self, query: Query, 
                     generation_config: Optional[GenerationConfig] = None) -> SystemOutput:
        """
        Process a query through the RAG pipeline.
        
        Args:
            query: Query object
            generation_config: Generation configuration
            
        Returns:
            SystemOutput with answer and retrieved documents
        """
        try:
            # Step 1: Retrieve documents
            logger.info(f"Processing query: {query.text[:100]}...")
            retrieval_results = self.retriever.retrieve(query, k=self.k_retrieve)
            
            if not retrieval_results:
                logger.warning("No documents retrieved")
                return SystemOutput(
                    id=query.id,
                    question=query.text,
                    answer="I don't have enough information to answer this question.",
                    retrieved_docs=[]
                )
            
            logger.info(f"Retrieved {len(retrieval_results)} documents")
            
            # Step 2: Format retrieved documents
            documents = []
            for result in retrieval_results:
                doc_id = result.document_id
                # Get document text from retriever's document store
                if hasattr(self.retriever, 'documents') and doc_id in self.retriever.documents:
                    doc = self.retriever.documents[doc_id]
                    documents.append({
                        'id': doc_id,
                        'text': doc.text
                    })
            
            # Step 3: Create prompt
            prompt = self.prompt_manager.create_basic_rag_prompt(
                question=query.text,
                documents=documents,
                max_context_length=self.max_context_length
            )
            
            logger.debug(f"Prompt length: {len(prompt)} characters")
            
            # Step 4: Generate answer
            try:
                answer = self.generator.generate(prompt, config=generation_config)
                logger.info("Answer generated successfully")
            except GenerationError as e:
                logger.error(f"Generation failed: {e}")
                answer = "I encountered an error while generating the answer."
            
            # Step 5: Format output
            retrieved_docs = [
                [result.document_id, float(result.score)]
                for result in retrieval_results
            ]
            
            output = SystemOutput(
                id=query.id,
                question=query.text,
                answer=answer,
                retrieved_docs=retrieved_docs,
                metadata={
                    'retrieval_method': self.retriever.get_method_name(),
                    'num_retrieved': len(retrieval_results),
                    'generation_model': self.generator.default_model
                }
            )
            
            return output
            
        except RetrievalError as e:
            logger.error(f"Retrieval error: {e}")
            return SystemOutput(
                id=query.id,
                question=query.text,
                answer="I encountered an error during document retrieval.",
                retrieved_docs=[]
            )
        except Exception as e:
            logger.error(f"Unexpected error in RAG pipeline: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return SystemOutput(
                id=query.id,
                question=query.text,
                answer="An unexpected error occurred.",
                retrieved_docs=[]
            )
    
    def process_batch(self, queries: List[Query],
                     generation_config: Optional[GenerationConfig] = None) -> List[SystemOutput]:
        """
        Process multiple queries.
        
        Args:
            queries: List of Query objects
            generation_config: Generation configuration
            
        Returns:
            List of SystemOutput objects
        """
        results = []
        for i, query in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}")
            result = self.process_query(query, generation_config)
            results.append(result)
        
        return results
    
    def save_results(self, results: List[SystemOutput], output_path: str) -> None:
        """
        Save results to JSONL file.
        
        Args:
            results: List of SystemOutput objects
            output_path: Path to output file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    # Convert to dict matching required format
                    output_dict = {
                        'id': result.id,
                        'question': result.question,
                        'answer': result.answer,
                        'retrieved_docs': result.retrieved_docs
                    }
                    f.write(json.dumps(output_dict, ensure_ascii=False) + '\n')
            
            logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise


class MultiHopRAGPipeline(BasicRAGPipeline):
    """RAG pipeline optimized for multi-hop questions."""
    
    def process_query(self, query: Query,
                     generation_config: Optional[GenerationConfig] = None) -> SystemOutput:
        """
        Process a multi-hop query.
        
        Args:
            query: Query object
            generation_config: Generation configuration
            
        Returns:
            SystemOutput with answer and retrieved documents
        """
        try:
            # Step 1: Retrieve documents
            logger.info(f"Processing multi-hop query: {query.text[:100]}...")
            retrieval_results = self.retriever.retrieve(query, k=self.k_retrieve)
            
            if not retrieval_results:
                logger.warning("No documents retrieved")
                return SystemOutput(
                    id=query.id,
                    question=query.text,
                    answer="I don't have enough information to answer this question.",
                    retrieved_docs=[]
                )
            
            logger.info(f"Retrieved {len(retrieval_results)} documents")
            
            # Step 2: Format retrieved documents
            documents = []
            for result in retrieval_results:
                doc_id = result.document_id
                if hasattr(self.retriever, 'documents') and doc_id in self.retriever.documents:
                    doc = self.retriever.documents[doc_id]
                    documents.append({
                        'id': doc_id,
                        'text': doc.text
                    })
            
            # Step 3: Create multi-hop prompt
            prompt = self.prompt_manager.create_multi_hop_prompt(
                question=query.text,
                documents=documents,
                max_context_length=self.max_context_length
            )
            
            logger.debug(f"Multi-hop prompt length: {len(prompt)} characters")
            
            # Step 4: Generate answer
            try:
                answer = self.generator.generate(prompt, config=generation_config)
                logger.info("Multi-hop answer generated successfully")
            except GenerationError as e:
                logger.error(f"Generation failed: {e}")
                answer = "I encountered an error while generating the answer."
            
            # Step 5: Format output
            retrieved_docs = [
                [result.document_id, float(result.score)]
                for result in retrieval_results
            ]
            
            output = SystemOutput(
                id=query.id,
                question=query.text,
                answer=answer,
                retrieved_docs=retrieved_docs,
                metadata={
                    'retrieval_method': self.retriever.get_method_name(),
                    'num_retrieved': len(retrieval_results),
                    'generation_model': self.generator.default_model,
                    'pipeline_type': 'multi_hop'
                }
            )
            
            return output
            
        except Exception as e:
            logger.error(f"Error in multi-hop RAG pipeline: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return SystemOutput(
                id=query.id,
                question=query.text,
                answer="An error occurred during processing.",
                retrieved_docs=[]
            )
