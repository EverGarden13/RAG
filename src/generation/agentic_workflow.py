"""
Agentic workflow implementation for RAG system.
Includes query decomposition, self-checking, and chain-of-thought reasoning.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from src.interfaces.base import BaseRetriever
from src.models.data_models import Query, SystemOutput
from src.generation.openrouter_client import OpenRouterClient, GenerationConfig
from src.generation.prompt_templates import PromptManager
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ReasoningStep:
    """Represents a single reasoning step."""
    step_number: int
    description: str
    action: str  # 'decompose', 'retrieve', 'generate', 'check'
    result: Any = None
    confidence: float = 0.0


class QueryDecomposer:
    """Decomposes complex queries into simpler sub-queries."""
    
    def __init__(self, generator: OpenRouterClient,
                 prompt_manager: Optional[PromptManager] = None):
        """
        Initialize query decomposer.
        
        Args:
            generator: OpenRouter client
            prompt_manager: Prompt manager
        """
        self.generator = generator
        self.prompt_manager = prompt_manager or PromptManager()
        
    def decompose_query(self, query: str) -> List[str]:
        """
        Decompose complex query into sub-queries.
        
        Args:
            query: Complex query to decompose
            
        Returns:
            List of sub-queries (empty if no decomposition needed)
        """
        try:
            # Create decomposition prompt
            template = self.prompt_manager.get_template("query_decomposition")
            prompt = template.format(question=query)
            
            # Generate sub-queries
            logger.info("Decomposing query...")
            response = self.generator.generate(
                prompt,
                config=GenerationConfig(temperature=0.1, max_tokens=200)
            )
            
            # Parse response
            response = response.strip()
            
            # Check if decomposition is needed
            if "NO_DECOMPOSITION_NEEDED" in response.upper():
                logger.info("Query doesn't need decomposition")
                return []
            
            # Extract sub-queries (one per line)
            sub_queries = []
            for line in response.split('\n'):
                line = line.strip()
                # Remove numbering and bullet points
                line = line.lstrip('0123456789.-) ')
                if line and len(line) > 5:
                    sub_queries.append(line)
            
            # Limit to 3 sub-queries
            sub_queries = sub_queries[:3]
            
            if sub_queries:
                logger.info(f"Decomposed into {len(sub_queries)} sub-queries")
                for i, sq in enumerate(sub_queries, 1):
                    logger.debug(f"  Sub-query {i}: {sq}")
            
            return sub_queries
            
        except Exception as e:
            logger.error(f"Error decomposing query: {e}")
            return []


class SelfChecker:
    """Verifies generated answers against retrieved evidence."""
    
    def __init__(self, generator: OpenRouterClient,
                 prompt_manager: Optional[PromptManager] = None):
        """
        Initialize self-checker.
        
        Args:
            generator: OpenRouter client
            prompt_manager: Prompt manager
        """
        self.generator = generator
        self.prompt_manager = prompt_manager or PromptManager()
        
    def check_answer(self, question: str, answer: str, 
                    context: str) -> Tuple[str, float]:
        """
        Check if answer is supported by context.
        
        Args:
            question: Original question
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            Tuple of (assessment, confidence_score)
        """
        try:
            # Create self-check prompt
            template = self.prompt_manager.get_template("self_check")
            prompt = template.format(
                context=context[:1500],  # Limit context length
                question=question,
                answer=answer
            )
            
            # Generate assessment
            logger.info("Performing self-check...")
            response = self.generator.generate(
                prompt,
                config=GenerationConfig(temperature=0.1, max_tokens=150)
            )
            
            response = response.strip().upper()
            
            # Parse assessment
            if "YES" in response[:50]:
                assessment = "SUPPORTED"
                confidence = 0.9
            elif "PARTIAL" in response[:50]:
                assessment = "PARTIAL"
                confidence = 0.6
            elif "NO" in response[:50]:
                assessment = "NOT_SUPPORTED"
                confidence = 0.3
            else:
                assessment = "UNCERTAIN"
                confidence = 0.5
            
            logger.info(f"Self-check result: {assessment} (confidence: {confidence})")
            return assessment, confidence
            
        except Exception as e:
            logger.error(f"Error in self-check: {e}")
            return "ERROR", 0.5


class ChainOfThoughtReasoner:
    """Implements chain-of-thought reasoning."""
    
    def __init__(self, generator: OpenRouterClient,
                 prompt_manager: Optional[PromptManager] = None):
        """
        Initialize CoT reasoner.
        
        Args:
            generator: OpenRouter client
            prompt_manager: Prompt manager
        """
        self.generator = generator
        self.prompt_manager = prompt_manager or PromptManager()
        
    def reason_through_query(self, question: str, context: str) -> Tuple[str, List[str]]:
        """
        Generate answer with explicit reasoning steps.
        
        Args:
            question: Question to answer
            context: Retrieved context
            
        Returns:
            Tuple of (answer, reasoning_steps)
        """
        try:
            # Create CoT prompt
            template = self.prompt_manager.get_template("chain_of_thought")
            prompt = template.format(
                context=context[:2000],  # Limit context
                question=question
            )
            
            # Generate reasoning
            logger.info("Generating chain-of-thought reasoning...")
            response = self.generator.generate(
                prompt,
                config=GenerationConfig(temperature=0.2, max_tokens=400)
            )
            
            # Parse reasoning and answer
            reasoning_steps = []
            answer = ""
            
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('Step') or line.startswith('Reasoning'):
                    reasoning_steps.append(line)
                elif line and not line.startswith('Final') and len(line) > 10:
                    # Potential answer line
                    if not answer:
                        answer = line
            
            # If no clear answer found, use last substantial line
            if not answer and lines:
                answer = lines[-1].strip()
            
            logger.info(f"Generated answer with {len(reasoning_steps)} reasoning steps")
            return answer, reasoning_steps
            
        except Exception as e:
            logger.error(f"Error in CoT reasoning: {e}")
            return "Error generating answer", []


class AgenticRAGPipeline:
    """RAG pipeline with agentic workflow capabilities."""
    
    def __init__(self, retriever: BaseRetriever,
                 generator: OpenRouterClient,
                 prompt_manager: Optional[PromptManager] = None,
                 enable_decomposition: bool = True,
                 enable_self_check: bool = True,
                 enable_cot: bool = True,
                 k_retrieve: int = 10):
        """
        Initialize agentic RAG pipeline.
        
        Args:
            retriever: Retriever instance
            generator: OpenRouter client
            prompt_manager: Prompt manager
            enable_decomposition: Enable query decomposition
            enable_self_check: Enable self-checking
            enable_cot: Enable chain-of-thought reasoning
            k_retrieve: Number of documents to retrieve
        """
        self.retriever = retriever
        self.generator = generator
        self.prompt_manager = prompt_manager or PromptManager()
        self.k_retrieve = k_retrieve
        
        # Agentic components
        self.enable_decomposition = enable_decomposition
        self.enable_self_check = enable_self_check
        self.enable_cot = enable_cot
        
        if enable_decomposition:
            self.decomposer = QueryDecomposer(generator, prompt_manager)
        if enable_self_check:
            self.checker = SelfChecker(generator, prompt_manager)
        if enable_cot:
            self.cot_reasoner = ChainOfThoughtReasoner(generator, prompt_manager)
        
        logger.info(f"Agentic RAG pipeline initialized (decomp={enable_decomposition}, "
                   f"check={enable_self_check}, cot={enable_cot})")
    
    def process_query(self, query: Query) -> SystemOutput:
        """
        Process query with agentic workflow.
        
        Args:
            query: Query object
            
        Returns:
            SystemOutput with answer and reasoning steps
        """
        reasoning_steps = []
        step_num = 0
        
        try:
            logger.info(f"Processing query with agentic workflow: {query.text[:100]}...")
            
            # Step 1: Query decomposition (if enabled)
            sub_queries = []
            if self.enable_decomposition:
                step_num += 1
                reasoning_steps.append(ReasoningStep(
                    step_number=step_num,
                    description="Analyzing query complexity",
                    action="decompose"
                ))
                
                sub_queries = self.decomposer.decompose_query(query.text)
                reasoning_steps[-1].result = sub_queries
            
            # Step 2: Retrieve documents
            step_num += 1
            reasoning_steps.append(ReasoningStep(
                step_number=step_num,
                description="Retrieving relevant documents",
                action="retrieve"
            ))
            
            if sub_queries:
                # Retrieve for each sub-query and combine
                all_results = []
                for sq in sub_queries:
                    sq_query = Query(id=f"{query.id}_sub", text=sq)
                    results = self.retriever.retrieve(sq_query, k=self.k_retrieve // len(sub_queries))
                    all_results.extend(results)
                
                # Remove duplicates and take top k
                seen_ids = set()
                unique_results = []
                for result in all_results:
                    if result.document_id not in seen_ids:
                        seen_ids.add(result.document_id)
                        unique_results.append(result)
                
                retrieval_results = unique_results[:self.k_retrieve]
            else:
                # Single retrieval
                retrieval_results = self.retriever.retrieve(query, k=self.k_retrieve)
            
            reasoning_steps[-1].result = f"Retrieved {len(retrieval_results)} documents"
            
            # Step 3: Format context
            documents = []
            for result in retrieval_results:
                doc_id = result.document_id
                if hasattr(self.retriever, 'documents') and doc_id in self.retriever.documents:
                    doc = self.retriever.documents[doc_id]
                    documents.append({'id': doc_id, 'text': doc.text})
            
            context = self.prompt_manager.format_context(documents, max_length=2000)
            
            # Step 4: Generate answer (with or without CoT)
            step_num += 1
            if self.enable_cot:
                reasoning_steps.append(ReasoningStep(
                    step_number=step_num,
                    description="Reasoning through the question step-by-step",
                    action="generate"
                ))
                
                answer, cot_steps = self.cot_reasoner.reason_through_query(query.text, context)
                reasoning_steps[-1].result = cot_steps
            else:
                reasoning_steps.append(ReasoningStep(
                    step_number=step_num,
                    description="Generating answer from context",
                    action="generate"
                ))
                
                prompt = self.prompt_manager.create_multi_hop_prompt(query.text, documents)
                answer = self.generator.generate(prompt)
                reasoning_steps[-1].result = "Answer generated"
            
            # Step 5: Self-check (if enabled)
            confidence = 0.8  # Default confidence
            if self.enable_self_check:
                step_num += 1
                reasoning_steps.append(ReasoningStep(
                    step_number=step_num,
                    description="Verifying answer against evidence",
                    action="check"
                ))
                
                assessment, confidence = self.checker.check_answer(query.text, answer, context)
                reasoning_steps[-1].result = f"Assessment: {assessment}"
                reasoning_steps[-1].confidence = confidence
                
                # If answer not supported, regenerate with warning
                if assessment == "NOT_SUPPORTED":
                    logger.warning("Answer not supported by evidence, adding disclaimer")
                    answer = f"[Low confidence] {answer}"
            
            # Format output
            retrieved_docs = [[r.document_id, float(r.score)] for r in retrieval_results]
            
            # Add reasoning steps to metadata
            reasoning_summary = [
                f"Step {step.step_number}: {step.description} - {step.result}"
                for step in reasoning_steps
            ]
            
            output = SystemOutput(
                id=query.id,
                question=query.text,
                answer=answer,
                retrieved_docs=retrieved_docs,
                metadata={
                    'retrieval_method': self.retriever.get_method_name(),
                    'pipeline_type': 'agentic',
                    'reasoning_steps': reasoning_summary,
                    'confidence': confidence,
                    'sub_queries': sub_queries
                }
            )
            
            logger.info("Agentic workflow completed successfully")
            return output
            
        except Exception as e:
            logger.error(f"Error in agentic workflow: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            return SystemOutput(
                id=query.id,
                question=query.text,
                answer="An error occurred during agentic processing.",
                retrieved_docs=[],
                metadata={'error': str(e)}
            )
