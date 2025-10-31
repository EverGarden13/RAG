"""
Prompt templates for RAG system.
Optimized for HotpotQA multi-hop question answering.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PromptTemplate:
    """Template for generating prompts."""
    name: str
    template: str
    description: str
    
    def format(self, **kwargs) -> str:
        """
        Format template with provided arguments.
        
        Args:
            **kwargs: Template variables
            
        Returns:
            Formatted prompt string
        """
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing template variable: {e}")
            raise ValueError(f"Missing required template variable: {e}")


class PromptManager:
    """Manager for prompt templates."""
    
    # Basic RAG template optimized for HotpotQA
    BASIC_RAG_TEMPLATE = PromptTemplate(
        name="basic_rag",
        template="""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Instructions:
1. Answer the question based ONLY on the information provided in the context above.
2. If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question."
3. Be concise and direct in your answer.
4. Do not make up information or use knowledge outside the provided context.

Answer:""",
        description="Basic RAG template for single-turn question answering"
    )
    
    # Multi-hop reasoning template for HotpotQA
    MULTI_HOP_TEMPLATE = PromptTemplate(
        name="multi_hop",
        template="""You are an expert at answering complex multi-hop questions that require reasoning across multiple pieces of information.

Context Documents:
{context}

Question: {question}

Instructions:
1. This question may require connecting information from multiple documents.
2. Carefully read all the context documents.
3. Identify the key facts needed to answer the question.
4. Combine the relevant information to form a complete answer.
5. Provide a clear, concise answer based on the context.
6. If the context doesn't provide enough information, state that clearly.

Answer:""",
        description="Template optimized for multi-hop reasoning questions"
    )
    
    # Structured output template
    STRUCTURED_OUTPUT_TEMPLATE = PromptTemplate(
        name="structured_output",
        template="""You are a helpful assistant that provides structured answers to questions.

Context:
{context}

Question: {question}

Instructions:
Provide your answer in the following JSON format:
{{
    "answer": "your concise answer here",
    "confidence": "high/medium/low",
    "supporting_facts": ["fact 1", "fact 2"]
}}

Response:""",
        description="Template for structured JSON output"
    )
    
    # Query reformulation template for multi-turn
    QUERY_REFORMULATION_TEMPLATE = PromptTemplate(
        name="query_reformulation",
        template="""You are an assistant that reformulates follow-up questions into self-contained queries.

Conversation History:
{history}

Current Question: {question}

Instructions:
Rewrite the current question as a self-contained query that includes all necessary context from the conversation history. The reformulated query should be understandable without the conversation history.

Reformulated Query:""",
        description="Template for reformulating queries in multi-turn conversations"
    )
    
    # Query decomposition template for agentic workflow
    QUERY_DECOMPOSITION_TEMPLATE = PromptTemplate(
        name="query_decomposition",
        template="""You are an assistant that breaks down complex questions into simpler sub-questions.

Question: {question}

Instructions:
If this question requires multiple steps or pieces of information to answer, break it down into 2-3 simpler sub-questions. Each sub-question should be answerable independently.

If the question is already simple and direct, respond with "NO_DECOMPOSITION_NEEDED".

Sub-questions (one per line):""",
        description="Template for decomposing complex queries"
    )
    
    # Self-checking template
    SELF_CHECK_TEMPLATE = PromptTemplate(
        name="self_check",
        template="""You are a fact-checker that verifies if an answer is supported by the given context.

Context:
{context}

Question: {question}

Proposed Answer: {answer}

Instructions:
Determine if the proposed answer is:
1. Fully supported by the context (YES)
2. Partially supported but contains unsupported claims (PARTIAL)
3. Not supported by the context (NO)

Provide your assessment and brief explanation.

Assessment:""",
        description="Template for self-checking answer consistency"
    )
    
    # Chain-of-thought template
    CHAIN_OF_THOUGHT_TEMPLATE = PromptTemplate(
        name="chain_of_thought",
        template="""You are an assistant that explains your reasoning step-by-step.

Context:
{context}

Question: {question}

Instructions:
1. Think through the question step by step.
2. Explain your reasoning process.
3. Provide the final answer.

Let's solve this step by step:

Step 1: Identify what the question is asking
Step 2: Find relevant information in the context
Step 3: Reason through the information
Step 4: Formulate the answer

Reasoning:""",
        description="Template for chain-of-thought reasoning"
    )
    
    def __init__(self):
        """Initialize prompt manager with default templates."""
        self.templates = {
            "basic_rag": self.BASIC_RAG_TEMPLATE,
            "multi_hop": self.MULTI_HOP_TEMPLATE,
            "structured_output": self.STRUCTURED_OUTPUT_TEMPLATE,
            "query_reformulation": self.QUERY_REFORMULATION_TEMPLATE,
            "query_decomposition": self.QUERY_DECOMPOSITION_TEMPLATE,
            "self_check": self.SELF_CHECK_TEMPLATE,
            "chain_of_thought": self.CHAIN_OF_THOUGHT_TEMPLATE
        }
    
    def get_template(self, name: str) -> PromptTemplate:
        """
        Get template by name.
        
        Args:
            name: Template name
            
        Returns:
            PromptTemplate instance
            
        Raises:
            ValueError: If template not found
        """
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found. Available: {list(self.templates.keys())}")
        
        return self.templates[name]
    
    def add_template(self, template: PromptTemplate) -> None:
        """
        Add custom template.
        
        Args:
            template: PromptTemplate instance
        """
        self.templates[template.name] = template
        logger.info(f"Added template: {template.name}")
    
    def list_templates(self) -> List[str]:
        """
        List available template names.
        
        Returns:
            List of template names
        """
        return list(self.templates.keys())
    
    def format_context(self, documents: List[Dict[str, Any]], 
                      max_length: Optional[int] = None) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            documents: List of document dictionaries with 'id' and 'text'
            max_length: Maximum context length in characters
            
        Returns:
            Formatted context string
        """
        context_parts = []
        total_length = 0
        
        for i, doc in enumerate(documents, 1):
            doc_text = doc.get('text', '')
            doc_id = doc.get('id', f'doc_{i}')
            
            # Format document
            formatted_doc = f"[Document {i} - ID: {doc_id}]\n{doc_text}\n"
            
            # Check length limit
            if max_length and (total_length + len(formatted_doc) > max_length):
                # Truncate if needed
                remaining = max_length - total_length
                if remaining > 100:  # Only add if meaningful space remains
                    formatted_doc = formatted_doc[:remaining] + "...\n"
                    context_parts.append(formatted_doc)
                break
            
            context_parts.append(formatted_doc)
            total_length += len(formatted_doc)
        
        return "\n".join(context_parts)
    
    def create_basic_rag_prompt(self, question: str, 
                               documents: List[Dict[str, Any]],
                               max_context_length: int = 2000) -> str:
        """
        Create basic RAG prompt.
        
        Args:
            question: User question
            documents: Retrieved documents
            max_context_length: Maximum context length
            
        Returns:
            Formatted prompt
        """
        context = self.format_context(documents, max_context_length)
        template = self.get_template("basic_rag")
        return template.format(context=context, question=question)
    
    def create_multi_hop_prompt(self, question: str,
                               documents: List[Dict[str, Any]],
                               max_context_length: int = 2500) -> str:
        """
        Create multi-hop reasoning prompt.
        
        Args:
            question: User question
            documents: Retrieved documents
            max_context_length: Maximum context length
            
        Returns:
            Formatted prompt
        """
        context = self.format_context(documents, max_context_length)
        template = self.get_template("multi_hop")
        return template.format(context=context, question=question)
