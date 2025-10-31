"""
Multi-turn conversation management for RAG system.
Handles conversation state, query reformulation, and context management.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

from src.models.data_models import Query, SystemOutput
from src.generation.openrouter_client import OpenRouterClient, GenerationConfig
from src.generation.prompt_templates import PromptManager
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    turn_number: int
    query: str
    answer: str
    retrieved_docs: List[List[Any]] = field(default_factory=list)
    entities: Dict[str, Any] = field(default_factory=dict)


class EntityTracker:
    """Tracks entities mentioned in conversation."""
    
    def __init__(self):
        """Initialize entity tracker."""
        self.entities = {}  # entity_name -> {type, value, turn_number}
        
    def extract_entities(self, text: str, turn_number: int) -> Dict[str, Any]:
        """
        Extract entities from text (simple implementation).
        
        Args:
            text: Text to extract entities from
            turn_number: Current turn number
            
        Returns:
            Dictionary of extracted entities
        """
        # Simple entity extraction based on capitalization and common patterns
        entities = {}
        
        # Look for capitalized words (potential names/places)
        words = text.split()
        for i, word in enumerate(words):
            if word and word[0].isupper() and len(word) > 1:
                # Check if it's not a sentence start
                if i > 0 or (i == 0 and not text[0].isupper()):
                    entity_name = word.strip('.,!?;:')
                    if entity_name:
                        entities[entity_name] = {
                            'type': 'ENTITY',
                            'value': entity_name,
                            'turn_number': turn_number
                        }
        
        return entities
    
    def update_entities(self, entities: Dict[str, Any]) -> None:
        """
        Update tracked entities.
        
        Args:
            entities: New entities to add/update
        """
        self.entities.update(entities)
    
    def get_entity(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get entity by name.
        
        Args:
            name: Entity name
            
        Returns:
            Entity dictionary or None
        """
        return self.entities.get(name)
    
    def get_recent_entities(self, n: int = 5) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Get most recent entities.
        
        Args:
            n: Number of entities to return
            
        Returns:
            List of (name, entity) tuples
        """
        sorted_entities = sorted(
            self.entities.items(),
            key=lambda x: x[1].get('turn_number', 0),
            reverse=True
        )
        return sorted_entities[:n]


class ConversationStateManager:
    """Manages conversation state and history."""
    
    def __init__(self, max_history: int = 10):
        """
        Initialize conversation state manager.
        
        Args:
            max_history: Maximum number of turns to keep in history
        """
        self.max_history = max_history
        self.conversation_id = None
        self.history = deque(maxlen=max_history)
        self.entity_tracker = EntityTracker()
        self.current_turn = 0
        
    def add_turn(self, query: str, answer: str, 
                 retrieved_docs: List[List[Any]] = None) -> ConversationTurn:
        """
        Add a turn to conversation history.
        
        Args:
            query: User query
            answer: System answer
            retrieved_docs: Retrieved documents
            
        Returns:
            ConversationTurn object
        """
        self.current_turn += 1
        
        # Extract entities from query and answer
        query_entities = self.entity_tracker.extract_entities(query, self.current_turn)
        answer_entities = self.entity_tracker.extract_entities(answer, self.current_turn)
        
        # Update entity tracker
        self.entity_tracker.update_entities(query_entities)
        self.entity_tracker.update_entities(answer_entities)
        
        # Create turn
        turn = ConversationTurn(
            turn_number=self.current_turn,
            query=query,
            answer=answer,
            retrieved_docs=retrieved_docs or [],
            entities={**query_entities, **answer_entities}
        )
        
        self.history.append(turn)
        logger.debug(f"Added turn {self.current_turn} to conversation history")
        
        return turn
    
    def get_history_text(self, n_turns: Optional[int] = None) -> str:
        """
        Get conversation history as formatted text.
        
        Args:
            n_turns: Number of recent turns to include (None for all)
            
        Returns:
            Formatted history string
        """
        turns_to_include = list(self.history)
        if n_turns:
            turns_to_include = turns_to_include[-n_turns:]
        
        history_parts = []
        for turn in turns_to_include:
            history_parts.append(f"Turn {turn.turn_number}:")
            history_parts.append(f"Q: {turn.query}")
            history_parts.append(f"A: {turn.answer}")
            history_parts.append("")
        
        return "\n".join(history_parts)
    
    def get_context_for_reformulation(self, n_turns: int = 3) -> str:
        """
        Get context for query reformulation.
        
        Args:
            n_turns: Number of recent turns to include
            
        Returns:
            Context string
        """
        recent_turns = list(self.history)[-n_turns:]
        
        context_parts = []
        for turn in recent_turns:
            context_parts.append(f"Previous Q: {turn.query}")
            context_parts.append(f"Previous A: {turn.answer}")
        
        return "\n".join(context_parts)
    
    def reset(self) -> None:
        """Reset conversation state."""
        self.history.clear()
        self.entity_tracker = EntityTracker()
        self.current_turn = 0
        logger.info("Conversation state reset")


class QueryReformulator:
    """Reformulates queries based on conversation context."""
    
    def __init__(self, generator: OpenRouterClient,
                 prompt_manager: Optional[PromptManager] = None):
        """
        Initialize query reformulator.
        
        Args:
            generator: OpenRouter client for reformulation
            prompt_manager: Prompt manager
        """
        self.generator = generator
        self.prompt_manager = prompt_manager or PromptManager()
        
    def reformulate_query(self, query: str, 
                         conversation_state: ConversationStateManager) -> str:
        """
        Reformulate query into self-contained version.
        
        Args:
            query: Current query
            conversation_state: Conversation state manager
            
        Returns:
            Reformulated query
        """
        try:
            # Check if query contains references that need resolution
            if not self._needs_reformulation(query):
                logger.debug("Query doesn't need reformulation")
                return query
            
            # Get conversation context
            context = conversation_state.get_context_for_reformulation()
            
            if not context:
                logger.debug("No conversation context available")
                return query
            
            # Create reformulation prompt
            template = self.prompt_manager.get_template("query_reformulation")
            prompt = template.format(history=context, question=query)
            
            # Generate reformulated query
            logger.info("Reformulating query...")
            reformulated = self.generator.generate(
                prompt,
                config=GenerationConfig(temperature=0.1, max_tokens=100)
            )
            
            # Clean up reformulated query
            reformulated = reformulated.strip()
            
            # Validate reformulation
            if len(reformulated) < 5 or len(reformulated) > 500:
                logger.warning("Reformulation invalid, using original query")
                return query
            
            logger.info(f"Reformulated: '{query}' -> '{reformulated}'")
            return reformulated
            
        except Exception as e:
            logger.error(f"Error reformulating query: {e}")
            return query
    
    def _needs_reformulation(self, query: str) -> bool:
        """
        Check if query needs reformulation.
        
        Args:
            query: Query text
            
        Returns:
            True if reformulation needed
        """
        # Check for pronouns and references
        reference_words = [
            'he', 'she', 'it', 'they', 'them', 'his', 'her', 'their',
            'this', 'that', 'these', 'those',
            'what about', 'how about', 'and', 'also'
        ]
        
        query_lower = query.lower()
        return any(word in query_lower for word in reference_words)


class ContextPruner:
    """Manages context window by pruning old passages."""
    
    def __init__(self, max_context_length: int = 3000):
        """
        Initialize context pruner.
        
        Args:
            max_context_length: Maximum context length in characters
        """
        self.max_context_length = max_context_length
        
    def prune_context(self, conversation_state: ConversationStateManager,
                     current_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prune context to fit within limits.
        
        Args:
            conversation_state: Conversation state
            current_docs: Current retrieved documents
            
        Returns:
            Pruned document list
        """
        # Calculate current context size
        history_text = conversation_state.get_history_text(n_turns=2)
        history_length = len(history_text)
        
        # Calculate available space for documents
        available_length = self.max_context_length - history_length
        
        if available_length <= 0:
            logger.warning("No space for documents after history")
            return []
        
        # Prune documents to fit
        pruned_docs = []
        current_length = 0
        
        for doc in current_docs:
            doc_text = doc.get('text', '')
            doc_length = len(doc_text)
            
            if current_length + doc_length <= available_length:
                pruned_docs.append(doc)
                current_length += doc_length
            else:
                # Try to fit truncated version
                remaining = available_length - current_length
                if remaining > 200:  # Only add if meaningful space remains
                    truncated_doc = {
                        'id': doc.get('id'),
                        'text': doc_text[:remaining] + "..."
                    }
                    pruned_docs.append(truncated_doc)
                break
        
        logger.debug(f"Pruned {len(current_docs)} docs to {len(pruned_docs)} docs")
        return pruned_docs
    
    def should_prune_history(self, conversation_state: ConversationStateManager) -> bool:
        """
        Check if history should be pruned.
        
        Args:
            conversation_state: Conversation state
            
        Returns:
            True if pruning recommended
        """
        history_text = conversation_state.get_history_text()
        return len(history_text) > self.max_context_length * 0.5
