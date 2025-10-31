"""
Terminal-based user interface for RAG system.
Provides interactive command-line interface for querying the system.
"""

import sys
from typing import Optional, List, Dict, Any
from pathlib import Path

from src.models.data_models import Query, SystemOutput
from src.generation.rag_pipeline import BasicRAGPipeline
from src.generation.multi_turn import ConversationStateManager
from src.utils.logging_config import get_logger
from src.utils.validation import sanitize_query, ValidationError

logger = get_logger(__name__)


class TerminalInterface:
    """Terminal-based interface for RAG system."""
    
    def __init__(self, rag_pipeline: BasicRAGPipeline,
                 enable_multi_turn: bool = False):
        """
        Initialize terminal interface.
        
        Args:
            rag_pipeline: RAG pipeline instance
            enable_multi_turn: Enable multi-turn conversation
        """
        self.rag_pipeline = rag_pipeline
        self.enable_multi_turn = enable_multi_turn
        self.conversation_state = ConversationStateManager() if enable_multi_turn else None
        self.query_count = 0
        
    def print_header(self) -> None:
        """Print interface header."""
        print("\n" + "=" * 80)
        print(" " * 25 + "RAG SYSTEM - Terminal Interface")
        print("=" * 80)
        print(f"Retrieval Method: {self.rag_pipeline.retriever.get_method_name()}")
        print(f"Generation Model: {self.rag_pipeline.generator.default_model}")
        print(f"Multi-turn: {'Enabled' if self.enable_multi_turn else 'Disabled'}")
        print("=" * 80)
        print("\nCommands:")
        print("  - Type your question and press Enter")
        print("  - Type 'quit' or 'exit' to quit")
        print("  - Type 'reset' to reset conversation (multi-turn mode)")
        print("  - Type 'help' for more commands")
        print("=" * 80 + "\n")
    
    def print_help(self) -> None:
        """Print help information."""
        print("\n" + "=" * 80)
        print("HELP - Available Commands")
        print("=" * 80)
        print("  quit, exit       - Exit the application")
        print("  reset            - Reset conversation history (multi-turn mode)")
        print("  help             - Show this help message")
        print("  stats            - Show system statistics")
        print("  history          - Show conversation history (multi-turn mode)")
        print("=" * 80 + "\n")
    
    def print_stats(self) -> None:
        """Print system statistics."""
        print("\n" + "=" * 80)
        print("SYSTEM STATISTICS")
        print("=" * 80)
        print(f"Total queries processed: {self.query_count}")
        
        if self.enable_multi_turn and self.conversation_state:
            print(f"Current conversation turn: {self.conversation_state.current_turn}")
            print(f"History length: {len(self.conversation_state.history)}")
        
        gen_stats = self.rag_pipeline.generator.get_stats()
        print(f"API requests made: {gen_stats['request_count']}")
        print("=" * 80 + "\n")
    
    def print_history(self) -> None:
        """Print conversation history."""
        if not self.enable_multi_turn or not self.conversation_state:
            print("\nMulti-turn mode is not enabled.\n")
            return
        
        if not self.conversation_state.history:
            print("\nNo conversation history yet.\n")
            return
        
        print("\n" + "=" * 80)
        print("CONVERSATION HISTORY")
        print("=" * 80)
        print(self.conversation_state.get_history_text())
        print("=" * 80 + "\n")
    
    def format_retrieved_docs(self, retrieved_docs: List[List[Any]],
                             max_display: int = 5) -> str:
        """
        Format retrieved documents for display.
        
        Args:
            retrieved_docs: List of [doc_id, score] pairs
            max_display: Maximum number of documents to display
            
        Returns:
            Formatted string
        """
        if not retrieved_docs:
            return "No documents retrieved"
        
        lines = []
        for i, (doc_id, score) in enumerate(retrieved_docs[:max_display], 1):
            lines.append(f"  {i}. Document ID: {doc_id} (Score: {score:.4f})")
            
            # Try to show document snippet
            if hasattr(self.rag_pipeline.retriever, 'documents'):
                if doc_id in self.rag_pipeline.retriever.documents:
                    doc = self.rag_pipeline.retriever.documents[doc_id]
                    snippet = doc.text[:100] + "..." if len(doc.text) > 100 else doc.text
                    lines.append(f"     {snippet}")
        
        if len(retrieved_docs) > max_display:
            lines.append(f"  ... and {len(retrieved_docs) - max_display} more documents")
        
        return "\n".join(lines)
    
    def display_result(self, result: SystemOutput, show_docs: bool = True,
                      show_metadata: bool = False) -> None:
        """
        Display query result.
        
        Args:
            result: SystemOutput object
            show_docs: Show retrieved documents
            show_metadata: Show metadata
        """
        print("\n" + "-" * 80)
        print("ANSWER:")
        print("-" * 80)
        print(result.answer)
        print("-" * 80)
        
        if show_docs and result.retrieved_docs:
            print("\nRETRIEVED DOCUMENTS:")
            print(self.format_retrieved_docs(result.retrieved_docs))
        
        if show_metadata and result.metadata:
            print("\nMETADATA:")
            for key, value in result.metadata.items():
                if key == 'reasoning_steps' and isinstance(value, list):
                    print(f"  {key}:")
                    for step in value:
                        print(f"    - {step}")
                else:
                    print(f"  {key}: {value}")
        
        print("-" * 80 + "\n")
    
    def process_command(self, command: str) -> bool:
        """
        Process user command.
        
        Args:
            command: User command
            
        Returns:
            True to continue, False to exit
        """
        command = command.strip().lower()
        
        if command in ['quit', 'exit', 'q']:
            print("\nGoodbye!\n")
            return False
        
        elif command == 'help':
            self.print_help()
        
        elif command == 'stats':
            self.print_stats()
        
        elif command == 'history':
            self.print_history()
        
        elif command == 'reset':
            if self.enable_multi_turn and self.conversation_state:
                self.conversation_state.reset()
                print("\nConversation history reset.\n")
            else:
                print("\nMulti-turn mode is not enabled.\n")
        
        elif command == '':
            pass  # Empty input, do nothing
        
        else:
            # Process as query - validate input first
            try:
                sanitized_command = sanitize_query(command)
            except ValidationError as e:
                print(f"\nInvalid query: {e}\n")
                return True
            
            self.query_count += 1
            query_id = f"terminal_q{self.query_count}"
            query = Query(id=query_id, text=sanitized_command)
            
            print("\nProcessing query...")
            try:
                result = self.rag_pipeline.process_query(query)
                self.display_result(result, show_docs=True, show_metadata=False)
                
                # Update conversation state if multi-turn
                if self.enable_multi_turn and self.conversation_state:
                    self.conversation_state.add_turn(
                        query=sanitized_command,
                        answer=result.answer,
                        retrieved_docs=result.retrieved_docs
                    )
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"\nError: Unable to process query. Please try again.\n")
        
        return True
    
    def run(self) -> None:
        """Run the terminal interface."""
        self.print_header()
        
        try:
            while True:
                try:
                    user_input = input("Query> ").strip()
                    
                    if not self.process_command(user_input):
                        break
                    
                except KeyboardInterrupt:
                    print("\n\nInterrupted by user. Type 'quit' to exit.\n")
                    continue
                
                except EOFError:
                    print("\n\nGoodbye!\n")
                    break
        
        except Exception as e:
            logger.error(f"Fatal error in terminal interface: {e}")
            print(f"\nFatal error: {e}\n")
            sys.exit(1)


def create_terminal_interface(rag_pipeline: BasicRAGPipeline,
                              enable_multi_turn: bool = False) -> TerminalInterface:
    """
    Factory function to create terminal interface.
    
    Args:
        rag_pipeline: RAG pipeline instance
        enable_multi_turn: Enable multi-turn conversation
        
    Returns:
        TerminalInterface instance
    """
    return TerminalInterface(rag_pipeline, enable_multi_turn)
