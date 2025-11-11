"""
Web-based user interface for RAG system using Streamlit.
Provides clean UI for query input, document display, and answer presentation.
"""

import streamlit as st
import time
import json
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.models.data_models import Query, SystemOutput
from src.generation.rag_pipeline import BasicRAGPipeline
from src.generation.multi_turn import ConversationStateManager
from src.utils.logging_config import get_logger
from src.utils.validation import sanitize_query, ValidationError

logger = get_logger(__name__)


class WebInterface:
    """Streamlit-based web interface for RAG system."""
    
    def __init__(self):
        """Initialize web interface."""
        self.setup_page_config()
        self.initialize_session_state()
    
    def setup_page_config(self) -> None:
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="RAG System",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            color: #1f77b4;
            margin-bottom: 2rem;
        }
        .query-box {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .answer-box {
            background-color: #e8f4fd;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
            margin: 1rem 0;
        }
        .doc-item {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            border-left: 3px solid #28a745;
        }
        .metadata-box {
            background-color: #fff3cd;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 3px solid #ffc107;
            margin: 1rem 0;
        }
        .reasoning-step {
            background-color: #f8f9fa;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 0.5rem;
            border-left: 4px solid #17a2b8;
        }
        .sub-query-item {
            background-color: #e3f2fd;
            padding: 0.5rem;
            margin: 0.25rem 0;
            border-radius: 0.3rem;
            border-left: 3px solid #2196f3;
        }
        .confidence-indicator {
            padding: 0.5rem;
            border-radius: 0.3rem;
            margin: 0.5rem 0;
        }
        .confidence-high { background-color: #d4edda; color: #155724; }
        .confidence-medium { background-color: #fff3cd; color: #856404; }
        .confidence-low { background-color: #f8d7da; color: #721c24; }
        .stButton > button {
            width: 100%;
            background-color: #1f77b4;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-weight: bold;
        }
        .stButton > button:hover {
            background-color: #1565c0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables."""
        if 'rag_pipeline' not in st.session_state:
            st.session_state.rag_pipeline = None
        
        if 'conversation_state' not in st.session_state:
            st.session_state.conversation_state = None
        
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        
        if 'query_count' not in st.session_state:
            st.session_state.query_count = 0
        
        if 'enable_multi_turn' not in st.session_state:
            st.session_state.enable_multi_turn = False
        
        if 'show_retrieved_docs' not in st.session_state:
            st.session_state.show_retrieved_docs = True
        
        if 'show_metadata' not in st.session_state:
            st.session_state.show_metadata = False
        
        if 'max_docs_display' not in st.session_state:
            st.session_state.max_docs_display = 5
    
    def render_header(self) -> None:
        """Render the main header."""
        st.markdown('<h1 class="main-header">🔍 RAG System - Web Interface</h1>', 
                   unsafe_allow_html=True)
        
        if st.session_state.rag_pipeline:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Retrieval:** {st.session_state.rag_pipeline.retriever.get_method_name()}")
            with col2:
                st.info(f"**Model:** {st.session_state.rag_pipeline.generator.default_model}")
            with col3:
                multi_turn_status = "Enabled" if st.session_state.enable_multi_turn else "Disabled"
                st.info(f"**Multi-turn:** {multi_turn_status}")
    
    def render_sidebar(self) -> None:
        """Render the sidebar with settings and controls."""
        st.sidebar.header("⚙️ Settings")
        
        # System status
        if st.session_state.rag_pipeline:
            st.sidebar.success("✅ System Ready")
        else:
            st.sidebar.error("❌ System Not Initialized")
            st.sidebar.info("Please initialize the system first using the main interface.")
        
        st.sidebar.divider()
        
        # Display settings
        st.sidebar.subheader("Display Options")
        st.session_state.show_retrieved_docs = st.sidebar.checkbox(
            "Show Retrieved Documents", 
            value=st.session_state.show_retrieved_docs
        )
        
        st.session_state.show_metadata = st.sidebar.checkbox(
            "Show Metadata", 
            value=st.session_state.show_metadata
        )
        
        if st.session_state.show_retrieved_docs:
            st.session_state.max_docs_display = st.sidebar.slider(
                "Max Documents to Display", 
                min_value=1, 
                max_value=10, 
                value=st.session_state.max_docs_display
            )
        
        st.sidebar.divider()
        
        # Multi-turn settings
        st.sidebar.subheader("Multi-turn Conversation")
        enable_multi_turn = st.sidebar.checkbox(
            "Enable Multi-turn", 
            value=st.session_state.enable_multi_turn
        )
        
        if enable_multi_turn != st.session_state.enable_multi_turn:
            st.session_state.enable_multi_turn = enable_multi_turn
            if enable_multi_turn and not st.session_state.conversation_state:
                st.session_state.conversation_state = ConversationStateManager()
        
        if st.session_state.enable_multi_turn and st.session_state.conversation_state:
            if st.sidebar.button("🔄 Reset Conversation"):
                st.session_state.conversation_state.reset()
                st.sidebar.success("Conversation reset!")
                st.rerun()
        
        st.sidebar.divider()
        
        # Statistics
        st.sidebar.subheader("📊 Statistics")
        st.sidebar.metric("Total Queries", st.session_state.query_count)
        
        if st.session_state.rag_pipeline:
            gen_stats = st.session_state.rag_pipeline.generator.get_stats()
            st.sidebar.metric("API Requests", gen_stats['request_count'])
        
        if st.session_state.enable_multi_turn and st.session_state.conversation_state:
            st.sidebar.metric("Conversation Turn", st.session_state.conversation_state.current_turn)
            st.sidebar.metric("History Length", len(st.session_state.conversation_state.history))
        
        st.sidebar.divider()
        
        # History
        if st.sidebar.button("📜 Show Query History"):
            self.show_query_history()
    
    def show_query_history(self) -> None:
        """Display query history in an expander."""
        if not st.session_state.query_history:
            st.info("No query history yet.")
            return
        
        with st.expander("📜 Query History", expanded=True):
            for i, entry in enumerate(reversed(st.session_state.query_history[-10:]), 1):
                st.markdown(f"**Query {len(st.session_state.query_history) - i + 1}:** {entry['query']}")
                st.markdown(f"**Time:** {entry['timestamp']}")
                with st.container():
                    st.markdown(f"**Answer:** {entry['answer'][:200]}...")
                st.divider()
    
    def render_query_input(self) -> Optional[str]:
        """Render query input section and return the query if submitted."""
        st.subheader("💬 Ask a Question")
        
        # Query input
        query_text = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="Type your question here... (e.g., 'What is the capital of France?')",
            key="query_input"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            submit_button = st.button("🔍 Submit Query", type="primary")
        
        with col2:
            if st.button("🧹 Clear"):
                st.session_state.query_input = ""
                st.rerun()
        
        with col3:
            if st.button("📋 Example"):
                example_queries = [
                    "What is the capital of France?",
                    "Who invented the telephone?",
                    "What are the benefits of renewable energy?",
                    "How does machine learning work?",
                    "What is the history of the Internet?"
                ]
                import random
                st.session_state.query_input = random.choice(example_queries)
                st.rerun()
        
        if submit_button and query_text.strip():
            return query_text.strip()
        
        return None
    
    def format_retrieved_docs(self, retrieved_docs: List[List[Any]]) -> None:
        """Format and display retrieved documents."""
        if not retrieved_docs:
            st.warning("No documents retrieved")
            return
        
        st.subheader(f"📄 Retrieved Documents (Top {min(len(retrieved_docs), st.session_state.max_docs_display)})")
        
        for i, (doc_id, score) in enumerate(retrieved_docs[:st.session_state.max_docs_display], 1):
            with st.container():
                st.markdown(f"""
                <div class="doc-item">
                    <strong>Document {i}</strong><br>
                    <strong>ID:</strong> {doc_id}<br>
                    <strong>Relevance Score:</strong> {score:.4f}
                </div>
                """, unsafe_allow_html=True)
                
                # Try to show document snippet
                if (st.session_state.rag_pipeline and 
                    hasattr(st.session_state.rag_pipeline.retriever, 'documents') and
                    doc_id in st.session_state.rag_pipeline.retriever.documents):
                    
                    doc = st.session_state.rag_pipeline.retriever.documents[doc_id]
                    snippet = doc.text[:300] + "..." if len(doc.text) > 300 else doc.text
                    
                    with st.expander(f"📖 View Document {i} Content"):
                        st.text(snippet)
        
        if len(retrieved_docs) > st.session_state.max_docs_display:
            st.info(f"... and {len(retrieved_docs) - st.session_state.max_docs_display} more documents")
    
    def display_result(self, result: SystemOutput) -> None:
        """Display query result with formatting."""
        # Answer section
        st.subheader("💡 Answer")
        st.markdown(f"""
        <div class="answer-box">
            {result.answer}
        </div>
        """, unsafe_allow_html=True)
        
        # Retrieved documents section
        if st.session_state.show_retrieved_docs and result.retrieved_docs:
            self.format_retrieved_docs(result.retrieved_docs)
        
        # Workflow visualization (bonus features)
        if result.metadata:
            self.display_workflow_visualization(result)
        
        # Metadata section
        if st.session_state.show_metadata and result.metadata:
            st.subheader("🔧 Metadata")
            
            metadata_text = ""
            for key, value in result.metadata.items():
                if key == 'reasoning_steps' and isinstance(value, list):
                    metadata_text += f"**{key}:**\n"
                    for step in value:
                        metadata_text += f"  - {step}\n"
                else:
                    metadata_text += f"**{key}:** {value}\n"
            
            st.markdown(f"""
            <div class="metadata-box">
                {metadata_text}
            </div>
            """, unsafe_allow_html=True)
    
    def display_workflow_visualization(self, result: SystemOutput) -> None:
        """Display workflow visualization with intermediate steps."""
        metadata = result.metadata or {}
        
        # Check if we have workflow data to visualize
        has_workflow_data = (
            metadata.get('reasoning_steps') or 
            metadata.get('sub_queries') or 
            metadata.get('pipeline_type') == 'agentic' or
            metadata.get('confidence') is not None
        )
        
        if not has_workflow_data:
            return
        
        st.subheader("🔧 Workflow Visualization")
        
        # Create tabs for different visualization aspects
        tab1, tab2, tab3, tab4 = st.tabs(["🧠 Reasoning", "🔍 Sub-queries", "📚 Retrieval", "✅ Verification"])
        
        with tab1:
            self.display_reasoning_steps(metadata.get('reasoning_steps', []))
        
        with tab2:
            self.display_sub_queries(metadata.get('sub_queries', []))
        
        with tab3:
            self.display_retrieval_details(result.retrieved_docs, metadata)
        
        with tab4:
            self.display_self_check_results(metadata)
    
    def display_reasoning_steps(self, reasoning_steps: List[str]) -> None:
        """Display reasoning steps visualization."""
        if not reasoning_steps:
            st.info("No reasoning steps available for this query.")
            return
        
        st.markdown("### Step-by-Step Reasoning Process")
        
        for i, step in enumerate(reasoning_steps, 1):
            # Parse step format: "Step X: Description - Result"
            step_parts = step.split(' - ')
            step_description = step_parts[0] if step_parts else step
            step_result = step_parts[1] if len(step_parts) > 1 else "Completed"
            
            st.markdown(f"""
            <div class="reasoning-step">
                <strong>Step {i}:</strong> {step_description}<br>
                <small><em>Result:</em> {step_result}</small>
            </div>
            """, unsafe_allow_html=True)
    
    def display_sub_queries(self, sub_queries: List[str]) -> None:
        """Display sub-queries visualization."""
        if not sub_queries:
            st.info("No sub-queries were generated for this query.")
            return
        
        st.markdown("### Query Decomposition")
        st.write("The complex query was broken down into the following sub-queries:")
        
        for i, sub_query in enumerate(sub_queries, 1):
            st.markdown(f"""
            <div class="sub-query-item">
                <strong>Sub-query {i}:</strong> {sub_query}
            </div>
            """, unsafe_allow_html=True)
    
    def display_retrieval_details(self, retrieved_docs: List[List[Any]], metadata: Dict[str, Any]) -> None:
        """Display detailed retrieval process information."""
        if not retrieved_docs:
            st.info("No retrieval details available.")
            return
        
        st.markdown("### Retrieval Process Details")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Retrieval Method", metadata.get('retrieval_method', 'Unknown'))
        with col2:
            st.metric("Documents Retrieved", len(retrieved_docs))
        
        # Show top retrieved documents with scores
        st.markdown("#### Top Retrieved Documents")
        
        for i, (doc_id, score) in enumerate(retrieved_docs[:5], 1):
            # Create a progress bar for the score
            score_normalized = min(score, 1.0)  # Normalize score to 0-1 range
            
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**Document {i}:** {doc_id}")
            with col2:
                st.write(f"Score: {score:.4f}")
            with col3:
                st.progress(score_normalized)
    
    def display_self_check_results(self, metadata: Dict[str, Any]) -> None:
        """Display self-checking and verification results."""
        confidence = metadata.get('confidence')
        
        if confidence is None:
            st.info("No verification results available for this query.")
            return
        
        st.markdown("### Answer Verification Results")
        
        # Determine confidence level and color
        if confidence > 0.8:
            confidence_level = "High"
            confidence_class = "confidence-high"
            confidence_text = "Answer is well supported by the retrieved evidence."
        elif confidence > 0.6:
            confidence_level = "Medium"
            confidence_class = "confidence-medium"
            confidence_text = "Answer is partially supported by the retrieved evidence."
        else:
            confidence_level = "Low"
            confidence_class = "confidence-low"
            confidence_text = "Answer may not be well supported by the retrieved evidence."
        
        # Display confidence metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence Level", confidence_level)
        with col2:
            st.metric("Confidence Score", f"{confidence * 100:.1f}%")
        
        # Display confidence indicator
        st.markdown(f"""
        <div class="confidence-indicator {confidence_class}">
            <strong>Assessment:</strong> {confidence_text}
        </div>
        """, unsafe_allow_html=True)
        
        # Show confidence as a progress bar
        st.progress(confidence)
    
    def process_query(self, query_text: str) -> None:
        """Process a user query and display results."""
        if not st.session_state.rag_pipeline:
            st.error("❌ RAG system not initialized. Please set up the system first.")
            return
        
        # Validate query
        try:
            sanitized_query = sanitize_query(query_text)
        except ValidationError as e:
            st.error(f"❌ Invalid query: {e}")
            return
        
        # Create query object
        st.session_state.query_count += 1
        query_id = f"web_q{st.session_state.query_count}"
        query = Query(id=query_id, text=sanitized_query)
        
        # Display query
        st.markdown(f"""
        <div class="query-box">
            <strong>Query {st.session_state.query_count}:</strong> {sanitized_query}
        </div>
        """, unsafe_allow_html=True)
        
        # Process with progress indicator
        with st.spinner("🔄 Processing your query..."):
            progress_bar = st.progress(0)
            
            try:
                # Simulate progress updates
                progress_bar.progress(25)
                time.sleep(0.1)
                
                # Process query
                result = st.session_state.rag_pipeline.process_query(query)
                progress_bar.progress(75)
                time.sleep(0.1)
                
                # Display result
                progress_bar.progress(100)
                time.sleep(0.1)
                progress_bar.empty()
                
                self.display_result(result)
                
                # Update conversation state if multi-turn
                if st.session_state.enable_multi_turn and st.session_state.conversation_state:
                    st.session_state.conversation_state.add_turn(
                        query=sanitized_query,
                        answer=result.answer,
                        retrieved_docs=result.retrieved_docs
                    )
                
                # Add to query history
                st.session_state.query_history.append({
                    'query': sanitized_query,
                    'answer': result.answer,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'retrieved_docs_count': len(result.retrieved_docs)
                })
                
                # Success message
                st.success("✅ Query processed successfully!")
                
            except Exception as e:
                progress_bar.empty()
                logger.error(f"Error processing query: {e}")
                st.error(f"❌ Error processing query: {str(e)}")
                st.error("Please try again or contact support if the problem persists.")
    
    def render_conversation_history(self) -> None:
        """Render conversation history for multi-turn mode."""
        if not st.session_state.enable_multi_turn or not st.session_state.conversation_state:
            return
        
        if not st.session_state.conversation_state.history:
            return
        
        with st.expander("💬 Conversation History", expanded=False):
            history_text = st.session_state.conversation_state.get_history_text()
            st.text_area("History", value=history_text, height=200, disabled=True)
    
    def run(self, rag_pipeline: Optional[BasicRAGPipeline] = None) -> None:
        """
        Run the web interface.
        
        Args:
            rag_pipeline: RAG pipeline instance (optional, can be set later)
        """
        # Set RAG pipeline if provided
        if rag_pipeline:
            st.session_state.rag_pipeline = rag_pipeline
        
        # Render interface
        self.render_header()
        self.render_sidebar()
        
        # Main content area
        if not st.session_state.rag_pipeline:
            st.warning("⚠️ RAG system not initialized. Please initialize the system to start querying.")
            st.info("Use the main application or CLI to initialize the RAG pipeline, then refresh this page.")
            return
        
        # Query input and processing
        query_text = self.render_query_input()
        
        if query_text:
            self.process_query(query_text)
        
        # Conversation history
        self.render_conversation_history()
        
        # Footer
        st.divider()
        st.markdown("""
        <div style="text-align: center; color: #666; margin-top: 2rem;">
            <p>RAG System Web Interface | Built with Streamlit</p>
        </div>
        """, unsafe_allow_html=True)


def create_web_interface() -> WebInterface:
    """
    Factory function to create web interface.
    
    Returns:
        WebInterface instance
    """
    return WebInterface()


def run_streamlit_app(rag_pipeline: Optional[BasicRAGPipeline] = None) -> None:
    """
    Run the Streamlit web application.
    
    Args:
        rag_pipeline: RAG pipeline instance
    """
    interface = create_web_interface()
    interface.run(rag_pipeline)


if __name__ == "__main__":
    # For standalone running
    run_streamlit_app()