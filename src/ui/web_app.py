"""
Web application launcher for RAG system.
Provides both Streamlit and Flask web interfaces.
"""

import os
import sys
import argparse
import threading
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset_loader import HQSmallLoader
from src.models.data_models import Query
from src.retrieval.sparse import BM25Retriever
from src.retrieval.dense_factory import create_dense_retriever
from src.retrieval.hybrid import HybridRetriever
from src.generation.openrouter_client import OpenRouterClient
from src.generation.rag_pipeline import BasicRAGPipeline, MultiHopRAGPipeline
from src.ui.web_interface import run_streamlit_app
from src.ui.flask_interface import create_flask_interface, create_templates_directory
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def setup_rag_system(args):
    """Set up the RAG system components."""
    logger.info("Setting up RAG system for web interface...")
    
    # Load documents
    loader = HQSmallLoader(cache_dir=args.data_dir)
    documents = loader.load_collection()
    logger.info(f"Loaded {len(documents)} documents")
    
    # Create retriever
    if args.retrieval_method == "bm25":
        retriever = BM25Retriever(index_path=os.path.join(args.index_dir, "bm25_index.pkl"))
    elif args.retrieval_method in ["e5-base", "e5-large", "bge-base", "bge-large", "gte-base"]:
        retriever = create_dense_retriever(
            method=args.retrieval_method,
            index_path=os.path.join(args.index_dir, f"{args.retrieval_method}_index.pkl")
        )
    elif args.retrieval_method == "hybrid":
        # Create hybrid retriever with BM25 + E5
        bm25 = BM25Retriever(index_path=os.path.join(args.index_dir, "bm25_index.pkl"))
        e5 = create_dense_retriever(
            method="e5-base",
            index_path=os.path.join(args.index_dir, "e5-base_index.pkl")
        )
        
        # Index documents if needed
        if not bm25.load_index():
            logger.info("Indexing documents with BM25...")
            bm25.index_documents(documents)
        
        if not e5.load_index():
            logger.info("Indexing documents with E5...")
            e5.index_documents(documents)
        
        retriever = HybridRetriever(
            retrievers=[bm25, e5],
            weights=[0.5, 0.5],
            fusion_method="rrf"
        )
    else:
        raise ValueError(f"Unknown retrieval method: {args.retrieval_method}")
    
    # Load or create index for non-hybrid retrievers
    if args.retrieval_method != "hybrid":
        if not retriever.load_index():
            logger.info("Index not found. Creating new index...")
            retriever.index_documents(documents)
        else:
            logger.info("Loaded existing index")
    
    # Set up generator
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in environment")
    
    generator = OpenRouterClient(
        api_key=api_key,
        default_model=args.model
    )
    
    # Create RAG pipeline
    if args.pipeline_type == "basic":
        pipeline = BasicRAGPipeline(
            retriever=retriever,
            generator=generator,
            k_retrieve=args.k_retrieve
        )
    elif args.pipeline_type == "multi_hop":
        pipeline = MultiHopRAGPipeline(
            retriever=retriever,
            generator=generator,
            k_retrieve=args.k_retrieve
        )
    else:
        # For now, use basic pipeline as fallback
        pipeline = BasicRAGPipeline(
            retriever=retriever,
            generator=generator,
            k_retrieve=args.k_retrieve
        )
    
    logger.info("RAG system setup complete")
    return pipeline


def run_streamlit(args, rag_pipeline):
    """Run Streamlit web interface."""
    logger.info(f"Starting Streamlit interface on port {args.port}")
    
    # Set Streamlit configuration
    os.environ['STREAMLIT_SERVER_PORT'] = str(args.port)
    os.environ['STREAMLIT_SERVER_ADDRESS'] = args.host
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    
    try:
        import streamlit.web.cli as stcli
        import sys
        
        # Create a temporary script to run
        script_content = f"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ui.web_interface import run_streamlit_app

# The RAG pipeline will be passed via session state
run_streamlit_app()
"""
        
        script_path = os.path.join(args.temp_dir, "streamlit_app.py")
        os.makedirs(args.temp_dir, exist_ok=True)
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Run Streamlit
        sys.argv = ["streamlit", "run", script_path, "--server.port", str(args.port), 
                   "--server.address", args.host]
        stcli.main()
        
    except ImportError:
        logger.error("Streamlit not available. Please install with: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running Streamlit: {e}")
        sys.exit(1)


def run_flask(args, rag_pipeline):
    """Run Flask web interface."""
    logger.info(f"Starting Flask interface on {args.host}:{args.port}")
    
    try:
        # Create templates directory
        create_templates_directory()
        
        # Create Flask interface
        flask_interface = create_flask_interface(
            rag_pipeline=rag_pipeline,
            enable_multi_turn=args.multi_turn
        )
        
        # Run Flask app
        flask_interface.run(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
        
    except Exception as e:
        logger.error(f"Error running Flask: {e}")
        sys.exit(1)


def main():
    """Main entry point for web application."""
    parser = argparse.ArgumentParser(description="RAG System Web Interface")
    
    # Web framework choice
    parser.add_argument('--framework', choices=['streamlit', 'flask'], default='streamlit',
                       help='Web framework to use')
    
    # Server settings
    parser.add_argument('--host', default='localhost',
                       help='Host address to bind to')
    parser.add_argument('--port', type=int, default=8501,
                       help='Port number to bind to')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode (Flask only)')
    
    # RAG system settings
    parser.add_argument('--retrieval-method', default='bm25',
                       choices=['bm25', 'e5-base', 'e5-large', 'bge-base', 'bge-large', 
                               'gte-base', 'hybrid'],
                       help='Retrieval method to use')
    parser.add_argument('--model', default='qwen/qwen-2.5-1.5b-instruct',
                       help='OpenRouter model to use')
    parser.add_argument('--pipeline-type', default='basic',
                       choices=['basic', 'multi_hop'],
                       help='Pipeline type')
    parser.add_argument('--k-retrieve', type=int, default=10,
                       help='Number of documents to retrieve')
    parser.add_argument('--multi-turn', action='store_true',
                       help='Enable multi-turn conversation')
    
    # Paths
    parser.add_argument('--data-dir', default='./data/cache',
                       help='Data directory')
    parser.add_argument('--index-dir', default='./data/indices',
                       help='Index directory')
    parser.add_argument('--temp-dir', default='./temp',
                       help='Temporary directory')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.index_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)
    
    try:
        # Set up RAG system
        rag_pipeline = setup_rag_system(args)
        
        # Run web interface
        if args.framework == 'streamlit':
            run_streamlit(args, rag_pipeline)
        elif args.framework == 'flask':
            run_flask(args, rag_pipeline)
        
    except KeyboardInterrupt:
        logger.info("\nWeb interface stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()