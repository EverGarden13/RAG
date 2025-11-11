#!/usr/bin/env python3
"""
Simple launcher for RAG system web interface.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main launcher for web interface."""
    parser = argparse.ArgumentParser(description="RAG System Web Interface Launcher")
    
    # Web framework choice
    parser.add_argument('--framework', choices=['streamlit', 'flask'], default='streamlit',
                       help='Web framework to use (default: streamlit)')
    
    # Server settings
    parser.add_argument('--host', default='localhost',
                       help='Host address (default: localhost)')
    parser.add_argument('--port', type=int, default=8501,
                       help='Port number (default: 8501 for Streamlit, 5000 for Flask)')
    
    # RAG system settings
    parser.add_argument('--retrieval', default='bm25',
                       choices=['bm25', 'e5-base', 'hybrid'],
                       help='Retrieval method (default: bm25)')
    parser.add_argument('--model', default='qwen/qwen-2.5-1.5b-instruct',
                       help='OpenRouter model (default: qwen-2.5-1.5b-instruct)')
    
    args = parser.parse_args()
    
    # Adjust default port for Flask
    if args.framework == 'flask' and args.port == 8501:
        args.port = 5000
    
    # Check for API key
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ Error: OPENROUTER_API_KEY environment variable not set.")
        print("Please set your OpenRouter API key:")
        print("  export OPENROUTER_API_KEY='your-api-key-here'")
        print("Or create a .env file with:")
        print("  OPENROUTER_API_KEY=your-api-key-here")
        sys.exit(1)
    
    # Build command
    cmd_args = [
        sys.executable, 'main.py',
        '--mode', 'web',
        '--web-framework', args.framework,
        '--host', args.host,
        '--port', str(args.port),
        '--retrieval-method', args.retrieval,
        '--model', args.model
    ]
    
    print(f"🚀 Starting RAG System Web Interface")
    print(f"   Framework: {args.framework}")
    print(f"   URL: http://{args.host}:{args.port}")
    print(f"   Retrieval: {args.retrieval}")
    print(f"   Model: {args.model}")
    print(f"   Press Ctrl+C to stop")
    print("-" * 50)
    
    # Run the main application
    os.execv(sys.executable, cmd_args)


if __name__ == "__main__":
    main()