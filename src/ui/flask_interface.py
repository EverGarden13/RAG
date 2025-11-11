"""
Flask-based web interface for RAG system.
Provides REST API and web UI for query processing with real-time streaming.
"""

import os
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from flask import Flask, render_template, request, jsonify, Response, stream_template
import threading
import queue

from src.models.data_models import Query, SystemOutput
from src.generation.rag_pipeline import BasicRAGPipeline
from src.generation.multi_turn import ConversationStateManager
from src.utils.logging_config import get_logger
from src.utils.validation import sanitize_query, ValidationError

logger = get_logger(__name__)


class FlaskInterface:
    """Flask-based web interface for RAG system."""
    
    def __init__(self, rag_pipeline: Optional[BasicRAGPipeline] = None,
                 enable_multi_turn: bool = False):
        """
        Initialize Flask interface.
        
        Args:
            rag_pipeline: RAG pipeline instance
            enable_multi_turn: Enable multi-turn conversation
        """
        self.app = Flask(__name__, 
                        template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
                        static_folder=os.path.join(os.path.dirname(__file__), 'static'))
        
        self.rag_pipeline = rag_pipeline
        self.enable_multi_turn = enable_multi_turn
        self.conversation_state = ConversationStateManager() if enable_multi_turn else None
        self.query_count = 0
        self.query_history = []
        
        # Thread-safe processing queue
        self.processing_queue = queue.Queue()
        self.results_cache = {}
        
        self.setup_routes()
    
    def setup_routes(self) -> None:
        """Set up Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main page."""
            system_info = self.get_system_info()
            return render_template('index.html', 
                                 system_info=system_info,
                                 enable_multi_turn=self.enable_multi_turn)
        
        @self.app.route('/api/query', methods=['POST'])
        def api_query():
            """Process query via API."""
            try:
                data = request.get_json()
                if not data or 'query' not in data:
                    return jsonify({'error': 'Query text required'}), 400
                
                query_text = data['query'].strip()
                if not query_text:
                    return jsonify({'error': 'Empty query'}), 400
                
                # Validate query
                try:
                    sanitized_query = sanitize_query(query_text)
                except ValidationError as e:
                    return jsonify({'error': f'Invalid query: {e}'}), 400
                
                # Process query
                result = self.process_query_sync(sanitized_query)
                
                if 'error' in result:
                    return jsonify(result), 500
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"API query error: {e}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/api/stream_query', methods=['POST'])
        def api_stream_query():
            """Process query with streaming response."""
            try:
                data = request.get_json()
                if not data or 'query' not in data:
                    return jsonify({'error': 'Query text required'}), 400
                
                query_text = data['query'].strip()
                if not query_text:
                    return jsonify({'error': 'Empty query'}), 400
                
                # Validate query
                try:
                    sanitized_query = sanitize_query(query_text)
                except ValidationError as e:
                    return jsonify({'error': f'Invalid query: {e}'}), 400
                
                # Stream processing
                def generate():
                    yield f"data: {json.dumps({'status': 'processing', 'message': 'Starting query processing...'})}\n\n"
                    
                    try:
                        # Process query with progress updates
                        result = self.process_query_with_progress(sanitized_query, generate)
                        
                        if 'error' in result:
                            yield f"data: {json.dumps({'status': 'error', 'error': result['error']})}\n\n"
                        else:
                            yield f"data: {json.dumps({'status': 'complete', 'result': result})}\n\n"
                    
                    except Exception as e:
                        logger.error(f"Streaming query error: {e}")
                        yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"
                
                return Response(generate(), mimetype='text/event-stream',
                              headers={'Cache-Control': 'no-cache'})
                
            except Exception as e:
                logger.error(f"Stream query error: {e}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/api/history')
        def api_history():
            """Get query history."""
            return jsonify({
                'history': self.query_history[-20:],  # Last 20 queries
                'total_queries': self.query_count
            })
        
        @self.app.route('/api/conversation')
        def api_conversation():
            """Get conversation history."""
            if not self.enable_multi_turn or not self.conversation_state:
                return jsonify({'error': 'Multi-turn not enabled'}), 400
            
            return jsonify({
                'history': self.conversation_state.get_history_dict(),
                'current_turn': self.conversation_state.current_turn
            })
        
        @self.app.route('/api/conversation/reset', methods=['POST'])
        def api_reset_conversation():
            """Reset conversation history."""
            if not self.enable_multi_turn or not self.conversation_state:
                return jsonify({'error': 'Multi-turn not enabled'}), 400
            
            self.conversation_state.reset()
            return jsonify({'message': 'Conversation reset successfully'})
        
        @self.app.route('/api/system/info')
        def api_system_info():
            """Get system information."""
            return jsonify(self.get_system_info())
        
        @self.app.route('/api/system/stats')
        def api_system_stats():
            """Get system statistics."""
            stats = {
                'query_count': self.query_count,
                'history_length': len(self.query_history),
                'multi_turn_enabled': self.enable_multi_turn
            }
            
            if self.rag_pipeline:
                gen_stats = self.rag_pipeline.generator.get_stats()
                stats.update({
                    'api_requests': gen_stats['request_count'],
                    'retrieval_method': self.rag_pipeline.retriever.get_method_name(),
                    'generation_model': self.rag_pipeline.generator.default_model
                })
            
            if self.enable_multi_turn and self.conversation_state:
                stats.update({
                    'conversation_turn': self.conversation_state.current_turn,
                    'conversation_history_length': len(self.conversation_state.history)
                })
            
            return jsonify(stats)
        
        @self.app.route('/api/workflow/capabilities')
        def api_workflow_capabilities():
            """Get workflow capabilities information."""
            capabilities = {
                'has_agentic': hasattr(self.rag_pipeline, 'enable_decomposition') if self.rag_pipeline else False,
                'has_multi_turn': self.enable_multi_turn,
                'supports_reasoning_steps': True,
                'supports_sub_queries': True,
                'supports_self_check': True
            }
            
            if self.rag_pipeline and hasattr(self.rag_pipeline, 'enable_decomposition'):
                capabilities.update({
                    'decomposition_enabled': getattr(self.rag_pipeline, 'enable_decomposition', False),
                    'self_check_enabled': getattr(self.rag_pipeline, 'enable_self_check', False),
                    'cot_enabled': getattr(self.rag_pipeline, 'enable_cot', False)
                })
            
            return jsonify(capabilities)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        info = {
            'status': 'ready' if self.rag_pipeline else 'not_initialized',
            'multi_turn_enabled': self.enable_multi_turn,
            'query_count': self.query_count
        }
        
        if self.rag_pipeline:
            info.update({
                'retrieval_method': self.rag_pipeline.retriever.get_method_name(),
                'generation_model': self.rag_pipeline.generator.default_model
            })
        
        return info
    
    def process_query_sync(self, query_text: str) -> Dict[str, Any]:
        """Process query synchronously."""
        if not self.rag_pipeline:
            return {'error': 'RAG system not initialized'}
        
        try:
            # Create query object
            self.query_count += 1
            query_id = f"flask_q{self.query_count}"
            query = Query(id=query_id, text=query_text)
            
            # Process query
            result = self.rag_pipeline.process_query(query)
            
            # Update conversation state if multi-turn
            if self.enable_multi_turn and self.conversation_state:
                self.conversation_state.add_turn(
                    query=query_text,
                    answer=result.answer,
                    retrieved_docs=result.retrieved_docs
                )
            
            # Add to history
            self.query_history.append({
                'id': query_id,
                'query': query_text,
                'answer': result.answer,
                'timestamp': datetime.now().isoformat(),
                'retrieved_docs_count': len(result.retrieved_docs),
                'metadata': result.metadata
            })
            
            # Format response
            response = {
                'id': result.id,
                'query': result.question,
                'answer': result.answer,
                'retrieved_docs': result.retrieved_docs,
                'metadata': result.metadata,
                'timestamp': datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {'error': f'Query processing failed: {str(e)}'}
    
    def process_query_with_progress(self, query_text: str, yield_func) -> Dict[str, Any]:
        """Process query with progress updates."""
        if not self.rag_pipeline:
            return {'error': 'RAG system not initialized'}
        
        try:
            # Create query object
            self.query_count += 1
            query_id = f"flask_q{self.query_count}"
            query = Query(id=query_id, text=query_text)
            
            # Check if this is an agentic pipeline for detailed progress
            is_agentic = hasattr(self.rag_pipeline, 'enable_decomposition')
            
            if is_agentic:
                # Detailed progress for agentic workflow
                yield_func(f"data: {json.dumps({'status': 'processing', 'message': 'Analyzing query complexity...'})}\n\n")
                time.sleep(0.1)
                
                yield_func(f"data: {json.dumps({'status': 'decomposing', 'message': 'Decomposing complex query...'})}\n\n")
                time.sleep(0.1)
            
            # Progress updates
            yield_func(f"data: {json.dumps({'status': 'retrieving', 'message': 'Retrieving relevant documents...'})}\n\n")
            time.sleep(0.1)
            
            # Process query
            result = self.rag_pipeline.process_query(query)
            
            if is_agentic:
                yield_func(f"data: {json.dumps({'status': 'reasoning', 'message': 'Applying chain-of-thought reasoning...'})}\n\n")
                time.sleep(0.1)
                
                yield_func(f"data: {json.dumps({'status': 'checking', 'message': 'Verifying answer against evidence...'})}\n\n")
                time.sleep(0.1)
            
            yield_func(f"data: {json.dumps({'status': 'generating', 'message': 'Generating final answer...'})}\n\n")
            time.sleep(0.1)
            
            # Update conversation state if multi-turn
            if self.enable_multi_turn and self.conversation_state:
                self.conversation_state.add_turn(
                    query=query_text,
                    answer=result.answer,
                    retrieved_docs=result.retrieved_docs
                )
            
            # Add to history
            self.query_history.append({
                'id': query_id,
                'query': query_text,
                'answer': result.answer,
                'timestamp': datetime.now().isoformat(),
                'retrieved_docs_count': len(result.retrieved_docs),
                'metadata': result.metadata
            })
            
            # Format response
            response = {
                'id': result.id,
                'query': result.question,
                'answer': result.answer,
                'retrieved_docs': result.retrieved_docs,
                'metadata': result.metadata,
                'timestamp': datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {'error': f'Query processing failed: {str(e)}'}
    
    def set_rag_pipeline(self, rag_pipeline: BasicRAGPipeline) -> None:
        """Set the RAG pipeline."""
        self.rag_pipeline = rag_pipeline
        logger.info("RAG pipeline set for Flask interface")
    
    def enable_multi_turn_mode(self, enable: bool = True) -> None:
        """Enable or disable multi-turn mode."""
        self.enable_multi_turn = enable
        if enable and not self.conversation_state:
            self.conversation_state = ConversationStateManager()
        elif not enable:
            self.conversation_state = None
        logger.info(f"Multi-turn mode {'enabled' if enable else 'disabled'}")
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False) -> None:
        """
        Run the Flask application.
        
        Args:
            host: Host address
            port: Port number
            debug: Debug mode
        """
        logger.info(f"Starting Flask web interface on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)


def create_flask_interface(rag_pipeline: Optional[BasicRAGPipeline] = None,
                          enable_multi_turn: bool = False) -> FlaskInterface:
    """
    Factory function to create Flask interface.
    
    Args:
        rag_pipeline: RAG pipeline instance
        enable_multi_turn: Enable multi-turn conversation
        
    Returns:
        FlaskInterface instance
    """
    return FlaskInterface(rag_pipeline, enable_multi_turn)


# HTML Templates (embedded for simplicity)
def create_templates_directory():
    """Create templates directory and HTML files."""
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Main template
    index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG System - Web Interface</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem 0;
            text-align: center;
            margin-bottom: 2rem;
            border-radius: 10px;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .system-info {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-top: 1rem;
        }
        
        .info-item {
            background: rgba(255, 255, 255, 0.2);
            padding: 0.5rem 1rem;
            border-radius: 5px;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 300px;
            gap: 2rem;
        }
        
        .query-section {
            background: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .sidebar {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            height: fit-content;
        }
        
        .form-group {
            margin-bottom: 1.5rem;
        }
        
        label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: bold;
            color: #555;
        }
        
        textarea {
            width: 100%;
            padding: 1rem;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 1rem;
            resize: vertical;
            min-height: 120px;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .button-group {
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        
        button {
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 5px;
            font-size: 1rem;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        
        .btn-primary {
            background-color: #667eea;
            color: white;
        }
        
        .btn-primary:hover {
            background-color: #5a6fd8;
        }
        
        .btn-secondary {
            background-color: #6c757d;
            color: white;
        }
        
        .btn-secondary:hover {
            background-color: #5a6268;
        }
        
        .btn-success {
            background-color: #28a745;
            color: white;
        }
        
        .btn-success:hover {
            background-color: #218838;
        }
        
        .result-section {
            margin-top: 2rem;
            padding: 1.5rem;
            background: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }
        
        .answer-box {
            background: white;
            padding: 1.5rem;
            border-radius: 5px;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .docs-section {
            margin-top: 1rem;
        }
        
        .doc-item {
            background: white;
            padding: 1rem;
            margin-bottom: 0.5rem;
            border-radius: 5px;
            border-left: 3px solid #28a745;
        }
        
        .loading {
            text-align: center;
            padding: 2rem;
            color: #667eea;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .stats-item {
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid #eee;
        }
        
        .history-item {
            padding: 1rem;
            margin-bottom: 0.5rem;
            background: #f8f9fa;
            border-radius: 5px;
            font-size: 0.9rem;
        }
        
        .error {
            background-color: #f8d7da;
            color: #721c24;
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        
        .success {
            background-color: #d4edda;
            color: #155724;
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            
            .system-info {
                flex-direction: column;
                gap: 1rem;
            }
            
            .button-group {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 RAG System</h1>
            <p>Web Interface for Retrieval-Augmented Generation</p>
            <div class="system-info">
                <div class="info-item">
                    <strong>Status:</strong> {{ system_info.status|title }}
                </div>
                {% if system_info.retrieval_method %}
                <div class="info-item">
                    <strong>Retrieval:</strong> {{ system_info.retrieval_method }}
                </div>
                {% endif %}
                {% if system_info.generation_model %}
                <div class="info-item">
                    <strong>Model:</strong> {{ system_info.generation_model }}
                </div>
                {% endif %}
                <div class="info-item">
                    <strong>Multi-turn:</strong> {{ 'Enabled' if enable_multi_turn else 'Disabled' }}
                </div>
            </div>
        </div>
        
        <div class="main-content">
            <div class="query-section">
                <h2>💬 Ask a Question</h2>
                
                <div id="error-message" class="error" style="display: none;"></div>
                <div id="success-message" class="success" style="display: none;"></div>
                
                <form id="query-form">
                    <div class="form-group">
                        <label for="query-input">Enter your question:</label>
                        <textarea id="query-input" name="query" 
                                placeholder="Type your question here... (e.g., 'What is the capital of France?')" 
                                required></textarea>
                    </div>
                    
                    <div class="button-group">
                        <button type="submit" class="btn-primary">🔍 Submit Query</button>
                        <button type="button" id="clear-btn" class="btn-secondary">🧹 Clear</button>
                        <button type="button" id="example-btn" class="btn-success">📋 Example</button>
                    </div>
                </form>
                
                <div id="loading" class="loading" style="display: none;">
                    <div class="spinner"></div>
                    <p id="loading-message">Processing your query...</p>
                </div>
                
                <div id="result-section" class="result-section" style="display: none;">
                    <h3>💡 Answer</h3>
                    <div id="answer-box" class="answer-box"></div>
                    
                    <div id="docs-section" class="docs-section" style="display: none;">
                        <h4>📄 Retrieved Documents</h4>
                        <div id="docs-container"></div>
                    </div>
                </div>
            </div>
            
            <div class="sidebar">
                <h3>📊 Statistics</h3>
                <div id="stats-container">
                    <div class="stats-item">
                        <span>Total Queries:</span>
                        <span id="query-count">{{ system_info.query_count }}</span>
                    </div>
                </div>
                
                <h3 style="margin-top: 2rem;">📜 Recent Queries</h3>
                <div id="history-container">
                    <p>No queries yet.</p>
                </div>
                
                {% if enable_multi_turn %}
                <h3 style="margin-top: 2rem;">💬 Conversation</h3>
                <button id="reset-conversation" class="btn-secondary" style="width: 100%;">
                    🔄 Reset Conversation
                </button>
                {% endif %}
            </div>
        </div>
    </div>
    
    <script>
        const queryForm = document.getElementById('query-form');
        const queryInput = document.getElementById('query-input');
        const clearBtn = document.getElementById('clear-btn');
        const exampleBtn = document.getElementById('example-btn');
        const loadingDiv = document.getElementById('loading');
        const loadingMessage = document.getElementById('loading-message');
        const resultSection = document.getElementById('result-section');
        const answerBox = document.getElementById('answer-box');
        const docsSection = document.getElementById('docs-section');
        const docsContainer = document.getElementById('docs-container');
        const errorMessage = document.getElementById('error-message');
        const successMessage = document.getElementById('success-message');
        const queryCountSpan = document.getElementById('query-count');
        const historyContainer = document.getElementById('history-container');
        
        const exampleQueries = [
            "What is the capital of France?",
            "Who invented the telephone?",
            "What are the benefits of renewable energy?",
            "How does machine learning work?",
            "What is the history of the Internet?"
        ];
        
        function showError(message) {
            errorMessage.textContent = message;
            errorMessage.style.display = 'block';
            successMessage.style.display = 'none';
        }
        
        function showSuccess(message) {
            successMessage.textContent = message;
            successMessage.style.display = 'block';
            errorMessage.style.display = 'none';
        }
        
        function hideMessages() {
            errorMessage.style.display = 'none';
            successMessage.style.display = 'none';
        }
        
        function updateStats() {
            fetch('/api/system/stats')
                .then(response => response.json())
                .then(data => {
                    queryCountSpan.textContent = data.query_count;
                })
                .catch(error => console.error('Error updating stats:', error));
        }
        
        function updateHistory() {
            fetch('/api/history')
                .then(response => response.json())
                .then(data => {
                    if (data.history.length === 0) {
                        historyContainer.innerHTML = '<p>No queries yet.</p>';
                        return;
                    }
                    
                    const historyHtml = data.history.slice(-5).reverse().map(item => `
                        <div class="history-item">
                            <strong>Q:</strong> ${item.query.substring(0, 50)}${item.query.length > 50 ? '...' : ''}<br>
                            <strong>A:</strong> ${item.answer.substring(0, 80)}${item.answer.length > 80 ? '...' : ''}<br>
                            <small>${new Date(item.timestamp).toLocaleString()}</small>
                        </div>
                    `).join('');
                    
                    historyContainer.innerHTML = historyHtml;
                })
                .catch(error => console.error('Error updating history:', error));
        }
        
        queryForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const query = queryInput.value.trim();
            if (!query) {
                showError('Please enter a question.');
                return;
            }
            
            hideMessages();
            loadingDiv.style.display = 'block';
            resultSection.style.display = 'none';
            
            try {
                // Use streaming API for real-time updates
                const response = await fetch('/api/stream_query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query: query })
                });
                
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\\n');
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                
                                if (data.status === 'processing') {
                                    loadingMessage.textContent = data.message;
                                } else if (data.status === 'retrieving') {
                                    loadingMessage.textContent = data.message;
                                } else if (data.status === 'generating') {
                                    loadingMessage.textContent = data.message;
                                } else if (data.status === 'complete') {
                                    loadingDiv.style.display = 'none';
                                    displayResult(data.result);
                                    showSuccess('Query processed successfully!');
                                    updateStats();
                                    updateHistory();
                                } else if (data.status === 'error') {
                                    loadingDiv.style.display = 'none';
                                    showError(data.error);
                                }
                            } catch (parseError) {
                                console.error('Error parsing SSE data:', parseError);
                            }
                        }
                    }
                }
                
            } catch (error) {
                loadingDiv.style.display = 'none';
                showError('Error processing query: ' + error.message);
            }
        });
        
        function displayResult(result) {
            answerBox.textContent = result.answer;
            resultSection.style.display = 'block';
            
            if (result.retrieved_docs && result.retrieved_docs.length > 0) {
                const docsHtml = result.retrieved_docs.slice(0, 5).map((doc, index) => `
                    <div class="doc-item">
                        <strong>Document ${index + 1}</strong><br>
                        <strong>ID:</strong> ${doc[0]}<br>
                        <strong>Score:</strong> ${doc[1].toFixed(4)}
                    </div>
                `).join('');
                
                docsContainer.innerHTML = docsHtml;
                docsSection.style.display = 'block';
            } else {
                docsSection.style.display = 'none';
            }
        }
        
        clearBtn.addEventListener('click', () => {
            queryInput.value = '';
            hideMessages();
            resultSection.style.display = 'none';
        });
        
        exampleBtn.addEventListener('click', () => {
            const randomExample = exampleQueries[Math.floor(Math.random() * exampleQueries.length)];
            queryInput.value = randomExample;
        });
        
        {% if enable_multi_turn %}
        const resetConversationBtn = document.getElementById('reset-conversation');
        resetConversationBtn.addEventListener('click', async () => {
            try {
                const response = await fetch('/api/conversation/reset', {
                    method: 'POST'
                });
                
                if (response.ok) {
                    showSuccess('Conversation reset successfully!');
                } else {
                    showError('Failed to reset conversation.');
                }
            } catch (error) {
                showError('Error resetting conversation: ' + error.message);
            }
        });
        {% endif %}
        
        // Initial load
        updateStats();
        updateHistory();
        
        // Auto-refresh stats and history every 30 seconds
        setInterval(() => {
            updateStats();
            updateHistory();
        }, 30000);
    </script>
</body>
</html>
    """
    
    with open(os.path.join(templates_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(index_html)


if __name__ == "__main__":
    # Create templates when running standalone
    create_templates_directory()
    
    # For standalone running
    interface = create_flask_interface()
    interface.run(debug=True)