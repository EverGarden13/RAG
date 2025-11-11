# RAG System Web Interface

This directory contains web-based user interfaces for the RAG system, providing both Streamlit and Flask implementations.

## Features

### Core Functionality
- **Clean UI**: Modern, responsive design for query input and result display
- **Real-time Processing**: Live updates during query processing with progress indicators
- **Document Display**: Shows retrieved documents with relevance scores and content snippets
- **Multi-turn Support**: Optional conversation history and context management
- **Statistics**: Real-time system statistics and query history

### Interface Options

#### 1. Streamlit Interface (`web_interface.py`)
- **Modern UI**: Clean, interactive interface with real-time updates
- **Built-in Components**: Uses Streamlit's native components for forms, displays, and controls
- **Session Management**: Automatic session state management
- **Easy Deployment**: Simple to deploy and share

#### 2. Flask Interface (`flask_interface.py`)
- **REST API**: Full REST API for programmatic access
- **Custom HTML/CSS**: Fully customizable interface design
- **Streaming Support**: Server-sent events for real-time updates
- **Production Ready**: Suitable for production deployment

## Quick Start

### Option 1: Using the Launcher Script
```bash
# Start Streamlit interface (default)
python run_web.py

# Start Flask interface
python run_web.py --framework flask

# Custom settings
python run_web.py --framework streamlit --port 8080 --retrieval hybrid
```

### Option 2: Using Main Application
```bash
# Streamlit interface
python main.py --mode web --web-framework streamlit --host localhost --port 8501

# Flask interface  
python main.py --mode web --web-framework flask --host localhost --port 5000
```

### Option 3: Direct Launch
```bash
# Streamlit
streamlit run src/ui/web_interface.py

# Flask
python src/ui/flask_interface.py
```

## Configuration

### Environment Variables
```bash
# Required: OpenRouter API key
export OPENROUTER_API_KEY="your-api-key-here"

# Optional: Streamlit configuration
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=localhost
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

### Command Line Options
- `--framework`: Choose between 'streamlit' or 'flask'
- `--host`: Host address (default: localhost)
- `--port`: Port number (default: 8501 for Streamlit, 5000 for Flask)
- `--retrieval-method`: Retrieval method to use
- `--model`: OpenRouter model for generation
- `--multi-turn`: Enable multi-turn conversation support

## API Endpoints (Flask)

### Query Processing
- `POST /api/query`: Process a single query
- `POST /api/stream_query`: Process query with streaming response

### History and State
- `GET /api/history`: Get query history
- `GET /api/conversation`: Get conversation history
- `POST /api/conversation/reset`: Reset conversation

### System Information
- `GET /api/system/info`: Get system information
- `GET /api/system/stats`: Get system statistics

## File Structure

```
src/ui/
├── __init__.py                 # Module initialization
├── web_interface.py           # Streamlit interface
├── flask_interface.py         # Flask interface
├── web_app.py                # Web application launcher
├── terminal_interface.py      # Terminal interface
├── templates/                 # Flask HTML templates
│   └── index.html            # Main web page
└── README.md                 # This file
```

## Usage Examples

### Basic Query Processing
1. Start the web interface
2. Enter your question in the text area
3. Click "Submit Query" 
4. View the generated answer and retrieved documents

### Multi-turn Conversations
1. Enable multi-turn mode in settings
2. Ask follow-up questions that reference previous context
3. Use "Reset Conversation" to start fresh

### Advanced Features
- View retrieved document content by expanding document items
- Monitor system statistics in the sidebar
- Review query history for previous interactions
- Adjust display settings for documents and metadata

## Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```
   Error: OPENROUTER_API_KEY not set
   Solution: Set the environment variable or create .env file
   ```

2. **Port Already in Use**
   ```
   Error: Address already in use
   Solution: Use a different port with --port option
   ```

3. **Import Errors**
   ```
   Error: Module not found
   Solution: Ensure you're running from the project root directory
   ```

4. **Streamlit Not Found**
   ```
   Error: Streamlit not available
   Solution: Install with pip install streamlit
   ```

### Performance Tips
- Use hybrid retrieval for better accuracy
- Enable caching for faster repeated queries
- Monitor memory usage with large document collections
- Use appropriate model size based on requirements

## Development

### Adding New Features
1. Extend the base interface classes
2. Add new API endpoints for Flask
3. Create corresponding UI components
4. Update the documentation

### Customization
- Modify CSS in templates for custom styling
- Add new Streamlit components for enhanced functionality
- Extend API endpoints for additional features
- Customize prompt templates and processing logic

## Dependencies

### Required
- `streamlit>=1.25.0` (for Streamlit interface)
- `flask>=2.3.0` (for Flask interface)
- `requests>=2.31.0` (for API calls)

### Optional
- `python-dotenv>=1.0.0` (for environment management)
- `pyyaml>=6.0` (for configuration files)

## License

This web interface is part of the RAG system project and follows the same licensing terms.