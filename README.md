# RAG System for Multi-Hop Question Answering

A comprehensive Retrieval-Augmented Generation (RAG) system optimized for multi-hop question answering on the HotpotQA dataset (HQ-small subset).

## Features

### Retrieval Methods (5 Categories)
1. **Sparse Retrieval**
   - BM25
   - TF-IDF

2. **Static Embedding Retrieval**
   - Word2Vec
   - GloVe
   - model2vec

3. **Dense Retrieval**
   - E5 (base, large)
   - BGE (base, large)
   - GTE (base)

4. **Instruction-based Dense Retrieval**
   - E5-Mistral (7B)
   - Qwen3-Embedding

5. **Multi-vector Retrieval**
   - ColBERT
   - GTE-ColBERT

### Generation Capabilities
- **OpenRouter API Integration** with Qwen3, DeepSeek, and other models
- **Prompt Engineering** with optimized templates for multi-hop reasoning
- **Multi-turn Conversations** with context management
- **Agentic Workflow** with query decomposition, self-checking, and chain-of-thought reasoning

### User Interface
- **Terminal Interface** with interactive CLI
- Commands: query, quit, reset, help, stats, history
- Real-time document retrieval display

### Evaluation
- **Exact Match (EM)** with answer normalization
- **nDCG@10** for retrieval quality
- Automated evaluation on validation set

## Installation

```bash
# Clone repository
git clone <repository-url>
cd "NLP Project"

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

## Quick Start

### Interactive Mode

```bash
# Basic RAG with BM25
python main.py --mode interactive --retrieval-method bm25

# With hybrid retrieval
python main.py --mode interactive --retrieval-method hybrid

# With agentic workflow and multi-turn
python main.py --mode interactive --pipeline-type agentic --multi-turn
```

### Batch Processing

```bash
# Process validation set
python main.py --mode batch --split validation --output validation_results.jsonl

# Process test set
python main.py --mode batch --split test --output test_results.jsonl

# With different retrieval method
python main.py --mode batch --retrieval-method e5-base --split validation
```

## Project Structure

```
NLP Project/
├── src/
│   ├── data/               # Dataset loading and processing
│   ├── retrieval/          # All retrieval implementations
│   │   ├── sparse.py       # BM25, TF-IDF
│   │   ├── static_embedding.py  # Word2Vec, GloVe, model2vec
│   │   ├── dense.py        # E5, BGE, GTE
│   │   ├── instruction_dense.py  # E5-Mistral, Qwen
│   │   ├── multi_vector.py # ColBERT
│   │   └── hybrid.py       # Hybrid retrieval with fusion
│   ├── generation/         # Generation and RAG pipelines
│   │   ├── openrouter_client.py  # API client
│   │   ├── prompt_templates.py   # Prompt engineering
│   │   ├── rag_pipeline.py       # Basic and multi-hop RAG
│   │   ├── multi_turn.py         # Conversation management
│   │   └── agentic_workflow.py   # Agentic reasoning
│   ├── ui/                 # User interfaces
│   │   └── terminal_interface.py
│   ├── evaluation/         # Evaluation metrics
│   │   └── metrics.py      # EM and nDCG@10
│   ├── models/             # Data models
│   ├── interfaces/         # Base interfaces
│   └── utils/              # Utilities
├── data/                   # Data directory
│   ├── cache/             # Cached datasets
│   └── indices/           # Saved indices
├── main.py                # Main entry point
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Usage Examples

### Example 1: Simple Query

```python
from src.models.data_models import Query, Document
from src.retrieval.sparse import BM25Retriever
from src.generation.openrouter_client import OpenRouterClient
from src.generation.rag_pipeline import BasicRAGPipeline

# Create documents
documents = [
    Document(id="doc1", text="Python was created by Guido van Rossum."),
    Document(id="doc2", text="The Eiffel Tower is in Paris, France."),
]

# Initialize retriever
retriever = BM25Retriever()
retriever.index_documents(documents)

# Initialize generator
generator = OpenRouterClient()

# Create pipeline
pipeline = BasicRAGPipeline(retriever, generator)

# Process query
query = Query(id="q1", text="Who created Python?")
result = pipeline.process_query(query)

print(f"Answer: {result.answer}")
```

### Example 2: Hybrid Retrieval

```python
from src.retrieval.sparse import BM25Retriever
from src.retrieval.dense_factory import create_dense_retriever
from src.retrieval.hybrid import HybridRetriever

# Create retrievers
bm25 = BM25Retriever()
e5 = create_dense_retriever("e5-base")

# Index documents
bm25.index_documents(documents)
e5.index_documents(documents)

# Create hybrid retriever
hybrid = HybridRetriever(
    retrievers=[bm25, e5],
    weights=[0.5, 0.5],
    fusion_method="rrf"  # Reciprocal Rank Fusion
)

# Use in pipeline
pipeline = BasicRAGPipeline(hybrid, generator)
```

### Example 3: Agentic Workflow

```python
from src.generation.agentic_workflow import AgenticRAGPipeline

# Create agentic pipeline
agentic_pipeline = AgenticRAGPipeline(
    retriever=retriever,
    generator=generator,
    enable_decomposition=True,  # Query decomposition
    enable_self_check=True,     # Answer verification
    enable_cot=True,            # Chain-of-thought
    k_retrieve=10
)

# Process complex query
query = Query(id="q1", text="What is the capital of the country where the Eiffel Tower is located?")
result = agentic_pipeline.process_query(query)

# Check reasoning steps
print(result.metadata['reasoning_steps'])
```

## Configuration

### Environment Variables (.env)

```bash
# OpenRouter API
OPENROUTER_API_KEY=your_api_key_here

# Model Configuration
QWEN_MODEL=qwen/qwen3-coder:free
MAX_TOKENS=512
TEMPERATURE=0.1

# Paths
CACHE_DIR=./data/cache
PROCESSED_DIR=./data/processed
```

### Command Line Arguments

```bash
# Retrieval methods
--retrieval-method {bm25,e5-base,e5-large,bge-base,bge-large,gte-base,hybrid}

# Pipeline types
--pipeline-type {basic,multi_hop,agentic}

# Generation model
--model qwen/qwen3-coder:free

# Number of documents to retrieve
--k-retrieve 10

# Enable multi-turn conversation
--multi-turn

# Data paths
--data-dir ./data/cache
--index-dir ./data/indices
```

## Evaluation

The system uses two metrics:

1. **Exact Match (EM)**: Measures answer accuracy
   - Normalizes answers (lowercase, remove punctuation/articles)
   - Binary score (1.0 for exact match, 0.0 otherwise)

2. **nDCG@10**: Measures retrieval quality
   - Evaluates top-10 retrieved documents
   - Considers ranking order

### Running Evaluation

```bash
# Automatic evaluation on validation set
python main.py --mode batch --split validation --output results.jsonl

# Manual evaluation
python -c "
from src.evaluation.metrics import RAGEvaluator
evaluator = RAGEvaluator(k=10)
results = evaluator.evaluate_from_files('predictions.jsonl', 'references.jsonl')
print(results)
"
```

## Performance Optimization

### Indexing
- Indices are automatically saved and loaded
- Use `--index-dir` to specify index location
- Reuse indices across runs for faster startup

### Rate Limiting
- Free tier models have rate limits
- System automatically retries with exponential backoff
- Adjust `retry_delay` in OpenRouterClient for slower/faster retries

### Memory Management
- Large models (E5-Mistral 7B) require significant GPU memory
- Use smaller models (E5-base, BGE-base) for CPU-only systems
- ColBERT stores token-level embeddings (memory-intensive)

## Troubleshooting

### OpenRouter API Issues

**Problem**: "No endpoints found matching your data policy"
**Solution**: Configure privacy settings at https://openrouter.ai/settings/privacy

**Problem**: Rate limit exceeded
**Solution**: Use paid tier or increase retry delays

### Model Loading Issues

**Problem**: CUDA out of memory
**Solution**: Use smaller models or CPU-only mode

**Problem**: Model download slow
**Solution**: Models are cached in `~/.cache/huggingface/`

### Retrieval Issues

**Problem**: No documents retrieved
**Solution**: Ensure documents are indexed before querying

**Problem**: Poor retrieval quality
**Solution**: Try hybrid retrieval or instruction-based models

## Testing

```bash
# Test OpenRouter connection
python test_simple_openrouter.py

# Test full pipeline
python test_full_pipeline.py

# Test specific retrieval method
python test_instruction_retrieval.py
```

## Dependencies

Key dependencies:
- `transformers` - HuggingFace models
- `sentence-transformers` - Dense embeddings
- `faiss-cpu` - Vector similarity search
- `bm25s` - BM25 implementation
- `requests` - API calls
- `numpy` - Numerical operations
- `torch` - Deep learning framework

See `requirements.txt` for complete list.

## Citation

If you use this system, please cite:

```bibtex
@misc{rag-system-2025,
  title={RAG System for Multi-Hop Question Answering},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/your-repo}}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- HotpotQA dataset creators
- HuggingFace for model hosting
- OpenRouter for API access
- Open-source community for retrieval implementations

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email].
