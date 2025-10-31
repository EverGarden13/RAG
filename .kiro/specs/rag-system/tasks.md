# Implementation Plan

- [x] 1. Set up project structure and data loading





  - Create directory structure for models, data, retrieval, generation, and UI components
  - Implement HQ-small dataset loading and preprocessing utilities
  - Set up configuration management for models, API keys, and system parameters
  - _Requirements: 8.1, 8.2, 8.3_

- [x] 1.1 Create core data models and interfaces


  - Define data classes for Query, Document, RetrievalResult, and SystemOutput
  - Implement base interfaces for retrieval and generation components
  - Create configuration classes for system settings and model parameters
  - _Requirements: 7.1, 7.2, 8.4_

- [x] 1.2 Implement dataset loading and document processing


  - Load HQ-small train, validation, and test splits from HuggingFace
  - Process document collection and create efficient storage format
  - Implement document preprocessing and text cleaning utilities
  - _Requirements: 1.1, 7.3_

- [x] 1.3 Set up logging and error handling framework


  - Create comprehensive logging system for debugging and monitoring
  - Implement error handling classes and recovery mechanisms
  - Set up performance monitoring and metrics collection
  - _Requirements: 8.5_

- [x] 2. Implement sparse retrieval methods





  - Implement BM25 retrieval using bm25s library for baseline performance
  - Create TF-IDF retrieval variant with optimized term weighting
  - Build indexing and search utilities for sparse methods
  - _Requirements: 2.1, 2.7_

- [x] 2.1 Create BM25 retrieval implementation


  - Set up bm25s library integration for document indexing
  - Implement query processing and document scoring for BM25
  - Create efficient storage and loading of BM25 indices
  - _Requirements: 2.1_

- [x] 2.2 Implement TF-IDF retrieval system

  - Build TF-IDF vectorizer using sklearn with custom term weighting
  - Create document indexing and similarity calculation methods
  - Implement query processing and ranking for TF-IDF retrieval
  - _Requirements: 2.1_

- [x] 3. Implement static embedding retrieval methods  -





 Integrate Word2Vec and GloVe embeddings using gensim library
  - Implement model2vec for efficient static embeddings
  - Create document embedding generation and similarity search
  - _Requirements: 2.2, 2.7_

- [x] 3.1 Set up Word2Vec and GloVe integration


  - Load pre-trained Word2Vec and GloVe models using gensim
  - Implement document embedding by averaging word vectors
  - Create similarity search using cosine similarity for static embeddings
  - _Requirements: 2.2_

- [x] 3.2 Implement model2vec static embeddings

  - Integrate model2vec library for efficient static embeddings
  - Generate document embeddings using model2vec approach
  - Implement fast similarity search for model2vec embeddings
  - _Requirements: 2.2_

- [x] 4. Implement dense retrieval with HuggingFace models





  - Set up sentence-transformers integration for E5, BGE, and GTE models
  - Create document embedding generation pipeline using HuggingFace models
  - Implement FAISS indexing for efficient dense retrieval search
  - _Requirements: 2.3, 2.7_

- [x] 4.1 Create E5 model integration


  - Load E5-base and E5-large models from HuggingFace using sentence-transformers
  - Generate document embeddings in batches for efficiency
  - Implement query embedding and similarity search for E5 models
  - _Requirements: 2.3_

- [x] 4.2 Implement BGE model retrieval

  - Integrate BAAI BGE models (base and large) from HuggingFace
  - Create efficient batch processing for document embedding generation
  - Set up similarity search and ranking for BGE embeddings
  - _Requirements: 2.3_

- [x] 4.3 Set up GTE model retrieval

  - Load Alibaba GTE-modernbert-base model from HuggingFace
  - Implement document embedding pipeline for GTE models
  - Create query processing and retrieval functionality for GTE
  - _Requirements: 2.3_

- [x] 5. Implement instruction-based dense retrieval
  - Integrate E5-Mistral and Qwen3-Embedding models from HuggingFace
  - Create instruction-aware query and document processing
  - Implement retrieval with instruction following capabilities
  - _Requirements: 2.4, 2.7_

- [x] 5.1 Create E5-Mistral integration
  - Load intfloat/e5-mistral-7b-instruct model from HuggingFace
  - Implement instruction-based query and document embedding
  - Create retrieval pipeline with instruction awareness
  - _Requirements: 2.4_

- [x] 5.2 Implement Qwen3-Embedding integration
  - Set up Qwen embedding models from HuggingFace if available
  - Create instruction-following embedding generation
  - Implement query processing with instruction capabilities
  - _Requirements: 2.4_

- [x] 6. Implement multi-vector retrieval (ColBERT)
  - Integrate ColBERT and GTE-ColBERT models from HuggingFace
  - Create token-level interaction and scoring mechanisms
  - Implement efficient multi-vector indexing and search
  - _Requirements: 2.5, 2.7_

- [x] 6.1 Set up ColBERT implementation
  - Load colbert-ir/colbertv2.0 model from HuggingFace
  - Implement token-level embedding generation for documents and queries
  - Create MaxSim scoring and ranking for ColBERT retrieval
  - _Requirements: 2.5_

- [x] 6.2 Implement GTE-ColBERT integration
  - Integrate lightonai/GTE-ModernColBERT model from HuggingFace
  - Create efficient multi-vector representation and storage
  - Implement fast similarity search for GTE-ColBERT embeddings
  - _Requirements: 2.5_

- [x] 7. Create hybrid retrieval system
  - Implement score fusion algorithms (RRF, CombSUM) for combining multiple retrievers
  - Create weighted combination system for different retrieval methods
  - Build retrieval ensemble with configurable method selection and weights
  - _Requirements: 2.6, 2.7_

- [x] 7.1 Implement score fusion algorithms
  - Create Reciprocal Rank Fusion (RRF) for combining retrieval results
  - Implement CombSUM and weighted score combination methods
  - Build configurable fusion system with different combination strategies
  - _Requirements: 2.6_

- [x] 7.2 Create hybrid retrieval manager
  - Build system to manage multiple retrieval methods simultaneously
  - Implement configurable weighting and method selection
  - Create efficient parallel retrieval execution and result combination
  - _Requirements: 2.6_

- [x] 8. Set up OpenRouter integration for generation
  - Create OpenRouter API client for Qwen2.5 model access
  - Implement API call management with rate limiting and error handling
  - Set up model selection and configuration for different Qwen2.5 variants
  - _Requirements: 3.1, 3.7_

- [x] 8.1 Create OpenRouter API client
  - Implement HTTP client for OpenRouter API with authentication
  - Create request/response handling for Qwen2.5 models
  - Set up error handling and retry logic for API calls
  - _Requirements: 3.1_

- [x] 8.2 Implement model management for Qwen2.5 variants
  - Create configuration system for different Qwen2.5 model sizes (0.5B, 1.5B, 3B, 7B)
  - Implement model selection based on performance and cost requirements
  - Set up parameter management for temperature, max_tokens, etc.
  - _Requirements: 3.1_

- [x] 9. Implement basic single-turn RAG pipeline
  - Create prompt engineering templates optimized for HotpotQA multi-hop questions
  - Implement context management for retrieved passages within API token limits
  - Build answer extraction and formatting for consistent JSONL output
  - _Requirements: 3.2, 3.3, 3.4, 3.7_

- [x] 9.1 Create prompt engineering system
  - Design prompt templates for different query types and complexity levels
  - Implement context formatting for retrieved passages and user queries
  - Create instruction templates optimized for HotpotQA-style questions
  - _Requirements: 3.2_

- [x] 9.2 Implement context management
  - Create passage selection and truncation for API token limits
  - Implement intelligent context windowing and passage prioritization
  - Build context formatting that maximizes information density
  - _Requirements: 3.3_

- [x] 9.3 Build answer extraction and formatting
  - Create robust parsing for extracting answers from LLM responses
  - Implement JSONL output formatting matching evaluation requirements
  - Set up answer validation and consistency checking
  - _Requirements: 3.4, 7.1, 7.2_

- [x] 10. Implement multi-turn search capability (Feature A)
  - Create conversation state management and entity tracking
  - Implement query reformulation for self-contained retrieval
  - Build context pruning and memory management for long conversations
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 10.1 Create conversation state manager
  - Implement dialogue history storage and entity tracking
  - Create conversation context management with efficient storage
  - Build state persistence and retrieval for multi-turn interactions
  - _Requirements: 5.1, 5.3_

- [x] 10.2 Implement query reformulation
  - Create reference resolution for pronouns and context-dependent terms
  - Implement query expansion using conversation history
  - Build self-contained query generation for effective retrieval
  - _Requirements: 5.2, 5.5_

- [x] 10.3 Build context pruning system
  - Implement intelligent context window management for long conversations
  - Create relevance-based pruning of older retrieved passages
  - Set up memory management to maintain conversation coherence
  - _Requirements: 5.4_

- [x] 11. Implement agentic workflow (Feature B)
  - Create query decomposition for complex multi-hop questions
  - Implement self-checking mechanisms for answer verification
  - Build chain-of-thought reasoning with explicit step tracking
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 11.1 Create query decomposition system
  - Implement complex query breakdown into sub-queries for parallel retrieval
  - Create sub-query generation optimized for HotpotQA multi-hop reasoning
  - Build result synthesis from multiple sub-query answers
  - _Requirements: 6.1_

- [x] 11.2 Implement self-checking mechanisms
  - Create answer verification against retrieved evidence
  - Implement hallucination detection through consistency checking
  - Build confidence scoring and answer validation systems
  - _Requirements: 6.2, 6.3_

- [x] 11.3 Build chain-of-thought reasoning
  - Implement explicit reasoning step generation and tracking
  - Create ReAct-style plan-act-reflect loops for complex queries
  - Build reasoning visualization and step-by-step explanation
  - _Requirements: 6.4_

- [x] 12. Create user interface with bonus features
  - Build terminal-based interface for basic query input and result display
  - Implement web-based interface using Flask or Streamlit
  - Create bonus features showing intermediate reasoning and workflow results
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

- [x] 12.1 Create terminal interface
  - Implement command-line interface for query input and result display
  - Create formatted output showing retrieved documents and generated answers
  - Build interactive mode for continuous querying and testing
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ] 12.2 Build web-based interface
  - Create Flask or Streamlit web application for user interaction
  - Implement clean UI for query input, document display, and answer presentation
  - Set up real-time processing and response streaming
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ] 12.3 Implement bonus UI features
  - Create visualization for intermediate reasoning steps and sub-queries
  - Implement display of retrieval outputs and self-checking results
  - Build workflow visualization showing complete agentic process
  - _Requirements: 4.5, 4.6, 4.7_

- [x] 13. Create evaluation and testing system
  - Implement official evaluation metrics (EM and nDCG@10) matching project requirements
  - Create validation pipeline using HQ-small validation set
  - Build performance optimization and hyperparameter tuning system
  - _Requirements: 7.3, 7.4, 7.5, 11.1, 11.2, 11.3, 11.4, 11.5_

- [x] 13.1 Implement evaluation metrics
  - Create Exact Match (EM) calculation for answer accuracy evaluation
  - Implement nDCG@10 calculation for retrieval quality assessment
  - Build evaluation pipeline matching official metrics_calculation.py script
  - _Requirements: 7.3, 7.4_

- [x] 13.2 Create validation and testing pipeline
  - Implement validation set evaluation for system tuning
  - Create automated testing for all system components
  - Build performance benchmarking and comparison tools
  - _Requirements: 7.5, 11.1_

- [ ] 13.3 Build performance optimization system
  - Implement hyperparameter tuning for retrieval and generation components
  - Create ablation study framework for component contribution analysis
  - Build optimization pipeline targeting top-tier ranking performance
  - _Requirements: 11.2, 11.3, 11.4, 11.5_

- [ ] 14. Create comprehensive documentation and deliverables
  - Write detailed README with setup instructions and code structure explanation
  - Create project report meeting all requirements (6+ pages, proper formatting)
  - Produce demo video showcasing system capabilities and technical understanding
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 10.1, 10.2, 10.3, 10.4, 10.5_

- [x] 14.1 Write comprehensive README and code documentation
  - Create detailed README explaining code structure and component organization
  - Write step-by-step setup and execution instructions
  - Document all dependencies, environment requirements, and API configurations
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [ ] 14.2 Create project report
  - Write 6+ page report with proper formatting (12pt font, single spacing)
  - Include team contributions, system design, flowchart, and result analysis
  - Document UI design, additional explorations, and performance optimization
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7_

- [ ] 14.3 Produce demo video
  - Create 5-minute MP4 video demonstrating system capabilities
  - Show real-time response, advanced features, and code structure overview
  - Highlight challenges addressed and technical understanding of implementation
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [ ] 15. Final system integration and optimization
  - Integrate all components into complete end-to-end RAG system
  - Perform final performance tuning and optimization for maximum ranking
  - Create final test_prediction.jsonl file for submission
  - _Requirements: 1.1, 1.2, 1.3, 7.1, 7.2, 11.1, 11.2, 11.3, 11.4, 11.5_

- [x] 15.1 Complete system integration
  - Integrate all retrieval methods, generation features, and UI components
  - Create unified system configuration and execution pipeline
  - Implement end-to-end testing and validation of complete system
  - _Requirements: 1.1, 1.2, 1.3_

- [ ] 15.2 Final performance optimization
  - Conduct comprehensive performance tuning using validation set
  - Optimize system parameters for maximum EM and nDCG@10 scores
  - Balance computational efficiency with accuracy for practical deployment
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

- [ ] 15.3 Generate final submission files
  - Create test_prediction.jsonl file in exact required format
  - Validate output format and ensure all 1052 test queries are processed
  - Package complete submission with code, report, video, and prediction file
  - _Requirements: 7.1, 7.2, 7.5_