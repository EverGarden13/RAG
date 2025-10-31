# Requirements Document

## Introduction

This document outlines the requirements for developing a Retrieval-Augmented Generation (RAG) system for the COMP5423 group project. The system will use the HotpotQA dataset subset (HQ-small) to build an end-to-end question answering system capable of handling complex multi-hop queries that require reasoning across multiple documents. The system integrates retrieval modules with generative language models to improve factual accuracy and explainability through evidence-grounded response generation.

## Glossary

- **RAG_System**: The complete Retrieval-Augmented Generation system that combines document retrieval with answer generation
- **Retrieval_Module**: Component responsible for finding relevant documents from the collection based on user queries
- **Generation_Module**: Component that uses retrieved documents and user queries to generate answers using LLMs
- **User_Interface**: The interface (terminal or web-based) that allows users to interact with the RAG system
- **HQ_Small_Dataset**: The HotpotQA subset containing 12,000 training, 1,500 validation, and 1,052 test samples
- **Document_Collection**: The corpus of 144,718 documents available for retrieval
- **Multi_Turn_Search**: Feature allowing context-aware retrieval across multiple conversation turns
- **Agentic_Workflow**: Advanced feature implementing intermediate reasoning steps and self-checking mechanisms

## Requirements

### Requirement 1

**User Story:** As a researcher, I want to query the RAG system with complex multi-hop questions, so that I can get accurate answers grounded in retrieved evidence.

#### Acceptance Criteria

1. WHEN a user submits a query, THE RAG_System SHALL retrieve relevant documents from the Document_Collection
2. WHEN documents are retrieved, THE RAG_System SHALL generate an answer using the retrieved evidence and user query
3. THE RAG_System SHALL return both the generated answer and the supporting documents with their relevance scores
4. THE RAG_System SHALL format output as JSON with id, question, answer, and retrieved_docs fields
5. THE RAG_System SHALL retrieve exactly 10 documents with their corresponding scores for each query

### Requirement 2

**User Story:** As a developer, I want to implement multiple retrieval methods to maximize scoring potential, so that I can achieve full marks by implementing all five retrieval categories.

#### Acceptance Criteria

1. THE Retrieval_Module SHALL implement sparse retrieval methods including TF-IDF, BM25, or predicted term weights from neural models
2. THE Retrieval_Module SHALL implement static embedding retrieval using Word2Vec, GloVe, or model2vec
3. THE Retrieval_Module SHALL implement dense retrieval with encoder-based models such as E5, BGE, or GTE
4. THE Retrieval_Module SHALL implement dense retrieval with instruction using LLM-based models like Qwen3-Embedding or E5-Mistral
5. THE Retrieval_Module SHALL implement multi-vector retrieval such as ColBERT
6. THE Retrieval_Module SHALL support hybrid retrieval systems combining multiple methods for optimal performance
7. THE Retrieval_Module SHALL return document IDs and relevance scores for all retrieved documents

### Requirement 3

**User Story:** As a user, I want the system to generate accurate answers with both basic and advanced features to maximize scoring, so that I can get comprehensive RAG capabilities.

#### Acceptance Criteria

1. THE Generation_Module SHALL use only Qwen2.5 model variants (0.5B, 1.5B, 3B, or 7B Instruct versions)
2. THE Generation_Module SHALL implement a basic single-turn RAG pipeline feeding retrieved passages and queries to the LLM
3. THE Generation_Module SHALL design optimized prompt templates considering instruction placement, question formatting, and retrieved passage ordering
4. THE Generation_Module SHALL extract answers in a consistent, parseable format for evaluation
5. THE Generation_Module SHALL implement Feature A (Multi-Turn Search) for context-aware retrieval across conversation turns
6. THE Generation_Module SHALL implement Feature B (Agentic Workflow) with query rewriting, self-checking, and explicit reasoning steps
7. THE Generation_Module SHALL generate responses that are grounded in retrieved evidence with explainable reasoning

### Requirement 4

**User Story:** As a user, I want an intuitive interface with bonus features to maximize UI scoring, so that I can interact effectively and see detailed system reasoning.

#### Acceptance Criteria

1. THE User_Interface SHALL accept user query input through either terminal-based or web-based interaction
2. THE User_Interface SHALL display retrieved documents with their relevance information and scores
3. THE User_Interface SHALL present the final generated answer from the LLM with clear formatting
4. THE User_Interface SHALL maintain a clean, readable layout optimized for user experience
5. THE User_Interface SHALL display intermediate reasoning processes including generated sub-questions
6. THE User_Interface SHALL show intermediate retrieval outputs and self-checking step results
7. THE User_Interface SHALL visualize the complete agentic workflow for transparency and debugging

### Requirement 5

**User Story:** As an advanced user, I want multi-turn conversation support, so that I can ask follow-up questions that reference previous context.

#### Acceptance Criteria

1. WHERE multi-turn search is implemented, THE RAG_System SHALL maintain dialogue memory across conversation turns
2. WHEN a follow-up query contains references, THE RAG_System SHALL reformulate queries into self-contained versions before retrieval
3. THE RAG_System SHALL track entities and conversation history for context-aware retrieval
4. THE RAG_System SHALL manage long contexts by pruning earlier retrieved passages when necessary
5. THE RAG_System SHALL correctly resolve references like pronouns to appropriate entities from conversation history

### Requirement 6

**User Story:** As a quality-conscious user, I want the system to implement advanced reasoning workflows, so that I can get higher quality answers with explicit reasoning steps.

#### Acceptance Criteria

1. WHERE agentic workflow is implemented, THE RAG_System SHALL decompose complex queries into sub-queries for parallel retrieval
2. THE RAG_System SHALL implement self-checking mechanisms to verify generated answers against retrieved evidence
3. THE RAG_System SHALL detect potential hallucinations through secondary verification checks
4. THE RAG_System SHALL implement explicit reasoning steps using chain-of-thought or ReAct-style workflows
5. THE RAG_System SHALL balance computational cost with performance improvements in workflow design

### Requirement 7

**User Story:** As an evaluator, I want the system to produce standardized output for performance assessment, so that I can measure retrieval quality and answer accuracy.

#### Acceptance Criteria

1. THE RAG_System SHALL generate predictions in the specified JSONL format for test data
2. THE RAG_System SHALL include exactly 10 retrieved document pairs with IDs and scores for each prediction
3. THE RAG_System SHALL produce answers that can be evaluated using Exact Match metrics
4. THE RAG_System SHALL support evaluation of retrieval quality using nDCG@10 metrics
5. THE RAG_System SHALL ensure all output fields (id, question, answer, retrieved_docs) are properly formatted

### Requirement 8

**User Story:** As a developer, I want comprehensive documentation and reproducible setup, so that I can understand, maintain, and reproduce the system.

#### Acceptance Criteria

1. THE RAG_System SHALL include a README.md file describing code structure and setup instructions
2. THE RAG_System SHALL provide step-by-step instructions for reproducing test predictions
3. THE RAG_System SHALL specify all required packages and versions for environment setup
4. THE RAG_System SHALL include appropriate code annotations and docstrings for major components
5. THE RAG_System SHALL ensure the complete system can be executed end-to-end following the provided instructions
### Req
uirement 9

**User Story:** As a project team, I want to create comprehensive documentation and deliverables, so that I can achieve maximum scores across all evaluation components.

#### Acceptance Criteria

1. THE RAG_System SHALL include a written report of at least 6 pages with 12-point font and single line spacing
2. THE RAG_System SHALL include team contribution percentages and roles for all members
3. THE RAG_System SHALL provide detailed system design methodology and integration approach
4. THE RAG_System SHALL include a labeled system flowchart visualizing the complete data pipeline
5. THE RAG_System SHALL present comprehensive result analysis with experimental outcomes under varying configurations
6. THE RAG_System SHALL include UI design documentation with screenshots and interaction explanations
7. THE RAG_System SHALL document additional explorations including prompt tuning and performance optimization attempts

### Requirement 10

**User Story:** As a project demonstrator, I want to create an effective demo video, so that I can clearly showcase system capabilities and technical understanding.

#### Acceptance Criteria

1. THE RAG_System SHALL include a demo video of maximum 5 minutes in MP4 format
2. THE RAG_System SHALL demonstrate real-time response capability and key solution aspects
3. THE RAG_System SHALL highlight main challenges addressed and advanced features beyond basic requirements
4. THE RAG_System SHALL provide code structure overview explaining preprocessing, retrieval, and generation components
5. THE RAG_System SHALL show clear understanding of implementation logic and workflow

### Requirement 11

**User Story:** As a performance evaluator, I want the system to achieve top-tier ranking, so that I can maximize the 20% performance component score.

#### Acceptance Criteria

1. THE RAG_System SHALL optimize for both Answer Accuracy (Exact Match) and Retrieval Quality (nDCG@10) metrics
2. THE RAG_System SHALL implement performance optimization strategies to achieve top 10% ranking if possible
3. THE RAG_System SHALL balance retrieval precision and generation quality for maximum combined score
4. THE RAG_System SHALL include performance tuning experiments and ablation studies
5. THE RAG_System SHALL target ranking in top 30% minimum to secure 70% of performance points (14/20 points)