# Module Classification System
A sophisticated system for classifying modules as either "script" or "product" using advanced RAG (Retrieval Augmented Generation) techniques with LLM integration.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [API Endpoints](#api-endpoints)
  - [Streamlit Clients](#streamlit-clients)
- [Advanced Retrieval Components](#advanced-retrieval-components)
  
  - [Multi-Query Retriever](#multi-query-retriever)
  - [Hybrid Retrieval System](#hybrid-retrieval-system)
  - [NVIDIA Reranker](#nvidia-reranker)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [API Reference](#api-reference)
- [License](#license)

## Overview

The Module Classification System is an AI-powered solution that analyzes module descriptions and classifies them as either "script" or "product" with confidence levels and supporting points. The system leverages state-of-the-art LLM technology from NVIDIA combined with advanced retrieval techniques to achieve high-accuracy classification.

## Architecture

The system is built on a modern tech stack with several key components:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                             Client Applications                             │
│                                                                             │
│  ┌───────────────────┐                            ┌────────────────────┐    │
│  │                   │                            │                    │    │
│  │  Streamlit Client │                            │ Evaluation Client  │    │
│  │    (client.py)    │                            │   (evaluate.py)    │    │
│  │                   │                            │                    │    │
│  └─────────┬─────────┘                            └──────────┬─────────┘    │
│            │                                                 │              │
└────────────┼─────────────────────────────────────────────────┼──────────────┘
             │                                                 │
             │                HTTP Requests                    │
             │                                                 │
┌────────────▼─────────────────────────────────────────────────▼──────────────┐
│                                                                             │
│                              FastAPI Server                                 │
│                                                                             │
│  ┌───────────────────┐      ┌────────────────────┐    ┌──────────────────┐  │
│  │                   │      │                    │    │                  │  │
│  │     app.py        ├──────►     routes.py     │    │     models.py     │  │
│  │ (FastAPI App)     │      │ (API Endpoints)   │    │(Pydantic Models)  │  │
│  │                   │      │                    │    │                  │  │
│  └─────────┬─────────┘      └──────────┬─────────┘    └──────────────────┘  │
│            │                           │                                    │
└────────────┼───────────────────────────┼────────────────────────────────────┘
             │                           │
             │                           │
┌────────────▼───────────────────────────▼─────────────────────────────────────┐
│                                                                              │
│                           LangChain Processing                               │
│                                                                              │
│  ┌───────────────────┐      ┌────────────────────┐    ┌──────────────────┐   │
│  │                   │      │                    │    │                  │   │
│  │     chains.py     │      │     utils.py       │    │     config.py    │   │
│  │(Classification    │      │(Helper Functions)  │    │ (Configuration)  │   │
│  │     Chain)        │      │                    │    │                  │   │
│  │                   │      │                    │    │                  │   │
│  └─────────┬─────────┘      └──────────┬─────────┘    └──────────────────┘   │
│            │                           │                                     │
└────────────┼───────────────────────────┼─────────────────────────────────────┘
             │                           │
             │                           │
┌────────────▼───────────────────────────▼─────────────────────────────────────┐
│                                                                             │
│                        NVIDIA AI & Vector Database                          │
│                                                                             │
│  ┌────────────────────────────┐      ┌────────────────────────────────────┐ │
│  │                            │      │                                    │ │
│  │  NVIDIA AI Endpoints       │      │  FAISS Vector Store                │ │
│  │  - ChatNVIDIA (LLM)        │      │  - Document Retrieval              │ │
│  │  - NVIDIAEmbeddings        │      │  - Semantic Search                 │ │
│  │  - NVIDIARerank            │      │  - ParentChild Retriever           │ │
│  │                            │      │                                    │ │
│  └────────────────────────────┘      └────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

- **Module Classification**: Classify modules as either "script" or "product"
- **Confidence Levels**: Get confidence ratings (High/Medium/Low) for classifications
- **Supporting Points**: Receive key points that support the classification decision
- **Advanced Retrieval**: Multiple retrieval techniques for improved accuracy
- **Evaluation**: Built-in evaluation system to measure classification accuracy
- **API & UI**: Both API endpoints and user interfaces available

## Installation

### Prerequisites

- Python 3.10.12+
- NVIDIA API key
- FAISS index (pre-built)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/module-classification-system.git
   cd module-classification-system
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your environment:
   - Update `config.py` with your NVIDIA API key and model preferences
   - Ensure your FAISS index is in the correct location (default: `C:\MultiAgent Assignment\RAG_Log\DBS-Faiss-Passage\faiss-llama-passage-32768-16348-new`)

## Usage

### Starting the server

Run the FastAPI server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### Classify a module

```bash
curl -X POST "http://localhost:8000/api/classify" \
     -H "Content-Type: application/json" \
     -d '{
           "synopsis": "Sample synopsis text",
           "description": "Detailed description of the issue or feature"
         }'
```

#### Health check

```bash
curl "http://localhost:8000/api/health"
```

### Streamlit Clients

#### Classification Client

Run the client UI:

```bash
streamlit run client.py
```

#### Evaluation Client

Run the evaluation UI:

```bash
streamlit run evaluate.py
```

## Advanced Retrieval Components

The system employs several advanced retrieval techniques to enhance accuracy:


### Multi-Query Retriever

Improves retrieval by generating multiple alternative phrasings of the original query:

```
Original Query → LLM → Multiple Alternative Queries → Multiple Retrievals → Combined Results
```

Key features:
- Reduces query phrasing sensitivity
- Captures different aspects of the query
- Configurable number of alternative queries

### Hybrid Retrieval System

Combines multiple retrieval approaches for comprehensive document search:

```
┌────────────────────────┐        ┌────────────────────────┐
│                        │        │                        │
│ ParentDocumentRetriever│        │    Semantic Retriever  │
│                        │        │                        │
└───────────┬────────────┘        └────────────┬───────────┘
            │                                  │
            ▼                                  ▼
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│                     EnsembleRetriever                              │
│            (Combines retrievers with weights)                      │
│                                                                    │
└──────────────────────────────┬─────────────────────────────────────┘
```

Key features:
- ParentDocumentRetriever for hierarchical document retrieval
- Semantic Retriever using FAISS for vector similarity search
- EnsembleRetriever with configurable weights

### NVIDIA Reranker

Improves relevance of retrieved documents:

```
Retrieved Documents → NVIDIA Reranker → Reordered Documents by Relevance
```

Key features:
- Uses NVIDIA's LLama 3.2 ReRankQA model
- Configurable number of top documents to retain
- Improves precision of retrieval results

## Configuration

All system components are configurable in `config.py`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `nvidia_api_key` | NVIDIA API key | - |
| `default_chat_model` | Default LLM model | "nvdev/meta/llama-3.1-405b-instruct" |
| `embed_model` | Embedding model | "nvdev/nvidia/llama-3.2-nv-embedqa-1b-v2" |
| `rerank_model` | Reranking model | "nvdev/nvidia/llama-3.2-nv-rerankqa-1b-v2" |
| `k_value` | Number of documents to retrieve | 50 |
| `score_threshold` | Minimum similarity score | 0.2 |
| `top_n_rerank` | Number of documents after reranking | 20 |
| `use_multi_query` | Enable MultiQuery | False |
| `num_alt_queries` | Number of alternative queries | 3 |
| `retrieval_method` | Retrieval method | "Full Hybrid (Parent + Semantic)" |

## Evaluation

The system includes a comprehensive evaluation framework:

- Sample from a test dataset
- Run classification on samples
- Calculate accuracy metrics
- Generate detailed reports
- Parallelized processing for efficiency

Run evaluation using the Streamlit client or API:

```bash
curl -X POST "http://localhost:8000/api/evaluate" \
     -H "Content-Type: application/json" \
     -d '{
           "sample_size": 100,
           "random_seed": 42,
           "num_threads": 10,
           "use_multi_query": false,
           "retrieval_method": "Full Hybrid (Parent + Semantic)"
         }'
```

## API Reference

### Classification Endpoint

**POST** `/api/classify`

Request body:
```json
{
  "synopsis": "string",
  "description": "string",
  "direct_query": "string (optional)"
}
```

Response:
```json
{
  "classification": "script|product",
  "confidence": "High|Medium|Low",
  "supporting_points": ["point1", "point2", "..."],
  "processing_details": {
    "num_documents_retrieved": 50,
    "num_documents_after_reranking": 20
  }
}
```

### Evaluation Endpoint

**POST** `/api/evaluate`

Request body:
```json
{
  "sample_size": 10,
  "random_seed": 42,
  "num_threads": 10,
  "use_multi_query": false,
  "num_alt_queries": 3,
  "retrieval_method": "Full Hybrid (Parent + Semantic)"
}
```

Response:
```json
{
  "accuracy": 0.85,
  "correct": 85,
  "total": 100,
  "results": [
    {
      "index": 0,
      "question": "sample question",
      "true_label": "script",
      "prediction": "script",
      "match": true,
      "confidence": "High"
    }
  ]
}
```

## License

[MIT License](LICENSE) 