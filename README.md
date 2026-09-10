# AI-Powered Local Document Analyzer

A privacy-focused AI document analysis application that allows users to upload PDF documents, process their content, search relevant information, and ask questions using a completely local Large Language Model (LLM).

The application uses a Retrieval-Augmented Generation (RAG) pipeline with local Hugging Face models, vector search, and document reranking.

---

## Features

- Upload PDF documents
- Extract text from PDF files
- Split documents into smaller chunks
- Generate semantic embeddings using BGE
- Store document embeddings in ChromaDB
- Retrieve the most relevant document chunks
- Rerank retrieved results for better relevance
- Ask natural-language questions about uploaded documents
- Generate answers using Qwen 2.5 3B Instruct
- Display source documents and page numbers
- Persistent local vector database
- Local model execution
- No Ollama required
- Streamlit-based user interface

---

## Architecture

The application follows a Retrieval-Augmented Generation (RAG) architecture:

```text
                 ┌─────────────────────┐
                 │      PDF Upload     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   PDF Text Extract  │
                 │    PyMuPDFLoader    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Text Chunking     │
                 │ Recursive Splitter  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  BGE Embeddings     │
                 │ bge-small-en-v1.5   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │     ChromaDB        │
                 │  Vector Database    │
                 └──────────┬──────────┘
                            │
                      User Question
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Semantic Retrieval  │
                 │    Top 10 Chunks    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │     Reranker        │
                 │ Cross Encoder       │
                 └──────────┬──────────┘
                            │
                      Top 3 Chunks
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Qwen 2.5 3B       │
                 │      Instruct       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │      Answer         │
                 │ + Source References │
                 └─────────────────────┘