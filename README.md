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
Technology Stack
Frontend
Streamlit
Programming Language
Python
Document Processing
PyMuPDF
LangChain Community
Recursive Character Text Splitter
Embedding Model
BAAI/bge-small-en-v1.5
Large Language Model
Qwen/Qwen2.5-3B-Instruct
Vector Database
ChromaDB
Reranking
cross-encoder/ms-marco-MiniLM-L-6-v2
Machine Learning
PyTorch
Sentence Transformers
Hugging Face Transformers
Environment
Python 3.13
Windows
uv package manager
Project Structure
PythonProject10/
│
├── .venv/
│
├── models/
│   ├── bge-small-en-v1.5/
│   │   ├── config.json
│   │   ├── modules.json
│   │   ├── model.safetensors
│   │   └── ...
│   │
│   └── qwen2.5-3b-instruct/
│       ├── config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── model-00001-of-00002.safetensors
│       ├── model-00002-of-00002.safetensors
│       └── ...
│
├── chroma_db/
│
├── uploads/
│
├── .env
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
Models
1. BGE Small English

The application uses:

BAAI/bge-small-en-v1.5

This model converts document chunks and user questions into vector embeddings.

The model is stored locally in:

models/bge-small-en-v1.5/

The embedding dimension is:

384
2. Qwen 2.5 3B Instruct

The application uses:

Qwen/Qwen2.5-3B-Instruct

The model is responsible for generating answers based on the retrieved document context.

It is stored locally in:

models/qwen2.5-3b-instruct/

The application loads the model using:

local_files_only=True

This allows the application to use the locally downloaded model instead of downloading it every time.

Installation
1. Clone the Repository
git clone <your-repository-url>
cd PythonProject10
2. Create a Virtual Environment

Using uv:

uv venv

Activate the environment on Windows:

.venv\Scripts\activate
3. Install Dependencies
uv pip install -r requirements.txt

If you don't have a requirements.txt yet, install the main dependencies:

uv pip install streamlit
uv pip install chromadb
uv pip install torch
uv pip install transformers
uv pip install accelerate
uv pip install sentence-transformers
uv pip install langchain
uv pip install langchain-community
uv pip install pymupdf
uv pip install python-dotenv
Download the Models

The models should be downloaded before running the application.

BGE Embedding Model
hf download BAAI/bge-small-en-v1.5 --local-dir .\models\bge-small-en-v1.5
Qwen LLM
hf download Qwen/Qwen2.5-3B-Instruct --local-dir .\models\qwen2.5-3b-instruct

After downloading, verify that both paths are directories and contain the model files.

Running the Application

From the project directory:

python -m streamlit run app.py

Alternatively:

streamlit run app.py

Streamlit will provide a local URL in the terminal.

Open that URL in your browser.

How the Application Works
Step 1: Upload a PDF

The user uploads a PDF document through the Streamlit interface.

The document is temporarily stored in:

uploads/
Step 2: Extract Text

The application uses PyMuPDF to extract text from the PDF.

Each page is processed along with its metadata.

Step 3: Split the Document

Large documents are divided into smaller chunks.

Current configuration:

Chunk Size: 800
Chunk Overlap: 150

Chunking makes semantic retrieval more effective.

Step 4: Generate Embeddings

Each chunk is converted into a vector using:

BAAI/bge-small-en-v1.5

The vectors are normalized before being stored.

Step 5: Store in ChromaDB

The generated embeddings and document metadata are stored in:

chroma_db/

The database uses cosine similarity for vector search.

Step 6: Ask a Question

The user enters a question about the uploaded documents.

The question is converted into an embedding using the same BGE model.

Step 7: Retrieve Relevant Chunks

ChromaDB performs semantic similarity search and retrieves the most relevant chunks.

The application initially retrieves:

Top 10 chunks
Step 8: Reranking

The retrieved chunks are passed through a cross-encoder reranker:

cross-encoder/ms-marco-MiniLM-L-6-v2

The most relevant:

Top 3 chunks

are selected.

Step 9: Generate the Answer

The selected chunks are provided as context to:

Qwen2.5-3B-Instruct

The model generates an answer based only on the retrieved document context.

Step 10: Display Sources

The application also displays the source information used to generate the answer, including:

Document name
Page number
Retrieved content

This makes the answer easier to verify.

RAG Pipeline

The core pipeline can be summarized as:

PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
BGE Embedding
 ↓
ChromaDB
 ↓
User Question
 ↓
Question Embedding
 ↓
Similarity Search
 ↓
Top 10 Results
 ↓
Cross-Encoder Reranking
 ↓
Top 3 Results
 ↓
Qwen 2.5 3B
 ↓
Final Answer
Privacy

This project is designed with local document processing in mind.

The main document analysis pipeline runs locally:

PDF
 ↓
Local Embeddings
 ↓
Local Vector Database
 ↓
Local LLM

The Qwen model is loaded from the local models/ directory.

No Ollama installation is required.

Note: The current reranker configuration uses the Hugging Face model ID
cross-encoder/ms-marco-MiniLM-L-6-v2, so the reranker may download its model
files from Hugging Face the first time it is used. For a fully offline setup,
download this model locally and change the application to load it from a local path.
