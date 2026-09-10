import os
from pathlib import Path

import streamlit as st
import chromadb
import torch

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Local models
EMBEDDING_MODEL_PATH = BASE_DIR / "models" / "bge-small-en-v1.5"
LLM_MODEL_PATH = BASE_DIR / "models" / "qwen2.5-3b-instruct"

# Reranker
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Local storage
CHROMA_PATH = BASE_DIR / "chroma_db"
UPLOAD_DIR = BASE_DIR / "uploads"

# Create required directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PATH.mkdir(parents=True, exist_ok=True)

# Load .env if available
load_dotenv()


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Document Analyzer",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("AI-Powered Document Analyzer")

st.write(
    "Upload a PDF and ask questions about its contents using "
    "local Hugging Face AI models."
)


# ============================================================
# CHECK MODEL DIRECTORIES
# ============================================================

if not EMBEDDING_MODEL_PATH.exists():
    st.error(
        f"Embedding model not found.\n\n"
        f"Expected location:\n`{EMBEDDING_MODEL_PATH}`"
    )
    st.stop()


if not LLM_MODEL_PATH.exists():
    st.error(
        f"LLM model not found.\n\n"
        f"Expected location:\n`{LLM_MODEL_PATH}`"
    )
    st.stop()


# ============================================================
# LOCAL BGE EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    st.info("Loading BGE embedding model...")

    model = SentenceTransformer(
        str(EMBEDDING_MODEL_PATH)
    )

    return model


embedding_model = load_embedding_model()


# ============================================================
# CHROMA EMBEDDING FUNCTION
# ============================================================

class BGEEmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def name(self):
        return "bge-small-en-v1.5"

    def __call__(self, input):
        embeddings = self.model.encode(
            input,
            normalize_embeddings=True
        )
        return embeddings.tolist()


embedding_function = BGEEmbeddingFunction(
    embedding_model
)


# ============================================================
# CHROMADB
# ============================================================

@st.cache_resource
def load_chroma():

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = client.get_or_create_collection(
        name="rag_app",
        metadata={
            "hnsw:space": "cosine"
        },
        embedding_function=embedding_function
    )

    return client, collection


chroma_client, collection = load_chroma()


# ============================================================
# CROSS ENCODER RERANKER
# ============================================================

@st.cache_resource
def load_reranker():

    st.info("Loading reranker model...")

    model = CrossEncoder(
        RERANKER_MODEL
    )

    return model


reranker = load_reranker()


# ============================================================
# LOCAL QWEN LLM
# ============================================================

@st.cache_resource
def load_llm():

    st.info("Loading Qwen2.5-3B-Instruct...")

    tokenizer = AutoTokenizer.from_pretrained(
        str(LLM_MODEL_PATH),
        local_files_only=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(LLM_MODEL_PATH),
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True
    )

    return tokenizer, model


tokenizer, llm_model = load_llm()


# ============================================================
# PDF PROCESSING
# ============================================================

def process_pdf(pdf_path):

    loader = PyMuPDFLoader(
        str(pdf_path)
    )

    documents = loader.load()

    if not documents:
        return []

    # --------------------------------------------------------
    # Split PDF into chunks
    # --------------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = text_splitter.split_documents(
        documents
    )

    return chunks


# ============================================================
# ADD DOCUMENT TO CHROMADB
# ============================================================

def add_documents_to_chroma(
    chunks,
    document_name
):

    if not chunks:
        return 0

    ids = []
    documents = []
    metadatas = []

    for index, chunk in enumerate(chunks):

        chunk_id = f"{document_name}_{index}"

        ids.append(chunk_id)

        documents.append(
            chunk.page_content
        )

        metadata = {
            "source": document_name,
            "page": chunk.metadata.get(
                "page",
                0
            ),
            "chunk": index
        }

        metadatas.append(metadata)

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    embeddings = embedding_model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = embeddings.tolist()

    # --------------------------------------------------------
    # Store in Chroma
    # --------------------------------------------------------

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(documents)


# ============================================================
# RETRIEVE DOCUMENT CHUNKS
# ============================================================

def retrieve_documents(
    query,
    top_k=10
):

    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False
    )[0].tolist()

    # --------------------------------------------------------
    # Search Chroma
    # --------------------------------------------------------

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    retrieved = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        retrieved.append(
            {
                "text": document,
                "metadata": metadata,
                "distance": distance
            }
        )

    return retrieved


# ============================================================
# RERANK DOCUMENTS
# ============================================================

def rerank_documents(
    query,
    documents,
    top_k=3
):

    if not documents:
        return []

    pairs = []

    for document in documents:

        pairs.append(
            (
                query,
                document["text"]
            )
        )

    scores = reranker.predict(
        pairs
    )

    for document, score in zip(
        documents,
        scores
    ):

        document["rerank_score"] = float(
            score
        )

    documents.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return documents[:top_k]


# ============================================================
# CREATE CONTEXT
# ============================================================

def create_context(
    documents
):

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1
    ):

        metadata = document["metadata"]

        source = metadata.get(
            "source",
            "Unknown"
        )

        page = metadata.get(
            "page",
            0
        )

        # Convert zero-based page number
        page_number = page + 1

        text = document["text"]

        context_parts.append(
            f"""
SOURCE {index}

Document: {source}
Page: {page_number}

Content:
{text}
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# LOCAL QWEN LLM
# ============================================================

def call_llm(
    prompt,
    context
):

    system_prompt = """
You are an AI document analysis assistant.

Your job is to answer the user's question using ONLY the
information provided in the document context.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not present in the context, clearly say:
   "The information is not available in the uploaded document."
4. Give a clear and concise answer.
5. Use the document context as your primary source.
6. If possible, mention the relevant page number.
7. Do not make assumptions that are not supported by the document.
"""

    user_prompt = f"""
DOCUMENT CONTEXT:

{context}


USER QUESTION:

{prompt}


Answer the user's question using only the document context.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    # --------------------------------------------------------
    # Apply Qwen chat template
    # --------------------------------------------------------

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    model_inputs = tokenizer(
        [text],
        return_tensors="pt"
    )

    # Move tensors to model device
    model_inputs = {
        key: value.to(llm_model.device)
        for key, value in model_inputs.items()
    }

    # --------------------------------------------------------
    # Generate response
    # --------------------------------------------------------

    with torch.no_grad():

        generated_ids = llm_model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.05
        )

    # --------------------------------------------------------
    # Remove input tokens
    # --------------------------------------------------------

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(
            model_inputs["input_ids"],
            generated_ids
        )
    ]

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    response = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return response.strip()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Document Analyzer")

    st.write(
        "Local AI pipeline"
    )

    st.markdown(
        """
### Models

**Embeddings**
- BAAI BGE-small-en-v1.5

**Reranker**
- MS MARCO MiniLM

**LLM**
- Qwen2.5-3B-Instruct

### Pipeline

PDF
↓
Chunking
↓
BGE Embeddings
↓
ChromaDB
↓
Top 10 Retrieval
↓
Reranking
↓
Top 3 Context
↓
Qwen
↓
Answer
"""
    )

    st.divider()

    if st.button(
        "Clear Vector Database",
        use_container_width=True
    ):

        try:

            chroma_client.delete_collection(
                "rag_app"
            )

            st.cache_resource.clear()

            st.success(
                "Vector database cleared."
            )

            st.rerun()

        except Exception as e:

            st.error(
                f"Error clearing database: {e}"
            )


# ============================================================
# PDF UPLOAD
# ============================================================

st.header("1. Upload PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    pdf_path = UPLOAD_DIR / uploaded_file.name

    # Save uploaded PDF
    with open(
        pdf_path,
        "wb"
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )

    st.success(
        f"PDF uploaded: {uploaded_file.name}"
    )

    if st.button(
        "Process Document",
        type="primary"
    ):

        with st.spinner(
            "Processing PDF..."
        ):

            try:

                # ------------------------------------------------
                # Load and chunk PDF
                # ------------------------------------------------

                chunks = process_pdf(
                    pdf_path
                )

                if not chunks:

                    st.error(
                        "No text could be extracted from the PDF."
                    )

                else:

                    # ------------------------------------------------
                    # Add to Chroma
                    # ------------------------------------------------

                    count = add_documents_to_chroma(
                        chunks,
                        uploaded_file.name
                    )

                    st.success(
                        f"Document processed successfully!"
                    )

                    st.info(
                        f"Created {count} document chunks."
                    )

            except Exception as e:

                st.error(
                    f"Error processing PDF:\n\n{e}"
                )


# ============================================================
# QUESTION SECTION
# ============================================================

st.divider()

st.header("2. Ask Questions")


question = st.text_input(
    "Ask something about your uploaded document",
    placeholder="Example: What is the main objective of this document?"
)


# ============================================================
# ASK QUESTION
# ============================================================

if st.button(
    "Ask AI",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching the document..."
        ):

            try:

                # ------------------------------------------------
                # Step 1: Retrieve
                # ------------------------------------------------

                retrieved_documents = retrieve_documents(
                    question,
                    top_k=10
                )

                if not retrieved_documents:

                    st.warning(
                        "No relevant information was found."
                    )

                else:

                    # ------------------------------------------------
                    # Step 2: Rerank
                    # ------------------------------------------------

                    reranked_documents = rerank_documents(
                        question,
                        retrieved_documents,
                        top_k=3
                    )

                    # ------------------------------------------------
                    # Step 3: Create context
                    # ------------------------------------------------

                    context = create_context(
                        reranked_documents
                    )

                    # ------------------------------------------------
                    # Step 4: Generate answer
                    # ------------------------------------------------

                    with st.spinner(
                        "Qwen is generating the answer..."
                    ):

                        answer = call_llm(
                            question,
                            context
                        )

                    # ------------------------------------------------
                    # Display answer
                    # ------------------------------------------------

                    st.subheader(
                        "Answer"
                    )

                    st.write(
                        answer
                    )

                    # ------------------------------------------------
                    # Sources
                    # ------------------------------------------------

                    st.subheader(
                        "Sources"
                    )

                    for index, document in enumerate(
                        reranked_documents,
                        start=1
                    ):

                        metadata = document[
                            "metadata"
                        ]

                        source = metadata.get(
                            "source",
                            "Unknown"
                        )

                        page = metadata.get(
                            "page",
                            0
                        ) + 1

                        score = document.get(
                            "rerank_score",
                            0
                        )

                        with st.expander(
                            f"Source {index} — {source} — Page {page}"
                        ):

                            st.write(
                                document["text"]
                            )

                            st.caption(
                                f"Rerank score: {score:.4f}"
                            )

            except Exception as e:
                st.error(
                    f"Error answering question:\n\n{e}"
                )


# ============================================================
# DATABASE INFORMATION
# ============================================================

st.divider()

try:

    collection_count = collection.count()

    st.caption(
        f"Vector database: {collection_count} chunks stored"
    )

except Exception:

    pass
