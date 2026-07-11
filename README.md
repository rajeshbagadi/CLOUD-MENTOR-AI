# CloudMentor AI – AWS Documentation RAG Assistant

CloudMentor AI is a production-grade Retrieval-Augmented Generation (RAG) assistant designed to help developers and solutions architects query AWS documentation, compare AWS services, and receive architecture recommendations.

The application utilizes LangChain, ChromaDB, sentence-transformers, Google Gemini API, and Streamlit to deliver grounded answers with inline source citations.

## Key Features
* **PDF Ingestion:** Fast page-by-page text extraction using pypdf.
* **Semantic Chunking:** Intelligent splitting using RecursiveCharacterTextSplitter to maintain paragraph boundaries.
* **Local Embedding Caching:** Saves computed vectors on disk to prevent redundant embedding operations.
* **ChromaDB Vector Store:** Persistent local vector database featuring deterministic record hashing and single-source purging.
* **Hallucination Prevention:** Prompts restrict Gemini to provided context blocks. Grounding failures trigger a fallback answer.
* **Source Citations:** Answers contain inline citations in the format `[source: file.pdf, page: X]`. Context source segments are rendered underneath the responses.
* **ChatGPT-style Interface:** Sleek Streamlit UI featuring dark themes, file upload progress bars, and historical conversation tracking.

## Folder Structure
```text
RAG Project/
│
├── data/
│   ├── chromadb/                  # Persistent Chroma DB local storage files
│   └── embeddings_cache/          # Local disk storage for cached text vectors
│
├── docs/
│   └── README.md                  # PDFs upload directory readme
│
├── src/
│   ├── core/
│   │   ├── exceptions.py          # Custom domain ingestion exceptions
│   │   └── logging_config.py      # Standardized logging setup
│   │
│   ├── ingestion/
│   │   ├── document_loader.py     # PDF parsing and directory scan loaders
│   │   └── text_splitter.py       # Document chunking logic
│   │
│   ├── model/
│   │   ├── embeddings.py          # Embedding manager with disk cache backing
│   │   └── gemini_client.py       # ChatGoogleGenerativeAI client manager
│   │
│   └── rag/
│       ├── retriever.py           # Semantic retriever and citation container
│       ├── rag_chain.py           # Core RAG orchestration pipeline
│       ├── chat_memory.py         # Conversational history & query condensation
│       ├── comparison.py          # Multi-query AWS service comparison engine
│       └── recommendation.py      # AWS Architecture recommendation & Mermaid builder
│
├── app.py                         # Streamlit graphical client application
├── requirements.txt               # Pin-point Python package list
├── test_loader.py                 # loader layer unit verification script
├── test_embeddings.py             # embeddings caching verification script
├── test_vector_store.py           # ChromaDB database verification script
├── test_retriever.py              # query retrieval verification script
├── test_rag.py                    # RAG chain integration verification script
├── test_memory.py                 # chat memory verification script
├── test_comparison.py             # service comparison verification script
└── test_recommendation.py         # architecture recommendation verification script
```

## Installation & Running

### 1. Configure the Environment
Ensure Python 3.9+ is installed, then set up your environment:

```bash
# Clone or enter the workspace directory
cd "RAG Project"

# Install all required libraries
pip install -r requirements.txt
```

### 2. Set Up Your Gemini API Key
Provide your credentials as environment variables:

**Windows Powershell:**
```powershell
$env:GEMINI_API_KEY="your-google-gemini-api-key-here"
```

**Windows CMD:**
```cmd
set GEMINI_API_KEY=your-google-gemini-api-key-here
```

**Linux/macOS:**
```bash
export GEMINI_API_KEY="your-google-gemini-api-key-here"
```

### 3. Run the Streamlit Application
Start the Streamlit application:

```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

## Testing & Verifying Individual Modules
You can verify separate layer components using the provided CLI scripts:

* **Verify Document Loader:**
  ```bash
  python test_loader.py
  ```
* **Verify Embeddings & Caching:**
  ```bash
  python test_embeddings.py
  ```
* **Verify ChromaDB Database:**
  ```bash
  python test_vector_store.py
  ```
* **Verify Context Retrieval:**
  ```bash
  python test_retriever.py
  ```
* **Verify Basic RAG Pipeline:**
  ```bash
  python test_rag.py
  ```
* **Verify Chat Memory / Rephrasing:**
  ```bash
  python test_memory.py
  ```
* **Verify Comparison Engine:**
  ```bash
  python test_comparison.py
  ```
* **Verify Architecture Recommendation:**
  ```bash
  python test_recommendation.py
  ```

## Production Test Suite (pytest)
To run the complete automated test suite (unit and integration tests) using pytest:

```bash
# Execute all tests
pytest

# Execute only unit tests
pytest tests/unit

# Execute only integration tests
pytest tests/integration

# Execute tests with detailed logs
pytest -v
```
