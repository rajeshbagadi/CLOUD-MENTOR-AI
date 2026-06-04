import os
import shutil
from pathlib import Path
import streamlit as st

# Configure Streamlit page layout and theme styles at the very beginning
st.set_page_config(
    page_title="CloudMentor AI - AWS RAG Assistant",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom styling for modern UI/UX design (Harmonious gradient headers, custom fonts, glassmorphism containers)
st.markdown(
    """
    <style>
    .main-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #FF9900 0%, #FF6100 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        color: #888888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .source-badge {
        display: inline-block;
        background-color: rgba(255, 153, 0, 0.15);
        border: 1px solid rgba(255, 153, 0, 0.3);
        color: #FF9900;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8rem;
        margin-right: 5px;
        margin-top: 5px;
        font-weight: 500;
    }
    .chat-timestamp {
        font-size: 0.75rem;
        color: #666;
        margin-top: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

from src.model.embeddings import EmbeddingManager
from src.model.gemini_client import GeminiClientManager
from src.vectorstore.vector_store import ChromaStoreManager
from src.rag.retriever import SemanticRetriever
from src.rag.rag_chain import AWSCloudMentorRAGChain
from src.rag.chat_memory import ChatMemoryManager
from src.ingestion.document_loader import PDFDocumentLoader
from src.ingestion.text_splitter import DocumentChunker

# Target paths definition
DB_DIR = Path("data/chromadb")
CACHE_DIR = Path("data/embeddings_cache")
UPLOAD_DIR = Path("docs")

# Ensure base directories exist
DB_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def get_rag_pipeline():
    """Initializes and caches the heavy RAG components to avoid reloading at every rerun."""
    try:
        # 1. Initialize embeddings manager
        embed_mgr = EmbeddingManager(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=CACHE_DIR
        )
        embeddings = embed_mgr.get_embeddings()

        # 2. Initialize ChromaDB persistent vector store manager
        db_mgr = ChromaStoreManager(
            persist_directory=DB_DIR,
            embedding_function=embeddings,
            collection_name="aws_docs_collection"
        )

        # 3. Initialize Gemini LLM manager
        gemini_mgr = GeminiClientManager(
            model_name="gemini-1.5-flash",
            temperature=0.2
        )

        # 3.5. Initialize memory manager
        memory_mgr = ChatMemoryManager(gemini_manager=gemini_mgr, max_history_turns=10)

        # 4. Initialize semantic retriever and RAG orchestrator chain
        retriever = SemanticRetriever(vector_store=db_mgr, default_k=4)
        rag_chain = AWSCloudMentorRAGChain(
            retriever=retriever,
            gemini_manager=gemini_mgr,
            memory_manager=memory_mgr
        )

        return db_mgr, retriever, rag_chain, gemini_mgr, memory_mgr
    except Exception as e:
        st.error(f"Critical error initializing RAG components: {str(e)}")
        return None, None, None, None, None


# Load cached RAG pipeline objects
db_mgr, retriever, rag_chain, gemini_mgr, memory_mgr = get_rag_pipeline()

# Initialize Streamlit Session state keys
if "messages" not in st.session_state:
    st.session_state.messages = []


def clear_chat():
    """Clears conversational session history."""
    st.session_state.messages = []
    if memory_mgr:
        memory_mgr.clear_session("streamlit_chat_session")


def get_db_count() -> int:
    """Safe helper to fetch total records in the database collection."""
    if db_mgr and db_mgr._db:
        try:
            return db_mgr._db._collection.count()
        except Exception:
            return 0
    return 0


# ==========================================
# SIDEBAR INTERFACE
# ==========================================
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/9/93/Amazon_Web_Services_Logo.svg", 
        width=80
    )
    st.markdown("## CloudMentor AI")
    st.markdown("AWS Architecture & Documentation Advisory")
    st.divider()

    # SECTION 1: PDF Document Ingestion
    st.markdown("### 📄 Document Ingestion")
    uploaded_file = st.file_uploader(
        "Upload AWS PDF Whitepapers", 
        type="pdf", 
        help="Upload AWS architectural guides or FAQs to index into ChromaDB."
    )

    if uploaded_file is not None:
        if st.button("Index Document", use_container_width=True):
            # Save uploaded bytes to local docs directory
            target_path = UPLOAD_DIR / uploaded_file.name
            with open(target_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Processing PDF: Parsing, splitting & embedding..."):
                try:
                    # Run parser
                    loader = PDFDocumentLoader(target_path)
                    pages = loader.load()

                    # Run splitter
                    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
                    chunks = chunker.split_documents(pages)

                    # Write to database (purging old file records first to prevent overlap duplicates)
                    db_mgr.delete_by_source(uploaded_file.name)
                    db_mgr.add_documents(chunks)

                    st.success(f"Indexed successfully: {len(chunks)} chunks registered!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to process document: {str(e)}")

    st.divider()

    # SECTION 2: System Status metrics
    st.markdown("### 📊 Vector DB Status")
    db_count = get_db_count()
    st.markdown(
        f"""
        <div class="metric-card">
            <span style="font-size: 0.9rem; color: #888;">Collection Name</span><br>
            <span style="font-size: 1.1rem; font-weight: bold; color: #FF9900;">aws_docs_collection</span>
        </div>
        <div class="metric-card">
            <span style="font-size: 0.9rem; color: #888;">Total Chunks Indexed</span><br>
            <span style="font-size: 1.8rem; font-weight: bold; color: #FF9900;">{db_count}</span>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.divider()

    # SECTION 3: Settings Configuration
    st.markdown("### ⚙️ Retriever Settings")
    retriever_k = st.slider(
        "Context Chunk Count (K)", 
        min_value=1, 
        max_value=8, 
        value=4, 
        help="Specify the maximum number of text chunks injected as context."
    )
    
    # Action buttons
    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True, on_click=clear_chat):
        st.success("Chat history cleared!")


# ==========================================
# MAIN INTERFACE
# ==========================================

# Display header banner
st.markdown('<div class="main-title">CloudMentor AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Production RAG Assistant for AWS Whitepapers & Architecture Decisions</div>', 
    unsafe_allow_html=True
)

# API key checkup warning alert
gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not gemini_key:
    st.warning(
        "⚠️ **Gemini API Key is Missing!** Please configure the `GEMINI_API_KEY` environmental "
        "variable to enable LLM response generation. You can still upload files to index them."
    )

# Standard empty state guidance
if not st.session_state.messages:
    st.info(
        "👋 **Welcome to CloudMentor AI!**\n\n"
        "To get started:\n"
        "1. Upload and index AWS architecture guides/PDFs in the sidebar.\n"
        "2. Ask queries comparing AWS services or requesting architecture advice.\n"
        "3. CloudMentor AI will retrieve context blocks and provide grounded answers with inline citations."
    )

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        # Format sources if they exist (only for assistant messages)
        if msg["role"] == "assistant" and msg.get("sources"):
            st.markdown("---")
            st.markdown("**Retrieved Source References:**")
            
            # Group sources by filename to prevent redundant listings
            unique_sources = {}
            for src in msg["sources"]:
                src_key = f"{src['source']} (Page {src['page']})"
                if src_key not in unique_sources:
                    unique_sources[src_key] = src["content"]
            
            # Render references as clean expanders containing context snippets
            for label, snippet in unique_sources.items():
                with st.expander(f"📖 {label}"):
                    st.write(snippet)


# Handle user chat prompt input
if user_prompt := st.chat_input(
    "Ask a question (e.g., 'Compare S3 vs EBS storage types' or 'Recommend a serverless architecture')"
):
    # Render user query bubble
    with st.chat_message("user"):
        st.write(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Execute search-and-generation RAG pipeline
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.spinner("Analyzing documentation and generating architecture recommendations..."):
            try:
                if not gemini_key:
                    st.error("Cannot query model: Gemini API key is missing.")
                elif db_count == 0:
                    st.error(
                        "No documents are indexed. Please upload and index a PDF guide in the sidebar "
                        "before querying."
                    )
                else:
                    # Query RAG Chain
                    response = rag_chain.query(
                        user_prompt, 
                        k=retriever_k, 
                        session_id="streamlit_chat_session"
                    )
                    
                    # Update assistant bubble
                    response_placeholder.write(response.answer)
                    
                    # Render sources expandable panels
                    st.markdown("---")
                    st.markdown("**Retrieved Source References:**")
                    
                    unique_sources = {}
                    for src in response.source_chunks:
                        src_key = f"{src.source} (Page {src.page})"
                        if src_key not in unique_sources:
                            unique_sources[src_key] = src.content

                    for label, snippet in unique_sources.items():
                        with st.expander(f"📖 {label}"):
                            st.write(snippet)

                    # Store message details in session state history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.answer,
                        "sources": [chunk.to_dict() for chunk in response.source_chunks]
                    })
                    
            except Exception as e:
                st.error(f"Error processing RAG execution query: {str(e)}")
