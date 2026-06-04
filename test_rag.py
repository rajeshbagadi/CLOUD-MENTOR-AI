import os
import sys
import shutil
from pathlib import Path
from langchain_core.documents import Document

# Add workspace root to Python path
sys.path.append(str(Path(__file__).parent))

from src.model.embeddings import EmbeddingManager
from src.model.gemini_client import GeminiClientManager
from src.vectorstore.vector_store import ChromaStoreManager
from src.rag.retriever import SemanticRetriever
from src.rag.rag_chain import AWSCloudMentorRAGChain


def main():
    print("Testing Complete RAG Pipeline Orchestrator...")
    
    # Check credentials before starting
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("\n[Warning]: GEMINI_API_KEY not configured. Execution of Gemini LLM requires standard key environments.")
        print("Please configure this environment variable to perform full RAG testing.")
        return

    persist_dir = Path(".chroma_test")
    cache_dir = Path(".embeddings_cache")
    
    # Clean up test directories
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        
    try:
        # Load embedding model
        print("\nInitializing Embedding Model...")
        embed_mgr = EmbeddingManager(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=cache_dir
        )
        embeddings = embed_mgr.get_embeddings()
        
        # Load Vector Store
        print("\nConnecting to Vector DB...")
        db_mgr = ChromaStoreManager(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="test_rag_collection"
        )
        
        # Write realistic AWS documentation chunks
        print("\nLoading mock AWS documentation pages...")
        aws_mock_docs = [
            Document(
                page_content=(
                    "Amazon S3 (Simple Storage Service) is an object storage service offering industry-leading "
                    "scalability, data availability, security, and performance. Recommended standard storage classes "
                    "include S3 Standard for frequently accessed data, S3 Standard-IA for infrequently accessed data, "
                    "and Amazon S3 Glacier for long-term archiving solutions."
                ),
                metadata={"source": "s3_user_guide.pdf", "page": 15, "chunk_index": 0}
            ),
            Document(
                page_content=(
                    "AWS Lambda is a serverless, event-driven compute service that lets you run code for virtually any "
                    "type of application or backend service without provisioning or managing servers. You pay only for "
                    "the compute time you consume - there is no charge when your code is not running."
                ),
                metadata={"source": "lambda_developer_guide.pdf", "page": 4, "chunk_index": 0}
            )
        ]
        db_mgr.add_documents(aws_mock_docs)
        
        # Initialize Gemini API Manager
        print("\nInitializing Gemini client...")
        gemini_mgr = GeminiClientManager(
            model_name="gemini-1.5-flash",
            temperature=0.0
        )
        
        # Initialize Retriever
        retriever = SemanticRetriever(vector_store=db_mgr, default_k=2)
        
        # Build RAG chain
        rag_chain = AWSCloudMentorRAGChain(retriever=retriever, gemini_manager=gemini_mgr)
        
        # Test Query 1: Fact in context
        print("\n--- Test 1: Querying details present in documentation ---")
        user_query = "What storage classes are recommended for Amazon S3?"
        response = rag_chain.query(user_query)
        
        print(f"User Query: '{user_query}'")
        print(f"RAG Answer:\n{response.answer}\n")
        print("Associated Sources Cited:")
        for idx, src in enumerate(response.source_chunks):
            print(f"  [{idx + 1}] Source: {src.source} | Page: {src.page}")
            
        # Test Query 2: Hallucination prevention check (Fact NOT in context)
        print("\n--- Test 2: Hallucination Prevention Check (Fact NOT in context) ---")
        unrelated_query = "What is the maximum payload size of Azure Blob Storage?"
        response_unrelated = rag_chain.query(unrelated_query)
        
        print(f"User Query: '{unrelated_query}'")
        print(f"RAG Answer:\n{response_unrelated.answer}\n")
        
    except Exception as e:
        print(f"\n[Test Error Occurred]: {e}")
    finally:
        # Tear down directories
        if persist_dir.exists():
            shutil.rmtree(persist_dir)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        print("\nCleaned up persistent database and cache directories successfully.")


if __name__ == "__main__":
    main()
