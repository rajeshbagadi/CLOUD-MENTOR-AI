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
    print("Testing AWS Service Comparison Engine...")
    
    # Check credentials
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("\n[Warning]: GEMINI_API_KEY not configured. LLM comparison query requires standard API environments.")
        return

    persist_dir = Path(".chroma_test")
    cache_dir = Path(".embeddings_cache")
    
    # Clean up directories
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        
    try:
        # Load embedding model
        print("Initializing Embeddings...")
        embed_mgr = EmbeddingManager(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=cache_dir
        )
        embeddings = embed_mgr.get_embeddings()
        
        # Load Vector Store
        print("Connecting to Vector DB...")
        db_mgr = ChromaStoreManager(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="test_comparison_collection"
        )
        
        # Write mock pages for S3 and EBS to perform comparison
        print("Writing S3 and EBS document chunks...")
        mock_docs = [
            Document(
                page_content=(
                    "Amazon S3 (Simple Storage Service) is a serverless object storage service designed for storing "
                    "unstructured data like images, logs, backups, and static website content. It offers 99.999999999% "
                    "durability, replicates data across multiple AZs automatically, and scales storage size infinitely. "
                    "Pricing is based on volume stored (GB/month) and network requests."
                ),
                metadata={"source": "s3_faq.pdf", "page": 1, "chunk_index": 0}
            ),
            Document(
                page_content=(
                    "Amazon EBS (Elastic Block Store) provides high-performance block storage volumes for use with "
                    "Amazon EC2 instances. It is structured like a virtual hard disk, mounted directly to virtual "
                    "servers, and designed for structured data like databases (MySQL, PostgreSQL) or operating system files. "
                    "It offers sub-millisecond latencies, requires manual backups via snapshots, and replication is restricted "
                    "within a single Availability Zone (AZ)."
                ),
                metadata={"source": "ebs_faq.pdf", "page": 3, "chunk_index": 0}
            )
        ]
        db_mgr.add_documents(mock_docs)
        
        # Initialize Gemini Client Manager
        print("Initializing Gemini client...")
        gemini_mgr = GeminiClientManager(
            model_name="gemini-1.5-flash",
            temperature=0.0
        )
        
        # Initialize Retriever
        retriever = SemanticRetriever(vector_store=db_mgr, default_k=2)
        
        # Build RAG chain
        rag_chain = AWSCloudMentorRAGChain(retriever=retriever, gemini_manager=gemini_mgr)
        
        # Test Query: Request comparison (Triggers routing)
        query = "Compare S3 vs EBS"
        print(f"\nSubmitting Comparison Query: '{query}'...")
        response = rag_chain.query(query)
        
        print("\n--- Generated Comparison Output ---\n")
        print(response.answer)
        
        print("\nAssociated Sources Cited:")
        for idx, src in enumerate(response.source_chunks):
            print(f"  [{idx + 1}] Source: {src.source} | Page: {src.page}")
            
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
