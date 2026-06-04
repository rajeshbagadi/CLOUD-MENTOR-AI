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
    print("Testing AWS Architecture Recommendation Engine...")
    
    # Check credentials
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("\n[Warning]: GEMINI_API_KEY not configured. LLM recommendation query requires standard API environments.")
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
            collection_name="test_recommendation_collection"
        )
        
        # Write mock pages for AWS scaling and multi-AZ deployments
        print("Writing AWS Architecture best practices chunks...")
        mock_docs = [
            Document(
                page_content=(
                    "For hosting scalable web applications, AWS recommends deploying in a Multi-AZ VPC configuration. "
                    "Incoming requests are received by Amazon Route 53 and resolved to Amazon CloudFront for edge caching. "
                    "Compute workloads are hosted on Amazon EC2 instances managed by an Auto Scaling Group behind an Application "
                    "Load Balancer (ALB). Relational database records are stored in Amazon RDS PostgreSQL operating in Multi-AZ "
                    "mode with automated replication."
                ),
                metadata={"source": "vpc_reference_architecture.pdf", "page": 2, "chunk_index": 0}
            ),
            Document(
                page_content=(
                    "AWS security best practices for 3-tier web applications require EC2 instances to reside in private subnets, "
                    "accessible only from the ALB security groups. Database endpoints are isolated in designated DB subnets "
                    "rejecting direct internet traffic. All stored data volumes inside Amazon RDS and EBS must be encrypted using "
                    "AWS KMS keys."
                ),
                metadata={"source": "aws_security_whitepaper.pdf", "page": 8, "chunk_index": 3}
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
        
        # Test Query: Request architecture recommendation
        query = "Recommend a scalable architecture design to host an ecommerce application"
        print(f"\nSubmitting Recommendation Query: '{query}'...")
        response = rag_chain.query(query)
        
        print("\n--- Generated Recommendation Blueprint Output ---\n")
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
