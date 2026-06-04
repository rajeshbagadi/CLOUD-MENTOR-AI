import sys
import shutil
from pathlib import Path
from langchain_core.documents import Document

# Add workspace root to Python path
sys.path.append(str(Path(__file__).parent))

from src.model.embeddings import EmbeddingManager
from src.vectorstore.vector_store import ChromaStoreManager
from src.rag.retriever import SemanticRetriever


def main():
    print("Testing SemanticRetriever and citation parsing integration...")
    persist_dir = Path(".chroma_test")
    cache_dir = Path(".embeddings_cache")
    
    # Clean up test directories
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
            collection_name="test_citation_collection"
        )
        
        # Write mock pages with detailed metadata
        print("Writing mock chunks with source metadata...")
        mock_docs = [
            Document(
                page_content="AWS Lambda runs code in response to events and automatically manages compute resource allocation.",
                metadata={"source": "lambda_faq.pdf", "page": 4, "chunk_index": 2}
            ),
            Document(
                page_content="Amazon S3 provides durable object storage with options for frequent, infrequent, and archive access.",
                metadata={"source": "s3_faq.pdf", "page": 12, "chunk_index": 7}
            ),
            Document(
                page_content="Amazon EC2 offers bare metal instances and virtualization capacities with customized instance configuration options.",
                metadata={"source": "ec2_faq.pdf", "page": 2, "chunk_index": 0}
            )
        ]
        db_mgr.add_documents(mock_docs)
        
        # Instantiate Retriever
        retriever = SemanticRetriever(vector_store=db_mgr, default_k=2)
        
        # Test Query 1: Top K = 2 (default)
        print("\n--- Test 1: Fetching default K = 2 chunks ---")
        chunks = retriever.retrieve_chunks("What storage classes does S3 support?")
        print(f"Retrieved {len(chunks)} chunks:")
        for idx, chunk in enumerate(chunks):
            print(f"  [{idx + 1}] Source   : {chunk.source}")
            print(f"      Page     : {chunk.page}")
            print(f"      Path     : {chunk.file_path}")
            print(f"      Content  : '{chunk.content[:70]}...'")
            
        # Test Query 2: Overriding K = 1
        print("\n--- Test 2: Overriding default K to fetch 1 chunk ---")
        single_chunk = retriever.retrieve_chunks("What is AWS Lambda computing?", k=1)
        print(f"Retrieved {len(single_chunk)} chunk:")
        if single_chunk:
            print(f"  [1] Source   : {single_chunk[0].source}")
            print(f"      Page     : {single_chunk[0].page}")
            print(f"      Content  : '{single_chunk[0].content[:70]}...'")
            
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
