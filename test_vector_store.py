import sys
import shutil
from pathlib import Path
from langchain_core.documents import Document

# Add workspace root to Python path
sys.path.append(str(Path(__file__).parent))

from src.model.embeddings import EmbeddingManager
from src.vectorstore.vector_store import ChromaStoreManager


def main():
    print("Testing ChromaStoreManager integration...")
    persist_dir = Path(".chroma_test")
    cache_dir = Path(".embeddings_cache")
    
    # Ensure clean workspace directories for validation
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        
    try:
        # Load embedding model
        print("\nInitializing Embedding model...")
        embed_mgr = EmbeddingManager(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=cache_dir
        )
        embeddings = embed_mgr.get_embeddings()
        
        # Open Chroma DB Connection
        print("\nOpening connection to local persistent ChromaDB...")
        db_mgr = ChromaStoreManager(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="test_collection"
        )
        
        # Mock document pages for chunk inserts
        print("\nWriting mock document chunks to database...")
        docs = [
            Document(
                page_content="AWS CloudFormation is an IaC service that helps define architectures.",
                metadata={"source": "cloudformation.pdf", "page": 1, "chunk_index": 0}
            ),
            Document(
                page_content="AWS EC2 is a compute service running resizable virtual servers.",
                metadata={"source": "ec2.pdf", "page": 1, "chunk_index": 0}
            ),
            Document(
                page_content="Amazon S3 is an object storage service offering industry-leading scalability.",
                metadata={"source": "s3.pdf", "page": 1, "chunk_index": 0}
            )
        ]
        
        # Save documents
        doc_ids = db_mgr.add_documents(docs)
        print(f"Added {len(docs)} documents. Return ID list: {doc_ids}")
        
        # Run Similarity Search
        print("\n--- Test 1: Similarity Search ---")
        query = "Show me compute and virtual instance servers on AWS"
        matches = db_mgr.similarity_search(query, k=1)
        if matches:
            match = matches[0]
            print(f"Top Query Match:")
            print(f"  Page Content  : '{match.page_content}'")
            print(f"  Source PDF    : {match.metadata.get('source')}")
            print(f"  Page Num      : {match.metadata.get('page')}")
        else:
            print("Error: No search results found.")
            
        # Run Deletion test
        print("\n--- Test 2: Purging records by source PDF name ---")
        print("Executing purge for source: 'cloudformation.pdf'")
        purged = db_mgr.delete_by_source("cloudformation.pdf")
        print(f"Operation status (Did delete records): {purged}")
        
        # Query again to verify records were cleared
        print("\nValidating deletion of CloudFormation chunks...")
        all_results = db_mgr.similarity_search("CloudFormation template resource schema", k=3)
        print("Available matching documents remaining:")
        for idx, m in enumerate(all_results):
            print(f"  Result {idx + 1}: Source='{m.metadata.get('source')}' | '{m.page_content[:40]}...'")
            
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
