import sys
import shutil
from pathlib import Path

# Add workspace root to Python path
sys.path.append(str(Path(__file__).parent))

from src.model.embeddings import EmbeddingManager


def main():
    print("Testing EmbeddingManager initialization and caching support...")
    cache_directory = Path(".embeddings_cache")
    
    # Ensure clean state for evaluation
    if cache_directory.exists():
        shutil.rmtree(cache_directory)
        
    try:
        # 1. Initialize without cache
        print("\n--- Test 1: Loading model without cache ---")
        manager_no_cache = EmbeddingManager(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=None
        )
        embeddings_no_cache = manager_no_cache.get_embeddings()
        
        # Test query embedding
        test_text = "AWS EC2 instances are virtual servers in the cloud."
        query_vector = embeddings_no_cache.embed_query(test_text)
        print(f"Generated query vector successfully. Vector dimension size: {len(query_vector)}")
        print(f"First 5 dimensions: {query_vector[:5]}")
        
        # 2. Initialize with cache
        print("\n--- Test 2: Loading model WITH local disk cache ---")
        manager_cached = EmbeddingManager(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=cache_directory
        )
        embeddings_cached = manager_cached.get_embeddings()
        
        # Run document vectors generation (First write pass)
        print("\nEncoding documents (1st run: compiling vector and caching)...")
        docs = [
            "Amazon S3 provides scalable object storage.",
            "AWS Lambda is a serverless compute service."
        ]
        vectors_1st = embeddings_cached.embed_documents(docs)
        print(f"Embedded {len(docs)} documents. Base vector length: {len(vectors_1st[0])}")
        
        # Run second document vectors generation (Second cache read pass)
        print("\nEncoding same documents (2nd run: should read directly from disk cache)...")
        vectors_2nd = embeddings_cached.embed_documents(docs)
        print("Success! Cache generation and lookup completed.")
        
    except Exception as e:
        print(f"\n[Test Error Occurred]: {e}")
    finally:
        # Clean up temporary test files
        if cache_directory.exists():
            shutil.rmtree(cache_directory)
            print("\nCleaned up temporary test cache directory.")


if __name__ == "__main__":
    main()
