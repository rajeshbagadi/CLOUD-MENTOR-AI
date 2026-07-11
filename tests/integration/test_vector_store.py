import pytest
import shutil
import tempfile
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.vectorstore.vector_store import ChromaStoreManager


class MockEmbeddings(Embeddings):
    """Fake embedding model to speed up tests without downloading models."""

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            emb = [0.0] * 384
            if any(kw in text.lower() for kw in ["ec2", "virtual server", "compute"]):
                emb[0] = 1.0
            elif any(kw in text.lower() for kw in ["s3", "storage", "serverless"]):
                emb[1] = 1.0
            else:
                emb[2] = 1.0
            embeddings.append(emb)
        return embeddings

    def embed_query(self, text):
        emb = [0.0] * 384
        if any(kw in text.lower() for kw in ["ec2", "virtual server", "compute"]):
            emb[0] = 1.0
        elif any(kw in text.lower() for kw in ["s3", "storage", "serverless"]):
            emb[1] = 1.0
        else:
            emb[2] = 1.0
        return emb


def test_chromadb_lifecycle():
    """Verify ChromaDB database creation, write, similarity search, and clear integrations."""
    # Setup temporary persistence directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        mock_embed = MockEmbeddings()
        db_mgr = ChromaStoreManager(
            persist_directory=temp_dir,
            embedding_function=mock_embed,
            collection_name="test_integration_collection"
        )
        
        # Test document loading
        docs = [
            Document(
                page_content="AWS EC2 is a virtual server instance hosting compute code.",
                metadata={"source": "ec2.pdf", "page": 1, "chunk_index": 0}
            ),
            Document(
                page_content="Amazon S3 is a serverless object storage storage class.",
                metadata={"source": "s3.pdf", "page": 2, "chunk_index": 0}
            )
        ]
        
        doc_ids = db_mgr.add_documents(docs)
        assert len(doc_ids) == 2
        
        # Test retrieval matching
        results = db_mgr.similarity_search("Tell me about virtual servers compute", k=1)
        assert len(results) == 1
        assert results[0].metadata["source"] == "ec2.pdf"
        
        # Test deleting matching source files
        purged = db_mgr.delete_by_source("ec2.pdf")
        assert purged is True
        
        # Query again to verify deletion
        results_after = db_mgr.similarity_search("EC2 virtual servers", k=2)
        assert len(results_after) > 0
        # The top result should no longer be ec2.pdf since it was deleted
        assert results_after[0].metadata["source"] != "ec2.pdf"
        
    finally:
        # Close database connection to release files on Windows
        if 'db_mgr' in locals() and db_mgr._db is not None:
            if hasattr(db_mgr._db, "_client") and hasattr(db_mgr._db._client, "close"):
                db_mgr._db._client.close()
        if 'db_mgr' in locals():
            del db_mgr
        import gc; gc.collect()
        
        # Retry deletion to avoid Windows file system locking race conditions
        import time
        for idx in range(5):
            try:
                shutil.rmtree(temp_dir)
                break
            except Exception:
                time.sleep(0.2 * (idx + 1))

