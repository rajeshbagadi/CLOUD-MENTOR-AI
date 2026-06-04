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
        # MiniLM-L6-v2 uses 384 dimensions
        return [[0.1] * 384 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 384


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
        # Tear down temporary workspace
        shutil.rmtree(temp_dir)
