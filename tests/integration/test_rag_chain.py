import pytest
import shutil
import tempfile
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.vectorstore.vector_store import ChromaStoreManager
from src.rag.retriever import SemanticRetriever
from src.rag.rag_chain import AWSCloudMentorRAGChain
from src.model.gemini_client import GeminiClientManager


class MockEmbeddings(Embeddings):
    """Fake embedding model to speed up tests without downloading models."""

    def embed_documents(self, texts):
        return [[0.1] * 384 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 384


def test_rag_pipeline_execution():
    """Verify end-to-end RAG pipeline, context formatting, LLM call, and response mapping."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Load database with mock vectors
        mock_embed = MockEmbeddings()
        db_mgr = ChromaStoreManager(
            persist_directory=temp_dir,
            embedding_function=mock_embed,
            collection_name="test_rag_integration"
        )
        
        # Load mock docs
        docs = [
            Document(
                page_content="AWS EC2 is a compute resource virtualization server.",
                metadata={"source": "ec2_doc.pdf", "page": 1, "chunk_index": 0}
            )
        ]
        db_mgr.add_documents(docs)
        
        # Mock Gemini Manager
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="AWS EC2 provides compute resource virtualization [source: ec2_doc.pdf, page: 1]."
        )
        
        mock_gemini = MagicMock(spec=GeminiClientManager)
        mock_gemini.get_llm.return_value = mock_llm
        
        # Setup pipeline components
        retriever = SemanticRetriever(vector_store=db_mgr, default_k=1)
        rag_chain = AWSCloudMentorRAGChain(retriever=retriever, gemini_manager=mock_gemini)
        
        # Run query
        response = rag_chain.query("Explain AWS EC2 virtual servers")
        
        # Verify RAG pipeline results
        assert response.answer == "AWS EC2 provides compute resource virtualization [source: ec2_doc.pdf, page: 1]."
        assert len(response.source_chunks) == 1
        assert response.source_chunks[0].source == "ec2_doc.pdf"
        assert response.source_chunks[0].page == 1
        
    finally:
        # Tear down directories
        shutil.rmtree(temp_dir)
