from langchain_core.documents import Document
from src.ingestion.text_splitter import DocumentChunker


def test_chunker_splitting_preserves_metadata():
    """Verify DocumentChunker splits text and augments sequential chunk index fields."""
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    
    # Input pages with metadata
    raw_pages = [
        Document(
            page_content="AWS EC2 is a compute service. It is designed to host virtual machines.",
            metadata={"source": "ec2_doc.pdf", "page": 1}
        ),
        Document(
            page_content="Amazon S3 provides durable storage. Use it for object backups.",
            metadata={"source": "s3_doc.pdf", "page": 5}
        )
    ]
    
    chunks = chunker.split_documents(raw_pages)
    
    assert len(chunks) >= 2
    
    # Verify metadata preservation
    assert chunks[0].metadata["source"] == "ec2_doc.pdf"
    assert chunks[0].metadata["page"] == 1
    assert "chunk_index" in chunks[0].metadata
    
    # Verify separate counters for unique source files
    assert chunks[0].metadata["chunk_index"] == 0
    
    # Verify metadata details of the second file
    second_file_chunk = [c for c in chunks if c.metadata["source"] == "s3_doc.pdf"][0]
    assert second_file_chunk.metadata["page"] == 5
    assert second_file_chunk.metadata["chunk_index"] == 0
