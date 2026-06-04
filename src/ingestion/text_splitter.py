from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.core.logging_config import setup_logger

logger = setup_logger("text_splitter")


class DocumentChunker:
    """Handles splitting of raw document pages into structured, overlapping text chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initializes the document chunker.
        
        Args:
            chunk_size (int): Maximum character size of each split chunk.
            chunk_overlap (int): Number of overlapping characters between adjacent chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Standard recursive splitter prioritizing paragraphs, sentences, and words
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits loaded LangChain documents into chunks, preserving and augmenting metadata.
        
        Preserves original document metadata:
            - source: The filename.
            - page: The specific page number of the chunk.
            - file_path: Absolute file location on disk.
            - total_pages: Total pages in the original document.
            
        Adds additional tracing metadata:
            - chunk_index: 0-indexed count of chunks for each specific source document.
            
        Args:
            documents (List[Document]): List of extracted document pages.
            
        Returns:
            List[Document]: List of text chunks with inherited and augmented metadata.
        """
        if not documents:
            logger.warning("Empty document list provided to split_documents.")
            return []
            
        logger.info(
            f"Splitting {len(documents)} page-documents. "
            f"Config: chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}"
        )
        
        try:
            chunks = self.splitter.split_documents(documents)
            
            # Post-process chunks to inject logical chunk ordering tracking
            source_chunk_counters = {}
            for chunk in chunks:
                source_file = chunk.metadata.get("source", "unknown")
                
                if source_file not in source_chunk_counters:
                    source_chunk_counters[source_file] = 0
                
                # Enrich metadata with ordinal index per source document
                chunk.metadata["chunk_index"] = source_chunk_counters[source_file]
                source_chunk_counters[source_file] += 1
                
            logger.info(
                f"Split operations complete. Generated {len(chunks)} chunks "
                f"across {len(source_chunk_counters)} unique source file(s)."
            )
            return chunks
            
        except Exception as e:
            logger.exception(f"Unexpected error during document chunking: {str(e)}")
            raise RuntimeError(f"Document split execution failed: {str(e)}") from e
