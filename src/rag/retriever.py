from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.vectorstore.vector_store import ChromaStoreManager
from src.core.logging_config import setup_logger

logger = setup_logger("retriever")


class RetrievedChunk:
    """Container representing a parsed vector match, optimized for inline citations."""

    def __init__(
        self,
        content: str,
        source: str,
        page: int,
        file_path: str,
        score: Optional[float] = None
    ):
        """Initializes a RetrievedChunk.
        
        Args:
            content (str): Text extracted from the matching document page.
            source (str): Original PDF source filename.
            page (int): Page number of the matching chunk.
            file_path (str): Local path location.
            score (Optional[float]): Relevance similarity metric score.
        """
        self.content = content
        self.source = source
        self.page = page
        self.file_path = file_path
        self.score = score

    def to_dict(self) -> Dict[str, Any]:
        """Converts the chunk metadata and content to a dictionary representation."""
        return {
            "content": self.content,
            "source": self.source,
            "page": self.page,
            "file_path": self.file_path,
            "score": self.score
        }


class SemanticRetriever:
    """Manages the query retrieval pipeline, wrapping ChromaDB to supply structured citation outputs."""

    def __init__(self, vector_store: ChromaStoreManager, default_k: int = 4):
        """Initializes the SemanticRetriever.
        
        Args:
            vector_store (ChromaStoreManager): Target vector database instance.
            default_k (int): Default number of documents to retrieve.
        """
        self.vector_store = vector_store
        self.default_k = default_k

    def retrieve_chunks(self, query: str, k: Optional[int] = None) -> List[RetrievedChunk]:
        """Queries the vector store and extracts chunks formatted for citation.
        
        Args:
            query (str): The search phrase.
            k (Optional[int]): Query-specific K parameter. Overrides default_k.
            
        Returns:
            List[RetrievedChunk]: List of structured chunks with text content and citations.
        """
        search_k = k if k is not None else self.default_k
        logger.info(f"Retrieving top {search_k} context chunks for query: '{query}'")
        
        try:
            # Query the database
            matching_docs = self.vector_store.similarity_search(query=query, k=search_k)
            
            retrieved_chunks: List[RetrievedChunk] = []
            for doc in matching_docs:
                content = doc.page_content
                source = doc.metadata.get("source", "unknown_source.pdf")
                page = doc.metadata.get("page", 1)
                file_path = doc.metadata.get("file_path", "")
                score = doc.metadata.get("score")  # Some configs append similarity metric scores
                
                retrieved_chunks.append(
                    RetrievedChunk(
                        content=content,
                        source=source,
                        page=page,
                        file_path=file_path,
                        score=score
                    )
                )
                
            logger.info(f"Successfully retrieved and parsed {len(retrieved_chunks)} citation-ready chunks.")
            return retrieved_chunks
            
        except Exception as e:
            logger.exception(f"Semantic retrieval operation failed for query '{query}': {str(e)}")
            raise RuntimeError(f"Retrieval operation failed: {str(e)}") from e

    def get_langchain_retriever(self, k: Optional[int] = None) -> BaseRetriever:
        """Retrieves the standard native LangChain Retriever class for integration with chains.
        
        Args:
            k (Optional[int]): Target Top-K documents.
        """
        search_k = k if k is not None else self.default_k
        return self.vector_store.get_retriever(search_kwargs={"k": search_k})
