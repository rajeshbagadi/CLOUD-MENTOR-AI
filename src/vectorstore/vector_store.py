import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from src.core.exceptions import CloudMentorError
from src.core.logging_config import setup_logger

logger = setup_logger("vector_store")


class VectorStoreError(CloudMentorError):
    """Exception raised for database operations failures inside ChromaDB."""
    pass


class ChromaStoreManager:
    """Manages persistent document storage, vector embedding generation, query retrieval, and updates using ChromaDB."""

    def __init__(
        self,
        persist_directory: Union[str, Path],
        embedding_function: Embeddings,
        collection_name: str = "aws_documentation"
    ):
        """Initializes the ChromaStoreManager.
        
        Args:
            persist_directory (Union[str, Path]): Directory path where Chroma database files are saved.
            embedding_function (Embeddings): Embedding model to convert text to vectors.
            collection_name (str): Name of the Chroma collection.
        """
        self.persist_directory = Path(persist_directory)
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self._db: Optional[Chroma] = None
        
        # Ensure database folder exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Instantiates the persistent Chroma vector store."""
        try:
            logger.info(
                f"Initializing ChromaDB connection. Collection: '{self.collection_name}', "
                f"Storage Path: '{self.persist_directory.resolve()}'"
            )
            self._db = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=str(self.persist_directory)
            )
        except Exception as e:
            logger.exception(f"Failed to initialize Chroma DB connection: {str(e)}")
            raise VectorStoreError(f"Database connection error: {str(e)}") from e

    def _generate_document_id(self, doc: Document) -> str:
        """Generates a unique deterministic ID for a document chunk using metadata and content hash.
        
        Ensures updates/upserts overwrite old records without duplication.
        """
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 0)
        chunk_idx = doc.metadata.get("chunk_index", 0)
        
        # Base logical identifier
        base_id = f"{source}_page_{page}_chunk_{chunk_idx}"
        
        # Append md5 hash of page content to ensure uniqueness if contents change
        content_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
        return f"{base_id}_{content_hash}"

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Saves documents to ChromaDB, automatically generating deterministic IDs.
        
        Args:
            documents (List[Document]): List of chunks/pages to write to store.
            
        Returns:
            List[str]: List of unique generated IDs for the documents stored.
        """
        if not documents:
            logger.warning("Empty document list passed to add_documents.")
            return []

        logger.info(f"Writing {len(documents)} document chunk(s) to ChromaDB...")
        try:
            ids = [self._generate_document_id(doc) for doc in documents]
            
            # Write/Upsert to ChromaDB
            self._db.add_documents(documents=documents, ids=ids)
            logger.info(f"Successfully saved {len(documents)} chunks to Vector Store.")
            return ids
            
        except Exception as e:
            logger.exception(f"Failed to write documents to vector database: {str(e)}")
            raise VectorStoreError(f"Failed to write documents: {str(e)}") from e

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Performs a similarity search query against the vector store.
        
        Args:
            query (str): The search query text.
            k (int): Number of top documents to retrieve.
            filter_metadata (Optional[Dict[str, Any]]): Metadata filtering dictionary.
            
        Returns:
            List[Document]: List of matching documents.
        """
        logger.debug(f"Executing similarity search for query: '{query}' with k={k}, filters={filter_metadata}")
        try:
            results = self._db.similarity_search(query=query, k=k, filter=filter_metadata)
            logger.debug(f"Search yielded {len(results)} matches.")
            return results
        except Exception as e:
            logger.exception(f"Similarity search failed: {str(e)}")
            raise VectorStoreError(f"Search execution failed: {str(e)}") from e

    def delete_documents(self, ids: List[str]) -> bool:
        """Deletes specific document records from the database using their unique IDs.
        
        Args:
            ids (List[str]): List of target record IDs to remove.
            
        Returns:
            bool: True if operation completed successfully, False otherwise.
        """
        if not ids:
            logger.warning("No IDs specified for delete operation.")
            return False

        logger.info(f"Deleting {len(ids)} document record(s) from ChromaDB...")
        try:
            self._db.delete(ids=ids)
            logger.info("Successfully deleted document records.")
            return True
        except Exception as e:
            logger.exception(f"Deletion failed for document IDs: {str(e)}")
            raise VectorStoreError(f"Delete operation failed: {str(e)}") from e

    def delete_by_source(self, source_filename: str) -> bool:
        """Queries and deletes all chunks originating from a specific source file.
        
        Extremely useful for cleaning up replaced/re-uploaded PDF documents.
        
        Args:
            source_filename (str): Name of the source file.
            
        Returns:
            bool: True if documents were cleared, False otherwise.
        """
        logger.info(f"Attempting to purge records matching source filename: '{source_filename}'")
        try:
            # Query all metadata matching this source
            collection = self._db._collection
            results = collection.get(where={"source": source_filename})
            
            ids_to_delete = results.get("ids", [])
            if not ids_to_delete:
                logger.info(f"No existing records found for source '{source_filename}'. Nothing to clear.")
                return False
                
            logger.info(f"Found {len(ids_to_delete)} record(s) for '{source_filename}'. Purging...")
            return self.delete_documents(ids_to_delete)
            
        except Exception as e:
            logger.exception(f"Failed to delete records for source '{source_filename}': {str(e)}")
            raise VectorStoreError(f"Purge by source failed: {str(e)}") from e

    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """Exposes the vector store as a standard LangChain Retriever.
        
        Args:
            search_kwargs (Optional[Dict[str, Any]]): Parameters configuration for retriever search 
                                                      (e.g., {'k': 4}).
        """
        kwargs = search_kwargs or {"k": 4}
        return self._db.as_retriever(search_kwargs=kwargs)
