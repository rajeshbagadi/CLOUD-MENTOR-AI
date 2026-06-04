import time
from pathlib import Path
from typing import Optional, Union, Dict, Any

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore

from src.core.logging_config import setup_logger

logger = setup_logger("embeddings")


class EmbeddingManager:
    """Manages the lifecycle, loading, execution, and local caching of text embeddings."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[Union[str, Path]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        encode_kwargs: Optional[Dict[str, Any]] = None
    ):
        """Initializes the EmbeddingManager.
        
        Args:
            model_name (str): Hugging Face model path or name.
            cache_dir (Optional[Union[str, Path]]): Directory path to write cached embeddings. 
                                                    If None, caching is disabled.
            model_kwargs (Optional[Dict[str, Any]]): Keyword configurations passed to the HF model 
                                                     (e.g., {'device': 'cpu'}).
            encode_kwargs (Optional[Dict[str, Any]]): Keyword configurations passed during model inference 
                                                      (e.g., {'normalize_embeddings': True}).
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model_kwargs = model_kwargs or {"device": "cpu"}
        self.encode_kwargs = encode_kwargs or {"normalize_embeddings": True}
        self._embeddings: Optional[Embeddings] = None

    def get_embeddings(self) -> Embeddings:
        """Instantiates and returns the configured embedding model, applying local caching if requested.
        
        Returns:
            Embeddings: LangChain core compatible embedding model.
            
        Raises:
            RuntimeError: If model fails to download, load, or cache directories cannot be configured.
        """
        if self._embeddings is not None:
            return self._embeddings

        start_time = time.time()
        logger.info(
            f"Initializing embedding model '{self.model_name}' "
            f"on execution device: '{self.model_kwargs.get('device', 'cpu')}'"
        )
        
        try:
            # Load the base HuggingFace Model
            base_embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs=self.model_kwargs,
                encode_kwargs=self.encode_kwargs
            )
            
            # Setup local file system cache backing if directory was provided
            if self.cache_dir:
                try:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Local embedding cache enabled at: {self.cache_dir.resolve()}")
                    
                    # Store vectors to local disk store
                    local_store = LocalFileStore(str(self.cache_dir))
                    
                    # Wrap base embeddings with CacheBackedEmbeddings
                    self._embeddings = CacheBackedEmbeddings.from_bytes_store(
                        underlying_embeddings=base_embeddings,
                        document_embedding_cache=local_store,
                        namespace=self.model_name.replace("/", "_")
                    )
                except Exception as cache_err:
                    logger.error(
                        f"Failed to set up embedding disk cache: {str(cache_err)}. "
                        "Falling back to non-cached execution."
                    )
                    self._embeddings = base_embeddings
            else:
                logger.info("Embedding cache is disabled (no cache_dir was specified).")
                self._embeddings = base_embeddings
                
            elapsed = time.time() - start_time
            logger.info(f"Embedding model loaded successfully in {elapsed:.2f} seconds.")
            return self._embeddings
            
        except Exception as e:
            logger.exception(f"Critical failure while loading model '{self.model_name}': {str(e)}")
            raise RuntimeError(f"Embedding model initialization failed: {str(e)}") from e
