from unittest.mock import MagicMock, patch
from src.model.embeddings import EmbeddingManager


@patch("src.model.embeddings.HuggingFaceEmbeddings")
def test_embedding_manager_initialization(mock_hf_embeddings):
    """Verify EmbeddingManager correctly instantiates HuggingFaceEmbeddings model."""
    mock_instance = MagicMock()
    mock_hf_embeddings.return_value = mock_instance
    
    manager = EmbeddingManager(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir=None
    )
    embeddings = manager.get_embeddings()
    
    assert embeddings == mock_instance
    mock_hf_embeddings.assert_called_once_with(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
