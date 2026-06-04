import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.ingestion.document_loader import PDFDocumentLoader
from src.core.exceptions import UnsupportedFileTypeError


def test_loader_invalid_file_extension():
    """Verify loader raises UnsupportedFileTypeError on non-PDF extensions."""
    invalid_path = Path("test_document.txt")
    with patch.object(Path, "exists", return_value=True):
        with pytest.raises(UnsupportedFileTypeError):
            PDFDocumentLoader(invalid_path)


def test_loader_file_not_found():
    """Verify loader raises FileNotFoundError when target path is missing."""
    missing_path = Path("docs/non_existing_file.pdf")
    with pytest.raises(FileNotFoundError):
        PDFDocumentLoader(missing_path)


@patch("src.ingestion.document_loader.pypdf.PdfReader")
def test_loader_successful_text_extraction(mock_pdf_reader):
    """Verify PDFDocumentLoader parses pages, extracts texts, and maps metadata."""
    # Set up mock reader return states
    mock_reader_instance = MagicMock()
    mock_pdf_reader.return_value = mock_reader_instance
    mock_reader_instance.is_encrypted = False
    
    # Configure mock page objects
    page_1 = MagicMock()
    page_1.extract_text.return_value = "Amazon EC2 instance compute services."
    page_2 = MagicMock()
    page_2.extract_text.return_value = "Auto Scaling Group configurations."
    mock_reader_instance.pages = [page_1, page_2]

    # Run loader with mocked filesystem checks
    with patch.object(Path, "exists", return_value=True):
        loader = PDFDocumentLoader("docs/mock_aws_guide.pdf")
        documents = loader.load()

        assert len(documents) == 2
        
        # Verify text mappings
        assert documents[0].page_content == "Amazon EC2 instance compute services."
        assert documents[1].page_content == "Auto Scaling Group configurations."
        
        # Verify citation metadata preservation
        assert documents[0].metadata["source"] == "mock_aws_guide.pdf"
        assert documents[0].metadata["page"] == 1
        assert documents[0].metadata["total_pages"] == 2
        assert documents[1].metadata["page"] == 2
