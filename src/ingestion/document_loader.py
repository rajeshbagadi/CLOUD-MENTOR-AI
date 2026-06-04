from pathlib import Path
from typing import List, Dict, Any, Union
import pypdf
from langchain_core.documents import Document

from src.core.exceptions import DocumentLoadError, UnsupportedFileTypeError, EmptyDocumentError
from src.core.logging_config import setup_logger

logger = setup_logger("document_loader")


class PDFDocumentLoader:
    """Loader responsible for reading a single PDF file and extracting text page by page."""

    def __init__(self, file_path: Union[str, Path]):
        """Initializes the loader with a target file path.
        
        Args:
            file_path (Union[str, Path]): The filesystem path to the PDF document.
            
        Raises:
            FileNotFoundError: If the specified file does not exist.
            UnsupportedFileTypeError: If the file does not have a .pdf extension.
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        if self.file_path.suffix.lower() != ".pdf":
            raise UnsupportedFileTypeError(
                f"Unsupported file format: {self.file_path.suffix}. Only PDFs are supported."
            )

    def load(self) -> List[Document]:
        """Loads and extracts text content from the PDF document page by page.
        
        Returns:
            List[Document]: List of LangChain Document objects containing the raw text
                            along with source and page metadata.
                            
        Raises:
            DocumentLoadError: If the PDF is encrypted and cannot be decrypted, or if parsing fails.
            EmptyDocumentError: If no text was extracted from any page (e.g. scanned image PDF).
        """
        logger.info(f"Starting text extraction for file: {self.file_path.name}")
        documents: List[Document] = []
        
        try:
            reader = pypdf.PdfReader(str(self.file_path))
            
            # Check if PDF is encrypted and attempt to decrypt with empty password
            if reader.is_encrypted:
                try:
                    decrypted = reader.decrypt("")
                    if decrypted == 0:
                        raise DocumentLoadError(
                            f"File {self.file_path.name} is encrypted and could not be decrypted."
                        )
                except Exception as decrypt_err:
                    raise DocumentLoadError(
                        f"Failed to decrypt PDF {self.file_path.name}: {str(decrypt_err)}"
                    ) from decrypt_err
            
            num_pages = len(reader.pages)
            logger.debug(f"PDF {self.file_path.name} contains {num_pages} pages.")
            
            total_text_extracted = 0
            for page_idx in range(num_pages):
                page = reader.pages[page_idx]
                page_num = page_idx + 1  # 1-indexed page numbering
                
                try:
                    text = page.extract_text() or ""
                    cleaned_text = text.strip()
                    total_text_extracted += len(cleaned_text)
                    
                    # Core metadata to preserve for source citation
                    metadata: Dict[str, Any] = {
                        "source": self.file_path.name,
                        "file_path": str(self.file_path.resolve()),
                        "page": page_num,
                        "total_pages": num_pages
                    }
                    
                    documents.append(
                        Document(
                            page_content=cleaned_text,
                            metadata=metadata
                        )
                    )
                except Exception as page_err:
                    logger.error(
                        f"Error extracting text from page {page_num} of {self.file_path.name}: {str(page_err)}"
                    )
                    # Proceed with other pages rather than failing the entire load
                    
            if total_text_extracted == 0:
                raise EmptyDocumentError(
                    f"No text could be extracted from PDF: {self.file_path.name}. "
                    "The document may be scanned (image-only) or contain only graphics."
                )
                
            logger.info(
                f"Successfully loaded {len(documents)} pages from {self.file_path.name} "
                f"(Extracted characters: {total_text_extracted})"
            )
            return documents
            
        except (DocumentLoadError, EmptyDocumentError):
            raise
        except Exception as e:
            logger.exception(f"Unexpected error loading PDF {self.file_path.name}: {str(e)}")
            raise DocumentLoadError(f"Failed to parse PDF {self.file_path.name}: {str(e)}") from e


class DirectoryPDFLoader:
    """Helper class to scan and ingest all PDF files from a specified directory."""
    
    def __init__(self, directory_path: Union[str, Path]):
        """Initializes the directory scanner.
        
        Args:
            directory_path (Union[str, Path]): The path to the directory containing PDFs.
            
        Raises:
            FileNotFoundError: If the directory does not exist.
            NotADirectoryError: If the path is a file instead of a directory.
        """
        self.directory_path = Path(directory_path)
        if not self.directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {self.directory_path}")
        if not self.directory_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.directory_path}")

    def load_all(self, fail_fast: bool = False) -> List[Document]:
        """Scans the directory and loads all PDF files.
        
        Args:
            fail_fast (bool): If True, halts execution on the first PDF loading failure.
                              If False, logs the failure and proceeds to the next PDF.
                              
        Returns:
            List[Document]: List of LangChain Document objects parsed across all successfully loaded PDFs.
        """
        all_documents: List[Document] = []
        pdf_files = list(self.directory_path.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in directory: {self.directory_path}")
            return []
            
        logger.info(f"Found {len(pdf_files)} PDF file(s) in {self.directory_path.name}")
        
        for pdf_path in pdf_files:
            try:
                loader = PDFDocumentLoader(pdf_path)
                docs = loader.load()
                all_documents.extend(docs)
            except Exception as e:
                logger.error(f"Failed to load document '{pdf_path.name}': {str(e)}")
                if fail_fast:
                    raise
                    
        logger.info(f"Ingestion complete. Total pages loaded: {len(all_documents)}")
        return all_documents
