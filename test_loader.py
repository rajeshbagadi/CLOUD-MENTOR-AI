import sys
import os
from pathlib import Path

# Add workspace root folder to Python path
sys.path.append(str(Path(__file__).parent))

from src.ingestion.document_loader import DirectoryPDFLoader
from src.core.exceptions import CloudMentorError


def main():
    docs_dir = Path("docs")
    if not docs_dir.exists():
        docs_dir.mkdir(parents=True)
        print(
            "Created empty 'docs/' directory. "
            "Please place some PDF files in it and run this script again."
        )
        return

    print("Scanning 'docs/' folder for PDF documents...")
    try:
        loader = DirectoryPDFLoader(docs_dir)
        documents = loader.load_all(fail_fast=False)
        
        if not documents:
            print("No PDF files were processed. Add PDFs to the 'docs/' directory.")
            return

        print(f"\nIngestion successful. Loaded {len(documents)} pages in total.")
        
        # Show configuration details of the first few loaded pages
        preview_limit = min(3, len(documents))
        print(f"\n--- Previewing first {preview_limit} page(s) ---")
        for idx in range(preview_limit):
            doc = documents[idx]
            print(f"\n[Page Record {idx + 1}]")
            print(f"  Source File   : {doc.metadata.get('source')}")
            print(f"  Page Number   : {doc.metadata.get('page')} of {doc.metadata.get('total_pages')}")
            print(f"  Absolute Path : {doc.metadata.get('file_path')}")
            print(f"  Snippet       : {repr(doc.page_content[:120])}...")
            
    except CloudMentorError as e:
        print(f"\n[Ingestion Exception Raised]: {e}")
    except Exception as e:
        print(f"\n[Unexpected Error Occurred]: {e}")


if __name__ == "__main__":
    main()
