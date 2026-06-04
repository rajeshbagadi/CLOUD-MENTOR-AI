import sys
from pathlib import Path

# Add workspace root to Python path
sys.path.append(str(Path(__file__).parent))

from src.ingestion.document_loader import DirectoryPDFLoader
from src.ingestion.text_splitter import DocumentChunker
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

    print("Scanning 'docs/' folder for PDF files to load and chunk...")
    try:
        # Load PDFs
        loader = DirectoryPDFLoader(docs_dir)
        documents = loader.load_all()
        
        if not documents:
            print("No PDF files were found. Add PDFs to the 'docs/' directory.")
            return

        # Chunk documents
        chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
        chunks = chunker.split_documents(documents)
        
        print(f"\nIngestion & Chunking successful.")
        print(f"Total original pages loaded: {len(documents)}")
        print(f"Total resulting chunks generated: {len(chunks)}")
        
        # Display preview of first few chunks
        preview_limit = min(3, len(chunks))
        print(f"\n--- Previewing first {preview_limit} chunk(s) ---")
        for idx in range(preview_limit):
            chunk = chunks[idx]
            print(f"\n[Chunk Record {idx + 1}]")
            print(f"  Source File   : {chunk.metadata.get('source')}")
            print(f"  Page Number   : {chunk.metadata.get('page')}")
            print(f"  Chunk Index   : {chunk.metadata.get('chunk_index')}")
            print(f"  Character Size: {len(chunk.page_content)}")
            print(f"  Snippet       : {repr(chunk.page_content[:120])}...")
            
    except CloudMentorError as e:
        print(f"\n[Ingestion Exception Raised]: {e}")
    except Exception as e:
        print(f"\n[Unexpected Error Occurred]: {e}")


if __name__ == "__main__":
    main()
