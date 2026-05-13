from pathlib import Path
from app.ingestion.loader import load_files
from app.ingestion.splitter import split_documents
from app.ingestion.vector_store import add_documents, list_collection_stats
from app.config import UPLOAD_DIR
def main():
    print("=== Document Ingestion Pipeline ===\n")
    
    file_paths = [str(p) for p in UPLOAD_DIR.glob("*.*") if p.suffix.lower() in [".pdf", ".txt", ".md"]]
    
    if not file_paths:
        print(f"No files found in {UPLOAD_DIR}")
        print("Supported formats: .pdf, .txt, .md")
        return
    
    print(f"Found {len(file_paths)} files:")
    for f in file_paths:
        print(f"  - {Path(f).name}")
    
    print("\nLoading documents...")
    docs = load_files(file_paths)
    print(f"Loaded {len(docs)} documents/pages")
    
    print("\nSplitting into chunks...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks")
    
    print("\nAdding to vector store...")
    result = add_documents(chunks)
    print(f"Added {result['chunks_added']} chunks to collection '{result['collection_name']}'")
    
    stats = list_collection_stats()
    print(f"\nVector store now has {stats['total_chunks']} total chunks")
    print(f"Location: {stats['persist_directory']}")
if __name__ == "__main__":
    main()