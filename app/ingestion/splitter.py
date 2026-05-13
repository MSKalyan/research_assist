# app/ingestion/splitter.py

from typing import List
from uuid import uuid4

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Recommended production defaults
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def get_text_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """
    Create and return a configured text splitter.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )


def split_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split input documents into chunks and enrich metadata.

    Added metadata:
    - chunk_id: globally unique identifier
    - chunk_index: index within the source document
    - source_id: stable identifier for document + chunk number

    Preserves existing metadata such as:
    - document_id
    - filename
    - page
    - file_type
    """
    splitter = get_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks: List[Document] = []

    for doc in documents:
        chunks = splitter.split_documents([doc])

        for chunk_index, chunk in enumerate(chunks):
            metadata = dict(chunk.metadata)
            chunk_id = str(uuid4())
            metadata.update(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "source_id": chunk_id,
                }
            )

            chunk.metadata = metadata
            all_chunks.append(chunk)

    return all_chunks


if __name__ == "__main__":
    import sys
    from app.ingestion.loader import load_files
    from app.ingestion.loader import SUPPORTED_EXTENSIONS
    from app.config import UPLOAD_DIR

    if len(sys.argv) > 1:
        file_paths = sys.argv[1:]
    else:
        file_paths = [str(p) for p in UPLOAD_DIR.glob("*.*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not file_paths:
        print(f"No files found in {UPLOAD_DIR}")
        print("Usage: python -m app.ingestion.splitter [file1.pdf file2.txt ...]")
        sys.exit(1)

    print(f"Loading: {file_paths}")
    docs = load_files(file_paths)
    chunks = split_documents(docs)

    print(f"Original documents/pages: {len(docs)}")
    print(f"Generated chunks: {len(chunks)}")

    if chunks:
        print("\nSample metadata:")
        print(chunks[0].metadata)

        print("\nChunk preview:")
        print(chunks[0].page_content[:500])