from pathlib import Path
from typing import List
from uuid import uuid4
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def _load_pdf(file_path: Path, document_id: str) -> List[Document]:
    """
    Load a PDF file and attach standard metadata.

    Metadata added:
    - document_id
    - filename
    - file_path
    - file_type
    - page (already provided by PyPDFLoader, zero-based)
    """
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()

    for doc in docs:
        doc.metadata.update(
            {
                "document_id": document_id,
                "filename": file_path.name,
                "file_path": str(file_path),
                "file_type": "pdf",
            }
        )

    return docs


def _load_text(file_path: Path, document_id: str) -> List[Document]:
    """
    Load a text/markdown file as a single document.

    Metadata added:
    - document_id
    - filename
    - file_path
    - file_type
    - page = 0 (for consistency with PDFs)
    """
    loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()

    for doc in docs:
        doc.metadata.update(
            {
                "document_id": document_id,
                "filename": file_path.name,
                "file_path": str(file_path),
                "file_type": file_path.suffix.lstrip("."),
                "page": 0,
            }
        )

    return docs


def load_file(file_path: str | Path) -> List[Document]:
    """
    Load a single file and return LangChain Document objects.

    Supported formats:
    - PDF
    - TXT
    - MD
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    document_id = str(uuid4())

    if extension == ".pdf":
        return _load_pdf(path, document_id)

    return _load_text(path, document_id)


def load_files(file_paths: List[str | Path]) -> List[Document]:
    """
    Load multiple files and return a combined list of Documents.

    Each file gets its own unique document_id.
    """
    all_docs: List[Document] = []
    for file_path in file_paths:
        all_docs.extend(load_file(file_path))
    return all_docs

if __name__ == "__main__":
    import sys
    from app.config import UPLOAD_DIR

    if len(sys.argv) > 1:
        file_paths = sys.argv[1:]
    else:
        file_paths = [str(p) for p in UPLOAD_DIR.glob("*.*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not file_paths:
        print(f"No files found in {UPLOAD_DIR}")
        print("Usage: python -m app.ingestion.loader [file1.pdf file2.txt ...]")
        sys.exit(1)

    print(f"Loading: {file_paths}")
    docs = load_files(file_paths)
    print(f"Loaded {len(docs)} documents/pages")
    if docs:
        print("Sample metadata:")
        print(docs[0].metadata)
        print("Preview:")
        print(docs[0].page_content[:500])