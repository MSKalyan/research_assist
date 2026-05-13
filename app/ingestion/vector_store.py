from typing import List, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import CHROMA_DIR, DEFAULT_COLLECTION

EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_embeddings():
    return EMBEDDINGS

def get_vector_store(
    collection_name: str = DEFAULT_COLLECTION,
) -> Chroma:
    """
    Create or load a persistent Chroma vector store.

    Parameters
    ----------
    collection_name : str
        Name of the Chroma collection.

    Returns
    -------
    Chroma
        Persistent vector store instance.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def add_documents(
    documents: List[Document],
    collection_name: str = DEFAULT_COLLECTION,
) -> dict:
    """
    Add document chunks to ChromaDB.

    Uses `source_id` as the unique ID so re-ingesting the same
    chunk can overwrite/update rather than create duplicates.

    Returns a summary dictionary.
    """
    if not documents:
        return {
            "status": "skipped",
            "message": "No documents provided.",
            "chunks_added": 0,
        }

    vector_store = get_vector_store(collection_name)

    ids = [
        doc.metadata.get("source_id")
        for doc in documents
    ]
    if any(doc_id is None for doc_id in ids):
        raise ValueError(
            "Each document must contain 'source_id' in metadata."
        )

    vector_store.add_documents(
        documents=documents,
        ids=ids,
    )

    return {
        "status": "success",
        "collection_name": collection_name,
        "chunks_added": len(documents),
    }


def similarity_search(
    query: str,
    k: int = 5,
    collection_name: str = DEFAULT_COLLECTION,
    filter: Optional[dict] = None,
) -> List[Document]:
    """
    Perform semantic similarity search.

    Parameters
    ----------
    query : str
        User question.
    k : int
        Number of chunks to retrieve.
    filter : dict | None
        Optional metadata filter, e.g.
        {"filename": "paper.pdf"}

    Returns
    -------
    List[Document]
    """
    vector_store = get_vector_store(collection_name)

    return vector_store.similarity_search(
        query=query,
        k=k,
        filter=filter,
    )


def similarity_search_with_scores(
    query: str,
    k: int = 5,
    collection_name: str = DEFAULT_COLLECTION,
    filter: Optional[dict] = None,
):
    """
    Return (Document, score) pairs.
    Lower score generally means closer match depending on metric.
    """
    vector_store = get_vector_store(collection_name)

    return vector_store.similarity_search_with_score(
        query=query,
        k=k,
        filter=filter,
    )


def as_retriever(
    k: int = 5,
    collection_name: str = DEFAULT_COLLECTION,
):
    """
    Return a LangChain retriever object.
    Useful for LangGraph nodes and RAG chains.
    """
    vector_store = get_vector_store(collection_name)

    return vector_store.as_retriever(
        search_kwargs={"k": k}
    )


def list_collection_stats(
    collection_name: str = DEFAULT_COLLECTION,
) -> dict:
    """
    Return collection statistics.
    """
    vector_store = get_vector_store(collection_name)

    count = vector_store._collection.count()

    return {
        "collection_name": collection_name,
        "total_chunks": count,
        "persist_directory": str(CHROMA_DIR),
    }


def list_available_documents(
    collection_name: str = DEFAULT_COLLECTION,
) -> List[dict]:
    """
    Return list of unique documents with their metadata.
    """
    vector_store = get_vector_store(collection_name)
    
    all_docs = vector_store.get()
    
    unique_docs = {}
    for idx, doc_id in enumerate(all_docs.get("ids", [])):
        metadata = all_docs.get("metadatas", [{}])[idx]
        filename = metadata.get("filename", "unknown")
        if filename not in unique_docs:
            unique_docs[filename] = {
                "filename": filename,
                "document_id": metadata.get("document_id"),
                "total_chunks": 0,
            }
        unique_docs[filename]["total_chunks"] += 1
    
    return list(unique_docs.values())


def search_documents(
    query: str,
    k: int = 5,
    filenames: Optional[List[str]] = None,
    collection_name: str = DEFAULT_COLLECTION,
) -> List[Document]:
    """
    Search with optional filename filter.
    """
    vector_store = get_vector_store(collection_name)
    
    filter_dict = None
    if filenames:
        filter_dict = {"filename": {"$in": filenames}}
    
    return vector_store.similarity_search(
        query=query,
        k=k,
        filter=filter_dict,
    )


def delete_collection(
    collection_name: str = DEFAULT_COLLECTION,
) -> dict:
    """
    Delete the entire collection.
    Useful for resetting during development.
    """
    vector_store = get_vector_store(collection_name)
    vector_store.delete_collection()

    return {
        "status": "success",
        "message": f"Collection '{collection_name}' deleted.",
    }


if __name__ == "__main__":
    import sys
    from app.ingestion.loader import load_files
    from app.ingestion.splitter import split_documents
    from app.config import UPLOAD_DIR

    if len(sys.argv) > 1:
        file_paths = [str(p) for p in sys.argv[1:]]
    else:
        file_paths = list(UPLOAD_DIR.glob("*.*"))
        file_paths = [str(p) for p in file_paths if p.suffix.lower() in [".pdf", ".txt", ".md"]]

    if not file_paths:
        print(f"No files found in {UPLOAD_DIR}")
        print("Usage: python -m app.ingestion.vector_store [file1.pdf file2.txt ...]")
        sys.exit(1)

    print(f"Loading files: {file_paths}")
    docs = load_files(file_paths)
    chunks = split_documents(docs)
    result = add_documents(chunks)
    print(result)
    print(list_collection_stats())
    results = similarity_search("What is the main topic?", k=3)

    print("\nTop Results:")
    for i, doc in enumerate(results, start=1):
        print(f"\nResult {i}")
        print("Metadata:", doc.metadata)
        print("Content:", doc.page_content[:300])