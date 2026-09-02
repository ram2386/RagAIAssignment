"""Text splitting and chunk display helpers."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split documents into overlapping chunks while preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk"] = index

    print(f"Total Pages: {len(documents)}")
    print(f"Total Chunks: {len(chunks)}")
    return chunks


def display_sample_chunks(chunks: list[Document], limit: int = 2) -> None:
    """Print a small sample of generated chunks."""
    for index, chunk in enumerate(chunks[:limit], start=1):
        page = chunk.metadata.get("page", "Unknown")
        print(f"\nChunk #{index}")
        print(f"Page: {page}\n")
        print(chunk.page_content)
        print("\n-------------------------------")

