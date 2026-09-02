"""PDF loading utilities."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


class DocumentLoadError(RuntimeError):
    """Raised when the HR policy PDF cannot be loaded."""


def load_hr_policy_pdf(pdf_path: Path) -> list[Document]:
    """Load the HR policy PDF and preserve source/page metadata."""
    if not pdf_path.exists():
        raise DocumentLoadError(
            f"HR Policy PDF was not found at {pdf_path}.\n"
            "Place the file at data/HRPolicy.pdf and try again."
        )

    try:
        documents = PyPDFLoader(str(pdf_path)).load()
    except Exception as exc:
        raise DocumentLoadError(f"Could not load HR Policy PDF: {exc}") from exc

    filename = pdf_path.name
    for document in documents:
        document.metadata["filename"] = filename
        document.metadata["source"] = str(pdf_path)
        if "page" in document.metadata:
            document.metadata["page"] = int(document.metadata["page"]) + 1

    print("HR Policy PDF loaded successfully.")
    print(f"Total Pages: {len(documents)}")
    return documents

