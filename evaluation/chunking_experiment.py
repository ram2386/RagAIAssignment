"""Compare retrieval behavior across chunk size and overlap settings."""

from pathlib import Path

from evaluation.retrieval_evaluation import QUESTIONS
from src.config import BASE_DIR, config
from src.document_loader import load_hr_policy_pdf
from src.embeddings import create_embeddings
from src.text_splitter import split_documents
from src.vector_store import build_faiss_index, similarity_search_with_scores


CONFIGURATIONS = [
    ("A", 300, 50),
    ("B", 500, 100),
    ("C", 800, 150),
]


def main() -> None:
    documents = load_hr_policy_pdf(config.pdf_path)
    embeddings = create_embeddings(config.embedding_model, config.ollama_base_url)

    for label, chunk_size, chunk_overlap in CONFIGURATIONS:
        print("\n========================================")
        print(f"Configuration {label}")
        print(f"Chunk Size: {chunk_size}")
        print(f"Overlap: {chunk_overlap}")

        chunks = split_documents(documents, chunk_size, chunk_overlap)
        index_path = Path(BASE_DIR / "vector_store" / f"faiss_index_{label.lower()}")
        vector_store = build_faiss_index(chunks, embeddings, index_path)

        print(f"Generated Chunks: {len(chunks)}")
        for item in QUESTIONS:
            results = similarity_search_with_scores(vector_store, item.question, config.top_k)
            pages = sorted(
                {
                    int(doc.metadata["page"])
                    for doc, _score in results
                    if doc.metadata.get("page") is not None
                }
            )
            scores = [round(float(score), 4) for _doc, score in results]
            correct_top_k = (
                "Not configured"
                if item.expected_page is None
                else "Yes" if item.expected_page in pages else "No"
            )

            print("\nQuestion:", item.question)
            print("Retrieved Pages:", pages)
            print("Score/Distance:", scores)
            print(f"Correct Chunk in Top-{config.top_k}:", correct_top_k)
            print("Answer Quality: Run main evaluation for generated answers")


if __name__ == "__main__":
    main()

