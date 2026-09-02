"""Run retrieval and RAG evaluation questions against the HR policy PDF.

This script records actual measured results. It intentionally does not invent
expected pages or pass/fail outcomes when the PDF content has not been reviewed.
"""

from dataclasses import dataclass

from src.config import config
from src.document_loader import load_hr_policy_pdf
from src.embeddings import create_embeddings
from src.rag_pipeline import answer_question, create_llm
from src.text_splitter import split_documents
from src.vector_store import get_or_create_faiss_index


@dataclass(frozen=True)
class EvaluationQuestion:
    question: str
    expected_page: int | None = None


QUESTIONS = [
    EvaluationQuestion("What is the attendance policy?"),
    EvaluationQuestion("What are the normal office timings?"),
    EvaluationQuestion("At what time are employees expected to reach the office?"),
    EvaluationQuestion("What is the leave policy?"),
    EvaluationQuestion("How can an employee apply for leave?"),
    EvaluationQuestion("What is the reimbursement policy?"),
    EvaluationQuestion("When does an employee become eligible for appraisal?"),
    EvaluationQuestion("What are the rules for salary deductions?"),
    EvaluationQuestion("What benefits are available to employees?"),
    EvaluationQuestion("What is the policy for working from home?"),
    EvaluationQuestion("Does the company provide employees with a free car?"),
    EvaluationQuestion("Can employees bring pets to the office?"),
]


def main() -> None:
    documents = load_hr_policy_pdf(config.pdf_path)
    chunks = split_documents(documents, config.chunk_size, config.chunk_overlap)
    embeddings = create_embeddings(config.embedding_model, config.ollama_base_url)
    vector_store = get_or_create_faiss_index(chunks, embeddings, config.index_path)
    llm = create_llm(
        config.local_llm,
        config.ollama_base_url,
        config.ollama_num_ctx,
        config.ollama_num_predict,
    )

    for item in QUESTIONS:
        result = answer_question(item.question, vector_store, llm, config.top_k)
        retrieved_pages = result.source_pages
        correct_top_k = (
            "Not configured"
            if item.expected_page is None
            else "Yes" if item.expected_page in retrieved_pages else "No"
        )

        print("\n========================================")
        print(f"Question: {item.question}")
        print(f"Expected Page: {item.expected_page or 'Not configured'}")
        print(f"Retrieved Pages: {retrieved_pages}")
        print(f"Correct Chunk in Top-{config.top_k}: {correct_top_k}")
        print("Answer Relevant: Review manually")
        print("Hallucination: Review manually")
        print(f"Retrieval Time: {result.retrieval_time:.2f} sec")
        print(f"Generation Time: {result.generation_time:.2f} sec")
        print(f"Total Response Time: {result.total_time:.2f} sec")
        print(f"Answer: {result.answer}")


if __name__ == "__main__":
    main()
