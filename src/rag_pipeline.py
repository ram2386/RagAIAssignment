"""End-to-end local RAG pipeline."""

from dataclasses import dataclass
from time import perf_counter

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from src.prompts import RAG_PROMPT
from src.vector_store import similarity_search_with_scores


@dataclass(frozen=True)
class RagResult:
    question: str
    answer: str
    retrieved_context: list[tuple[Document, float]]
    retrieval_time: float
    generation_time: float
    total_time: float

    @property
    def source_pages(self) -> list[int]:
        pages = {
            doc.metadata.get("page")
            for doc, _ in self.retrieved_context
            if doc.metadata.get("page") is not None
        }
        return sorted(int(page) for page in pages)


def format_context(retrieved_context: list[tuple[Document, float]]) -> str:
    """Format retrieved chunks for the strict prompt."""
    context_parts = []
    for doc, _score in retrieved_context:
        page = doc.metadata.get("page", "Unknown")
        context_parts.append(f"Page {page}:\n{doc.page_content}")
    return "\n\n---\n\n".join(context_parts)


def create_llm(model: str, base_url: str) -> ChatOllama:
    """Create the local Ollama chat model."""
    return ChatOllama(model=model, base_url=base_url, temperature=0)


def answer_question(
    question: str,
    vector_store: FAISS,
    llm: ChatOllama,
    top_k: int,
) -> RagResult:
    """Retrieve context, ask the local LLM, and measure real timings."""
    total_start = perf_counter()

    retrieval_start = perf_counter()
    retrieved_context = similarity_search_with_scores(vector_store, question, top_k)
    retrieval_time = perf_counter() - retrieval_start

    if not retrieved_context:
        total_time = perf_counter() - total_start
        return RagResult(
            question=question,
            answer="I could not find this information in the available HR Policy document.",
            retrieved_context=[],
            retrieval_time=retrieval_time,
            generation_time=0,
            total_time=total_time,
        )

    generation_start = perf_counter()
    prompt = RAG_PROMPT.format_messages(
        context=format_context(retrieved_context),
        question=question,
    )
    response = llm.invoke(prompt)
    generation_time = perf_counter() - generation_start
    total_time = perf_counter() - total_start

    return RagResult(
        question=question,
        answer=str(response.content).strip(),
        retrieved_context=retrieved_context,
        retrieval_time=retrieval_time,
        generation_time=generation_time,
        total_time=total_time,
    )


def generate_answer(
    question: str,
    retrieved_context: list[tuple[Document, float]],
    llm: ChatOllama,
) -> tuple[str, float]:
    """Generate an answer from already retrieved context."""
    if not retrieved_context:
        return "I could not find this information in the available HR Policy document.", 0

    generation_start = perf_counter()
    prompt = RAG_PROMPT.format_messages(
        context=format_context(retrieved_context),
        question=question,
    )
    response = llm.invoke(prompt)
    generation_time = perf_counter() - generation_start
    return str(response.content).strip(), generation_time


def print_retrieved_context(retrieved_context: list[tuple[Document, float]]) -> None:
    """Print retrieved chunks with page numbers and raw FAISS distances."""
    print("\n========================================")
    print("Retrieved Context")
    print("========================================\n")

    for rank, (doc, score) in enumerate(retrieved_context, start=1):
        print(f"Chunk #{rank}")
        print(f"Page: {doc.metadata.get('page', 'Unknown')}")
        print(f"Score/Distance: {score:.4f}\n")
        print(doc.page_content)
        print("\n----------------------------------------\n")


def print_answer(result: RagResult) -> None:
    """Print final answer and timing details."""
    print("========================================")
    print("Answer")
    print("========================================\n")
    print(result.answer)
    pages = ", ".join(str(page) for page in result.source_pages) or "None"
    print(f"\nSource Pages: {pages}")
    print(f"Retrieval Time     : {result.retrieval_time:.2f} sec")
    print(f"Generation Time    : {result.generation_time:.2f} sec")
    print(f"Total Response Time: {result.total_time:.2f} sec")
