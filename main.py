"""Interactive CLI for the local HR Policy RAG assistant."""

from argparse import ArgumentParser
from time import perf_counter

from src.cli_feedback import Spinner
from src.config import config
from src.document_loader import DocumentLoadError, load_hr_policy_pdf
from src.embeddings import create_embeddings
from src.rag_pipeline import RagResult, create_llm, generate_answer, print_answer, print_retrieved_context
from src.text_splitter import display_sample_chunks, split_documents
from src.vector_store import VectorStoreError, get_or_create_faiss_index, similarity_search_with_scores


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Local HR Policy RAG CLI")
    parser.add_argument("--rebuild-index", action="store_true", help="Recreate the FAISS index")
    parser.add_argument("--show-chunks", action="store_true", help="Print sample chunks after splitting")
    parser.add_argument("--search-only", action="store_true", help="Run retrieval without LLM generation")
    return parser


def print_header() -> None:
    print("========================================")
    print("       Innvonix HR Policy Assistant")
    print("========================================\n")
    print("Local RAG System\n")
    print(f"Embedding Model : {config.embedding_model}")
    print(f"LLM             : {config.local_llm}")
    print("Vector Store    : FAISS\n")
    print("Ask your HR Policy question or type 'exit' to close.\n")


def main() -> None:
    args = build_parser().parse_args()

    try:
        documents = load_hr_policy_pdf(config.pdf_path)
        chunks = split_documents(documents, config.chunk_size, config.chunk_overlap)
        if args.show_chunks:
            display_sample_chunks(chunks)

        embeddings = create_embeddings(config.embedding_model, config.ollama_base_url)
        vector_store = get_or_create_faiss_index(
            chunks,
            embeddings,
            config.index_path,
            rebuild=args.rebuild_index,
        )
        llm = None if args.search_only else create_llm(config.local_llm, config.ollama_base_url)
    except (DocumentLoadError, VectorStoreError) as exc:
        print(f"\nError: {exc}")
        return
    except Exception as exc:
        print("\nError: Could not initialize the local RAG system.")
        print("Please confirm Ollama is running and both models are installed:")
        print("  ollama pull nomic-embed-text")
        print("  ollama pull qwen3:4b")
        print(f"\nDetails: {exc}")
        return

    print_header()
    while True:
        question = input("> ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not question:
            continue

        try:
            total_start = perf_counter()
            with Spinner("Retrieving relevant HR policy chunks"):
                retrieval_start = perf_counter()
                retrieved_context = similarity_search_with_scores(vector_store, question, config.top_k)
                retrieval_time = perf_counter() - retrieval_start

            print_retrieved_context(retrieved_context)

            if args.search_only:
                continue

            if llm is None:
                print("\nError: Local LLM was not initialized.\n")
                continue

            with Spinner("Generating answer with local Ollama model"):
                answer, generation_time = generate_answer(question, retrieved_context, llm)

            result = RagResult(
                question=question,
                answer=answer,
                retrieved_context=retrieved_context,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                total_time=perf_counter() - total_start,
            )
            print_answer(result)
            print()
        except Exception as exc:
            print(f"\nError: LLM generation or retrieval failed: {exc}\n")


if __name__ == "__main__":
    main()
