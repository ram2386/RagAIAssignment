# HR Policy RAG CLI

A fully local Python CLI that answers questions from `data/HRPolicy.pdf` using LangChain, FAISS, Ollama embeddings, and an Ollama local LLM. It does not use OpenAI, Gemini, Anthropic, Pinecone, or any paid cloud API.

## Architecture

```text
     HR Policy PDF
          ↓
     PyPDFLoader
          ↓
       Documents
          ↓
RecursiveCharacterTextSplitter
          ↓
        Chunks
          ↓
 Ollama Embeddings
  nomic-embed-text
          ↓
        FAISS
          ↓
 Similarity Search
          ↓
      Retriever
          ↓
  Relevant Context
          ↓
     RAG Prompt
          ↓
   Ollama qwen3:4b
          ↓
        Answer
```

## Components

- RAG: Retrieves policy text first, then asks the LLM to answer from that context only.
- Document Loader: `PyPDFLoader` reads the HR Policy PDF.
- Document: A LangChain object containing page text plus metadata like source and page number.
- Text Splitter: `RecursiveCharacterTextSplitter` breaks long pages into smaller pieces.
- Chunk: A smaller text segment used for embedding and retrieval.
- Chunk Size: Maximum target characters per chunk.
- Chunk Overlap: Shared text between neighboring chunks to preserve context.
- Embedding: Numeric representation of text created locally by Ollama.
- Vector: The stored embedding for a chunk or query.
- Vector Store: A searchable database of vectors.
- FAISS: The local vector index used for similarity search.
- Similarity Search: Finds chunks closest to the question embedding.
- Retriever: LangChain interface for returning top-k relevant chunks.
- Context Retrieval: The selected chunks passed to the prompt.
- LLM: The local model that writes the final answer.
- Ollama: Local runtime for `nomic-embed-text` and `qwen3:4b`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install and start Ollama, then pull the local models:

```bash
ollama pull nomic-embed-text
ollama pull qwen3:4b
```

Place the PDF here:

```text
data/HRPolicy.pdf
```

Optional configuration:

```bash
cp .env.example .env
```

## Run

Build or rebuild the FAISS index:

```bash
python main.py --rebuild-index --show-chunks
```

Start the interactive CLI:

```bash
python main.py
```

Ask a question:

```text
> What is the attendance policy?
```

The CLI prints retrieved chunks, page numbers, raw FAISS score/distance values, the final answer, source pages, retrieval time, generation time, and total response time.

For faster local responses, the app caps Ollama generation with:

```text
OLLAMA_NUM_CTX=4096
OLLAMA_NUM_PREDICT=300
```

This avoids accidentally using Qwen's very large default context window for short HR policy answers.

## Evaluation

Run retrieval and answer evaluation:

```bash
python -m evaluation.retrieval_evaluation
```

The script records actual retrieved pages and timings. Fill `expected_page` values in `evaluation/retrieval_evaluation.py` after reviewing the provided PDF, then rerun to mark whether the correct chunk appears in Top-K.

Run the chunking experiment:

```bash
python -m evaluation.chunking_experiment
```

It compares:

- Configuration A: `300 / 50`
- Configuration B: `500 / 100`
- Configuration C: `800 / 150`

The experiment reports generated chunk counts, retrieved pages, FAISS score/distance values, and whether expected pages appeared in Top-K when expected pages are configured.

## Notes

- The HR Policy PDF is the only HR knowledge source.
- The strict prompt tells the LLM to respond with: `I could not find this information in the available HR Policy document.` when retrieved context does not contain the answer.
- FAISS indexes are saved under `vector_store/` and loaded on later runs to avoid re-embedding the PDF every time.
