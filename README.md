# Pure-Python RAG Pipeline with Recall Evaluation

A from-scratch Retrieval-Augmented Generation (RAG) pipeline built in pure Python and NumPy with empirical recall evaluation.

---

## Headline Result

> **`Recall@1`: `0.765` → `1.0`**  
> *(Evaluated on a 17-question / 6-chunk test set under default sampling temperature—a proof-of-method validation subject to run-to-run variance, not a deterministic benchmark claim).*

---

## The Reranking Debug Story

The gain from `0.765` to `1.0` was diagnostic, not architectural—the reranker was already in place, but running inert due to a silent failure mode in the prompt contract:

1. **Reranking Added**: To push target context into the `#1` candidate slot, an LLM-based reranking step (`rerank.py`) was introduced between initial cosine retrieval and answer generation.
2. **Recall Didn't Move**: After adding reranking, `Recall@1` stayed byte-identical at `0.765`—matching the un-reranked baseline to four decimal places. That precision was the tell: a real reranker might underperform or fluctuate, but an exact match meant the stage was not executing at all.
3. **Probed Intermediate Output**: Treating a byte-identical null result as suspicious rather than accepting that "reranking didn't work," I bypassed the final generated answers to inspect raw `rerank()` list outputs.
4. **Discovered Silent No-Op**: The reranker was silently failing validation on every single request and falling back to the default cosine ordering (`candidates[:k]`), rendering the reranker completely inert.
5. **Traced to Prompt Flaw**: The original prompt permitted the LLM to output a filtered subset of top indices rather than ranking the full list. Because the extracted indices failed the strict permutation check (`sorted(extracted) == list(range(1, N+1))`), the code defaulted back to un-reranked cosine order.
6. **Fixed Prompt Constraints**: Updated the prompt to explicitly enforce a complete permutation:
   > *"Rank all N candidates... Return every candidate number from 1 to N, each exactly once, reordered from most to least relevant."*
7. **Re-Measured**: With the LLM actually reordering candidates, `Recall@1` immediately moved from `0.765` to `1.0`.

---

## Architecture

The pipeline consists of five spine components, plus reranking and evaluation:

```
Source Doc ──► chunker ──► embed_chunks ──► store ──┐
                                                    │
Question   ──► embed_query ──► retrieve_chunks ◄────┘
                                     │
                                     ▼
                                  rerank
                                ┌────┴────┐
                                ▼         ▼
                            generate    eval
                                │         │
                                ▼         ▼
                             Answer    Recall@k Metric
```

### Spine Components

1. **`chunker.py`**
   - **Takes**: `doc` (`str`), `chunk_size` (`int`), `overlap` (`int`)
   - **Returns**: `chunks` (`list[str]`)
   - *Description*: Splits raw document text into overlapping character windows using a sliding stride (`chunk_size - overlap`).

2. **`embed_chunks.py`**
   - **Takes**: `chunks` (`list[str]`), `client` (`genai.Client`)
   - **Returns**: `vectors` (`list[list[float]]`)
   - *Description*: Generates dense vector representations using `gemini-embedding-001` with `task_type='RETRIEVAL_DOCUMENT'`.

3. **`store.py`**
   - **Takes**: `chunks` (`list[str]`), `vectors` (`list[list[float]]`)
   - **Returns**: `storage` (`list[dict]`)
   - *Description*: Pairs chunks with their corresponding embedding vectors into an in-memory storage list (`[{"text": ..., "vector": ...}]`).

4. **`retrieve.py`**
   - **`embed_query(query: str, client: genai.Client) -> list[float]`**
     - Generates the query embedding vector using `gemini-embedding-001` with `task_type='RETRIEVAL_QUERY'`.
   - **`retrieve_chunks(query_vector: list[float], storage: list[dict], k: int) -> list[dict]`**
     - Computes dot-product cosine similarity (`cosine.py`) between the query vector and all stored chunk vectors, returning top-$k$ candidate dicts.
   - *Design*: Decoupling API embedding calls from vector similarity math keeps similarity ranking unit-testable locally without invoking external network requests.

5. **`generate.py`**
   - **Takes**: `question` (`str`), `chunks` (`list[str]`), `client` (`genai.Client`)
   - **Returns**: `answer` (`str`)
   - *Description*: Synthesizes a final answer using `gemini-3.5-flash-lite` under strict system instructions to answer *only* from the supplied context.

### Reranking & Evaluation

6. **`rerank.py`**
   - **Takes**: `query` (`str`), `candidates` (`list[dict]`), `client` (`genai.Client`), `k` (`int`)
   - **Returns**: `reordered_candidates` (`list[dict]`)
   - *Description*: Sends numbered candidate snippets to `gemini-3.5-flash-lite` to rank by relevance, parses the returned index permutation, and returns the top-$k$ reordered items.

7. **`eval.py`**
   - **Takes**: `labelled_set` (`list[dict]`), `storage` (`list[dict]`), `client` (`genai.Client`), `k` (`int`)
   - **Returns**: `metrics` (`dict` with `"recall"` float and `"misses"` list)
   - *Description*: Iterates through ground-truth question/answer-phrase pairs, running retrieval and reranking for each question to evaluate whether the required phrase exists in the reranked top-$k$ context.

---

## Why Hand-Built (From-Scratch Rationale)

Frameworks like LangChain or LlamaIndex add layers of indirection between the developer and intermediate state. When a component fails silently—such as a reranker failing index parsing and falling back to default ordering—the failure is swallowed behind framework-specific abstractions and data structures you didn't write and wouldn't think to inspect. Debugging requires tracing deeply nested multi-class call stacks instead of directly inspecting raw data.

By hand-building the pipeline using basic Python data structures (`list`, `dict`) and NumPy math, every intermediate state remains transparent. This level of visibility was the precise prerequisite for probing the reranking layer and catching the silent fallback failure.

---

## Setup & Execution

### Prerequisites
- Python 3.10+
- Google Gemini API Key

### 1. Environment Setup

```bash
# Clone or navigate to project directory
cd RAG

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install google-genai numpy python-dotenv
```

### 2. Configure API Key

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Run Pipeline & Evaluation

```bash
python main.py
```

*Note: Running `python main.py` executes an interactive question prompt, outputs the generated answer, and then executes `evaluate_recall()` across `k=1, 2, 3`, printing the resulting recall metrics (`1.0`) directly to stdout.*

---

## Future Work & Intentional Exclusions

### Why Hybrid Search (BM25 + Dense) Was Deferred

Hybrid search (combining keyword BM25 scoring with dense vector retrieval) is standard practice for handling exact keyword matches and domain-specific terminology.

However, in this implementation:
- Dense retrieval paired with LLM reranking already achieved **`Recall@1 = 1.0`** and **`Recall@3 = 1.0`** on the evaluation dataset.
- Because retrieval was already fully saturated, there was **no metric gap left for hybrid search to close**.

Engineering effort was intentionally directed toward evaluation and reranker validation rather than adding unnecessary complexity. Hybrid search remains a candidate upgrade once evaluation scales to larger, noisier corpora where dense retrieval alone exhibits recall degradation.
