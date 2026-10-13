# Pure-Python RAG Pipeline with hit@k Evaluation

A from-scratch Retrieval-Augmented Generation (RAG) pipeline built in pure Python and NumPy, with an empirical hit@k evaluation harness.

---

## Headline Result

> **`hit@1`: `13/17` → `17/17`** (`0.765` → `1.0`)  
> *(Evaluated on a 17-question / 6-chunk test set under default sampling temperature—a proof-of-method validation subject to run-to-run variance, not a deterministic benchmark claim.)*

### Metric: hit@k, not recall@k

For each question, the harness checks whether its labelled answer phrase appears in **any** of the top-k reranked chunks, and reports the fraction of questions that pass. That is **hit@k** (hit rate). It is not recall@k, which would measure the fraction of *all* relevant chunks per question that were retrieved; here each question is labelled with a single answer phrase, so only hit@k is measured.

*Naming note:* the code predates this distinction—the function is `evaluate_recall()` and the returned key is `"recall"`, but the value it computes is hit@k as defined above.

---

## The Reranking Debug Story

The gain from `13/17` to `17/17` was diagnostic, not architectural—the reranker was already in place, but was having no effect because of a flaw in its prompt contract:

1. **Reranking Added**: To push the target context into the `#1` slot, an LLM-based reranking step (`rerank.py`) was introduced between initial cosine retrieval and answer generation.
2. **hit@1 Didn't Move**: After adding reranking, hit@1 stayed at exactly `13/17`—identical to the un-reranked cosine baseline. A working reranker might help, hurt or fluctuate, but an exact match suggested it wasn't changing the order at all.
3. **Read the Fallback Path**: `rerank()` only applies the LLM's ordering if the extracted indices form a complete permutation (`sorted(extracted) == list(range(1, N+1))`); otherwise it silently returns the original cosine order (`candidates[:k]`). An unchanged score was consistent with that fallback firing.
4. **Traced to Prompt Flaw**: The original prompt allowed the LLM to return a filtered subset of the most relevant indices instead of ranking the full list. A subset fails the permutation check, so the code fell back to cosine order and the reranker was effectively inert.
5. **Fixed Prompt Constraints**: Updated the prompt to explicitly require a complete permutation:
   > *"Rank all N candidates... Return every candidate number from 1 to N, each exactly once, reordered from most to least relevant."*
6. **Re-Measured**: Probing `rerank()` directly now returned a reordered candidate list, and hit@1 moved from `13/17` to `17/17`.

**Lesson:** a silent fallback makes a broken component look like one that simply doesn't help. An identical-to-baseline metric is a signal to check whether the stage is executing, not a verdict on the technique.

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
                             Answer    hit@k metric
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
     - Computes dot-product cosine similarity (`cosine.py`) between the query vector and all stored chunk vectors, returning the top-$k$ candidate dicts.
   - *Design*: Decoupling API embedding calls from vector similarity math keeps similarity ranking unit-testable locally without invoking external network requests.

5. **`generate.py`**
   - **Takes**: `question` (`str`), `chunks` (`list[str]`), `client` (`genai.Client`)
   - **Returns**: `answer` (`str`)
   - *Description*: Synthesizes a final answer using `gemini-3.5-flash-lite` under strict system instructions to answer *only* from the supplied context.

### Reranking & Evaluation

6. **`rerank.py`**
   - **Takes**: `query` (`str`), `candidates` (`list[dict]`), `client` (`genai.Client`), `k` (`int`)
   - **Returns**: `reordered_candidates` (`list[dict]`)
   - *Description*: Sends numbered candidate snippets to `gemini-3.5-flash-lite` to rank by relevance, parses the returned index permutation, and returns the top-$k$ reordered items. If the reply is not a complete permutation, it falls back to the original cosine order.

7. **`eval.py`**
   - **Takes**: `labelled_set` (`list[dict]`), `storage` (`list[dict]`), `client` (`genai.Client`), `k` (`int`)
   - **Returns**: `metrics` (`dict` with `"recall"` — the hit@k rate — and `"misses"` list)
   - *Description*: For each labelled question, retrieves 5 candidates by cosine similarity, reranks them, keeps the top-$k$, and counts a hit if the answer phrase appears (case-sensitive substring match) in any of those $k$ chunks.

---

## Why Hand-Built (From-Scratch Rationale)

Frameworks like LangChain or LlamaIndex add layers of indirection between the developer and intermediate state. When a component fails silently—such as a reranker failing index parsing and falling back to default ordering—the failure is swallowed behind framework-specific abstractions and data structures you didn't write and wouldn't think to inspect. Debugging requires tracing deeply nested multi-class call stacks instead of directly inspecting raw data.

By hand-building the pipeline using basic Python data structures (`list`, `dict`) and NumPy math, every intermediate state remains transparent. That visibility is what made the reranker's fallback path easy to find and reason about.

---

## Setup & Execution

### Prerequisites
- Python 3.10+
- Google Gemini API Key

### 1. Environment Setup

```bash
# Clone or navigate to project directory
cd rag-from-scratch

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

Create a `.env` file in the root directory (it is git-ignored—never commit it):

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Run Pipeline & Evaluation

```bash
python main.py
```

*Note: `python main.py` prompts for a question, prints the generated answer, then runs `evaluate_recall()` at `k=1, 2, 3` and prints the hit@k rate for each (`1.0` on the current test set).*

---

## Limitations

- **Small test set**: 17 questions over a 6-chunk corpus. This validates the method and the debugging story; it is not a benchmark.
- **Substring matching**: a question counts as a hit only if its exact answer phrase appears in a retrieved chunk, so paraphrased evidence counts as a miss.
- **Run-to-run variance**: reranking runs at default sampling temperature, so scores can vary between runs.

---

## Future Work & Intentional Exclusions

### Why Hybrid Search (BM25 + Dense) Was Deferred

Hybrid search (combining keyword BM25 scoring with dense vector retrieval) is standard practice for handling exact keyword matches and domain-specific terminology.

However, in this implementation:
- Dense retrieval paired with LLM reranking already achieved **`hit@1 = 1.0`** and **`hit@3 = 1.0`** on the evaluation set.
- Because retrieval was already saturated on this set, there was **no metric gap left for hybrid search to close**.

Engineering effort was intentionally directed toward evaluation and reranker validation rather than adding unnecessary complexity. Hybrid search remains a candidate upgrade once evaluation scales to larger, noisier corpora where dense retrieval alone shows hit-rate degradation.