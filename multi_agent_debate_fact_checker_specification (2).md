# Project Specification: Multi-Agent "Debate & Consensus" Fact-Checker with Hybrid Retrieval

## 1. Executive Summary & Problem Statement

### 1.1 Problem Statement
Single-pass Large Language Model (LLM) fact-checkers and standard Retrieval-Augmented Generation (RAG) pipelines suffer from three well-documented failure modes:
1. **Sycophancy & Confirmation Bias:** A single LLM tends to agree with the framing of a user's claim rather than critically stress-testing it.
2. **Shallow Evidence Synthesis:** Standard RAG retrieves top-$k$ chunks and synthesizes them in one pass, often missing subtle contradictions or failing to distinguish between *refuted* claims and *insufficient evidence*.
3. **Opaque Verdicts:** End users receive a black-box conclusion without an auditable trail of competing arguments or chunk-level citations.

### 1.2 Proposed Solution
Build a **Multi-Agent Debate & Consensus Fact-Checking System** grounded in a **Hybrid Retrieval + Cross-Encoder Reranking** pipeline. Given a natural-language claim:
1. A hybrid retriever fetches and reranks the most relevant evidence chunks from a curated scientific/factual corpus.
2. Two adversarial agents—a **Proponent Agent** and a **Skeptic Agent**—conduct a structured, multi-round debate where every argument must explicitly cite retrieved `chunk_id`s.
3. A **Judge Agent** evaluates the full debate transcript alongside the raw evidence, flags logical fallacies or unsupported extrapolations, and outputs a strictly structured verdict (`SUPPORTED`, `REFUTED`, or `NOT_ENOUGH_INFO`) with a calibrated confidence score and cited evidence list.
4. An automated **Evaluation Suite** benchmarks the Multi-Agent Debate system against a **Single-LLM + RAG Baseline** on the exact same dataset and retrieval outputs to quantify accuracy, macro-F1, and citation faithfulness gains.

---

## 2. End-to-End System Architecture & Design Flow

### 2.1 High-Level Data Flow

```text
[User Claim]
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: HYBRID RETRIEVAL & RERANKING                      │
│  ┌──────────────────────┐     ┌──────────────────────────┐  │
│  │ Dense Vector Search  │     │ Sparse Keyword Search    │  │
│  │ (ChromaDB + BGE)     │     │ (BM25Okapi)              │  │
│  └──────────┬───────────┘     └────────────┬─────────────┘  │
│             └──────────────┬───────────────┘                │
│                            ▼                                │
│             Reciprocal Rank Fusion (RRF, k=60)              │
│                     Top-15 Candidates                       │
│                            │                                │
│                            ▼                                │
│           Cross-Encoder Reranker (MiniLM-L6-v2)             │
│                 Top-5 Evidence Chunks                       │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: MULTI-AGENT DEBATE GRAPH (LangGraph)              │
│                                                             │
│  Round 1..N (default N=2):                                  │
│  ┌─────────────────────┐       ┌─────────────────────────┐  │
│  │   Proponent Agent   │──────▶│      Skeptic Agent      │  │
│  │ (Argues FOR claim,  │       │ (Argues AGAINST claim & │  │
│  │  cites chunk_ids)   │◀──────│  rebuts Proponent)      │  │
│  └─────────────────────┘       └────────────┬────────────┘  │
│        ▲        (Loop until round == N)     │               │
│        └────────────────────────────────────┘               │
│                                             │ (round == N)  │
│                                             ▼               │
│                                ┌─────────────────────────┐  │
│                                │       Judge Agent       │  │
│                                │ (Audits transcript +    │  │
│                                │  raw evidence -> JSON)  │  │
│                                └────────────┬────────────┘  │
└─────────────────────────────────────────────┼───────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: OUTPUT, CITATION VERIFICATION & UI                │
│  • Programmatic citation check (all cited IDs in Top-5)     │
│  • Structured Verdict Card + Side-by-Side Debate Transcript │
│  • Automated Baseline vs. Debate Benchmark Dashboard        │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Datasets & Knowledge Base Specification

### 3.1 Primary Dataset: SciFact (Direct Official Release)
* **Source:** Official AllenAI SciFact release tarball (`https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz`, ~3.1 MB, open under CC BY-NC 2.0).
* **Why Direct Download:** Avoids Hugging Face `datasets` legacy Python loading-script errors (`allenai/scifact` uses a deprecated `scifact.py` builder script). Download and extract directly in `src/ingestion.py` using Python's built-in `urllib.request` and `tarfile`, then parse the JSONL files natively with `json.loads()`.
* **Extracted Files & Exact JSONL Schemas:**
  1. `data/corpus.jsonl` (~5,183 scientific abstracts):
     ```json
     {
       "doc_id": 4983,
       "title": "Microstructural development of human newborn cerebral white matter...",
       "abstract": ["Sentence 1...", "Sentence 2..."],
       "structured": false
     }
     ```
  2. `data/claims_train.jsonl` & `data/claims_dev.jsonl`:
     ```json
     {
       "id": 1,
       "claim": "0-dimensional biomaterials show inductive properties.",
       "evidence": {
         "31715818": [{"sentences": [2, 5], "label": "SUPPORT"}]
       },
       "cited_doc_ids": [31715818]
     }
     ```
     *Note:* `evidence` is a dictionary keyed by stringified `doc_id`. An empty dict (`{}`) indicates no supporting or refuting evidence in the corpus (`NOT_ENOUGH_INFO`).

### 3.2 Label Normalization
Normalize all ground-truth labels from `evidence` into a strict 3-class enum:
* If `evidence` is empty (`{}`): `NOT_ENOUGH_INFO`
* Otherwise, inspect the `label` fields across `evidence.values()`:
  * `"SUPPORT"` -> `SUPPORTED`
  * `"CONTRADICT"` -> `REFUTED`
  * *(Skip any rare edge-case claim containing mixed `SUPPORT` and `CONTRADICT` labels across different docs during evaluation sampling.)*

### 3.3 Evaluation Split Construction
* Construct a **balanced evaluation set of 60 claims** (20 `SUPPORTED`, 20 `REFUTED`, 20 `NOT_ENOUGH_INFO`) sampled deterministically (`seed=42`) from `claims_dev.jsonl` (supplementing from `claims_train.jsonl` only if any class has fewer than 20 dev examples).
* Index the full `corpus.jsonl` (~5,183 abstracts) so all gold `doc_id`s and realistic distractor abstracts are present in the retrieval store.

---

## 4. Models & Technology Stack

All models and libraries are **free, open-access, active, and reproducible**.

| Component | Recommended Default | Alternatives / Notes |
| :--- | :--- | :--- |
| **Embedding Model** | `BAAI/bge-base-en-v1.5` (768-dim) | `BAAI/bge-small-en-v1.5` (384-dim, lighter) |
| **Sparse Retriever** | `rank_bm25` (`BM25Okapi`) | Pure Python, fast tokenization |
| **Cross-Encoder Reranker** | `cross-encoder/ms-marco-MiniLM-L6-v2` | `BAAI/bge-reranker-base` |
| **Vector Database** | `ChromaDB` (persistent local directory) | `FAISS` (`faiss-cpu`) |
| **Primary LLM Backend** | **Gemini API:** `gemini-3.8-flash` (Judge) & `gemini-3.5-flash-lite` (Debaters) | Optional fallback: **Groq API** (verify active model IDs at `console.groq.com/docs/models` before configuring) |
| **Agent Orchestration** | `LangGraph` + `LangChain` (`langchain-google-genai`, optional `langchain-groq`) | Pydantic v2 for structured outputs |
| **Web Interface** | `Gradio` (`>=4.40`) | `Streamlit` |
| **Evaluation & Plotting** | `scikit-learn`, `pandas`, `matplotlib`, `seaborn` | Classification report + confusion matrix |

---

## 5. Module-by-Module Technical Specification

### 5.1 Repository Structure

```text
debate-fact-checker/
├── .env.example              # API keys and configuration variables
├── requirements.txt          # Pinned dependencies
├── config.py                 # Centralized hyperparameters & model IDs
├── run_colab.ipynb           # Lightweight Google Colab launcher notebook
├── data/
│   ├── raw/                  # Extracted SciFact corpus.jsonl & claims_*.jsonl
│   └── eval_set.json         # Deterministic 60-claim evaluation benchmark
├── chroma_db/                # Persisted ChromaDB vector store & BM25 pickle
├── src/
│   ├── __init__.py
│   ├── schemas.py            # Pydantic models & LangGraph AgentState
│   ├── ingestion.py          # Tarball download, JSONL parsing, chunking, indexing
│   ├── retriever.py          # Hybrid (Dense + BM25 + RRF) + Cross-Encoder
│   ├── llm_client.py         # Provider-agnostic LLM wrapper with retry/backoff
│   ├── prompts.py            # System & user prompts for all agents + baseline
│   ├── agents.py             # Node functions: proponent, skeptic, judge
│   ├── graph.py              # LangGraph StateGraph compilation
│   ├── baseline.py           # Single-LLM + RAG baseline for comparison
│   └── evaluate.py           # Batch runner, metrics computation, chart generator
├── results/
│   ├── eval_results.json     # Per-claim predictions, transcripts, latencies
│   └── charts/               # Saved PNG charts (accuracy, F1, confusion matrix)
├── app.py                    # Gradio 2-tab interactive UI
└── README.md                 # Setup, usage, and architecture summary
```

---

### 5.2 Module 1: Data Schemas (`src/schemas.py`)

Define strict Pydantic v2 models and the LangGraph `TypedDict` state:

```python
from typing import List, Literal, Optional, TypedDict
from pydantic import BaseModel, Field

VerdictType = Literal["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]

class EvidenceChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0

class DebateTurn(BaseModel):
    round_num: int
    role: Literal["PROPONENT", "SKEPTIC"]
    argument: str
    cited_chunk_ids: List[str] = Field(default_factory=list)

class JudgeVerdict(BaseModel):
    verdict: VerdictType
    confidence: float = Field(..., ge=0.0, le=1.0)
    cited_chunk_ids: List[str] = Field(default_factory=list)
    proponent_strengths: str
    skeptic_strengths: str
    detected_fallacies: List[str] = Field(default_factory=list)
    reasoning: str

class AgentState(TypedDict):
    claim: str
    evidence: List[dict]               # Serialized EvidenceChunk dicts
    transcript: List[dict]             # Serialized DebateTurn dicts
    current_round: int
    max_rounds: int
    verdict: Optional[dict]            # Serialized JudgeVerdict dict
    citation_validity_rate: float      # Fraction of cited IDs present in retrieved evidence
```

---

### 5.3 Module 2: Ingestion & Indexing (`src/ingestion.py`)

1. **Download & Extract:** Download `https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz` into `data/raw/` (skip if `data/raw/corpus.jsonl` already exists) and extract `corpus.jsonl`, `claims_train.jsonl`, and `claims_dev.jsonl`.
2. **Evaluation Set Generation:** Parse `claims_dev.jsonl` (and `claims_train.jsonl` if needed), normalize labels per Section 3.2, sample 20 claims per class (`seed=42`), and save to `data/eval_set.json` with fields `{"id", "claim", "label", "gold_doc_ids"}`.
3. **Document Formatting:** For each record in `corpus.jsonl`, concatenate `title + "\n" + " ".join(abstract)`.
4. **Chunking Strategy:**
   * Use `RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64, separators=["\n\n", "\n", ". ", " "])`.
   * Assign deterministic IDs: `chunk_id = f"{doc_id}_c{idx}"`.
   * Store metadata: `{"chunk_id": chunk_id, "doc_id": str(doc_id), "title": title}`.
5. **Dual Indexing:**
   * **Dense Index:** Embed all chunks with `BAAI/bge-base-en-v1.5` (`normalize_embeddings=True`) and persist to `chroma_db/`.
   * **Sparse Index:** Tokenize chunks (lowercased alphanumeric tokens) and save a pickled `BM25Okapi` index + chunk lookup map to `chroma_db/bm25_index.pkl`.

---

### 5.4 Module 3: Hybrid Retrieval & Reranking (`src/retriever.py`)

Expose a single clean interface:

```python
def retrieve_evidence(claim: str, top_k: int = 5) -> List[EvidenceChunk]:
    ...
```

**Algorithm:**
1. **Dense Search:** Query ChromaDB with the claim embedding (prefix `"Represent this sentence for searching relevant passages: "` for BGE queries) to get top-20 candidates.
2. **Sparse Search:** Query `BM25Okapi` to get top-20 candidates.
3. **Reciprocal Rank Fusion (RRF):** Merge both ranked lists using:
   $$\text{RRF}(d) = \frac{1}{k + \text{rank}_{\text{dense}}(d)} + \frac{1}{k + \text{rank}_{\text{bm25}}(d)}, \quad k = 60$$
   Keep the top-15 fused candidates.
4. **Cross-Encoder Reranking:** Score all `(claim, chunk.text)` pairs with `cross-encoder/ms-marco-MiniLM-L6-v2`. Sort descending by `rerank_score` and return the top-`top_k` (default `5`) `EvidenceChunk` objects.

---

### 5.5 Module 4: Prompts & Multi-Agent Debate (`src/prompts.py`, `src/agents.py`, `src/graph.py`)

#### Agent Behavioral Rules
* **Strict Grounding:** Every agent receives the exact Top-5 retrieved chunks formatted as `[chunk_id] (Title): text`. Agents are forbidden from using outside knowledge to invent facts.
* **Proponent Agent (`gemini-3.5-flash-lite`):**
  * *Round 1:* Build the strongest possible case that the claim is `SUPPORTED` using only the provided chunks. Cite specific `[chunk_id]` tags inline. If no chunk supports the claim, explicitly state the weakest plausible inference without fabricating evidence.
  * *Round 2+:* Directly rebut the Skeptic's previous turn and reinforce supported links with citations.
* **Skeptic Agent (`gemini-3.5-flash-lite`):**
  * *Round 1+:* Critically attack both the claim and the Proponent's latest argument. Highlight contradictions, scope mismatches (e.g., correlation vs. causation, wrong species/population, missing dosage/condition), or absence of evidence. Cite `[chunk_id]` tags inline.
* **Judge Agent (`gemini-3.8-flash`):**
  * Receives: Original `claim`, raw Top-5 `evidence`, and the complete `transcript`.
  * Task: Ignore rhetorical persuasion; verify whether each cited chunk actually says what the debater claimed. Identify any logical fallacies (e.g., overgeneralization, cherry-picking, unsupported leap).
  * Output: Strict JSON matching `JudgeVerdict` (enforced via `llm.with_structured_output(JudgeVerdict)` with a regex/JSON-repair fallback).
  * **Decision Boundary:**
    * `SUPPORTED`: Evidence directly confirms the claim.
    * `REFUTED`: Evidence directly contradicts the claim.
    * `NOT_ENOUGH_INFO`: Evidence is tangential, inconclusive, or does not address the core assertion.

#### LangGraph Topology (`src/graph.py`)
* **Nodes:** `retrieve` -> `proponent` -> `skeptic` -> `judge` -> `verify_citations`.
* **Conditional Edge after `skeptic`:**
  * If `state["current_round"] < state["max_rounds"]`: increment `current_round` and route back to `proponent`.
  * Else: route to `judge`.
* **`verify_citations` Node (Programmatic Guardrail):**
  * Extract all `cited_chunk_ids` from the Judge's verdict and debaters' turns.
  * Compute `citation_validity_rate = (# of cited IDs present in retrieved top_k) / max(1, total cited IDs)`.
  * Strip any hallucinated IDs from the final displayed citation list.

---

### 5.6 Module 5: Single-LLM Baseline (`src/baseline.py`)

To prove that multi-agent debate adds measurable value over standard RAG:
* Uses the **exact same `retrieve_evidence(claim, top_k=5)` output** and the **exact same LLM (`gemini-3.8-flash`)** as the Judge.
* Single prompt: *"Given the claim and evidence chunks below, classify the claim as SUPPORTED, REFUTED, or NOT_ENOUGH_INFO, provide a confidence score (0–1), cite chunk IDs, and explain your reasoning in JSON."*
* Returns a `JudgeVerdict`-compatible object for apples-to-apples comparison.

---

### 5.7 Module 6: Evaluation & Benchmarking (`src/evaluate.py`)

Run both `baseline` and `debate_graph` on all 60 claims in `data/eval_set.json` (360 total LLM calls: 1 baseline + 5 debate calls per claim at `max_rounds=2`) and save raw outputs to `results/eval_results.json`.

**Metrics to Compute:**
1. **Overall Accuracy** (Baseline vs. Multi-Agent Debate)
2. **Macro-F1 & Per-Class F1** (`SUPPORTED`, `REFUTED`, `NOT_ENOUGH_INFO`)
3. **Citation Validity Rate** (% of cited `chunk_id`s that exist in the retrieved top-5)
4. **Average Latency & Token Usage** per claim (for accuracy-vs-cost tradeoff analysis)
5. **Disagreement Analysis:** Save a JSON list of cases where the Baseline was wrong and Multi-Agent Debate was right (and vice versa) for qualitative inspection.

**Charts to Auto-Generate (`results/charts/`):**
1. `accuracy_f1_comparison.png` — Grouped bar chart comparing Baseline vs. Debate on Accuracy and Macro-F1.
2. `per_class_f1.png` — Grouped bar chart across `SUPPORTED`, `REFUTED`, `NOT_ENOUGH_INFO`.
3. `confusion_matrices.png` — Side-by-side confusion matrices (Baseline vs. Debate).

---

### 5.8 Module 7: Interactive Web UI (`app.py`)

Build a clean **2-tab Gradio interface** that supports both local execution and public sharing (`demo.launch(share=True)` when `--share` CLI flag or `GRADIO_SHARE=true` env var is set):

* **Tab 1: Live Fact-Checker**
  * **Inputs:** Claim text box, dropdown of 5 pre-loaded example claims from `SciFact`, slider for `Debate Rounds` (1–3, default 2), slider for `Top-K Evidence` (3–8, default 5).
  * **Outputs:**
    1. **Verdict Banner:** Color-coded badge (`SUPPORTED` = Green, `REFUTED` = Red, `NOT_ENOUGH_INFO` = Amber) + Confidence bar + Citation Validity score.
    2. **Judge's Reasoning & Fallacies Card:** Concise summary, detected fallacies, and cited `chunk_id`s.
    3. **Side-by-Side Debate Transcript:** Formatted Markdown showing Round 1 & Round 2 Proponent vs. Skeptic arguments.
    4. **Retrieved Evidence Accordion:** Expandable table/list showing the Top-5 chunks with their `chunk_id`, document title, and reranker score.
* **Tab 2: Benchmark Dashboard**
  * Displays the summary metrics table (Baseline vs. Debate) loaded from `results/eval_results.json`.
  * Renders the 3 generated PNG charts from `results/charts/`.
  * Displays a selectable table of "Debate Wins" (claims where Multi-Agent Debate corrected a Baseline error).

---

### 5.9 Module 8: Google Colab Launcher Notebook (`run_colab.ipynb`)

Create a clean, well-documented **5-cell Jupyter/Colab launcher notebook** at the repository root (`run_colab.ipynb`) so the entire modular repository can be executed end-to-end inside Google Colab without duplicating any `src/` code into notebook cells.

**Required Notebook Cells:**
1. **Cell 1 — Environment Setup & Repository Access:**
   * Optionally mount Google Drive (`from google.colab import drive; drive.mount('/content/drive')`) to persist `chroma_db/` and `results/` across sessions.
   * Clone the repository (or `cd` into the uploaded project folder in Drive) and verify the working directory.
2. **Cell 2 — Dependency Installation:**
   * Run `!pip install -q -r requirements.txt` to install all pinned dependencies.
3. **Cell 3 — Secure API Key & Config Loading:**
   * Load `GEMINI_API_KEY` (or optional `GROQ_API_KEY`) securely using Colab Secrets (`from google.colab import userdata; os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")`) with a fallback to `getpass` if the secret is not configured.
   * Set `os.environ["LLM_PROVIDER"] = "gemini"`. Never print secret values in cell output.
4. **Cell 4 — Data Ingestion & Evaluation Pipeline:**
   * Run `!python -m src.ingestion` (skippable if `chroma_db/` already exists).
   * Run `!python -m src.evaluate --limit 10` (smoke test) or full `!python -m src.evaluate`.
   * Display the generated benchmark PNG charts (`results/charts/*.png`) inline using `IPython.display.Image`.
5. **Cell 5 — Launch Live Gradio Demo:**
   * Run `!python app.py --share` (or import `demo` from `app` and call `demo.launch(share=True, debug=True)`) to expose a public, shareable Gradio URL directly from Colab.

---

## 6. Reliability, Error Handling & Configuration

### 6.1 Environment Variables (`.env.example`)
```env
LLM_PROVIDER=gemini                           # Default: "gemini" (optional fallback: "groq")
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=                                 # Optional: only needed if LLM_PROVIDER=groq
DEBATER_MODEL=gemini-3.5-flash-lite
JUDGE_MODEL=gemini-3.8-flash
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
MAX_DEBATE_ROUNDS=2
TOP_K_EVIDENCE=5
GRADIO_SHARE=false
```

### 6.2 Rate-Limiting & Failure Guardrails (`src/llm_client.py`)
* **Exponential Backoff:** Wrap all LLM calls with `tenacity` (retry on HTTP 429 / 5xx / `ResourceExhausted`, `wait_exponential(min=2, max=30)`, `stop_after_attempt=5`).
* **Structured Output Fallback:** If `with_structured_output()` fails to parse JSON, fall back to raw text generation + regex JSON extraction + Pydantic validation; if still invalid, retry once with an explicit error-correction prompt.
* **Checkpointing in Evaluation:** `src/evaluate.py` must append results incrementally after each claim so an interrupted batch run resumes from the last completed claim without re-calling the API.

---

## 7. Implementation Acceptance Criteria

An implementation is considered complete when all of the following pass:
- [ ] `python -m src.ingestion` downloads and extracts the official SciFact `data.tar.gz`, parses `corpus.jsonl` and `claims_*.jsonl`, builds both `ChromaDB` and `BM25` indices cleanly, and creates `data/eval_set.json` (60 balanced claims).
- [ ] `retrieve_evidence("test claim", top_k=5)` returns 5 valid `EvidenceChunk` objects with non-zero rerank scores from `cross-encoder/ms-marco-MiniLM-L6-v2`.
- [ ] Running the LangGraph debate pipeline on a single claim executes `max_rounds` of Proponent/Skeptic turns (`gemini-3.5-flash-lite`), produces a valid `JudgeVerdict` JSON (`gemini-3.8-flash`), and computes `citation_validity_rate`.
- [ ] `python -m src.evaluate --limit 10` (smoke test) and full 60-claim run complete without unhandled exceptions and generate `results/eval_results.json` plus all 3 PNG charts.
- [ ] `python app.py` launches the 2-tab Gradio UI with working live fact-checking and benchmark visualization, and `python app.py --share` generates a public Gradio link.
- [ ] `run_colab.ipynb` executes cleanly top-to-bottom in Google Colab using Colab Secrets (`userdata.get`) and launches the live Gradio UI without modifying `src/` files.