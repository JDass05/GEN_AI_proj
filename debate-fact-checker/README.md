# 🔬 ConsensusAI: Multi-Agent Debate & Consensus Fact-Checker

A modern, production-ready fact-checking platform powered by **adversarial multi-agent debate** over scientific evidence with hybrid dense-sparse retrieval (ChromaDB + BM25 + Cross-Encoder reranking) and an interactive **React frontend**.

---

## 💡 Can you add custom files to the RAG knowledge base?

**Yes, absolutely!** 

The system includes dynamic file ingestion in [`src/doc_uploader.py`](file:///c:/GEN_AI_proj/debate-fact-checker/src/doc_uploader.py) and via the **React UI**:
1. You can upload any **PDF, DOCX, Markdown, or TXT** file.
2. The file is split into chunks, embedded using **`BAAI/bge-base-en-v1.5`** on your local **NVIDIA RTX 3050 GPU**, and added to the persistent **ChromaDB** vector store.
3. The **BM25Okapi** sparse index is updated in real-time.
4. The adversarial debaters (**Proponent** and **Skeptic**) immediately retrieve and cite your uploaded documents when arguing a claim.

---

## 🏗️ System Architecture

```text
[User Claim] ─────────┐
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: HYBRID RETRIEVAL & RERANKING                      │
│  • Dense Vector Search: ChromaDB + BGE (CUDA RTX 3050)      │
│  • Sparse Keyword Search: BM25Okapi                         │
│  • Fusion: Reciprocal Rank Fusion (RRF, k=60)               │
│  • Reranking: Cross-Encoder (MiniLM-L6-v2)                  │
│  • Knowledge: 24,420 SciFact chunks + User-uploaded files   │
└────────────────────────────┬────────────────────────────────┘
                             │ Top-K Evidence Chunks
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: MULTI-AGENT DEBATE (LangGraph)                    │
│  Round 1..N:                                                │
│  ┌────────────────────────┐      ┌────────────────────────┐ │
│  │    Proponent Agent     │ ───► │     Skeptic Agent      │ │
│  │ (Argues FOR the claim, │ ◄─── │  (Attacks claim and    │ │
│  │   cites [chunk_id]s)   │      │   rebuts Proponent)    │ │
│  └────────────────────────┘      └───────────┬────────────┘ │
│                ▲ (Loop N rounds)             │              │
│                └─────────────────────────────┘              │
│                                              ▼              │
│                                  ┌────────────────────────┐ │
│                                  │      Judge Agent       │ │
│                                  │ (Evaluates debate,     │ │
│                                  │  audits citations,     │ │
│                                  │  detects fallacies)    │ │
│                                  └───────────┬────────────┘ │
└──────────────────────────────────────────────┼──────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: CITATION GUARDRAIL & AUDIT                        │
│  • Programmatically strips hallucinated chunk IDs           │
│  • Computes Citation Faithfulness Rate                      │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start

### 1. Configure Your API Key

Create or edit `.env` in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
LLM_PROVIDER=gemini
DEBATER_MODEL=gemini-2.0-flash-lite
JUDGE_MODEL=gemini-2.0-flash
```

### 2. Launch the Application

Run the unified backend + frontend server:
```bash
python server.py
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

The server hosts the React application and all backend APIs:
- **`POST /api/fact-check`**: Triggers the multi-agent debate pipeline.
- **`POST /api/upload-document`**: Ingests new PDF, DOCX, TXT, or MD files into the RAG index.
- **`GET /api/benchmark`**: Returns evaluation metrics and comparison charts.

Alternatively, you can run the Gradio interface:
```bash
python app.py
```

### 3. Run the Evaluation Benchmark

To benchmark the Multi-Agent Debate system against the Single-LLM Baseline:
```bash
# Smoke test (10 claims)
python -m src.evaluate --limit 10

# Full 60-claim balanced benchmark
python -m src.evaluate
```

Charts and metrics will be saved in `results/charts/` and displayed in the React dashboard.

---

## 🖥️ React Frontend Features

- **Fact Checker Tab**:
  - Live query input with pre-loaded scientific claims.
  - Sliders for debate rounds (1–3) and top-K evidence chunks (3–8).
  - Color-coded verdict card (`SUPPORTED`, `REFUTED`, `NOT_ENOUGH_INFO`).
  - Confidence and citation faithfulness meters.
  - Detected fallacies list.
  - Full side-by-side debate transcript with inline chunk citations.
  - Expandable evidence cards with reranker relevance scores.
- **Upload Knowledge Tab**:
  - Drag-and-drop file uploader for PDFs, Word documents, Markdown, and text files.
  - Real-time chunking, GPU embedding, and dual-index update.
- **Benchmark Dashboard Tab**:
  - Direct comparison table: Multi-Agent Debate vs. Single-LLM Baseline.
  - Rendered accuracy, macro-F1, and confusion matrix charts.
  - Table of claims where debate corrected baseline errors.

---

## 📁 Repository Structure

```text
debate-fact-checker/
├── server.py                 # FastAPI server (serves API + React frontend)
├── config.py                 # Central configuration and hyperparameters
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
├── run_colab.ipynb           # Google Colab launcher notebook
├── app.py                    # Gradio 2-tab interface alternative
├── frontend/                 # Clean, modern React application
│   ├── src/
│   │   ├── App.jsx           # Main React component
│   │   ├── main.jsx          # React DOM entry point
│   │   └── index.css         # Modern dark-mode styling
│   ├── dist/                 # Pre-built production bundle
│   ├── package.json
│   └── vite.config.js
├── src/
│   ├── schemas.py            # Pydantic models & LangGraph AgentState
│   ├── ingestion.py          # SciFact dataset downloader & index builder
│   ├── retriever.py          # Hybrid RAG (Dense + BM25 + RRF + Cross-Encoder)
│   ├── doc_uploader.py       # Custom document ingestion (PDF, DOCX, TXT, MD)
│   ├── llm_client.py         # LLM wrapper with exponential backoff
│   ├── prompts.py            # Agent prompt templates
│   ├── agents.py             # Proponent, Skeptic, and Judge nodes
│   ├── graph.py              # LangGraph debate topology
│   ├── baseline.py           # Single-LLM RAG baseline
│   └── evaluate.py           # Benchmark evaluator & chart generator
├── chroma_db/                # Persisted ChromaDB vectors & BM25 pickle
├── data/
│   ├── raw/                  # SciFact corpus & claim files
│   └── eval_set.json         # 60-claim balanced evaluation set
└── results/
    ├── eval_results.json     # Evaluation benchmark results
    └── charts/               # Generated comparison plots
```
