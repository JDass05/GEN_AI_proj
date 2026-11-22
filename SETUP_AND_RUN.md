# 🚀 Quick Start Guide: Running the Debate Fact-Checker

This guide walks anyone through running the **Multi-Agent Debate & Consensus Fact-Checker with Hybrid RAG and React UI** from scratch after cloning the repository.

---

## 📋 Prerequisites

Before starting, ensure you have the following installed on your machine:

1. **Python 3.10 to 3.13** ([Download Python](https://www.python.org/downloads/))
2. **Node.js 18+ & npm** ([Download Node.js](https://nodejs.org/))
3. **Git** ([Download Git](https://git-scm.com/))
4. **Google AI Studio Gemini API Key** (Free from [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey))
5. *(Optional but recommended)*: **NVIDIA GPU with CUDA** for fast vector embeddings (CPU works too).

---

## ⚙️ Step-by-Step Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/JDass05/GEN_AI_proj.git
cd GEN_AI_proj/debate-fact-checker
```

---

### Step 2: Install Python Dependencies

It is recommended to use a virtual environment:

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv

# Activate the virtual environment
# On Windows (PowerShell):
venv\Scripts\Activate.ps1
# On Windows (CMD):
venv\Scripts\activate.bat
# On macOS / Linux:
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

> **Note for NVIDIA GPU users:** If you have an NVIDIA GPU (e.g., RTX 3050, 3060, 4060, etc.) and want GPU acceleration, install the CUDA-enabled PyTorch build:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
> ```

---

### Step 3: Configure Environment Variables

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` in a text editor and configure your preferred provider:

#### Option 1: Local Ollama (100% Free & Local GPU/CPU Inference)
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
DEBATER_MODEL=mistral
JUDGE_MODEL=phi3:mini
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
MAX_DEBATE_ROUNDS=1
TOP_K_EVIDENCE=5
GRADIO_SHARE=false
```

To prepare Ollama:
```bash
ollama pull mistral
ollama pull phi3:mini
```

#### Option 2: Cloud Google Gemini API
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
DEBATER_MODEL=gemini-3.5-flash-lite
JUDGE_MODEL=gemini-3.5-flash
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
MAX_DEBATE_ROUNDS=2
TOP_K_EVIDENCE=5
GRADIO_SHARE=false
```

---

### Step 4: Build the Knowledge Base & RAG Index

Run the ingestion script to download the SciFact dataset (~5,183 scientific abstracts), generate ~24,420 chunks, embed them with `BAAI/bge-base-en-v1.5`, and construct both the **ChromaDB** vector store and the **BM25Okapi** sparse index:

```bash
python -m src.ingestion
```

*This step runs once and automatically caches the index in `chroma_db/` for all future runs.*

---

### Step 5: Build the React Frontend

Install the frontend dependencies and build the production bundle:

```bash
cd frontend
npm install
npm run build
cd ..
```

---

## 🏃 Running the Application

### Option A: Modern React Web Application (Recommended)

Start the unified FastAPI + React server:

```bash
python server.py
```

Now open **[http://localhost:8000](http://localhost:8000)** in your browser!

You will have access to:
- 🔍 **Fact Checker Tab**: Enter scientific claims or pick preloaded examples to trigger the live adversarial debate (Proponent vs. Skeptic vs. Judge).
- 📤 **Upload Knowledge Tab**: Drag-and-drop any custom **PDF, DOCX, TXT, or Markdown** document to dynamically chunk, embed, and index it into the active RAG pipeline in real time.
- 📊 **Benchmark Dashboard Tab**: View quantitative accuracy, Macro-F1, and confusion matrix charts comparing the Multi-Agent Debate system against the Single-LLM Baseline.

---

### Option B: Gradio Web Interface Alternative

If you prefer the Gradio interface:

```bash
# Run locally
python app.py

# Or create a public shareable URL
python app.py --share
```

---

### Option C: Run Benchmark & Evaluation Suite

To test the system against the SciFact benchmark and generate performance comparison charts:

```bash
# Smoke test on 3 to 10 claims
python -m src.evaluate --limit 10

# Full 60-claim balanced evaluation (20 Supported, 20 Refuted, 20 Not Enough Info)
python -m src.evaluate
```

Generated charts will be saved to `results/charts/`:
- `accuracy_f1_comparison.png`
- `per_class_f1.png`
- `confusion_matrices.png`

---

### Option D: Run in Google Colab (Zero Local Setup)

If you don't want to install anything locally:
1. Open [`debate-fact-checker/run_colab.ipynb`](file:///c:/GEN_AI_proj/debate-fact-checker/run_colab.ipynb) in **Google Colab**.
2. Add your `GEMINI_API_KEY` to Colab Secrets (the 🔑 icon in the sidebar).
3. Select **Runtime > Run all**.
4. Click the generated public Gradio URL to use the app in your browser.

---

## 🛠️ Project Structure at a Glance

```text
debate-fact-checker/
├── server.py              # Unified FastAPI server (serves API + React frontend)
├── config.py              # Configuration and hyperparameters
├── requirements.txt       # Python dependencies
├── .env.example           # Environment template
├── frontend/              # Modern React 18 UI
│   ├── src/App.jsx        # Main UI component
│   └── dist/              # Production web bundle
├── src/
│   ├── ingestion.py       # Downloads & indexes corpus into ChromaDB & BM25
│   ├── retriever.py       # Hybrid retrieval (Dense + BM25 + RRF + Cross-Encoder)
│   ├── doc_uploader.py    # Custom document ingestion (PDF, DOCX, TXT, MD)
│   ├── agents.py          # Proponent, Skeptic, and Judge node functions
│   ├── graph.py           # LangGraph state machine orchestrating debate
│   ├── baseline.py        # Single-LLM baseline for benchmark comparison
│   ├── evaluate.py        # Evaluation suite & chart generator
│   ├── prompts.py         # Agent prompts & guardrail guidelines
│   ├── llm_client.py      # LLM invocation with retry & backoff
│   └── schemas.py         # Pydantic models & state schemas
├── results/               # Saved benchmark metrics & generated PNG charts
└── run_colab.ipynb        # Google Colab launcher notebook
```

---

## ❓ Troubleshooting

1. **`ModuleNotFoundError: No module named 'src'`**
   - Make sure you are running commands from inside the `debate-fact-checker` directory:
     ```bash
     cd debate-fact-checker
     python server.py
     ```

2. **`CUDA out of memory` during retrieval**
   - The hybrid retriever uses CPU for query cross-encoding by default to keep VRAM free for other tasks. Ingestion can safely use CUDA.

3. **Port 8000 already in use**
   - You can launch the server on a different port:
     ```bash
     uvicorn server:app --host 0.0.0.0 --port 8080
     ```
