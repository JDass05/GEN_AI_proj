"""FastAPI Backend Server for Multi-Agent Debate Fact-Checker.
Serves:
- Live multi-agent debate fact-checking API
- Dynamic document upload and ingestion into RAG (PDF, DOCX, TXT, MD)
- Benchmark evaluation metrics & static chart distribution
- Production React Frontend distribution
"""

import os
import shutil
import tempfile
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from src.doc_uploader import add_document_to_index
from src.graph import run_debate
from src.retriever import retrieve_evidence

app = FastAPI(
    title="ConsensusAI - Multi-Agent Debate Fact-Checker",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure results/charts directory exists and mount static charts
os.makedirs(config.CHARTS_DIR, exist_ok=True)
app.mount("/static/charts", StaticFiles(directory=config.CHARTS_DIR), name="charts")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class FactCheckRequest(BaseModel):
    claim: str
    rounds: Optional[int] = 2
    top_k: Optional[int] = 5


class FactCheckResponse(BaseModel):
    claim: str
    verdict: Optional[dict]
    evidence: List[dict]
    transcript: List[dict]
    citation_validity_rate: float
    current_round: int
    max_rounds: int


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health_check():
    has_api_key = bool(config.GEMINI_API_KEY)
    return {
        "status": "healthy",
        "has_api_key": has_api_key,
        "llm_provider": config.LLM_PROVIDER,
        "debater_model": config.DEBATER_MODEL,
        "judge_model": config.JUDGE_MODEL,
    }


@app.post("/api/fact-check", response_model=FactCheckResponse)
def run_fact_check(req: FactCheckRequest):
    if not req.claim or not req.claim.strip():
        raise HTTPException(status_code=400, detail="Claim cannot be empty.")

    try:
        original_top_k = config.TOP_K_EVIDENCE
        if req.top_k:
            config.TOP_K_EVIDENCE = req.top_k

        state = run_debate(req.claim.strip(), max_rounds=req.rounds or 2)
        config.TOP_K_EVIDENCE = original_top_k

        return FactCheckResponse(
            claim=state["claim"],
            verdict=state.get("verdict"),
            evidence=state.get("evidence", []),
            transcript=state.get("transcript", []),
            citation_validity_rate=state.get("citation_validity_rate", 0.0),
            current_round=state.get("current_round", 1),
            max_rounds=state.get("max_rounds", req.rounds or 2),
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """Upload any text, markdown, docx, or pdf file and dynamically index it into ChromaDB + BM25."""
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        result = add_document_to_index(tmp_path)
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("message"))

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/benchmark")
def get_benchmark():
    import json
    results_path = os.path.join(config.RESULTS_DIR, "eval_results.json")
    if not os.path.exists(results_path):
        return {
            "status": "not_run",
            "message": "Evaluation benchmark has not been run yet. Run 'python -m src.evaluate' to generate.",
            "metrics": None,
            "debate_wins": [],
            "baseline_wins": [],
            "charts": [],
        }

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    charts = []
    for chart in ["accuracy_f1_comparison.png", "per_class_f1.png", "confusion_matrices.png"]:
        if os.path.exists(os.path.join(config.CHARTS_DIR, chart)):
            charts.append(f"/static/charts/{chart}")

    return {
        "status": "ready",
        "metrics": data.get("metrics"),
        "debate_wins": data.get("debate_wins", []),
        "baseline_wins": data.get("baseline_wins", []),
        "charts": charts,
    }


# ---------------------------------------------------------------------------
# Serve Production React Frontend
# ---------------------------------------------------------------------------
dist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
if os.path.exists(dist_dir):
    app.mount("/assets", StaticFiles(directory=os.path.join(dist_dir, "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_react(full_path: str):
        file_path = os.path.join(dist_dir, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(dist_dir, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
