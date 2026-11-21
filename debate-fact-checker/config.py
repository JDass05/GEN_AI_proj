"""Centralized hyperparameters & model IDs for the Debate Fact-Checker."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM Provider ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- Model IDs ---
DEBATER_MODEL = os.getenv("DEBATER_MODEL", "gemini-3.5-flash-lite")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")

# --- Retrieval Hyperparameters ---
TOP_K_EVIDENCE = int(os.getenv("TOP_K_EVIDENCE", "5"))
DENSE_TOP_N = 20          # Dense search candidates before fusion
BM25_TOP_N = 20           # BM25 candidates before fusion
RRF_K = 60                # Reciprocal Rank Fusion k parameter
RRF_TOP_N = 15            # Candidates after RRF, before reranking

# --- Debate Hyperparameters ---
MAX_DEBATE_ROUNDS = int(os.getenv("MAX_DEBATE_ROUNDS", "2"))

# --- Chunking ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
EVAL_SET_PATH = os.path.join(DATA_DIR, "eval_set.json")
CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
BM25_INDEX_PATH = os.path.join(CHROMA_DB_DIR, "bm25_index.pkl")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")

# --- SciFact Dataset ---
SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"

# --- Gradio ---
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"

# --- LLM Retry ---
LLM_RETRY_MIN_WAIT = 2
LLM_RETRY_MAX_WAIT = 30
LLM_RETRY_MAX_ATTEMPTS = 5
