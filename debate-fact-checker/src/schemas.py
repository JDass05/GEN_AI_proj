"""Pydantic v2 data models and LangGraph AgentState for the Debate Fact-Checker."""

from typing import List, Literal, Optional, TypedDict
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Type alias for verdict labels
# ---------------------------------------------------------------------------
VerdictType = Literal["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------
class EvidenceChunk(BaseModel):
    """A single chunk of evidence retrieved from the knowledge base."""
    chunk_id: str
    doc_id: str
    title: str
    text: str
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0


class DebateTurn(BaseModel):
    """A single turn in the adversarial debate."""
    round_num: int
    role: Literal["PROPONENT", "SKEPTIC"]
    argument: str
    cited_chunk_ids: List[str] = Field(default_factory=list)


class JudgeVerdict(BaseModel):
    """Structured output from the Judge Agent."""
    verdict: VerdictType
    confidence: float = Field(..., ge=0.0, le=1.0)
    cited_chunk_ids: List[str] = Field(default_factory=list)
    proponent_strengths: str = ""
    skeptic_strengths: str = ""
    detected_fallacies: List[str] = Field(default_factory=list)
    reasoning: str = ""


class BaselineResult(BaseModel):
    """Structured output from the single-LLM baseline."""
    verdict: VerdictType
    confidence: float = Field(..., ge=0.0, le=1.0)
    cited_chunk_ids: List[str] = Field(default_factory=list)
    reasoning: str = ""


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    """Shared state passed through the LangGraph debate pipeline."""
    claim: str
    evidence: List[dict]                # Serialized EvidenceChunk dicts
    transcript: List[dict]              # Serialized DebateTurn dicts
    current_round: int
    max_rounds: int
    verdict: Optional[dict]             # Serialized JudgeVerdict dict
    citation_validity_rate: float       # Fraction of cited IDs present in retrieved evidence


# ---------------------------------------------------------------------------
# Evaluation record
# ---------------------------------------------------------------------------
class EvalClaim(BaseModel):
    """A single claim from the evaluation set."""
    id: int
    claim: str
    label: VerdictType
    gold_doc_ids: List[int] = Field(default_factory=list)


class EvalResult(BaseModel):
    """Result of evaluating a single claim with both systems."""
    claim_id: int
    claim_text: str
    gold_label: VerdictType
    baseline_verdict: Optional[VerdictType] = None
    baseline_confidence: Optional[float] = None
    baseline_reasoning: Optional[str] = None
    debate_verdict: Optional[VerdictType] = None
    debate_confidence: Optional[float] = None
    debate_reasoning: Optional[str] = None
    debate_transcript: List[dict] = Field(default_factory=list)
    citation_validity_rate: float = 0.0
    baseline_latency_s: float = 0.0
    debate_latency_s: float = 0.0
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
