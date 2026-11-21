"""
Single-LLM + RAG baseline for apples-to-apples comparison with the debate system.

Uses the exact same retrieval output and the same LLM as the Judge (gemini-2.0-flash).
"""

import config
from src.schemas import BaselineResult, EvidenceChunk
from src.llm_client import get_llm, invoke_llm_structured
from src.prompts import (
    BASELINE_SYSTEM_PROMPT,
    BASELINE_USER_TEMPLATE,
    format_evidence_for_prompt,
)


def run_baseline(claim: str, evidence_chunks: list) -> BaselineResult:
    """
    Run the single-LLM baseline on a claim with pre-retrieved evidence.

    Args:
        claim: The natural-language claim.
        evidence_chunks: List of EvidenceChunk dicts (same evidence used by debate).

    Returns:
        BaselineResult with verdict, confidence, citations, and reasoning.
    """
    evidence_text = format_evidence_for_prompt(evidence_chunks)

    llm = get_llm(config.JUDGE_MODEL)
    user_prompt = BASELINE_USER_TEMPLATE.format(
        claim=claim,
        evidence=evidence_text,
    )

    result = invoke_llm_structured(
        llm, BASELINE_SYSTEM_PROMPT, user_prompt, BaselineResult
    )

    return result
