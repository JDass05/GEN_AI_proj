"""
Agent node functions for the LangGraph debate pipeline.
Each function takes AgentState and returns a partial state update.
"""

import re
from typing import List

import config
from src.schemas import AgentState, DebateTurn, JudgeVerdict, EvidenceChunk
from src.llm_client import get_llm, invoke_llm, invoke_llm_structured
from src.prompts import (
    PROPONENT_SYSTEM_PROMPT,
    PROPONENT_ROUND1_USER_TEMPLATE,
    PROPONENT_REBUTTAL_USER_TEMPLATE,
    SKEPTIC_SYSTEM_PROMPT,
    SKEPTIC_USER_TEMPLATE,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    format_evidence_for_prompt,
    format_transcript_for_prompt,
)
from src.retriever import retrieve_evidence


# ---------------------------------------------------------------------------
# Helper: Extract cited chunk IDs from agent argument text
# ---------------------------------------------------------------------------
def extract_cited_ids(text: str) -> List[str]:
    """Extract [chunk_id] references from agent argument text."""
    return re.findall(r"\[(\d+_c\d+)\]", text)


# ---------------------------------------------------------------------------
# Node: Retrieve Evidence
# ---------------------------------------------------------------------------
def retrieve_node(state: AgentState) -> dict:
    """Retrieve top-K evidence chunks for the claim."""
    claim = state["claim"]
    top_k = config.TOP_K_EVIDENCE

    chunks = retrieve_evidence(claim, top_k=top_k)
    evidence = [chunk.model_dump() for chunk in chunks]

    return {
        "evidence": evidence,
        "transcript": [],
        "current_round": 1,
        "verdict": None,
        "citation_validity_rate": 0.0,
    }


# ---------------------------------------------------------------------------
# Node: Proponent Agent
# ---------------------------------------------------------------------------
def proponent_node(state: AgentState) -> dict:
    """Proponent argues FOR the claim."""
    claim = state["claim"]
    evidence = state["evidence"]
    transcript = list(state["transcript"])
    current_round = state["current_round"]

    evidence_text = format_evidence_for_prompt(evidence)
    llm = get_llm(config.DEBATER_MODEL)

    if current_round == 1:
        user_prompt = PROPONENT_ROUND1_USER_TEMPLATE.format(
            claim=claim,
            evidence=evidence_text,
        )
    else:
        # Find the last skeptic argument
        skeptic_arg = ""
        for turn in reversed(transcript):
            t = turn if isinstance(turn, dict) else turn.model_dump()
            if t.get("role") == "SKEPTIC":
                skeptic_arg = t.get("argument", "")
                break

        user_prompt = PROPONENT_REBUTTAL_USER_TEMPLATE.format(
            claim=claim,
            evidence=evidence_text,
            skeptic_argument=skeptic_arg,
        )

    argument = invoke_llm(llm, PROPONENT_SYSTEM_PROMPT, user_prompt)
    cited_ids = extract_cited_ids(argument)

    turn = DebateTurn(
        round_num=current_round,
        role="PROPONENT",
        argument=argument,
        cited_chunk_ids=cited_ids,
    )
    transcript.append(turn.model_dump())

    return {"transcript": transcript}


# ---------------------------------------------------------------------------
# Node: Skeptic Agent
# ---------------------------------------------------------------------------
def skeptic_node(state: AgentState) -> dict:
    """Skeptic argues AGAINST the claim / rebuts Proponent."""
    claim = state["claim"]
    evidence = state["evidence"]
    transcript = list(state["transcript"])
    current_round = state["current_round"]

    evidence_text = format_evidence_for_prompt(evidence)

    # Find the last proponent argument
    proponent_arg = ""
    for turn in reversed(transcript):
        t = turn if isinstance(turn, dict) else turn.model_dump()
        if t.get("role") == "PROPONENT":
            proponent_arg = t.get("argument", "")
            break

    llm = get_llm(config.DEBATER_MODEL)
    user_prompt = SKEPTIC_USER_TEMPLATE.format(
        claim=claim,
        evidence=evidence_text,
        proponent_argument=proponent_arg,
    )

    argument = invoke_llm(llm, SKEPTIC_SYSTEM_PROMPT, user_prompt)
    cited_ids = extract_cited_ids(argument)

    turn = DebateTurn(
        round_num=current_round,
        role="SKEPTIC",
        argument=argument,
        cited_chunk_ids=cited_ids,
    )
    transcript.append(turn.model_dump())

    return {"transcript": transcript}


# ---------------------------------------------------------------------------
# Node: Judge Agent
# ---------------------------------------------------------------------------
def judge_node(state: AgentState) -> dict:
    """Judge evaluates the full debate and renders a structured verdict."""
    claim = state["claim"]
    evidence = state["evidence"]
    transcript = state["transcript"]

    evidence_text = format_evidence_for_prompt(evidence)
    transcript_text = format_transcript_for_prompt(transcript)

    llm = get_llm(config.JUDGE_MODEL)
    user_prompt = JUDGE_USER_TEMPLATE.format(
        claim=claim,
        evidence=evidence_text,
        transcript=transcript_text,
    )

    verdict = invoke_llm_structured(
        llm, JUDGE_SYSTEM_PROMPT, user_prompt, JudgeVerdict
    )

    return {"verdict": verdict.model_dump()}


# ---------------------------------------------------------------------------
# Node: Citation Verification (Programmatic Guardrail)
# ---------------------------------------------------------------------------
def verify_citations_node(state: AgentState) -> dict:
    """
    Verify that all cited chunk IDs exist in the retrieved evidence.
    Compute citation_validity_rate and strip hallucinated IDs.
    """
    evidence = state["evidence"]
    transcript = state["transcript"]
    verdict = state["verdict"]

    # Collect valid chunk IDs from evidence
    valid_ids = set()
    for chunk in evidence:
        cid = chunk.get("chunk_id") if isinstance(chunk, dict) else chunk.chunk_id
        valid_ids.add(cid)

    # Collect all cited IDs from transcript + verdict
    all_cited = []

    for turn in transcript:
        t = turn if isinstance(turn, dict) else turn
        cited = t.get("cited_chunk_ids", []) if isinstance(t, dict) else t.cited_chunk_ids
        all_cited.extend(cited)

    if verdict:
        v = verdict if isinstance(verdict, dict) else verdict
        verdict_cited = v.get("cited_chunk_ids", []) if isinstance(v, dict) else v.cited_chunk_ids
        all_cited.extend(verdict_cited)

    # Compute validity rate
    if all_cited:
        valid_count = sum(1 for cid in all_cited if cid in valid_ids)
        citation_validity_rate = valid_count / len(all_cited)
    else:
        citation_validity_rate = 1.0  # No citations = vacuously valid

    # Strip hallucinated IDs from verdict
    if verdict and isinstance(verdict, dict):
        verdict["cited_chunk_ids"] = [
            cid for cid in verdict.get("cited_chunk_ids", [])
            if cid in valid_ids
        ]

    return {
        "verdict": verdict,
        "citation_validity_rate": round(citation_validity_rate, 4),
    }


# ---------------------------------------------------------------------------
# Conditional edge: should we loop or go to judge?
# ---------------------------------------------------------------------------
def should_continue_debate(state: AgentState) -> str:
    """
    Routing function for the conditional edge after the skeptic node.
    Returns 'continue' to loop back to proponent, or 'judge' to proceed to verdict.
    """
    current_round = state["current_round"]
    max_rounds = state["max_rounds"]

    if current_round < max_rounds:
        return "continue"
    else:
        return "judge"
