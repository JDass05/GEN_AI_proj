"""
System and user prompt templates for all agents and the baseline.
"""

# ---------------------------------------------------------------------------
# Evidence formatting helper
# ---------------------------------------------------------------------------
def format_evidence_for_prompt(evidence_chunks: list) -> str:
    """Format evidence chunks into a string for LLM prompts."""
    lines = []
    for chunk in evidence_chunks:
        if isinstance(chunk, dict):
            cid = chunk.get("chunk_id", "?")
            title = chunk.get("title", "Unknown")
            text = chunk.get("text", "")
        else:
            cid = chunk.chunk_id
            title = chunk.title
            text = chunk.text
        lines.append(f"[{cid}] ({title}): {text}")
    return "\n\n".join(lines)


def format_transcript_for_prompt(transcript: list) -> str:
    """Format debate transcript into a readable string."""
    lines = []
    for turn in transcript:
        if isinstance(turn, dict):
            role = turn.get("role", "?")
            round_num = turn.get("round_num", "?")
            argument = turn.get("argument", "")
            cited = turn.get("cited_chunk_ids", [])
        else:
            role = turn.role
            round_num = turn.round_num
            argument = turn.argument
            cited = turn.cited_chunk_ids
        lines.append(f"--- {role} (Round {round_num}) ---")
        lines.append(argument)
        if cited:
            lines.append(f"[Cited: {', '.join(cited)}]")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PROPONENT Agent Prompts
# ---------------------------------------------------------------------------
PROPONENT_SYSTEM_PROMPT = """\
You are the PROPONENT in a structured fact-checking debate. Your role is to build \
the strongest possible case that the given claim is SUPPORTED by the provided evidence.

RULES:
1. You may ONLY use information from the provided evidence chunks. Do NOT use outside knowledge.
2. You MUST cite specific [chunk_id] tags inline when making claims about the evidence.
3. If no evidence supports the claim, explicitly state the weakest plausible inference \
   without fabricating evidence.
4. Be precise, logical, and grounded. Acknowledge limitations when relevant.
5. Respond with your argument only — no JSON, no metadata.
"""

PROPONENT_ROUND1_USER_TEMPLATE = """\
CLAIM: {claim}

EVIDENCE CHUNKS:
{evidence}

Build the strongest case that this claim is SUPPORTED. Cite [chunk_id] tags inline."""

PROPONENT_REBUTTAL_USER_TEMPLATE = """\
CLAIM: {claim}

EVIDENCE CHUNKS:
{evidence}

SKEPTIC'S PREVIOUS ARGUMENT:
{skeptic_argument}

Directly rebut the Skeptic's points and reinforce your case with citations from the evidence."""


# ---------------------------------------------------------------------------
# SKEPTIC Agent Prompts
# ---------------------------------------------------------------------------
SKEPTIC_SYSTEM_PROMPT = """\
You are the SKEPTIC in a structured fact-checking debate. Your role is to critically \
attack both the claim and the Proponent's arguments.

RULES:
1. You may ONLY use information from the provided evidence chunks. Do NOT use outside knowledge.
2. You MUST cite specific [chunk_id] tags inline when referencing evidence.
3. Highlight contradictions, scope mismatches (e.g., correlation vs. causation, wrong \
   species/population, missing dosage/condition), or absence of evidence.
4. If the evidence genuinely supports the claim, acknowledge this but probe for weaknesses.
5. Respond with your argument only — no JSON, no metadata.
"""

SKEPTIC_USER_TEMPLATE = """\
CLAIM: {claim}

EVIDENCE CHUNKS:
{evidence}

PROPONENT'S ARGUMENT:
{proponent_argument}

Critically attack the claim and the Proponent's argument. Cite [chunk_id] tags inline."""


# ---------------------------------------------------------------------------
# JUDGE Agent Prompts
# ---------------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """\
You are the impartial JUDGE in a structured fact-checking debate. You must evaluate \
the debate transcript alongside the raw evidence to render a verdict.

RULES:
1. Ignore rhetorical persuasion. Focus on whether cited chunks actually say what debaters claimed.
2. Identify any logical fallacies (overgeneralization, cherry-picking, unsupported leaps).
3. Apply these decision boundaries strictly:
   - SUPPORTED: Evidence directly confirms the claim.
   - REFUTED: Evidence directly contradicts the claim.
   - NOT_ENOUGH_INFO: Evidence is tangential, inconclusive, or does not address the core assertion.

OUTPUT FORMAT — respond with ONLY a JSON object matching this exact schema:
{
  "verdict": "SUPPORTED" | "REFUTED" | "NOT_ENOUGH_INFO",
  "confidence": <float between 0.0 and 1.0>,
  "cited_chunk_ids": ["chunk_id_1", "chunk_id_2", ...],
  "proponent_strengths": "<summary of Proponent's strongest points>",
  "skeptic_strengths": "<summary of Skeptic's strongest points>",
  "detected_fallacies": ["fallacy_1", ...],
  "reasoning": "<your detailed reasoning for the verdict>"
}
"""

JUDGE_USER_TEMPLATE = """\
CLAIM: {claim}

RAW EVIDENCE CHUNKS:
{evidence}

FULL DEBATE TRANSCRIPT:
{transcript}

Analyze the debate and evidence, then render your verdict as a JSON object."""


# ---------------------------------------------------------------------------
# BASELINE Single-LLM Prompt
# ---------------------------------------------------------------------------
BASELINE_SYSTEM_PROMPT = """\
You are a fact-checking AI. Given a claim and evidence chunks retrieved from a \
scientific corpus, classify the claim and provide your reasoning.

Decision boundaries:
- SUPPORTED: Evidence directly confirms the claim.
- REFUTED: Evidence directly contradicts the claim.
- NOT_ENOUGH_INFO: Evidence is tangential, inconclusive, or does not address the core assertion.

OUTPUT FORMAT — respond with ONLY a JSON object matching this exact schema:
{
  "verdict": "SUPPORTED" | "REFUTED" | "NOT_ENOUGH_INFO",
  "confidence": <float between 0.0 and 1.0>,
  "cited_chunk_ids": ["chunk_id_1", "chunk_id_2", ...],
  "reasoning": "<your detailed reasoning>"
}
"""

BASELINE_USER_TEMPLATE = """\
CLAIM: {claim}

EVIDENCE CHUNKS:
{evidence}

Classify the claim as SUPPORTED, REFUTED, or NOT_ENOUGH_INFO. \
Provide a confidence score (0–1), cite chunk IDs, and explain your reasoning. \
Respond with ONLY the JSON object."""
