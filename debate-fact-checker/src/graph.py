"""
LangGraph StateGraph compilation for the multi-agent debate pipeline.

Topology:
  retrieve -> proponent -> skeptic -> [conditional] -> judge -> verify_citations
                 ^                        |
                 |--- (continue) ---------+

Usage:
    from src.graph import run_debate
    result = run_debate("Some claim to fact-check", max_rounds=2)
"""

from langgraph.graph import StateGraph, END

import config
from src.schemas import AgentState
from src.agents import (
    retrieve_node,
    proponent_node,
    skeptic_node,
    judge_node,
    verify_citations_node,
    should_continue_debate,
)


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_debate_graph() -> StateGraph:
    """Construct and compile the LangGraph debate pipeline."""

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("proponent", proponent_node)
    graph.add_node("skeptic", skeptic_node)
    graph.add_node("increment_round", _increment_round)
    graph.add_node("judge", judge_node)
    graph.add_node("verify_citations", verify_citations_node)

    # Set entry point
    graph.set_entry_point("retrieve")

    # Edges
    graph.add_edge("retrieve", "proponent")
    graph.add_edge("proponent", "skeptic")

    # Conditional edge after skeptic: continue debate or go to judge
    graph.add_conditional_edges(
        "skeptic",
        should_continue_debate,
        {
            "continue": "increment_round",
            "judge": "judge",
        },
    )

    # Increment round then loop back to proponent
    graph.add_edge("increment_round", "proponent")

    # Judge -> verify citations -> END
    graph.add_edge("judge", "verify_citations")
    graph.add_edge("verify_citations", END)

    return graph.compile()


def _increment_round(state: AgentState) -> dict:
    """Increment the round counter."""
    return {"current_round": state["current_round"] + 1}


# ---------------------------------------------------------------------------
# Compiled graph singleton
# ---------------------------------------------------------------------------
_compiled_graph = None


def get_debate_graph():
    """Get or create the compiled debate graph (singleton)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_debate_graph()
    return _compiled_graph


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_debate(
    claim: str,
    max_rounds: int = None,
    top_k: int = None,
) -> AgentState:
    """
    Run the full debate pipeline on a single claim.

    Args:
        claim: The natural-language claim to fact-check.
        max_rounds: Number of debate rounds (default from config).
        top_k: Number of evidence chunks to retrieve (default from config).

    Returns:
        Final AgentState with verdict, transcript, and citation validity.
    """
    max_rounds = max_rounds or config.MAX_DEBATE_ROUNDS

    initial_state: AgentState = {
        "claim": claim,
        "evidence": [],
        "transcript": [],
        "current_round": 1,
        "max_rounds": max_rounds,
        "verdict": None,
        "citation_validity_rate": 0.0,
    }

    graph = get_debate_graph()
    final_state = graph.invoke(initial_state)

    return final_state
