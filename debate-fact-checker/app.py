"""
Gradio 2-tab interactive UI for the Multi-Agent Debate Fact-Checker.

Tab 1: Live Fact-Checker — submit a claim and see the full debate + verdict.
Tab 2: Benchmark Dashboard — view evaluation metrics and charts.

Usage:
    python app.py              # Local launch
    python app.py --share      # Public Gradio link
"""

import argparse
import json
import os
import sys

import gradio as gr

import config

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.graph import run_debate
from src.prompts import format_evidence_for_prompt, format_transcript_for_prompt


# ---------------------------------------------------------------------------
# Example claims from SciFact
# ---------------------------------------------------------------------------
EXAMPLE_CLAIMS = [
    "0-dimensional biomaterials show inductive properties.",
    "Epigenetic regulation is associated with sporadic cancers.",
    "ALDH1 expression is associated with better prognosis in breast cancer.",
    "Decreased expression of CXCR4 is associated with disease-free survival in breast cancer.",
    "MicroRNA-155 is upregulated in certain cancers.",
]


# ---------------------------------------------------------------------------
# Tab 1: Live Fact-Checker
# ---------------------------------------------------------------------------
def fact_check_claim(claim: str, debate_rounds: int, top_k: int):
    """Run the debate pipeline and format outputs for Gradio."""
    if not claim or not claim.strip():
        return "⚠️ Please enter a claim.", "", "", ""

    try:
        # Temporarily override config
        original_top_k = config.TOP_K_EVIDENCE
        config.TOP_K_EVIDENCE = top_k

        state = run_debate(claim.strip(), max_rounds=debate_rounds)

        config.TOP_K_EVIDENCE = original_top_k

        verdict = state.get("verdict", {}) or {}
        evidence = state.get("evidence", [])
        transcript = state.get("transcript", [])
        citation_rate = state.get("citation_validity_rate", 0.0)

        # --- Output 1: Verdict Banner ---
        verdict_label = verdict.get("verdict", "UNKNOWN")
        confidence = verdict.get("confidence", 0.0)

        color_map = {
            "SUPPORTED": "🟢",
            "REFUTED": "🔴",
            "NOT_ENOUGH_INFO": "🟡",
        }
        emoji = color_map.get(verdict_label, "⚪")

        banner = f"""## {emoji} Verdict: **{verdict_label}**

| Metric | Value |
|--------|-------|
| **Confidence** | {confidence:.1%} |
| **Citation Validity** | {citation_rate:.1%} |
| **Debate Rounds** | {debate_rounds} |
| **Evidence Chunks** | {len(evidence)} |
"""

        # --- Output 2: Judge's Reasoning ---
        fallacies = verdict.get("detected_fallacies", [])
        fallacy_text = ", ".join(fallacies) if fallacies else "None detected"
        cited = verdict.get("cited_chunk_ids", [])
        cited_text = ", ".join(f"`{c}`" for c in cited) if cited else "None"

        reasoning = f"""### Judge's Analysis

**Reasoning:** {verdict.get("reasoning", "N/A")}

**Proponent Strengths:** {verdict.get("proponent_strengths", "N/A")}

**Skeptic Strengths:** {verdict.get("skeptic_strengths", "N/A")}

**Detected Fallacies:** {fallacy_text}

**Cited Evidence:** {cited_text}
"""

        # --- Output 3: Debate Transcript ---
        transcript_md = "### Debate Transcript\n\n"
        for turn in transcript:
            role = turn.get("role", "?")
            round_num = turn.get("round_num", "?")
            argument = turn.get("argument", "")
            cited_ids = turn.get("cited_chunk_ids", [])

            role_emoji = "🟦" if role == "PROPONENT" else "🟥"
            transcript_md += f"#### {role_emoji} {role} — Round {round_num}\n\n"
            transcript_md += f"{argument}\n\n"
            if cited_ids:
                transcript_md += f"*Cited: {', '.join(f'`{c}`' for c in cited_ids)}*\n\n"
            transcript_md += "---\n\n"

        # --- Output 4: Retrieved Evidence ---
        evidence_md = "### Retrieved Evidence Chunks\n\n"
        for chunk in evidence:
            cid = chunk.get("chunk_id", "?")
            title = chunk.get("title", "Unknown")
            text = chunk.get("text", "")
            score = chunk.get("rerank_score", 0.0)
            evidence_md += f"<details><summary><b>[{cid}]</b> {title} (score: {score:.4f})</summary>\n\n"
            evidence_md += f"{text}\n\n</details>\n\n"

        return banner, reasoning, transcript_md, evidence_md

    except Exception as e:
        import traceback
        error_msg = f"⚠️ Error: {str(e)}\n\n```\n{traceback.format_exc()}\n```"
        return error_msg, "", "", ""


# ---------------------------------------------------------------------------
# Tab 2: Benchmark Dashboard
# ---------------------------------------------------------------------------
def load_benchmark_data():
    """Load benchmark results and return formatted outputs."""
    results_path = os.path.join(config.RESULTS_DIR, "eval_results.json")

    if not os.path.exists(results_path):
        return ("⚠️ No evaluation results found. Run `python -m src.evaluate` first.",
                None, None, None, "")

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    debate_wins = data.get("debate_wins", [])

    if not metrics:
        return ("⚠️ No metrics found in results.", None, None, None, "")

    # Summary table
    summary = f"""## Evaluation Summary

| Metric | Baseline (Single-LLM) | Multi-Agent Debate |
|--------|:---------------------:|:------------------:|
| **Accuracy** | {metrics.get('baseline_accuracy', 0):.3f} | {metrics.get('debate_accuracy', 0):.3f} |
| **Macro-F1** | {metrics.get('baseline_macro_f1', 0):.3f} | {metrics.get('debate_macro_f1', 0):.3f} |
| **Avg Latency (s)** | {metrics.get('avg_baseline_latency_s', 0):.2f} | {metrics.get('avg_debate_latency_s', 0):.2f} |
| **Avg Citation Validity** | — | {metrics.get('avg_citation_validity', 0):.3f} |
| **Claims Evaluated** | {metrics.get('n_claims', 0)} | {metrics.get('n_claims', 0)} |

**Debate corrected {metrics.get('debate_win_count', 0)} baseline errors** | Baseline corrected {metrics.get('baseline_win_count', 0)} debate errors
"""

    # Load charts
    chart1_path = os.path.join(config.CHARTS_DIR, "accuracy_f1_comparison.png")
    chart2_path = os.path.join(config.CHARTS_DIR, "per_class_f1.png")
    chart3_path = os.path.join(config.CHARTS_DIR, "confusion_matrices.png")

    chart1 = chart1_path if os.path.exists(chart1_path) else None
    chart2 = chart2_path if os.path.exists(chart2_path) else None
    chart3 = chart3_path if os.path.exists(chart3_path) else None

    # Debate wins table
    wins_md = "### Debate Wins (Corrected Baseline Errors)\n\n"
    if debate_wins:
        wins_md += "| Claim ID | Claim | Gold | Baseline | Debate |\n"
        wins_md += "|----------|-------|------|----------|--------|\n"
        for w in debate_wins:
            claim_short = w["claim"][:60] + "..." if len(w["claim"]) > 60 else w["claim"]
            wins_md += f"| {w['claim_id']} | {claim_short} | {w['gold']} | {w['baseline_pred']} | {w['debate_pred']} |\n"
    else:
        wins_md += "No debate wins recorded yet."

    return summary, chart1, chart2, chart3, wins_md


# ---------------------------------------------------------------------------
# Build Gradio App
# ---------------------------------------------------------------------------
def create_app() -> gr.Blocks:
    """Build the 2-tab Gradio interface."""

    with gr.Blocks(
        title="Multi-Agent Debate Fact-Checker",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown(
            "# 🔬 Multi-Agent Debate & Consensus Fact-Checker\n"
            "A fact-checking system powered by adversarial AI debate over scientific evidence."
        )

        # --- Tab 1: Live Fact-Checker ---
        with gr.Tab("🔍 Live Fact-Checker"):
            with gr.Row():
                with gr.Column(scale=3):
                    claim_input = gr.Textbox(
                        label="Enter a claim to fact-check",
                        placeholder="e.g., 0-dimensional biomaterials show inductive properties.",
                        lines=2,
                    )
                with gr.Column(scale=1):
                    rounds_slider = gr.Slider(
                        minimum=1, maximum=3, value=2, step=1,
                        label="Debate Rounds",
                    )
                    topk_slider = gr.Slider(
                        minimum=3, maximum=8, value=5, step=1,
                        label="Top-K Evidence",
                    )

            gr.Examples(
                examples=[[c] for c in EXAMPLE_CLAIMS],
                inputs=[claim_input],
                label="Example SciFact Claims",
            )

            check_btn = gr.Button("🔎 Fact-Check", variant="primary", size="lg")

            verdict_output = gr.Markdown(label="Verdict")
            reasoning_output = gr.Markdown(label="Judge's Reasoning")

            with gr.Accordion("📜 Debate Transcript", open=False):
                transcript_output = gr.Markdown()

            with gr.Accordion("📚 Retrieved Evidence", open=False):
                evidence_output = gr.Markdown()

            check_btn.click(
                fn=fact_check_claim,
                inputs=[claim_input, rounds_slider, topk_slider],
                outputs=[verdict_output, reasoning_output, transcript_output, evidence_output],
            )

        # --- Tab 2: Benchmark Dashboard ---
        with gr.Tab("📊 Benchmark Dashboard"):
            refresh_btn = gr.Button("🔄 Refresh Dashboard", variant="secondary")

            summary_output = gr.Markdown()
            with gr.Row():
                chart1_img = gr.Image(label="Accuracy & Macro-F1", type="filepath")
                chart2_img = gr.Image(label="Per-Class F1", type="filepath")
            chart3_img = gr.Image(label="Confusion Matrices", type="filepath")
            wins_output = gr.Markdown()

            # Load on tab open
            demo.load(
                fn=load_benchmark_data,
                outputs=[summary_output, chart1_img, chart2_img, chart3_img, wins_output],
            )

            refresh_btn.click(
                fn=load_benchmark_data,
                outputs=[summary_output, chart1_img, chart2_img, chart3_img, wins_output],
            )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link")
    args = parser.parse_args()

    share = args.share or config.GRADIO_SHARE

    demo = create_app()
    demo.launch(share=share, server_name="0.0.0.0")
