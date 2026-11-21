"""
Evaluation & benchmarking: run both baseline and debate on eval_set.json,
compute metrics, generate charts.

Usage:
    python -m src.evaluate                # Full 60-claim run
    python -m src.evaluate --limit 10     # Smoke test with 10 claims
"""

import argparse
import json
import os
import sys
import time
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import config
from src.schemas import EvalClaim, EvalResult, VerdictType
from src.graph import run_debate
from src.baseline import run_baseline


LABELS = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]
RESULTS_PATH = os.path.join(config.RESULTS_DIR, "eval_results.json")


# ---------------------------------------------------------------------------
# Load eval set
# ---------------------------------------------------------------------------
def load_eval_set(limit: Optional[int] = None) -> List[EvalClaim]:
    """Load evaluation claims from eval_set.json."""
    with open(config.EVAL_SET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    claims = [EvalClaim(**c) for c in data]
    if limit:
        claims = claims[:limit]
    return claims


# ---------------------------------------------------------------------------
# Checkpointing: load existing results to resume
# ---------------------------------------------------------------------------
def load_existing_results() -> dict:
    """Load previously saved results for resume support. Returns {claim_id: EvalResult}."""
    if not os.path.exists(RESULTS_PATH):
        return {}
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Handle both list and dict formats
    if isinstance(data, list):
        return {r["claim_id"]: r for r in data}
    elif isinstance(data, dict) and "results" in data:
        return {r["claim_id"]: r for r in data["results"]}
    return {}


def save_results(results: List[dict]) -> None:
    """Incrementally save results to disk."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
def run_evaluation(claims: List[EvalClaim]) -> List[dict]:
    """Run baseline + debate on all claims with checkpointing."""
    existing = load_existing_results()
    results = []

    for i, claim_obj in enumerate(claims):
        # Check if already completed
        if claim_obj.id in existing:
            result = existing[claim_obj.id]
            # Verify both systems ran
            if result.get("baseline_verdict") and result.get("debate_verdict"):
                print(f"[eval] [{i+1}/{len(claims)}] Claim {claim_obj.id}: "
                      f"resuming from checkpoint (skipped)")
                results.append(result)
                continue

        print(f"[eval] [{i+1}/{len(claims)}] Processing claim {claim_obj.id}: "
              f"{claim_obj.claim[:80]}...")

        result = {
            "claim_id": claim_obj.id,
            "claim_text": claim_obj.claim,
            "gold_label": claim_obj.label,
            "baseline_verdict": None,
            "baseline_confidence": None,
            "baseline_reasoning": None,
            "debate_verdict": None,
            "debate_confidence": None,
            "debate_reasoning": None,
            "debate_transcript": [],
            "citation_validity_rate": 0.0,
            "baseline_latency_s": 0.0,
            "debate_latency_s": 0.0,
            "retrieved_chunk_ids": [],
        }

        # --- Run Debate first (to get evidence for baseline) ---
        try:
            t0 = time.time()
            debate_state = run_debate(claim_obj.claim)
            debate_latency = time.time() - t0

            verdict = debate_state.get("verdict", {})
            result["debate_verdict"] = verdict.get("verdict") if verdict else None
            result["debate_confidence"] = verdict.get("confidence") if verdict else None
            result["debate_reasoning"] = verdict.get("reasoning") if verdict else None
            result["debate_transcript"] = debate_state.get("transcript", [])
            result["citation_validity_rate"] = debate_state.get("citation_validity_rate", 0.0)
            result["debate_latency_s"] = round(debate_latency, 2)
            result["retrieved_chunk_ids"] = [
                c.get("chunk_id", "") if isinstance(c, dict) else c.chunk_id
                for c in debate_state.get("evidence", [])
            ]

            # --- Run Baseline with same evidence ---
            evidence = debate_state.get("evidence", [])
            t1 = time.time()
            baseline_result = run_baseline(claim_obj.claim, evidence)
            baseline_latency = time.time() - t1

            result["baseline_verdict"] = baseline_result.verdict
            result["baseline_confidence"] = baseline_result.confidence
            result["baseline_reasoning"] = baseline_result.reasoning
            result["baseline_latency_s"] = round(baseline_latency, 2)

        except Exception as e:
            print(f"[eval] ⚠ Error on claim {claim_obj.id}: {e}")
            import traceback
            traceback.print_exc()

        results.append(result)

        # Save after each claim (checkpointing)
        save_results(results)
        print(f"[eval]   → Baseline: {result['baseline_verdict']} | "
              f"Debate: {result['debate_verdict']} | "
              f"Gold: {claim_obj.label}")

    return results


# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------
def compute_metrics(results: List[dict]) -> dict:
    """Compute accuracy, F1, citation validity, and latency metrics."""
    # Filter results with both verdicts
    valid = [r for r in results
             if r.get("baseline_verdict") in LABELS
             and r.get("debate_verdict") in LABELS]

    if not valid:
        print("[eval] ⚠ No valid results to compute metrics.")
        return {}

    gold = [r["gold_label"] for r in valid]
    baseline_preds = [r["baseline_verdict"] for r in valid]
    debate_preds = [r["debate_verdict"] for r in valid]

    metrics = {
        "n_claims": len(valid),
        "baseline_accuracy": accuracy_score(gold, baseline_preds),
        "debate_accuracy": accuracy_score(gold, debate_preds),
        "baseline_macro_f1": f1_score(gold, baseline_preds, labels=LABELS, average="macro"),
        "debate_macro_f1": f1_score(gold, debate_preds, labels=LABELS, average="macro"),
        "baseline_per_class_f1": dict(zip(
            LABELS,
            f1_score(gold, baseline_preds, labels=LABELS, average=None).tolist(),
        )),
        "debate_per_class_f1": dict(zip(
            LABELS,
            f1_score(gold, debate_preds, labels=LABELS, average=None).tolist(),
        )),
        "avg_citation_validity": sum(r.get("citation_validity_rate", 0) for r in valid) / len(valid),
        "avg_baseline_latency_s": sum(r.get("baseline_latency_s", 0) for r in valid) / len(valid),
        "avg_debate_latency_s": sum(r.get("debate_latency_s", 0) for r in valid) / len(valid),
    }

    # Disagreement analysis
    debate_wins = []
    baseline_wins = []
    for r in valid:
        bl_correct = r["baseline_verdict"] == r["gold_label"]
        db_correct = r["debate_verdict"] == r["gold_label"]
        if db_correct and not bl_correct:
            debate_wins.append({
                "claim_id": r["claim_id"],
                "claim": r["claim_text"],
                "gold": r["gold_label"],
                "baseline_pred": r["baseline_verdict"],
                "debate_pred": r["debate_verdict"],
            })
        elif bl_correct and not db_correct:
            baseline_wins.append({
                "claim_id": r["claim_id"],
                "claim": r["claim_text"],
                "gold": r["gold_label"],
                "baseline_pred": r["baseline_verdict"],
                "debate_pred": r["debate_verdict"],
            })

    metrics["debate_wins"] = debate_wins
    metrics["baseline_wins"] = baseline_wins
    metrics["debate_win_count"] = len(debate_wins)
    metrics["baseline_win_count"] = len(baseline_wins)

    return metrics


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------
def generate_charts(results: List[dict], metrics: dict) -> None:
    """Generate and save evaluation charts."""
    os.makedirs(config.CHARTS_DIR, exist_ok=True)

    valid = [r for r in results
             if r.get("baseline_verdict") in LABELS
             and r.get("debate_verdict") in LABELS]

    if not valid:
        print("[eval] ⚠ No valid results to generate charts.")
        return

    gold = [r["gold_label"] for r in valid]
    baseline_preds = [r["baseline_verdict"] for r in valid]
    debate_preds = [r["debate_verdict"] for r in valid]

    sns.set_theme(style="whitegrid")

    # --- Chart 1: Accuracy & Macro-F1 Comparison ---
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    width = 0.35
    baseline_vals = [metrics["baseline_accuracy"], metrics["baseline_macro_f1"]]
    debate_vals = [metrics["debate_accuracy"], metrics["debate_macro_f1"]]

    bars1 = ax.bar(x - width/2, baseline_vals, width, label="Baseline (Single-LLM)", color="#5B9BD5")
    bars2 = ax.bar(x + width/2, debate_vals, width, label="Multi-Agent Debate", color="#70AD47")

    ax.set_ylabel("Score")
    ax.set_title("Accuracy & Macro-F1: Baseline vs. Multi-Agent Debate")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Macro-F1"])
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", fontsize=9)
    ax.bar_label(bars2, fmt="%.3f", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "accuracy_f1_comparison.png"), dpi=150)
    plt.close()

    # --- Chart 2: Per-Class F1 ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(LABELS))
    bl_f1 = [metrics["baseline_per_class_f1"][l] for l in LABELS]
    db_f1 = [metrics["debate_per_class_f1"][l] for l in LABELS]

    bars1 = ax.bar(x - width/2, bl_f1, width, label="Baseline", color="#5B9BD5")
    bars2 = ax.bar(x + width/2, db_f1, width, label="Multi-Agent Debate", color="#70AD47")

    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1: Baseline vs. Multi-Agent Debate")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", fontsize=9)
    ax.bar_label(bars2, fmt="%.3f", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "per_class_f1.png"), dpi=150)
    plt.close()

    # --- Chart 3: Side-by-Side Confusion Matrices ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, preds, title in [
        (axes[0], baseline_preds, "Baseline (Single-LLM)"),
        (axes[1], debate_preds, "Multi-Agent Debate"),
    ]:
        cm = confusion_matrix(gold, preds, labels=LABELS)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=LABELS, yticklabels=LABELS, ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True Label")

    plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHARTS_DIR, "confusion_matrices.png"), dpi=150)
    plt.close()

    print(f"[eval] ✅ Charts saved to {config.CHARTS_DIR}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate Debate vs Baseline")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of claims (for smoke testing)")
    args = parser.parse_args()

    print("[eval] Loading evaluation set ...")
    claims = load_eval_set(limit=args.limit)
    print(f"[eval] Running on {len(claims)} claims ...")

    # Run evaluation
    results = run_evaluation(claims)

    # Compute metrics
    metrics = compute_metrics(results)

    if metrics:
        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Claims evaluated: {metrics['n_claims']}")
        print(f"\n{'Metric':<25} {'Baseline':>10} {'Debate':>10}")
        print("-" * 50)
        print(f"{'Accuracy':<25} {metrics['baseline_accuracy']:>10.3f} {metrics['debate_accuracy']:>10.3f}")
        print(f"{'Macro-F1':<25} {metrics['baseline_macro_f1']:>10.3f} {metrics['debate_macro_f1']:>10.3f}")
        for label in LABELS:
            bl = metrics['baseline_per_class_f1'][label]
            db = metrics['debate_per_class_f1'][label]
            print(f"  F1 {label:<20} {bl:>10.3f} {db:>10.3f}")
        print(f"\n{'Avg Citation Validity':<25} {metrics['avg_citation_validity']:>10.3f}")
        print(f"{'Avg Baseline Latency (s)':<25} {metrics['avg_baseline_latency_s']:>10.2f}")
        print(f"{'Avg Debate Latency (s)':<25} {metrics['avg_debate_latency_s']:>10.2f}")
        print(f"\nDebate wins (corrected baseline errors): {metrics['debate_win_count']}")
        print(f"Baseline wins (debate regression): {metrics['baseline_win_count']}")
        print("=" * 60)

        # Save metrics alongside results
        final_output = {
            "results": results,
            "metrics": {k: v for k, v in metrics.items()
                       if k not in ("debate_wins", "baseline_wins")},
            "debate_wins": metrics.get("debate_wins", []),
            "baseline_wins": metrics.get("baseline_wins", []),
        }
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2)

        # Generate charts
        generate_charts(results, metrics)
    else:
        print("[eval] ⚠ No metrics could be computed.")

    print("[eval] ✅ Evaluation complete!")


if __name__ == "__main__":
    main()
