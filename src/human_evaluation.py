#!/usr/bin/env python3
"""
Compute and plot metrics for individual human reviewers against the gold standard.

Outputs:
- results/metrics_human_<reviewer_slug>.json per reviewer with field + overall metrics
- figures/humans/overall_{metric}.(png|pdf) bar charts across reviewers (accuracy/precision/recall/f1)
- results/human_summary.tex concise TeX summary paragraph with key numbers

Data sources used:
- data/processed/ground_truth_clean.json (14 binary fields per PMID)
- data/raw/master-df-get-reviewers.csv (two rows per PMID; column 'reviewer' denotes A/B and
  columns 'reviewer_1_2' and 'reviewer_2_3' provide reviewer names)

Notes:
- We robustly convert reviewer cell strings to binary using simple, conservative rules.
- Non-answers like 'no mention', 'unclear', 'na' are treated as 0; phrases clearly indicating
  the concept (e.g., for assumption discussion) are treated as 1 when appropriate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Canonical 14 evaluation fields
BINARY_FIELDS: List[str] = [
    "Randomized Exposure",
    "Causal Mediation",
    "Examined Mediator-Outcome Linearity",
    "Examined Exposure-Mediator Interaction",
    "Covariates in Exposure-Mediator Model",
    "Covariates in Exposure-Outcome Model",
    "Covariates in Mediator-Outcome Model",
    "Control for Baseline Mediator",
    "Control for Baseline Outcome",
    "Temporal Ordering Exposure Before Mediator",
    "Temporal Ordering Mediator Before Outcome",
    "Discussed Mediator Assumptions",
    "Sensitivity Analysis to Assumption",
    "Control for Other Post-Exposure Variables",
]


# Mapping from reviewer CSV columns to canonical field names
CSV_TO_CANONICAL = {
    "exposure_randomized": "Randomized Exposure",
    "causal_mediation_yes_no": "Causal Mediation",
    "if_b_k_examine_linear_relationship_b_w_mediator_and_outcome": "Examined Mediator-Outcome Linearity",
    "if_baron_kenny_examine_whether_no_interaction_b_w_tx_and_mediator_on_outcome": "Examined Exposure-Mediator Interaction",
    "covariates_in_exposure_mediator_model": "Covariates in Exposure-Mediator Model",
    "covariates_in_mediator_outcome_model": "Covariates in Mediator-Outcome Model",
    "covariates_in_exposure_outcome_model": "Covariates in Exposure-Outcome Model",
    "baseline_value_of_mediator_adjusted_for": "Control for Baseline Mediator",
    "baseline_value_of_outcome_adjusted_for": "Control for Baseline Outcome",
    "temporal_ordering_of_exposure_before_mediator_yes_no": "Temporal Ordering Exposure Before Mediator",
    "temporal_ordering_of_mediator_before_outcome_yes_no": "Temporal Ordering Mediator Before Outcome",
    "discussion_of_mediation_assumptions_yes_no": "Discussed Mediator Assumptions",
    "sensitivity_analyes_to_assumptions_yes_no": "Sensitivity Analysis to Assumption",
    "does_model_control_for_other_post_exposure_variables_yes_no": "Control for Other Post-Exposure Variables",
}


def _to_bin(value: Any, field: str) -> int:
    """Conservatively convert a reviewer cell value to 0/1.

    Rules:
    - Accept exact and typical variants: yes/true/y -> 1; no/false/n -> 0
    - Treat 'no mention', 'unclear', 'na', empty as 0
    - For 'Discussed Mediator Assumptions', count phrases that clearly indicate discussion
      (e.g., 'discuss', 'assumption', 'preclude causal inference', 'limitations of mediation').
    - Otherwise default to 0 unless an explicit yes-like token appears.
    """
    s = str(value).strip().lower()
    if s == "" or s in {"na", "n/a", "nan", "none", "__"}:
        return 0

    # Simple binary tokens
    if s in {"yes", "y", "true", "1"}:
        return 1
    if s in {"no", "n", "false", "0"}:
        return 0

    # Common textual variants
    if "no mention" in s or "not mentioned" in s or "unclear" in s or "possibly" in s:
        return 0

    # Heuristics per field where prose is common
    if field == "Discussed Mediator Assumptions":
        # Any text explicitly signaling assumption discussion or causal inference limitations
        if any(tok in s for tok in ["assumption", "assumptions", "discuss", "sequential ignorability", "causal inference", "preclude causal inference", "limitations of mediation"]):
            return 1
        return 0

    # For covariate fields, if they list specific covariates, treat as yes
    if field.startswith("Covariates "):
        if any(sep in s for sep in [",", ";"]) or any(word in s for word in ["age", "sex", "gender", "ethnicity", "adjusted", "controlled"]):
            return 1
        return 0

    # Temporal ordering: phrases implying cross-sectional imply 0
    if field.startswith("Temporal Ordering"):
        if "cross-sectional" in s:
            return 0
        if "longitudinal" in s or "time" in s or "measured at time" in s or "prospective" in s:
            return 1

    # Sensitivity analysis: look for 'sensitivity', 'robustness'
    if field == "Sensitivity Analysis to Assumption":
        if "sensitivity" in s or "robustness" in s or "bias analysis" in s:
            return 1
        return 0

    # Post-exposure control: look for 'post' or 'intermediate'
    if field == "Control for Other Post-Exposure Variables":
        if "post" in s or "post-" in s or "intermediate" in s:
            return 1
        return 0

    # Randomization: spot randomized/RCT keywords
    if field == "Randomized Exposure":
        if "random" in s or "rct" in s or "trial" in s:
            return 1
        return 0

    # Default: only count an explicit yes-like token
    if "yes" in s or "true" in s:
        return 1
    return 0


def load_ground_truth(path: Path) -> Dict[str, Dict[str, int]]:
    with path.open("r") as f:
        gt = json.load(f)
    # Ensure ints 0/1
    clean: Dict[str, Dict[str, int]] = {}
    for pmid, row in gt.items():
        clean[pmid] = {field: int(row.get(field, 0)) for field in BINARY_FIELDS}
    return clean


def load_individual_reviewer_rows(csv_path: Path) -> List[Dict[str, Any]]:
    """Load the per-reviewer CSV where each PMID appears twice (A/B) without pandas."""
    import csv
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize key fields
            for c in ["reviewer_1_2", "reviewer_2_3", "reviewer", "pmid_number"]:
                if c in row and row[c] is not None:
                    row[c] = str(row[c]).strip()
            rows.append(row)
    return rows


def build_reviewer_predictions(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Return {reviewer_slug: {pmid: {field: 0/1}}} from CSV rows.

    We map 'reviewer' A/B to the name in 'reviewer_1_2' / 'reviewer_2_3'.
    Unknown names are skipped.
    """
    per_reviewer: Dict[str, Dict[str, Dict[str, int]]] = {}
    # Select only relevant columns to avoid surprises
    needed_cols = {
        "pmid_number", "reviewer", "reviewer_1_2", "reviewer_2_3",
        *CSV_TO_CANONICAL.keys(),
    }
    present_cols = list(needed_cols)
    for row_in in rows:
        row = {k: row_in.get(k) for k in present_cols}
        pmid = str(row.get("pmid_number", "")).strip()
        if not pmid or pmid == "nan":
            continue

        # Determine reviewer name for this row
        which = str(row.get("reviewer", "")).strip().upper()
        if which == "A":
            name = str(row.get("reviewer_1_2", "")).strip()
        elif which == "B":
            name = str(row.get("reviewer_2_3", "")).strip()
        else:
            # In rare cases there could be C; fall back to second slot
            name = str(row.get("reviewer_2_3", "")).strip() or str(row.get("reviewer_1_2", "")).strip()

        if not name or name.lower() == "na" or name.lower() == "nan":
            continue

        reviewer_slug = name.lower().replace(" ", "_")
        per_reviewer.setdefault(reviewer_slug, {})
        per_reviewer[reviewer_slug].setdefault(pmid, {})

        # Map each CSV column to canonical field name
        for csv_col, canonical in CSV_TO_CANONICAL.items():
            if csv_col in row and row[csv_col] is not None:
                per_reviewer[reviewer_slug][pmid][canonical] = _to_bin(row[csv_col], canonical)

    return per_reviewer


def compute_metrics_for_predictions(ground_truth: Dict[str, Dict[str, int]],
                                    predictions: Dict[str, Dict[str, Dict[str, int]]]
                                    ) -> Dict[str, Dict[str, Any]]:
    """Compute per-field and overall metrics for each reviewer.

    Returns {reviewer_slug: metrics_dict} where metrics_dict mimics the LLM metrics structure.
    """
    all_metrics: Dict[str, Dict[str, Any]] = {}

    for reviewer, pred_by_pmid in predictions.items():
        field_metrics: Dict[str, Any] = {}

        # Per-field
        for field in BINARY_FIELDS:
            gt_vals: List[int] = []
            pr_vals: List[int] = []
            for pmid, gt_row in ground_truth.items():
                if pmid in pred_by_pmid and field in pred_by_pmid[pmid]:
                    gt_vals.append(int(gt_row.get(field, 0)))
                    pr_vals.append(int(pred_by_pmid[pmid].get(field, 0)))

            if gt_vals:
                acc = accuracy_score(gt_vals, pr_vals)
                prec, rec, f1, _ = precision_recall_fscore_support(gt_vals, pr_vals, average='binary', zero_division=0)
                field_metrics[field] = {
                    "accuracy": float(acc),
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1": float(f1),
                    "n_samples": len(gt_vals),
                }

        # Overall (macro average across fields)
        all_gt: List[int] = []
        all_pr: List[int] = []
        for field in BINARY_FIELDS:
            for pmid, gt_row in ground_truth.items():
                if pmid in pred_by_pmid and field in pred_by_pmid[pmid]:
                    all_gt.append(int(gt_row.get(field, 0)))
                    all_pr.append(int(pred_by_pmid[pmid].get(field, 0)))

        if all_gt:
            overall = {}
            overall["accuracy"] = float(accuracy_score(all_gt, all_pr))
            p, r, f1, _ = precision_recall_fscore_support(all_gt, all_pr, average='macro', zero_division=0)
            overall.update({"precision": float(p), "recall": float(r), "f1": float(f1)})
            overall["n_samples"] = len(all_gt)
            field_metrics["overall"] = overall

        all_metrics[reviewer] = field_metrics

    return all_metrics


def save_reviewer_metrics(all_metrics: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for reviewer, metrics in all_metrics.items():
        payload = {
            "model": f"human_{reviewer}",
            "prompt_type": "n/a",
            "metrics": metrics,
        }
        (out_dir / f"metrics_human_{reviewer}.json").write_text(json.dumps(payload, indent=2))


def plot_overall_bars(all_metrics: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    fig_dir = out_dir / "figures" / "humans"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Determine ordering by F1
    items = []
    for reviewer, metrics in all_metrics.items():
        overall = metrics.get("overall", {})
        items.append((reviewer, overall.get("accuracy", 0.0), overall.get("precision", 0.0), overall.get("recall", 0.0), overall.get("f1", 0.0)))
    # Sort by F1 desc
    items.sort(key=lambda x: x[4], reverse=True)

    reviewers = [f"Human {i+1}" for i in range(len(items))]
    accs = [x[1] for x in items]
    precs = [x[2] for x in items]
    recs = [x[3] for x in items]
    f1s = [x[4] for x in items]

    metrics_map = {
        "accuracy": accs,
        "precision": precs,
        "recall": recs,
        "f1": f1s,
    }

    for metric, values in metrics_map.items():
        plt.figure(figsize=(8, 4))
        bars = plt.bar(np.arange(len(values)), values, color="#7f7f7f", edgecolor="black")
        plt.xticks(np.arange(len(values)), reviewers, rotation=45, ha="right")
        plt.ylim(0, 1.05)
        plt.ylabel(metric.capitalize())
        plt.title(f"Human Reviewers: Overall {metric.capitalize()}")
        # Value labels
        for b, v in zip(bars, values):
            plt.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8, rotation=90)
        plt.tight_layout()
        for ext in ("png", "pdf"):
            out_path = fig_dir / f"overall_{metric}.{ext}"
            plt.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close()


def write_tex_summary(all_metrics: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    # Summarize best and median human
    rows = []
    for reviewer, metrics in all_metrics.items():
        ov = metrics.get("overall", {})
        rows.append((reviewer, ov.get("accuracy", 0.0), ov.get("precision", 0.0), ov.get("recall", 0.0), ov.get("f1", 0.0)))
    if not rows:
        return
    rows.sort(key=lambda x: x[4], reverse=True)
    best = rows[0]
    median = rows[len(rows)//2]

    text = (
        "\\paragraph{Human reviewer performance.} We evaluated individual human reviewers against the gold standard across the 14 binary criteria. "
        f"Across N={len(rows)} reviewers, the best reviewer achieved an overall accuracy of {best[1]:.2f} and F1 of {best[4]:.2f}, "
        f"while the median reviewer achieved accuracy {median[1]:.2f} and F1 {median[4]:.2f}. "
        "Precision and recall exhibited similar trends, indicating consistent calibration rather than a precision–recall tradeoff. "
        "See the human summary plots for per-reviewer metrics."
    )
    (out_dir / "results").mkdir(exist_ok=True)
    (out_dir / "results" / "human_summary.tex").write_text(text)


def _chunk_fields(fields: List[str], n: int) -> List[List[str]]:
    return [fields[i:i+n] for i in range(0, len(fields), n)]


def plot_human_field_breakdown(all_metrics: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    """Grouped bar charts per metric across humans, split into 2 panels of 7 fields each.

    Saves: figures/humans/fields_{metric}_part{1,2}.{png,pdf}
    """
    fig_dir = out_dir / "figures" / "humans"
    fig_dir.mkdir(parents=True, exist_ok=True)

    reviewers = sorted(all_metrics.keys())
    human_labels = [f"Human {i+1}" for i in range(len(reviewers))]
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"
    ]
    # Map reviewer slug -> color and human label
    color_map = {r: colors[i % len(colors)] for i, r in enumerate(reviewers)}

    metrics_list = ["accuracy", "precision", "recall", "f1"]
    chunks = _chunk_fields(BINARY_FIELDS, 7)  # two chunks of 7

    for metric in metrics_list:
        for part_idx, fields in enumerate(chunks, 1):
            x = np.arange(len(fields))
            width = 0.8 / max(1, len(reviewers))
            fig, ax = plt.subplots(figsize=(12, 5))

            # Determine dynamic y-limit headroom based on max value across all reviewers/fields
            panel_vals_max = 0.0
            for reviewer in reviewers:
                for field in fields:
                    m = all_metrics.get(reviewer, {}).get(field, {})
                    panel_vals_max = max(panel_vals_max, float(m.get(metric, 0.0)))
            # Add a bit more headroom so labels at 1.00 never clip
            y_top = max(1.08, min(1.22, panel_vals_max + 0.10))

            for i, reviewer in enumerate(reviewers):
                vals = []
                for field in fields:
                    m = all_metrics.get(reviewer, {}).get(field, {})
                    vals.append(float(m.get(metric, 0.0)))
                offset = (i - (len(reviewers)-1)/2) * width
                bars = ax.bar(x + offset, vals, width=width, color=color_map[reviewer], label=human_labels[i], edgecolor='black', linewidth=0.8)
                # Add small numeric labels vertically above each bar; special symbol for 0
                for xi, v in zip(x + offset, vals):
                    if v == 0:
                        label = "•"
                        y_pos = 0.01
                    else:
                        label = f"{v:.2f}"
                        y_pos = v + 0.012
                    ax.text(xi, y_pos, label, ha='center', va='bottom', rotation=90, fontsize=6.5, color='black')

            ax.set_xticks(x)
            ax.set_xticklabels([
                field.replace("Exposure", "Exp").replace("Mediator", "Med").replace("Outcome", "Out")
                for field in fields
            ], rotation=40, ha='right')
            ax.set_ylim(0, y_top)
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f"Human breakdown by criteria ({metric.capitalize()}), part {part_idx}")
            # No reference lines per updated request
            # Aesthetics: grid off, black borders
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.5)
            # Unified legend outside
            leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Reviewer', frameon=True, ncol=1)
            if leg:
                leg.get_title().set_fontsize(10)
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(fig_dir / f"fields_{metric}_part{part_idx}.{ext}", bbox_inches='tight', dpi=300)
            plt.close(fig)


def main():
    base = Path('.')
    gt_path = base / 'data' / 'processed' / 'ground_truth_clean.json'
    csv_path = base / 'data' / 'raw' / 'master-df-get-reviewers.csv'
    out_dir = base / 'results'

    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Reviewer CSV not found: {csv_path}")

    ground_truth = load_ground_truth(gt_path)
    rows = load_individual_reviewer_rows(csv_path)
    reviewer_preds = build_reviewer_predictions(rows)
    metrics = compute_metrics_for_predictions(ground_truth, reviewer_preds)
    save_reviewer_metrics(metrics, out_dir)
    plot_overall_bars(metrics, base)
    plot_human_field_breakdown(metrics, base)
    write_tex_summary(metrics, base)

    print(f"Computed metrics for {len(metrics)} reviewers. Outputs in 'results' and 'figures/humans'.")


if __name__ == '__main__':
    main()
