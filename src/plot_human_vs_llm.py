#!/usr/bin/env python3
"""
Plot best human vs best LLM per field for Accuracy and F1.

Creates two scatter plots with a dashed y=x line:
 - figures/humans/human_vs_llm_accuracy.(png|pdf)
 - figures/humans/human_vs_llm_f1.(png|pdf)

Assumptions:
- Human metrics are in results/metrics_human_*.json (produced by src/human_evaluation.py)
- LLM metrics are in results/metrics_*.json (excluding human files)
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Dict, Any, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def load_human_metrics() -> List[Dict[str, Any]]:
    files = sorted(glob.glob("results/metrics_human_*.json"))
    out = []
    for fp in files:
        try:
            with open(fp, "r") as f:
                out.append(json.load(f))
        except Exception:
            pass
    return out


def load_llm_metrics(allowed_models: List[str] | None = None) -> List[Dict[str, Any]]:
    files = sorted(glob.glob("results/metrics_*.json"))
    out = []
    for fp in files:
        if "metrics_human_" in fp:
            continue
        try:
            with open(fp, "r") as f:
                data = json.load(f)
                model_name = data.get('model') or ''
                # Filter by allowed models if provided
                if allowed_models is not None and model_name not in allowed_models:
                    continue
                # Some files store metrics under key 'metrics'
                out.append(data if 'metrics' in data else {'metrics': data, 'model': model_name})
        except Exception:
            pass
    return out


def best_by_field(metric_name: str, bundles: List[Dict[str, Any]]) -> Dict[str, float]:
    """Return {field: best_value} across a list of metric bundles having .['metrics'][field][metric_name]."""
    best: Dict[str, float] = {f: 0.0 for f in BINARY_FIELDS}
    for bundle in bundles:
        md = bundle.get('metrics', {})
        for f in BINARY_FIELDS:
            if f in md and isinstance(md[f], dict) and metric_name in md[f]:
                try:
                    v = float(md[f][metric_name])
                except Exception:
                    v = 0.0
                if v > best[f]:
                    best[f] = v
    return best

def avg_by_field(metric_name: str, bundles: List[Dict[str, Any]]) -> Dict[str, float]:
    """Return {field: average_value} across all bundles."""
    sums: Dict[str, float] = {f: 0.0 for f in BINARY_FIELDS}
    counts: Dict[str, int] = {f: 0 for f in BINARY_FIELDS}
    for bundle in bundles:
        md = bundle.get('metrics', {})
        for f in BINARY_FIELDS:
            if f in md and isinstance(md[f], dict) and metric_name in md[f]:
                try:
                    v = float(md[f][metric_name])
                except Exception:
                    continue
                sums[f] += v
                counts[f] += 1
    return {f: (sums[f] / counts[f] if counts[f] > 0 else 0.0) for f in BINARY_FIELDS}

def worst_by_field(metric_name: str, bundles: List[Dict[str, Any]]) -> Dict[str, float]:
    """Return {field: min_value} across bundles (useful for worst human)."""
    worst: Dict[str, float] = {f: 1.0 for f in BINARY_FIELDS}
    seen: Dict[str, bool] = {f: False for f in BINARY_FIELDS}
    for bundle in bundles:
        md = bundle.get('metrics', {})
        for f in BINARY_FIELDS:
            if f in md and isinstance(md[f], dict) and metric_name in md[f]:
                try:
                    v = float(md[f][metric_name])
                except Exception:
                    continue
                seen[f] = True
                if v < worst[f]:
                    worst[f] = v
    for f in BINARY_FIELDS:
        if not seen[f]:
            worst[f] = 0.0
    return worst

def shorten(field: str) -> str:
    mapping = {
        "Randomized Exposure": "Randomized",
        "Causal Mediation": "Causal Med",
        "Examined Mediator-Outcome Linearity": "Linearity",
        "Examined Exposure-Mediator Interaction": "Interaction",
        "Covariates in Exposure-Mediator Model": "Cov Exp→Med",
        "Covariates in Exposure-Outcome Model": "Cov Exp→Out",
        "Covariates in Mediator-Outcome Model": "Cov Med→Out",
        "Control for Baseline Mediator": "Baseline Med",
        "Control for Baseline Outcome": "Baseline Out",
        "Temporal Ordering Exposure Before Mediator": "Time Exp→Med",
        "Temporal Ordering Mediator Before Outcome": "Time Med→Out",
        "Discussed Mediator Assumptions": "Assumptions",
        "Sensitivity Analysis to Assumption": "Sensitivity",
        "Control for Other Post-Exposure Variables": "Post-exposure",
    }
    return mapping.get(field, field)


def plot_scatter(
    hvals: Dict[str, float],
    lvals: Dict[str, float],
    metric: str,
    out_dir: Path,
    *,
    title_prefix: str = "Best Human vs Best LLM",
    fname_suffix: str = "",
    x_label: str | None = None,
    y_label: str | None = None,
    show_diag: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    xs = []
    ys = []
    labels = []
    rows = []
    for f in BINARY_FIELDS:
        hv = hvals.get(f, 0.0)
        lv = lvals.get(f, 0.0)
        xs.append(hv)
        ys.append(lv)
        lbl = shorten(f)
        labels.append(lbl)
        rows.append((f, lbl, hv, lv))

    plt.figure(figsize=(7.5, 6.2))
    plt.scatter(xs, ys, c="#1f77b4", edgecolors="black", s=40)
    ax = plt.gca()
    # Aesthetics similar to final plots
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
    # Place labels with simple force-directed repulsion and leader lines
    import math
    xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
    # initial positions: slight offset
    pos = [(x + 0.015, y + 0.015) for x, y in zip(xs, ys)]
    def clamp(p):
        px = min(max(xmin, p[0]), xmax)
        py = min(max(ymin, p[1]), ymax)
        return (px, py)
    # Iterative repulsion between labels
    for _ in range(120):
        moved = False
        for i in range(len(pos)):
            fx = fy = 0.0
            xi, yi = pos[i]
            # repel from other labels
            for j in range(len(pos)):
                if i == j: continue
                xj, yj = pos[j]
                dx = xi - xj; dy = yi - yj
                d2 = dx*dx + dy*dy + 1e-9
                d = math.sqrt(d2)
                # target min spacing
                r0 = 0.04
                if d < r0:
                    k = (r0 - d) / r0
                    fx += (dx / d) * k * 0.02
                    fy += (dy / d) * k * 0.02
            # repel from its point a bit so text isn't on the marker
            dxp = xi - xs[i]; dyp = yi - ys[i]
            dp2 = dxp*dxp + dyp*dyp + 1e-9
            dp = math.sqrt(dp2)
            rp = 0.02
            if dp < rp:
                k = (rp - dp) / rp
                fx += (dxp / dp) * k * 0.02
                fy += (dyp / dp) * k * 0.02
            # boundary push
            margin = 0.005
            if xi - xmin < margin: fx += 0.01
            if xmax - xi < margin: fx -= 0.01
            if yi - ymin < margin: fy += 0.01
            if ymax - yi < margin: fy -= 0.01
            if abs(fx) > 1e-6 or abs(fy) > 1e-6:
                xi_new = xi + fx
                yi_new = yi + fy
                pos[i] = clamp((xi_new, yi_new))
                moved = True
        if not moved:
            break
    # Draw texts with leader lines (bias labels slightly left)
    left_bias = 0.01
    for (x, y, lbl), (tx, ty) in zip(zip(xs, ys, labels), pos):
        # apply left bias and clamp
        tx = max(xmin, min(xmax, tx - left_bias))
        # leader line only if displaced
        if abs(tx - x) > 1e-3 or abs(ty - y) > 1e-3:
            ax.plot([x, tx], [y, ty], color='gray', linewidth=0.6, alpha=0.6, zorder=0)
        ax.text(tx, ty, lbl, fontsize=7.5, ha='left', va='bottom', clip_on=True,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.5))
    # Auto-zoom axes independently with small padding within [0,1]
    if xs and ys:
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        pad_x = max(0.02, 0.05 * (xmax - xmin + 1e-6))
        pad_y = max(0.02, 0.05 * (ymax - ymin + 1e-6))
        x_hi = xmax + pad_x
        y_hi = ymax + pad_y
        # Allow slight extension above 1.0 for aesthetics when values are at ceiling
        if x_hi > 0.98:
            x_hi = min(1.08, x_hi)
        if y_hi > 0.98:
            y_hi = min(1.08, y_hi)
        plt.xlim(max(0.0, xmin - pad_x), x_hi)
        plt.ylim(max(0.0, ymin - pad_y), y_hi)
    # Labels
    if x_label is None:
        x_label = f"Human {metric.capitalize()}"
    if y_label is None:
        y_label = f"LLM {metric.capitalize()}"
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # y=x diagonal for average vs average view only (optional)
    if show_diag:
        xmin_c, xmax_c = ax.get_xlim(); ymin_c, ymax_c = ax.get_ylim()
        lo = max(min(xmin_c, xmax_c), min(ymin_c, ymax_c))
        hi = min(max(xmin_c, xmax_c), max(ymin_c, ymax_c))
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1.0, alpha=0.8)
    # Pearson correlation r
    if len(xs) > 1 and len(ys) > 1:
        try:
            import numpy as np
            r = np.corrcoef(xs, ys)[0, 1]
            plt.title(f"{title_prefix} by Field ({metric.capitalize()})  r={r:.2f}")
        except Exception:
            plt.title(f"{title_prefix} by Field ({metric.capitalize()})")
    else:
        plt.title(f"{title_prefix} by Field ({metric.capitalize()})")
    plt.tight_layout()
    suffix = f"_{fname_suffix}" if fname_suffix else ""
    for ext in ("png", "pdf"):
        plt.savefig(out_dir / f"human_vs_llm_{metric}{suffix}.{ext}", bbox_inches="tight", dpi=300)
    plt.close()

    # Save plotted points for verification
    try:
        import csv
        points_dir = Path('results')
        points_dir.mkdir(parents=True, exist_ok=True)
        with (points_dir / f"human_vs_llm_points_{metric}{suffix}.csv").open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['field', 'label', 'human_value', 'llm_value'])
            for r in rows:
                w.writerow(r)
    except Exception:
        pass


def main():
    base = Path('.')
    fig_dir = base / 'figures' / 'humans'

    human = load_human_metrics()
    # Restrict LLMs to GPT-5 and O3 for clean comparison
    llm = load_llm_metrics(allowed_models=["gpt-5-2025-08-07", "o3-2025-04-16"]) 
    if not human or not llm:
        print("No human or LLM metrics found. Run human_evaluation.py and recompute_metrics first.")
        return

    for metric in ("accuracy", "f1"):
        # Best vs Best (restricted to GPT-5 and O3)
        best_h = best_by_field(metric, human)
        best_l = best_by_field(metric, llm)
        plot_scatter(
            best_h, best_l, metric, fig_dir,
            title_prefix="Best Human vs Best LLM",
            fname_suffix="",
            x_label=f"Best Human {metric.capitalize()}",
            y_label=f"Best LLM {metric.capitalize()}",
            show_diag=False,
        )
        print(f"Saved scatter (best) for {metric} at {fig_dir}")
        # Average vs Average
        avg_h = avg_by_field(metric, human)
        avg_l = avg_by_field(metric, llm)
        plot_scatter(
            avg_h, avg_l, metric, fig_dir,
            title_prefix="Average Human vs Average LLM",
            fname_suffix="avg",
            x_label=f"Avg Human {metric.capitalize()}",
            y_label=f"Avg LLM (GPT-5+O3) {metric.capitalize()}",
            show_diag=True,
        )
        print(f"Saved scatter (avg) for {metric} at {fig_dir}")
        # Worst human vs Average LLM
        worst_h = worst_by_field(metric, human)
        plot_scatter(
            worst_h, avg_l, metric, fig_dir,
            title_prefix="Worst Human vs Average LLM",
            fname_suffix="worst_vs_avg",
            x_label=f"Worst Human {metric.capitalize()}",
            y_label=f"Avg LLM (GPT-5+O3) {metric.capitalize()}",
            show_diag=False,
        )
        print(f"Saved scatter (worst vs avg) for {metric} at {fig_dir}")
        # Worst human vs Best LLM
        plot_scatter(
            worst_h, best_l, metric, fig_dir,
            title_prefix="Worst Human vs Best LLM",
            fname_suffix="worst_vs_best",
            x_label=f"Worst Human {metric.capitalize()}",
            y_label=f"Best LLM {metric.capitalize()}",
            show_diag=False,
        )
        print(f"Saved scatter (worst vs best) for {metric} at {fig_dir}")


if __name__ == '__main__':
    main()
