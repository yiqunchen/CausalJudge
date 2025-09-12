# Development Notes

This repo historically accumulated multiple scripts per task. To reduce confusion, here are the canonical entry points and how they relate to older files.

- Canonical evaluation:
  - `scripts/run_evaluation.py` (single or multi)
  - `scripts/run_single_evaluation.py`
  - `scripts/run_multi_evaluation.py`

- Metrics recomputation and plots:
  - `src/recompute_metrics.py` (recompute from saved predictions)
  - `src/generate_final_plots_with_values.py` (publication-style plots; supports human baseline files)

- Human reviewer analysis (new):
  - `src/human_evaluation.py` (per-reviewer metrics, plots, TeX summary)
  - `src/plot_human_vs_llm.py` (scatter: best human vs best LLM per field)

Older or duplicative helpers remain for backward compatibility but are not required for the paper pipeline. Consider archiving/removing them in a follow-up once no longer referenced:

- `scripts/run_with_checkpoints.py` (similar purpose to single/multi runners)
- `synthesize_reasoning_patterns.py` (kept for qualitative analysis)

If you want me to hard-delete or move legacy files into an `archive/` folder, I can do that next.

