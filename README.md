# Evaluating LLM Performance on Methods Review for Causal Mediation Analysis

A comprehensive evaluation framework for assessing large language models' ability to extract causal mediation information from scientific literature.

![Prior Work Compression](figures/compress_prior_work.drawio.png)

## Quick Start

### End-to-End
```bash
# 1) Verify data and environment
python scripts/test_setup.py

# 2) Run a single evaluation (writes predictions + metrics under results/)
python scripts/run_single_evaluation.py --model gpt-4o --prompt_type detailed

# 3) Generate publication-style plots from metrics
python src/generate_final_plots_with_values.py
```

### Main Evaluation Commands
```bash
# Optional: Set API key (if attempting to run locally)
export OPENAI_API_KEY="your-key-here"

# Single model evaluation
python scripts/run_evaluation.py --mode single --model gpt-4o --prompt_type detailed

# Multi-model batch evaluation
python scripts/run_evaluation.py --mode multi
```

## Project Structure

```
CausalJudge/
├── scripts/                             # Executable entry points
│   ├── run_evaluation.py               # Main evaluation runner (single/multi)
│   ├── run_single_evaluation.py        # Single model evaluation (checkpointed)
│   ├── run_multi_evaluation.py         # Multi-model multi-run evaluator
│   └── test_setup.py                   # Data integrity tests (no API needed)
├── src/                                 # Core library code
│   ├── causal_evaluation.py            # Evaluation engine (OpenAI calls + metrics)
│   ├── extract_ground_truth.py         # Build clean ground truth JSON from Excel
│   ├── recompute_metrics.py            # Recompute metrics from predictions
│   ├── generate_final_plots_with_values.py  # Plots + human baseline overlay
│   ├── human_evaluation.py             # Per-reviewer metrics + plots + TeX
│   └── plot_human_vs_llm.py            # Scatter plots: human vs LLM per field
├── data/
│   ├── processed/
│   │   ├── PMID_all_text.jsonl         # 180 research articles (input)
│   │   └── ground_truth_clean.json     # Clean ground truth (generated)
│   └── raw/                            # Optional reviewer inputs
├── results/                            # Generated (gitignored)
│   ├── predictions_*.json              # Model outputs
│   ├── predictions_*_confidence.json   # Per-field confidences
│   ├── metrics_*.json                  # Run-level metrics
│   └── metrics_human_*.json            # Human metrics (generated)
└── figures/                            # Generated plots (gitignored)
    └── compress_prior_work.drawio.png  # Preview image (kept)
```

## Installation

```bash
git clone https://github.com/yiqunchen/CausalJudge.git
cd CausalJudge
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
```

## E2E Pipeline (Reference)

- Prepare data:
  - Place `PMID/180FinalResult_Jun17.xlsx` (source) and `data/processed/PMID_all_text.jsonl` (articles) locally.
  - Build clean ground truth: `python src/extract_ground_truth.py` (writes `data/processed/ground_truth_clean.json`).
- Run evaluation (may not function in archived state):
  - Single run: `python scripts/run_single_evaluation.py --model gpt-4o --prompt_type detailed`
  - Multi-run: `python scripts/run_evaluation.py --mode multi`
- Recompute metrics (optional):
  - `python src/recompute_metrics.py`
- Visualize:
  - `python src/generate_final_plots_with_values.py` → figures under `figures/`


## Evaluation Criteria

The framework evaluates 14 binary criteria across 180 research papers:
- **Design**: Randomized Exposure, Causal Mediation
- **Assumptions**: Linearity Tests, Interaction Effects  
- **Controls**: Covariate Adjustment, Baseline Controls
- **Temporal**: Exposure→Mediator, Mediator→Outcome ordering
- **Robustness**: Assumption Discussion, Sensitivity Analysis, Post-Exposure Variables

## License

MIT License - See LICENSE file for details
