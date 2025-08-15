# CausalJudge: Evaluating LLM Performance on Causal Mediation Analysis

A comprehensive evaluation framework for assessing large language models' ability to extract causal mediation information from scientific literature.

## Overview

This repository contains the code and data for evaluating LLM performance on 14 key criteria in causal mediation analysis, including temporal ordering, covariate adjustment, assumption discussions, and sensitivity analyses.

## Key Results

- **Human Baseline**: 97.7% accuracy across all criteria (based on expert consensus)
- **Best LLM Performance**: O3 (88.3% detailed prompts), GPT-5 (88.1%), GPT-4o (77.6%)
- **Performance Gap**: LLMs trail human accuracy by 9-15 percentage points on average
- **Prompting Strategy**: Detailed prompts consistently improve performance over basic prompts

## Quick Start

### Prerequisites

```bash
# Clone the repository
git clone [repository-url]
cd CausalJudge

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export CHEN_OPENAI_API_KEY="your-api-key-here"
```

### Running Evaluations

```bash
# Single model evaluation
python run_single_evaluation.py --model gpt-4o --prompt_type detailed --max_concurrency 4

# Multi-model evaluation (all models, both prompt types)
python run_multi_evaluation.py --inner_max_concurrency 6

# Generate final plots
python generate_final_plots_with_human.py
```

## Core Files

### Evaluation Engine
- `causal_evaluation.py` - Main evaluation system with async OpenAI integration
- `run_single_evaluation.py` - Single model evaluation with checkpoint support
- `run_multi_evaluation.py` - Multi-model evaluation orchestrator
- `run_evaluation.py` - Main entry point

### Data Processing
- `extract_ground_truth.py` - Extract ground truth from Excel annotations
- `ground_truth_clean.json` - Processed ground truth data (180 papers)
- `PMID_all_text.jsonl` - Full text of research papers
- `Ian_original_plus_filtered_final.xlsx` - Human baseline comparison data

### Analysis & Visualization  
- `generate_final_plots_with_human.py` - Publication-quality plots with human baseline
- `aesthetic_plots.py` - Alternative plotting with model+temperature combinations
- `recompute_metrics.py` - Recompute evaluation metrics from predictions
- `statistical_analysis.py` - Statistical significance testing

### Utilities
- `enhanced_plot_results.py` - Extended analysis plots
- `joint_regression_analysis.py` - Regression analysis across models

## Evaluation Criteria

The framework evaluates 14 key aspects of causal mediation analysis:

1. **Randomized Exposure** - Use of randomized experimental design
2. **Causal Mediation** - Explicit causal mediation analysis
3. **Mediator-Outcome Linearity** - Testing of linearity assumptions
4. **Exposure-Mediator Interaction** - Assessment of interaction effects
5. **Covariate Adjustment** - Control for confounding variables (3 models)
6. **Baseline Controls** - Adjustment for baseline mediator/outcome
7. **Temporal Ordering** - Proper sequencing of exposure→mediator→outcome
8. **Assumption Discussion** - Discussion of key mediation assumptions
9. **Sensitivity Analysis** - Robustness checks for assumptions
10. **Post-Exposure Control** - Control for post-treatment variables

## Prompting Strategies

- **Basic Prompts**: Minimal instructions with JSON structure
- **Detailed Prompts**: Comprehensive guidelines with examples and confidence scoring
- **Async Processing**: Concurrent API calls for improved throughput

## Results Structure

```
results/
├── metrics_[model]_[prompt]_run[N]_temp[X].json    # Evaluation metrics
├── predictions_[model]_[prompt]_run[N]_temp[X].json # Model predictions
└── checkpoint_[model]_[prompt]_run[N]_temp[X].pkl   # Resume checkpoints

figures/
└── final_plots/                                     # Publication plots
    ├── accuracy_basic.pdf
    ├── accuracy_detailed.pdf
    └── [other metrics]_[basic|detailed].pdf
```

## Model Performance Summary

| Model | Prompt Type | Accuracy | F1 Score | Precision | Recall |
|-------|-------------|----------|----------|-----------|--------|
| Human | Basic/Detailed | 97.7% | 89.3% | 91.8% | 89.0% |
| O3 | Detailed | 88.3% | 65.0% | 61.0% | 75.4% |
| GPT-5 | Detailed | 88.1% | 65.6% | 62.1% | 76.1% |
| GPT-4o | Detailed | 77.6% | 53.6% | 45.8% | 79.9% |
| GPT-4o-mini | Basic | 57.6% | 47.8% | 35.9% | 89.1% |

## Citation

```bibtex
@article{causaljudge2024,
  title={CausalJudge: Evaluating LLM Performance on Causal Mediation Analysis},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

[License information]

## Contributing

[Contributing guidelines]