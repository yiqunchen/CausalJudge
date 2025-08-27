# CausalJudge: Evaluating LLM Performance on Causal Mediation Analysis

A comprehensive evaluation framework for assessing large language models' ability to extract causal mediation information from scientific literature.

## Quick Start

### End-to-End Test
```bash
# Test system: data check → single evaluation → visualization
python scripts/test_setup.py && \
python scripts/run_single_evaluation.py --model gpt-4o --prompt_type detailed && \
python src/generate_final_plots_with_human.py
```

### Main Evaluation Commands
```bash
# Set API key
export CHEN_OPENAI_API_KEY="your-key-here"

# Single model evaluation
python scripts/run_evaluation.py --mode single --model gpt-4o --prompt_type detailed

# Multi-model batch evaluation
python scripts/run_evaluation.py --mode multi
```

## Project Structure

```
CausalJudge/
├── scripts/                            # Executable entry points
│   ├── run_evaluation.py              # Main evaluation runner
│   ├── run_single_evaluation.py       # Single model evaluation
│   ├── run_multi_evaluation.py        # Parallel evaluation
│   └── test_setup.py                  # Data integrity tests
├── src/                                # Core library code
│   ├── causal_evaluation.py           # Main evaluation engine
│   ├── statistical_analysis.py        # Statistical testing
│   └── generate_final_plots_with_human.py # Plotting utilities
├── data/
│   ├── processed/
│   │   ├── PMID_all_text.jsonl        # 180 research articles
│   │   └── ground_truth_clean.json    # Ground truth annotations
├── results/[model]/predictions_*.json  # Model outputs
├── synthesize_reasoning_patterns.py    # Error analysis (root level)
└── qualitative_analysis_for_paper.md   # Analysis documentation
```

## Installation

```bash
git clone https://github.com/yiqunchen/CausalJudge.git
cd CausalJudge
pip install -r requirements.txt
export CHEN_OPENAI_API_KEY="your-key-here"
```

## Error Analysis

```bash
# Analyze model reasoning patterns and extract systematic error types
python synthesize_reasoning_patterns.py
```

Outputs: `reasoning_pattern_synthesis.json` and `qualitative_analysis_for_paper.md`

## Evaluation Criteria

The framework evaluates 14 binary criteria across 180 research papers:
- **Design**: Randomized Exposure, Causal Mediation
- **Assumptions**: Linearity Tests, Interaction Effects  
- **Controls**: Covariate Adjustment, Baseline Controls
- **Temporal**: Exposure→Mediator, Mediator→Outcome ordering
- **Robustness**: Assumption Discussion, Sensitivity Analysis, Post-Exposure Variables

## Advanced Usage

```bash
# Custom parallel evaluation
python scripts/run_multi_evaluation.py --models gpt-5 o3 --prompt_types detailed --num_runs 3

# Resume interrupted evaluation
python scripts/run_with_checkpoints.py --checkpoint_file results/checkpoint_*.pkl

# Statistical analysis
python src/statistical_analysis.py
```

## License

MIT License - See LICENSE file for details

## Branch Protection

This repository has a pre-push hook that prevents pushing to `main` branch unless data integrity tests pass:

```bash
# The hook automatically runs: python scripts/test_setup.py
# Requires: 4/5 or 5/5 tests to pass (no OpenAI API key needed)
# Tests: data files exist, proper format, PMID alignment, system initialization
```

To bypass for emergency (not recommended):
```bash
git push --no-verify origin main
```

## Contributing

Report issues at: https://github.com/yiqunchen/CausalJudge/issues