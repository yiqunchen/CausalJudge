# CausalJudge: Evaluating LLM Performance on Causal Mediation Analysis

A comprehensive evaluation framework for assessing large language models' ability to extract causal mediation information from scientific literature.

## Quick Start

### End-to-End Test Command
```bash
# Complete end-to-end test: data integrity → single evaluation → results visualization
python scripts/test_setup.py && \
python scripts/run_single_evaluation.py --model gpt-4o --prompt_type detailed && \
python generate_final_figures.py
```

### Standard Evaluation Pipeline
```bash
# Set API key
export CHEN_OPENAI_API_KEY="your-key-here"

# Run comprehensive multi-model evaluation (50 configurations)
python scripts/run_evaluation.py --mode multi

# Or run single model evaluation
python scripts/run_evaluation.py --mode single --model gpt-5 --prompt_type detailed
```

## Repository Structure

```
CausalJudge/
├── scripts/                        # Main execution scripts
│   ├── run_evaluation.py          # Main entry point (single/multi mode)
│   ├── run_single_evaluation.py   # Single model evaluation with checkpoints
│   ├── run_multi_evaluation.py    # Parallel multi-model evaluation
│   └── test_setup.py             # Data integrity and system verification
├── src/                           # Core evaluation system
│   └── causal_evaluation.py      # Main evaluation engine (async OpenAI)
├── data/
│   ├── raw/                      # Original data files
│   │   └── GoldenStandard180.csv # Human expert annotations
│   └── processed/
│       └── PMID_all_text.jsonl   # Processed article texts (auto-truncated)
├── results/                       # Model outputs
│   └── [model_name]/             # Per-model results
│       ├── predictions_*.json    # Raw predictions
│       ├── metrics_*.json        # Computed metrics
│       └── checkpoint_*.pkl      # Resume checkpoints
├── figures/                       # Visualizations
│   └── final_plots/              # Publication-ready figures
└── analysis/                      # Analysis scripts
    ├── qualitative_analysis_for_paper.md  # Detailed error analysis
    ├── reasoning_pattern_synthesis.json   # Systematic error patterns
    └── extract_reasoning_analysis.py      # Reasoning extraction pipeline

## Qualitative Analysis Pipeline

Our qualitative pipeline systematically analyzes model reasoning patterns:

### 1. Error Pattern Extraction
```bash
# Extract reasoning patterns from model predictions
python extract_reasoning_analysis.py

# Synthesize patterns across models
python synthesize_reasoning_patterns.py
```

### 2. Error Categories Identified
- **Overinterpretation**: Inferring methodological elements from weak evidence
- **Underinterpretation**: Missing clear methodological evidence
- **Technical Misunderstanding**: Confusing related but distinct concepts
- **Keyword Bias**: Over-reliance on specific terms without context
- **Ambiguity**: Genuinely unclear textual evidence

### 3. Model-Specific Patterns
- **GPT-5**: Sophisticated but overconfident (false positives on randomization)
- **GPT-4o**: Balanced but struggles with temporal ordering concepts
- **GPT-4o-mini**: Conservative with high keyword dependency

### 4. Field-Specific Insights
- **High Performance (>80%)**: Randomized Exposure, Causal Mediation
- **Moderate (60-80%)**: Covariate Controls, Temporal Ordering
- **Challenging (<60%)**: Linearity Tests, Sensitivity Analysis

## Installation & Setup

### Prerequisites

```bash
# Clone repository
git clone https://github.com/yiqunchen/CausalJudge.git
cd CausalJudge

# Install Python dependencies
pip install -r requirements.txt

# Set OpenAI API key
export CHEN_OPENAI_API_KEY="your-api-key-here"
```


### Data Preparation

```bash
# Verify data integrity
python scripts/test_setup.py

# Expected output:
# ✓ PMID_all_text.jsonl exists (180 articles)
# ✓ PMID/GoldenStandard180.csv exists
# ✓ Articles truncated at References section: 165/180
```

## Evaluation Framework

### 14 Evaluation Criteria
Each paper is evaluated on binary (0/1) indicators:

| Category | Criteria | Human Accuracy | Best LLM |
|----------|----------|----------------|----------|
| **Design** | Randomized Exposure | 98.9% | 93.3% (GPT-5) |
| **Analysis** | Causal Mediation | 97.8% | 89.4% (O3) |
| **Assumptions** | Linearity Tests | 96.1% | 53.3% (GPT-5) |
| | Interaction Effects | 98.3% | 72.8% (O3) |
| **Controls** | Covariate Adjustment (3 types) | 97.2% | 78.9% (GPT-5) |
| | Baseline Controls | 98.9% | 85.6% (O3) |
| **Temporal** | Exposure→Mediator | 96.7% | 68.9% (GPT-5) |
| | Mediator→Outcome | 97.2% | 71.1% (O3) |
| **Robustness** | Assumption Discussion | 98.3% | 76.7% (GPT-5) |
| | Sensitivity Analysis | 97.8% | 58.9% (O3) |
| | Post-Exposure Variables | 98.9% | 91.1% (GPT-5) |

## Advanced Usage

### Parallel Evaluation with Custom Parameters
```bash
# Run with custom concurrency limits
python scripts/run_multi_evaluation.py \
  --outer_max_concurrency 4 \
  --inner_max_concurrency 8 \
  --models gpt-5 o3 gpt-4o \
  --prompt_types detailed \
  --num_runs 3
```

### Checkpoint Recovery
```bash
# Resume interrupted evaluation
python scripts/run_with_checkpoints.py \
  --checkpoint_file results/checkpoint_gpt-5_detailed_run1.pkl
```

### Statistical Analysis
```bash
# Comprehensive statistical testing
python statistical_analysis.py

# Joint regression analysis
python joint_regression_analysis.py

# Generate extended plots with confidence intervals
python extended_analysis_plots.py
```

### Prompting Strategies
- **Basic**: Minimal JSON-structured prompts (baseline)
- **Detailed**: Enhanced with methodological guidance (+10-15% accuracy)
- **Examples**: Pattern-based with specific text markers (experimental)

## Results & Outputs

### Metrics Computed
- **Accuracy**: Overall correctness across 14 criteria
- **F1 Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve
- **PR-AUC**: Precision-Recall AUC
- **Field-specific**: Per-criterion accuracy and confidence scores

### Output Files
```
results/
├── [model_name]/
│   ├── predictions_*.json     # Raw predictions with confidence scores
│   ├── metrics_*.json         # Computed performance metrics
│   └── checkpoint_*.pkl       # Resume points for interrupted runs
│
figures/
├── final_plots/               # Publication-ready figures
│   ├── accuracy_[prompt].pdf
│   ├── f1_score_[prompt].pdf
│   └── field_comparison.pdf
│
analysis/
├── reasoning_pattern_synthesis.json  # Systematic error analysis
└── qualitative_findings.md          # Detailed insights
```

## Performance Comparison

### Overall Performance (Detailed Prompts)
| Model | Accuracy | F1 | AUC | PR-AUC | Key Strength | Key Weakness |
|-------|----------|-----|-----|--------|--------------|---------------|
| **Human** | 97.7% | 89.3% | - | - | Consistency | - |
| **O3** | 88.3% | 65.0% | 0.82 | 0.74 | Balanced | Sensitivity Analysis |
| **GPT-5** | 88.1% | 65.6% | 0.80 | 0.73 | Sophistication | Overinterpretation |
| **GPT-4o** | 77.6% | 53.6% | 0.77 | 0.70 | Balance | Technical Concepts |
| **GPT-4o-mini** | 68.3%* | 47.8% | 0.75 | 0.67 | Precision | Underinterpretation |

*Best performance with basic prompts

### Statistical Significance
- All model differences significant at p < 0.05
- Detailed prompts improve accuracy by 10-15% (p < 0.001)
- Performance gap with humans remains consistent across criteria

## Future Work & Limitations

### Current Limitations
- Models struggle with implicit methodological evidence
- Technical concepts (temporal ordering, post-exposure variables) remain challenging
- Overinterpretation in advanced models limits precision for systematic reviews

### Recommended Improvements
1. **Ensemble Methods**: Combine models to leverage complementary error patterns
2. **Domain-Specific Fine-Tuning**: Train on causal methodology examples
3. **Hybrid Workflows**: Allocate tasks based on model strengths
4. **Confidence Calibration**: Improve reliability of confidence scores

## License

MIT License - See LICENSE file for details

## Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines.

### Reporting Issues
Report bugs and feature requests at: https://github.com/yiqunchen/CausalJudge/issues

## Contact

For questions about the research or collaboration opportunities, please contact the authors through the GitHub repository.