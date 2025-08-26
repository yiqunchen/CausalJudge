#!/usr/bin/env python3
"""
Statistical analysis: accuracy ~ length + prompt_style + model
"""

import pandas as pd
import numpy as np
import json
import glob
import pickle
from scipy import stats

def run_statistical_analysis():
    """Run comprehensive statistical analysis of model performance factors."""
    
    print('Running comprehensive statistical analysis: accuracy ~ length + prompt_style + model')
    print('=' * 80)

    # Load extended metadata with paper characteristics
    with open('paper_metadata_extended.pkl', 'rb') as f:
        paper_data = pickle.load(f)

    # Load all metrics files
    metrics_files = glob.glob('results/metrics_*.json')
    all_results = []

    for file_path in metrics_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if 'metrics' in data and isinstance(data['metrics'], dict):
                model = data.get('model', 'unknown')
                prompt_type = data.get('prompt_type', 'basic')
                
                # Get PMID-level accuracies from individual field performance
                for field, field_metrics in data['metrics'].items():
                    if field != 'overall' and isinstance(field_metrics, dict):
                        if 'accuracy' in field_metrics:
                            all_results.append({
                                'model': model,
                                'prompt_type': prompt_type,
                                'field': field,
                                'accuracy': field_metrics['accuracy'],
                                'file': file_path
                            })
        except Exception as e:
            print(f'Error processing {file_path}: {e}')
            continue

    # Create comprehensive dataset
    df = pd.DataFrame(all_results)

    if df.empty:
        print('No data available for analysis')
        return

    print(f'Dataset summary:')
    print(f'- Models analyzed: {df["model"].nunique()}')
    print(f'- Prompt types: {df["prompt_type"].nunique()}') 
    print(f'- Fields evaluated: {df["field"].nunique()}')
    print(f'- Total observations: {len(df)}')

    print(f'\n=== DETAILED PROMPT EFFECTS BY MODEL ===')

    # Calculate exact statistics for each model
    for model in sorted(df['model'].unique()):
        model_data = df[df['model'] == model]
        basic_scores = model_data[model_data['prompt_type'] == 'basic']['accuracy']
        detailed_scores = model_data[model_data['prompt_type'] == 'detailed']['accuracy']
        
        if len(basic_scores) > 5 and len(detailed_scores) > 5:
            # Calculate statistics
            basic_mean = basic_scores.mean()
            detailed_mean = detailed_scores.mean()
            basic_std = basic_scores.std()
            detailed_std = detailed_scores.std()
            
            # Effect size
            abs_diff = detailed_mean - basic_mean
            rel_improvement = (abs_diff / basic_mean) * 100
            
            # Statistical test
            t_stat, p_val = stats.ttest_ind(detailed_scores, basic_scores)
            
            # Cohen's d
            pooled_std = np.sqrt((basic_scores.var() + detailed_scores.var()) / 2)
            cohens_d = abs_diff / pooled_std
            
            print(f'\n{model}:')
            print(f'  Basic prompt:    {basic_mean:.3f} ± {basic_std:.3f} (n={len(basic_scores)})')
            print(f'  Detailed prompt: {detailed_mean:.3f} ± {detailed_std:.3f} (n={len(detailed_scores)})')
            print(f'  Absolute improvement: +{abs_diff:.3f}')
            print(f'  Relative improvement: +{rel_improvement:.1f}%')
            print(f'  Statistical significance: p = {p_val:.4f}')
            print(f'  Effect size (Cohen\'s d): {cohens_d:.3f}')

    # Overall analysis
    print(f'\n=== OVERALL PROMPTING EFFECTS ===')
    overall_basic = df[df['prompt_type'] == 'basic']['accuracy']
    overall_detailed = df[df['prompt_type'] == 'detailed']['accuracy']

    overall_t, overall_p = stats.ttest_ind(overall_detailed, overall_basic)
    overall_cohens_d = (overall_detailed.mean() - overall_basic.mean()) / np.sqrt((overall_basic.var() + overall_detailed.var()) / 2)

    print(f'Across all models and fields:')
    print(f'  Basic prompts:    {overall_basic.mean():.3f} ± {overall_basic.std():.3f} (n={len(overall_basic)})')
    print(f'  Detailed prompts: {overall_detailed.mean():.3f} ± {overall_detailed.std():.3f} (n={len(overall_detailed)})')
    print(f'  Overall improvement: +{overall_detailed.mean() - overall_basic.mean():.3f}')
    print(f'  Statistical significance: p = {overall_p:.4f}')
    print(f'  Overall effect size: {overall_cohens_d:.3f}')

    # Model ranking
    print(f'\n=== MODEL PERFORMANCE RANKING ===')
    model_means = df.groupby('model')['accuracy'].agg(['mean', 'std', 'count']).round(3)
    model_means_sorted = model_means.sort_values('mean', ascending=False)

    print('Rank  Model                 Mean_Accuracy  Std    N_obs')
    print('-' * 60)
    for i, (model, row) in enumerate(model_means_sorted.iterrows(), 1):
        print(f'{i:2d}   {model:20s} {row["mean"]:8.3f}      {row["std"]:5.3f}  {row["count"]:4.0f}')

    # Field difficulty ranking
    print(f'\n=== EVALUATION CRITERIA DIFFICULTY RANKING ===')
    field_means = df.groupby('field')['accuracy'].agg(['mean', 'std', 'count']).round(3)
    field_means_sorted = field_means.sort_values('mean', ascending=False)

    print('\nEasiest 5 criteria:')
    for i, (field, row) in enumerate(field_means_sorted.head().iterrows(), 1):
        short_field = field[:40] + '...' if len(field) > 40 else field
        print(f'{i}. {short_field:43s} {row["mean"]:.3f} ± {row["std"]:.3f}')

    print('\nHardest 5 criteria:')
    for i, (field, row) in enumerate(field_means_sorted.tail().iterrows(), 1):
        short_field = field[:40] + '...' if len(field) > 40 else field
        print(f'{i}. {short_field:43s} {row["mean"]:.3f} ± {row["std"]:.3f}')

    # Return the dataframe for further analysis
    return df

if __name__ == "__main__":
    df = run_statistical_analysis()