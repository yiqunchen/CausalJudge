#!/usr/bin/env python3
"""
Generate publication-quality plots with error bars showing mean ± SE across runs.
Separate plots for basic and detailed prompts.
Now includes human baseline performance as grey bars.
Displays metric values on top of bars vertically in bold black.
"""

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import seaborn as sns
import pandas as pd
import numpy as np
import json
import glob
from pathlib import Path
from collections import defaultdict

# Set style for publication quality
plt.rcParams.update({
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,  # Ensure fonts are embedded in PDF
    'ps.fonttype': 42
})

def load_human_baseline():
    """Load human baseline metrics from Ian's Excel comparison."""
    human_metrics = {}
    
    # Try to load human metrics files
    human_files = ['results/metrics_human_basic_run1_temp0.0.json', 
                   'results/metrics_human_detailed_run1_temp0.0.json']
    
    for filepath in human_files:
        if Path(filepath).exists():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                prompt_type = data.get('prompt_type', 'basic')
                metrics = data.get('metrics', {})
                
                human_metrics[f"Human_{prompt_type}"] = metrics
                print(f"✓ Loaded human baseline for {prompt_type} prompts")
                
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    
    return human_metrics

def load_metrics_data():
    """Load all metrics files and organize by model, run, and temperature."""
    data = defaultdict(lambda: defaultdict(list))
    
    # Find all metrics files (excluding human files - we'll handle those separately)
    metrics_files = glob.glob("results/metrics_*.json")
    metrics_files = [f for f in metrics_files if "human" not in f]
    
    for filepath in metrics_files:
        # Skip confidence files
        if "confidence" in filepath:
            continue
            
        # Parse filename - expected format: metrics_MODEL_PROMPT_runN_tempX.X.json
        filename = Path(filepath).stem.replace("metrics_", "")
        
        # Find prompt type (basic or detailed)
        prompt_type = None
        if "_basic_" in filename:
            prompt_type = "basic"
            model_part, rest = filename.split("_basic_", 1)
        elif "_detailed_" in filename:
            prompt_type = "detailed"
            model_part, rest = filename.split("_detailed_", 1)
        else:
            print(f"No prompt type found in: {filepath}")
            continue
        
        # Parse run and temperature from rest
        try:
            run_temp_parts = rest.split("_")
            run_num = None
            temp = None
            
            for i, part in enumerate(run_temp_parts):
                if part.startswith("run"):
                    run_num = int(part.replace("run", ""))
                elif part.startswith("temp"):
                    temp_str = part.replace("temp", "")
                    if temp_str:
                        temp = float(temp_str)
                    elif i + 1 < len(run_temp_parts):
                        try:
                            temp = float(run_temp_parts[i + 1].split(".")[0] + "." + run_temp_parts[i + 1].split(".")[1] if "." in run_temp_parts[i + 1] else run_temp_parts[i + 1])
                        except:
                            pass
            
            if temp is None and "temp" in rest:
                temp_match = rest.split("temp")[-1]
                temp_match = temp_match.replace(".json", "")
                try:
                    temp = float(temp_match)
                except:
                    pass
            
            if run_num is None or temp is None:
                continue
                
        except Exception as e:
            continue
        
        # Keep original model names for data keys
        model = model_part
        
        # Load the metrics
        try:
            with open(filepath, 'r') as f:
                file_data = json.load(f)
            
            # Extract metrics from the file structure
            if 'metrics' in file_data:
                metrics = file_data['metrics']
            elif 'field_metrics' in file_data:
                metrics = file_data['field_metrics']
            else:
                metrics = file_data
                
            # Store the data
            key = f"{model}_{prompt_type}"
            data[key][(run_num, temp)].append(metrics)
            
        except Exception as e:
            continue
    
    return data

def calculate_field_statistics(data, temp_filter=None):
    """Calculate mean and standard error for each model-field-metric combination."""
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    # Field names to use
    field_names = [
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
        "Control for Other Post-Exposure Variables"
    ]
    
    # Metrics to analyze
    metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'pr_auc']
    
    for model_prompt, runs_data in data.items():
        # Determine which temperature to use
        if temp_filter is not None:
            # Use specific temperature filter
            target_temps = temp_filter
        elif "gpt-5" in model_prompt.lower() or "o3" in model_prompt.lower():
            # Use temp=1.0 for GPT-5 and O3
            target_temps = [1.0]
        else:
            # Use temp=0.5 for GPT-4o models
            target_temps = [0.5]
        
        # Collect all runs for the target temperatures
        runs_at_temp = [(run, temp) for (run, temp) in runs_data.keys() if temp in target_temps]
        
        if not runs_at_temp:
            continue
        
        # For each field and metric, collect values across runs
        for field in field_names:
            for metric in metrics_list:
                values = []
                
                for run_key in runs_at_temp:
                    for metrics_dict in runs_data[run_key]:
                        if field in metrics_dict and isinstance(metrics_dict[field], dict):
                            if metric in metrics_dict[field]:
                                values.append(metrics_dict[field][metric])
                
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values, ddof=1) if len(values) > 1 else 0
                    se_val = std_val / np.sqrt(len(values))
                    
                    stats[model_prompt][field][metric] = {
                        'mean': mean_val,
                        'std': std_val,
                        'se': se_val,
                        'n': len(values),
                        'values': values
                    }
    
    return stats

def add_human_baseline_to_stats(stats, human_metrics):
    """Add human baseline data to the stats dictionary."""
    
    field_names = [
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
        "Control for Other Post-Exposure Variables"
    ]
    
    metrics_list = ['accuracy', 'precision', 'recall', 'f1']  # Exclude AUC and PR-AUC per request
    
    for human_key, human_data in human_metrics.items():
        for field in field_names:
            if field in human_data:
                field_data = human_data[field]
                for metric in metrics_list:
                    if metric in field_data:
                        # Human baseline has no error bars (single measurement)
                        stats[human_key][field][metric] = {
                            'mean': field_data[metric],
                            'std': 0,
                            'se': 0,
                            'n': 1,
                            'values': [field_data[metric]]
                        }
    
    return stats

def plot_metric_by_prompt(stats, metric, prompt_type, save_dir="figures", suffix=""):
    """Create a single plot for a given metric and prompt type with values displayed."""
    
    # Create save directory
    save_path = Path(save_dir) / "final_plots"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Field labels (shortened for x-axis)
    field_labels = [
        "Random\nExposure",
        "Causal\nMediation",
        "Linearity\nTests",
        "Interaction\nEffects",
        "Cov\nExp-Med",
        "Cov\nExp-Out",
        "Cov\nMed-Out",
        "Baseline\nMediator",
        "Baseline\nOutcome",
        "Temporal\nExp→Med",
        "Temporal\nMed→Out",
        "Assumption\nDiscussion",
        "Sensitivity\nAnalysis",
        "Post-Exp\nControl"
    ]
    
    field_names = [
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
        "Control for Other Post-Exposure Variables"
    ]
    
    # Define colors and order for models (including human)
    model_order = ['GPT-4o-mini', 'GPT-4o', 'O3', 'GPT-5', 'Human']
    model_colors = {
        'GPT-4o-mini': '#2ca02c',  # Green
        'GPT-4o': '#ff7f0e',        # Orange
        'O3': '#d62728',            # Red
        'GPT-5': '#1f77b4',         # Blue
        'Human': '#7f7f7f'          # Grey
    }
    
    # Skip human for AUC and PR-AUC
    if metric in ['auc', 'pr_auc']:
        model_order = ['GPT-4o-mini', 'GPT-4o', 'O3', 'GPT-5']
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    
    # Prepare data for plotting
    x_positions = np.arange(len(field_names))
    width = 0.16 if 'Human' in model_order else 0.2
    
    # Filter models for this prompt type
    models_to_plot = []
    for model in model_order:
        if model == 'Human':
            model_key = f"Human_{prompt_type}"
        else:
            # Map display names to actual data keys
            model_map = {
                'GPT-4o-mini': 'gpt-4o-mini',
                'GPT-4o': 'gpt-4o', 
                'O3': 'o3-2025-04-16',
                'GPT-5': 'gpt-5-2025-08-07'
            }
            actual_model = model_map.get(model, model)
            model_key = f"{actual_model}_{prompt_type}"
        
        if model_key in stats:
            models_to_plot.append((model, model_key))
            print(f"✓ Will plot {model} using data key {model_key}")
        else:
            print(f"✗ No data found for {model} (looked for {model_key})")
    
    # Plot each model
    for idx, (model_display, model_key) in enumerate(models_to_plot):
        means = []
        errors = []
        
        for field in field_names:
            if field in stats[model_key] and metric in stats[model_key][field]:
                means.append(stats[model_key][field][metric]['mean'])
                errors.append(stats[model_key][field][metric]['se'])
            else:
                means.append(0)
                errors.append(0)
        
        # Calculate position offset
        offset = (idx - len(models_to_plot)/2 + 0.5) * width
        
        # Plot bars with error bars
        bars = ax.bar(x_positions + offset, means, width,
                     label=model_display,
                     color=model_colors[model_display],
                     alpha=0.8,
                     edgecolor='black',
                     linewidth=0.5)
        
        # Add error bars (skip for GPT-5 and Human since they only have one run)
        if model_display not in ['GPT-5', 'Human']:
            ax.errorbar(x_positions + offset, means, yerr=errors,
                       fmt='none', color='black', capsize=3, capthick=1,
                       alpha=0.6, linewidth=1)
        
        # Add value labels on top of bars
        for i, (bar, mean_val) in enumerate(zip(bars, means)):
            # For visualization purposes, show 0.01 height for 0 values but keep true value
            display_val = mean_val
            bar_height = max(mean_val, 0.01) if mean_val == 0 else mean_val
            
            # Format the value to 2 decimal places
            if display_val == 0:
                value_text = "0.00"
            else:
                value_text = f"{display_val:.2f}"
            
            # Position text slightly above the bar (considering error bars)
            y_pos = bar_height + (errors[i] if model_display not in ['GPT-5', 'Human'] else 0) + 0.02
            
            # Add text vertically
            ax.text(bar.get_x() + bar.get_width()/2, y_pos, value_text,
                   ha='center', va='bottom', rotation=90,
                   fontsize=8, fontweight='bold', color='black')
    
    # Customize plot
    ax.set_title(f"{metric.upper()} - {prompt_type.capitalize()} Prompts{suffix}", 
                fontsize=14, fontweight='bold')
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_xlabel("Evaluation Criteria", fontsize=12)
    ax.set_ylim(0, 1.15)  # Increased to accommodate text labels
    
    # X-axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels(field_labels, rotation=45, ha='right')
    
    # Legend at bottom
    # Add legend only if we have models to plot
    if models_to_plot:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                 ncol=len(models_to_plot), frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    # Save the plot
    output_filename = f"{metric}_{prompt_type}{suffix}"
    
    # Save as PDF
    pdf_path = save_path / f"{output_filename}.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"✓ Saved {pdf_path}")
    
    # Save as PNG
    png_path = save_path / f"{output_filename}.png"
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    print(f"✓ Saved {png_path}")
    
    plt.close()

def main():
    """Main function to generate all plots."""
    print("=" * 60)
    print("GENERATING FINAL PLOTS WITH VALUES")
    print("=" * 60)
    
    # Load human baseline data
    print("\n1. Loading human baseline data...")
    human_metrics = load_human_baseline()
    
    # Load all metrics data
    print("\n2. Loading model metrics data...")
    data = load_metrics_data()
    
    # Calculate statistics for field-level metrics
    print("\n3. Calculating field statistics...")
    stats = calculate_field_statistics(data)
    
    # Add human baseline to stats
    print("\n4. Adding human baseline to stats...")
    stats = add_human_baseline_to_stats(stats, human_metrics)
    
    # Debug: Show available models
    print("\n5. Available model-prompt combinations:")
    for key in sorted(stats.keys()):
        sample_fields = list(stats[key].keys())[:3]
        print(f"  - {key}: {len(stats[key])} fields (e.g., {', '.join(sample_fields[:2])}...)")
    
    # Generate plots for each metric and prompt type
    print("\n6. Generating plots...")
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'pr_auc']
    prompt_types = ['basic', 'detailed']
    
    for metric in metrics_to_plot:
        for prompt_type in prompt_types:
            print(f"\nPlotting {metric} for {prompt_type} prompts...")
            plot_metric_by_prompt(stats, metric, prompt_type)
    
    print("\n" + "=" * 60)
    print("✓ ALL PLOTS GENERATED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()