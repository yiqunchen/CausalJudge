#!/usr/bin/env python3
"""
Generate publication-quality plots with error bars showing mean ± SE across runs.
Separate plots for basic and detailed prompts.
Now includes human baseline performance as grey bars.
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
        
        # Normalize model names
        model = model_part
        model = model.replace("o3-2025-04-16", "O3")
        model = model.replace("gpt-5-2025-08-07", "GPT-5")
        model = model.replace("gpt-4o-mini", "GPT-4o-mini")
        model = model.replace("gpt-4o", "GPT-4o")
        
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
        elif "GPT-5" in model_prompt or "O3" in model_prompt:
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
    """Create a single plot for a given metric and prompt type."""
    
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
            model_key = f"{model}_{prompt_type}"
        if model_key in stats:
            models_to_plot.append((model, model_key))
    
    # Debug output removed
    
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
    
    # Customize plot
    ax.set_title(f"{metric.upper()} - {prompt_type.capitalize()} Prompts{suffix}", 
                fontsize=14, fontweight='bold')
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_xlabel("Evaluation Criteria", fontsize=12)
    ax.set_ylim(0, 1.05)
    
    # Grid (removed per user request)
    # ax.grid(True, alpha=0.2, axis='y')
    
    # X-axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels(field_labels, rotation=45, ha='right')
    
    # Legend at bottom
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
             ncol=len(models_to_plot), frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    # Save figure
    filename_base = f"{metric}_{prompt_type}{suffix.replace(' ', '_').replace('(', '').replace(')', '')}"
    
    # Save PNG
    output_file_png = save_path / f"{filename_base}.png"
    plt.savefig(output_file_png, bbox_inches='tight', dpi=150)
    
    # Save PDF with proper backend
    output_file_pdf = save_path / f"{filename_base}.pdf"
    plt.savefig(output_file_pdf, bbox_inches='tight', format='pdf', backend='pdf')
    
    plt.close()
    
    print(f"✓ Saved {metric} {prompt_type} plot{suffix}")

def main():
    print("Loading metrics data...")
    data = load_metrics_data()
    
    print("Loading human baseline...")
    human_metrics = load_human_baseline()
    
    if not data:
        print("❌ No metrics data found!")
        return
    
    print(f"Found data for {len(data)} model-prompt combinations")
    if human_metrics:
        print(f"Found human baseline for {len(human_metrics)} prompt types\n")
    else:
        print("⚠️ No human baseline data found\n")
    
    # Calculate statistics for main plots (temp=0.5 for GPT-4o models, temp=1.0 for GPT-5 and O3)
    print("Calculating statistics for main plots...")
    stats_main = calculate_field_statistics(data)
    
    # Add human baseline to stats
    if human_metrics:
        stats_main = add_human_baseline_to_stats(stats_main, human_metrics)
    
    # List of metrics to plot
    metrics_to_plot = ['accuracy', 'f1', 'precision', 'recall', 'auc', 'pr_auc']
    
    print("\nGenerating main plots (separate for basic and detailed)...")
    for metric in metrics_to_plot:
        for prompt_type in ['basic', 'detailed']:
            plot_metric_by_prompt(stats_main, metric, prompt_type)
    
    # Create summary table
    print("\nCreating summary table...")
    save_path = Path("figures") / "final_plots"
    save_path.mkdir(parents=True, exist_ok=True)
    
    summary_data = []
    for model_key in stats_main.keys():
        model_name = model_key.replace("_basic", " (Basic)").replace("_detailed", " (Detailed)")
        
        for metric in ['accuracy', 'f1', 'precision', 'recall']:
            values_all_fields = []
            for field in stats_main[model_key].keys():
                if metric in stats_main[model_key][field]:
                    values_all_fields.extend(stats_main[model_key][field][metric]['values'])
            
            if values_all_fields:
                mean_val = np.mean(values_all_fields)
                summary_data.append({
                    'Model': model_name,
                    'Metric': metric.upper(),
                    'Mean': f"{mean_val:.3f}"
                })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        pivot_df = df.pivot_table(index='Model', columns='Metric', values='Mean', aggfunc='first')
        pivot_df.to_csv(save_path / "summary_statistics_with_human.csv")
        
        print("\n" + "="*60)
        print("OVERALL PERFORMANCE SUMMARY (INCLUDING HUMAN BASELINE)")
        print("="*60)
        print(pivot_df.to_string())
        print("="*60)
    
    print("\n✅ All plots generated successfully!")
    print("📁 Check figures/final_plots/ for results with human baseline")

if __name__ == "__main__":
    main()
