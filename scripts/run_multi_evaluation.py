#!/usr/bin/env python3
"""
Multi-Model Multi-Run Evaluation Script
Calls the single evaluation routine for each model/prompt/run combination.

Now supports asynchronous execution with a configurable maximum concurrency.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from run_single_evaluation import run_single_evaluation, is_evaluation_complete

def load_master_checkpoint():
    """Load master checkpoint to track overall progress."""
    checkpoint_file = "results/master_checkpoint.json"
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {
        'completed_tasks': [],
        'failed_tasks': [],
        'timestamp': datetime.now().isoformat(),
        'total_expected_tasks': 0
    }

def save_master_checkpoint(checkpoint_data):
    """Save master checkpoint."""
    checkpoint_file = "results/master_checkpoint.json"
    os.makedirs('results', exist_ok=True)
    checkpoint_data['last_updated'] = datetime.now().isoformat()
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

def print_summary_stats(model, prompt_type, results):
    """Print summary statistics for a model/prompt combination."""
    
    if not results:
        return
    
    print(f"\n  📊 {model} ({prompt_type}) - {len(results)} runs:")
    
    # Extract overall metrics from all runs
    overall_metrics = []
    for result in results:
        if 'metrics' in result and 'overall' in result['metrics']:
            overall_metrics.append(result['metrics']['overall'])
    
    if not overall_metrics:
        print("    No valid results to summarize")
        return
    
    # Calculate statistics for key metrics
    metrics_to_analyze = ['accuracy', 'f1', 'auc', 'pr_auc']
    
    for metric in metrics_to_analyze:
        values = [m[metric] for m in overall_metrics if metric in m]
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            print(f"    {metric.upper():<8}: {mean_val:.3f} ± {std_val:.3f} (range: {min_val:.3f}-{max_val:.3f})")

def generate_variability_report(all_results, timestamp):
    """Generate a comprehensive variability analysis report."""
    
    report_file = f"results/variability_analysis_report_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("CausalJudge Multi-Model Multi-Run Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("SUMMARY OF RUNS:\n")
        f.write("-" * 20 + "\n")
        
        for model_key, results in all_results.items():
            if not results:
                continue
                
            model_name = results[0]['model']
            prompt_type = results[0]['prompt_type']
            n_runs = len(results)
            
            f.write(f"\n{model_name} ({prompt_type}): {n_runs} runs\n")
            
            # Extract overall metrics
            overall_metrics = []
            for result in results:
                if 'metrics' in result and 'overall' in result['metrics']:
                    overall_metrics.append(result['metrics']['overall'])
            
            if overall_metrics:
                metrics_to_analyze = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'pr_auc']
                
                for metric in metrics_to_analyze:
                    values = [m[metric] for m in overall_metrics if metric in m]
                    if values:
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        min_val = np.min(values)
                        max_val = np.max(values)
                        
                        f.write(f"  {metric.capitalize():<10}: {mean_val:.4f} ± {std_val:.4f} (range: {min_val:.4f}-{max_val:.4f})\n")
        
        f.write(f"\n{'='*60}\n")
        f.write("INTERPRETATION NOTES:\n")
        f.write("-" * 20 + "\n")
        f.write("• Low standard deviation (<0.01) suggests high consistency\n")
        f.write("• Medium standard deviation (0.01-0.05) suggests moderate variability\n")
        f.write("• High standard deviation (>0.05) suggests high variability\n")
        f.write("• O1 models may show more variability due to temperature=1.0\n")
        f.write("• Other models use temperature=0.0 for more deterministic behavior\n")
    
    print(f"📋 Variability analysis report saved to: {report_file}")

def run_multi_evaluation(inner_max_concurrency: int = 1):
    """Run multi-model multi-run evaluation with optional concurrency.

    Args:
        max_concurrency: Maximum number of concurrent evaluations. Default 1 (sequential).
    """
    
    # Get API key
    api_key = os.getenv('OPEN_AI_KEY')
    if not api_key:
        print("❌ Please set OPEN_AI_KEY environment variable")
        return False
    
    # Configuration
    models = ["gpt-4o", "gpt-4o-mini"]
    o3_models = ["o3-2025-04-16"]  # Special handling for O3 models
    gpt5_models = ["gpt-5-2025-08-07"]  # Special handling for GPT-5 models
    prompt_types = ["basic", "detailed"] 
    temperatures = [0.0, 0.5]  # Add both temp=0 and temp=0.5
    n_runs_per_temp = {0.0: 1, 0.5: 5}  # 1 run for temp=0, 5 runs for temp=0.5
    # O3 models: only temperature=1.0 with 5 runs
    o3_temperatures = [1.0]
    o3_runs_per_temp = {1.0: 5}  # 5 runs for temp=1.0 for o3 models
    # GPT-5 models: only temperature=1.0 with 1 run
    gpt5_temperatures = [1.0]
    gpt5_runs_per_temp = {1.0: 1}  # 1 run for temp=1.0 for gpt-5 models
    
    # Load master checkpoint
    master_checkpoint = load_master_checkpoint()
    # Calculate total tasks: regular models get 1 run at temp=0 + 5 runs at temp=0.5
    # O3 models get 5 runs at temp=1.0
    # GPT-5 models get 1 run at temp=1.0
    regular_tasks = len(models) * len(prompt_types) * (1 + 5)  # 6 runs per model/prompt combo
    o3_tasks = len(o3_models) * len(prompt_types) * 5  # 5 runs per model/prompt combo
    gpt5_tasks = len(gpt5_models) * len(prompt_types) * 1  # 1 run per model/prompt combo
    total_tasks = regular_tasks + o3_tasks + gpt5_tasks
    master_checkpoint['total_expected_tasks'] = total_tasks
    
    print(f"🚀 CausalJudge Multi-Model Multi-Run Evaluation")
    print(f"=" * 60)
    print(f"📊 Regular Models: {len(models)} ({', '.join(models)})")
    print(f"🧠 O3 Models: {len(o3_models)} ({', '.join(o3_models)})")
    print(f"🤖 GPT-5 Models: {len(gpt5_models)} ({', '.join(gpt5_models)})")
    print(f"📝 Prompt types: {len(prompt_types)} ({', '.join(prompt_types)})")
    print(f"🌡️ Regular Temperatures: {temperatures} (1 run @ temp=0, 5 runs @ temp=0.5)")
    print(f"🌡️ O3 Temperature: {o3_temperatures} (5 runs @ temp=1.0)")
    print(f"🌡️ GPT-5 Temperature: {gpt5_temperatures} (1 run @ temp=1.0)")
    print(f"📈 Total evaluation tasks: {total_tasks} ({regular_tasks} regular + {o3_tasks} O3 + {gpt5_tasks} GPT-5)")
    print(f"🔁 Max in-task OpenAI concurrency: {inner_max_concurrency}")
    print(f"")
    
    # Check for existing completed work
    completed_tasks = []
    remaining_tasks = []
    
    # Check regular models
    for model in models:
        for prompt_type in prompt_types:
            # All models: multiple temperatures and runs
            for temp in temperatures:
                n_runs = n_runs_per_temp[temp]
                for run_id in range(1, n_runs + 1):
                    task_id = f"{model}_{prompt_type}_run{run_id}_temp{temp}"
                    model_safe = model.replace('-', '_')
                    output_file = f"results/predictions_{model_safe}_{prompt_type}_run{run_id}_temp{temp}.json"
                    conf_file = f"results/predictions_{model_safe}_{prompt_type}_run{run_id}_temp{temp}_confidence.json"
                    metrics_file = f"results/metrics_{model}_{prompt_type}_run{run_id}_temp{temp}.json"
                    
                    if is_evaluation_complete(output_file, conf_file, metrics_file):
                        completed_tasks.append(task_id)
                    else:
                        remaining_tasks.append((model, prompt_type, run_id, temp, task_id))
    
    # Check O3 models
    for model in o3_models:
        for prompt_type in prompt_types:
            for temp in o3_temperatures:
                n_runs = o3_runs_per_temp[temp]
                for run_id in range(1, n_runs + 1):
                    task_id = f"{model}_{prompt_type}_run{run_id}_temp{temp}"
                    model_safe = model.replace('-', '_')
                    output_file = f"results/predictions_{model_safe}_{prompt_type}_run{run_id}_temp{temp}.json"
                    conf_file = f"results/predictions_{model_safe}_{prompt_type}_run{run_id}_temp{temp}_confidence.json"
                    metrics_file = f"results/metrics_{model}_{prompt_type}_run{run_id}_temp{temp}.json"
                    
                    if is_evaluation_complete(output_file, conf_file, metrics_file):
                        completed_tasks.append(task_id)
                    else:
                        remaining_tasks.append((model, prompt_type, run_id, temp, task_id))
    
    # Check GPT-5 models
    for model in gpt5_models:
        for prompt_type in prompt_types:
            for temp in gpt5_temperatures:
                n_runs = gpt5_runs_per_temp[temp]
                for run_id in range(1, n_runs + 1):
                    task_id = f"{model}_{prompt_type}_run{run_id}_temp{temp}"
                    model_safe = model.replace('-', '_')
                    output_file = f"results/predictions_{model_safe}_{prompt_type}_run{run_id}_temp{temp}.json"
                    conf_file = f"results/predictions_{model_safe}_{prompt_type}_run{run_id}_temp{temp}_confidence.json"
                    metrics_file = f"results/metrics_{model}_{prompt_type}_run{run_id}_temp{temp}.json"
                    
                    if is_evaluation_complete(output_file, conf_file, metrics_file):
                        completed_tasks.append(task_id)
                    else:
                        remaining_tasks.append((model, prompt_type, run_id, temp, task_id))
    
    print(f"✅ Already completed: {len(completed_tasks)}/{total_tasks} tasks")
    print(f"⏳ Remaining: {len(remaining_tasks)} tasks")
    print(f"")
    
    if len(completed_tasks) == total_tasks:
        print("🎉 All evaluation tasks already completed!")
    else:
        print(f"🔄 Resuming evaluation... {len(remaining_tasks)} tasks to process")
    
    # Update master checkpoint
    master_checkpoint['completed_tasks'] = completed_tasks
    save_master_checkpoint(master_checkpoint)
    
    # Process remaining tasks sequentially (outer concurrency disabled)
    for i, (model, prompt_type, run_id, temp, task_id) in enumerate(remaining_tasks, 1):
        print(f"\n{'='*60}")
        print(f"🎯 Task {i}/{len(remaining_tasks)}: {task_id}")
        print(f"{'='*60}")

        try:
            # Call single evaluation routine with in-task concurrency
            run_single_evaluation(
                api_key=api_key,
                model=model,
                prompt_type=prompt_type,
                run_id=run_id,
                temperature=temp if temp != "default" else None,
                resume_checkpoint=True,
                max_concurrency=inner_max_concurrency,
            )

            # Mark as completed
            if task_id not in master_checkpoint['completed_tasks']:
                master_checkpoint['completed_tasks'].append(task_id)
            if task_id in master_checkpoint['failed_tasks']:
                master_checkpoint['failed_tasks'].remove(task_id)
            save_master_checkpoint(master_checkpoint)

            completed_count = len(master_checkpoint['completed_tasks'])
            remaining = total_tasks - completed_count
            print(f"📊 Progress: {completed_count}/{total_tasks} completed ({remaining} remaining)")

        except Exception as e:
            print(f"❌ Task {task_id} failed: {e}")
            if task_id not in master_checkpoint['failed_tasks']:
                master_checkpoint['failed_tasks'].append(task_id)
            save_master_checkpoint(master_checkpoint)
            continue
    
    # Collect and analyze results
    print(f"\n{'='*60}")
    print("📊 COLLECTING RESULTS FOR ANALYSIS")
    print(f"{'='*60}")
    
    all_results = {}
    
    # Collect results from regular models
    for model in models:
        for prompt_type in prompt_types:
            # All models: group by temperature
            for temp in temperatures:
                model_key = f"{model}_{prompt_type}_temp{temp}"
                model_results = []
                n_runs = n_runs_per_temp[temp]
                
                # Collect all completed runs for this model/prompt/temp combination
                for run_id in range(1, n_runs + 1):
                    metrics_file = f"results/metrics_{model}_{prompt_type}_run{run_id}_temp{temp}.json"
                    if os.path.exists(metrics_file):
                        try:
                            with open(metrics_file, 'r') as f:
                                run_result = json.load(f)
                            model_results.append(run_result)
                        except Exception as e:
                            print(f"⚠️ Could not load {metrics_file}: {e}")
                
                if model_results:
                    all_results[model_key] = model_results
                    print_summary_stats(f"{model} (temp={temp})", prompt_type, model_results)
    
    # Collect results from O3 models
    for model in o3_models:
        for prompt_type in prompt_types:
            for temp in o3_temperatures:
                model_key = f"{model}_{prompt_type}_temp{temp}"
                model_results = []
                n_runs = o3_runs_per_temp[temp]
                
                # Collect all completed runs for this model/prompt/temp combination
                for run_id in range(1, n_runs + 1):
                    metrics_file = f"results/metrics_{model}_{prompt_type}_run{run_id}_temp{temp}.json"
                    if os.path.exists(metrics_file):
                        try:
                            with open(metrics_file, 'r') as f:
                                run_result = json.load(f)
                            model_results.append(run_result)
                        except Exception as e:
                            print(f"⚠️ Could not load {metrics_file}: {e}")
                
                if model_results:
                    all_results[model_key] = model_results
                    print_summary_stats(f"{model} (temp={temp})", prompt_type, model_results)
    
    # Collect results from GPT-5 models
    for model in gpt5_models:
        for prompt_type in prompt_types:
            for temp in gpt5_temperatures:
                model_key = f"{model}_{prompt_type}_temp{temp}"
                model_results = []
                n_runs = gpt5_runs_per_temp[temp]
                
                # Collect all completed runs for this model/prompt/temp combination
                for run_id in range(1, n_runs + 1):
                    metrics_file = f"results/metrics_{model}_{prompt_type}_run{run_id}_temp{temp}.json"
                    if os.path.exists(metrics_file):
                        try:
                            with open(metrics_file, 'r') as f:
                                run_result = json.load(f)
                            model_results.append(run_result)
                        except Exception as e:
                            print(f"⚠️ Could not load {metrics_file}: {e}")
                
                if model_results:
                    all_results[model_key] = model_results
                    print_summary_stats(f"{model} (temp={temp})", prompt_type, model_results)
    
    # Final results
    final_completed = len(master_checkpoint['completed_tasks'])
    final_failed = len(master_checkpoint['failed_tasks'])
    
    print(f"\n{'='*60}")
    print("🏁 MULTI-MODEL MULTI-RUN EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Completed: {final_completed}/{total_tasks}")
    print(f"❌ Failed: {final_failed}")
    
    # Save complete results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    complete_results_file = f"results/variability_analysis_results_{timestamp}.json"
    
    with open(complete_results_file, "w") as f:
        json.dump({
            'all_results': all_results,
            'master_checkpoint': master_checkpoint,
            'summary': {
                'total_tasks': total_tasks,
                'completed_tasks': final_completed,
                'failed_tasks': final_failed,
                'timestamp': timestamp,
                'models': models,
                'o3_models': o3_models,
                'gpt5_models': gpt5_models,
                'prompt_types': prompt_types,
                'temperatures': temperatures,
                'runs_per_temp': n_runs_per_temp,
                'o3_temperatures': o3_temperatures,
                'o3_runs_per_temp': o3_runs_per_temp,
                'gpt5_temperatures': gpt5_temperatures,
                'gpt5_runs_per_temp': gpt5_runs_per_temp
            }
        }, f, indent=2)
    
    print(f"💾 Complete results saved to: {complete_results_file}")
    
    if final_completed == total_tasks:
        print(f"🎉 All {total_tasks} evaluation tasks completed successfully!")
        
        # Generate variability report
        generate_variability_report(all_results, timestamp)
        
        # Run final analysis
        print(f"\n🔄 Running final analysis pipeline...")
        try:
            import subprocess
            
            # Recompute all metrics
            result1 = subprocess.run([sys.executable, 'src/recompute_metrics.py'], 
                                   capture_output=True, text=True)
            if result1.returncode == 0:
                print("✅ Metrics recomputed")
            
            # Generate plots (final publication style)
            result2 = subprocess.run([sys.executable, 'src/generate_final_plots_with_values.py'], 
                                   capture_output=True, text=True)
            if result2.returncode == 0:
                print("✅ Plots generated (final values plots)")
                print("📊 Check 'figures/final_plots' for visualization results")
                
        except Exception as e:
            print(f"⚠️ Final analysis failed: {e}")
            print("💡 You can run 'python recompute_metrics.py' and 'python plot_results.py' manually")
        
        return True
    else:
        print(f"⚠️ {total_tasks - final_completed} tasks incomplete")
        if final_failed > 0:
            print(f"❌ {final_failed} tasks failed - check error messages above")
        print("🔄 Run this script again to retry failed tasks")
        return False

def main():
    """Main function."""

    import argparse

    parser = argparse.ArgumentParser(description="CausalJudge multi-model, multi-run evaluator")
    parser.add_argument(
        "--inner_max_concurrency",
        type=int,
        default=1,
        help="Maximum concurrent OpenAI calls per task (JSONL only, default: 1)",
    )
    args = parser.parse_args()

    print("CausalJudge Multi-Model Multi-Run Evaluation")
    print("=" * 50)
    print("This runs comprehensive evaluation with checkpoint resume")
    print("")
    print("🔧 CONFIGURATION:")
    print("  • Regular models (GPT-4o, GPT-4o-mini): 6 runs each × 2 prompts = 24 tasks")
    print("  • O3 models (o3-2025-04-16): 5 runs each × 2 prompts = 10 tasks")
    print("  • GPT-5 models (gpt-5-2025-08-07): 1 run each × 2 prompts = 2 tasks")
    print("  • Total: 36 tasks")
    print("  • Checkpoint-safe: interrupt and resume anytime")
    print("  • Auto-skips completed tasks")
    print(f"  • Max in-task OpenAI concurrency: {args.inner_max_concurrency}")
    print("")
    print("⏱️ Expected runtime: Several hours for complete run")
    print("💡 You can Ctrl+C anytime and resume later")

    try:
        success = run_multi_evaluation(inner_max_concurrency=args.inner_max_concurrency)
        if success:
            print(f"\n🎉 Multi-model evaluation completed successfully!")
        else:
            print(f"\n⚠️ Multi-model evaluation completed with some failures")

    except KeyboardInterrupt:
        print(f"\n\n⚠️ Evaluation interrupted by user")
        print("💾 Progress saved - run again to resume from checkpoint")
        return

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        print("💾 Progress saved - run again to resume from checkpoint")
        return

if __name__ == "__main__":
    main()
