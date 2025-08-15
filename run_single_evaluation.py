#!/usr/bin/env python3
"""
Single Model Evaluation Script with Checkpoint Support
This is the core single evaluation routine called by the multi-run script.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from causal_evaluation import CausalEvaluationSystem

def run_single_evaluation(api_key, model, prompt_type, run_id=1, temperature=None, resume_checkpoint=True, max_concurrency: int = 1):
    """
    Run a single model evaluation with checkpoint support.
    
    Args:
        api_key: OpenAI API key
        model: Model name (e.g., 'gpt-4o-mini')
        prompt_type: Prompt type ('basic' or 'detailed')
        run_id: Run identifier for this specific run
        resume_checkpoint: Whether to resume from checkpoint if exists
        
    Returns:
        Dictionary with results and metrics
    """
    
    # Configuration
    input_file = "PMID_all_text.jsonl"
    ground_truth_file = "ground_truth_clean.json" 
    max_cases = 180
    
    # Output files with temperature suffix
    model_safe = model.replace('-', '_')
    temp_suffix = f"_temp{temperature}" if temperature is not None else "_temp_default"
    output_file = f"results/predictions_{model_safe}_{prompt_type}_run{run_id}{temp_suffix}.json"
    conf_file = f"results/predictions_{model_safe}_{prompt_type}_run{run_id}{temp_suffix}_confidence.json"
    checkpoint_file = f"results/checkpoint_{model_safe}_{prompt_type}_run{run_id}{temp_suffix}.pkl"
    metrics_file = f"results/metrics_{model}_{prompt_type}_run{run_id}{temp_suffix}.json"
    
    # Create temperature description for logging
    temp_desc = f" temp={temperature}" if temperature is not None else " (default temp)"
    
    # Check if already completed
    if resume_checkpoint and is_evaluation_complete(output_file, conf_file, metrics_file):
        print(f"✓ {model} ({prompt_type}) Run {run_id}{temp_desc} already completed, loading existing results...")
        try:
            with open(metrics_file, 'r') as f:
                existing_result = json.load(f)
            return existing_result
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}. Re-running...")
    
    print(f"Starting evaluation: {model} ({prompt_type}) Run {run_id}{temp_desc}")
    print(f"  Input: {input_file}")
    print(f"  Ground truth: {ground_truth_file}")
    print(f"  Max cases: {max_cases}")
    print(f"  Checkpoint: {checkpoint_file}")
    
    try:
        # Initialize evaluator
        evaluator = CausalEvaluationSystem(api_key, model=model)
        
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)
        
        # Process articles with checkpoint support
        print(f"Processing {max_cases} articles...")
        results = evaluator.process_articles(
            input_file=input_file,
            max_cases=max_cases,
            prompt_type=prompt_type,
            output_file=output_file,
            skip_existing=False,  # Always process this specific run
            checkpoint_file=checkpoint_file,
            checkpoint_interval=5,  # Save checkpoint every 5 articles
            temperature=temperature,
            max_concurrency=max_concurrency,
        )
        
        # Evaluate performance
        print(f"Evaluating performance...")
        
        # Ensure ground truth exists
        if not os.path.exists(ground_truth_file):
            print(f"Ground truth not found, extracting from Excel file...")
            import subprocess
            result = subprocess.run([sys.executable, 'extract_ground_truth.py'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Ground truth extraction failed: {result.stderr}")
        
        metrics = evaluator.evaluate_performance(
            ground_truth_file=ground_truth_file,
            predictions_file=output_file,
            confidence_file=conf_file
        )
        
        # Create result summary
        task_id_suffix = temp_suffix.replace('_temp', '_temp') if temperature is not None else '_temp_default'
        run_result = {
            'run_id': run_id,
            'model': model,
            'prompt_type': prompt_type,
            'temperature': temperature,
            'metrics': metrics,
            'n_predictions': len(results['predictions']),
            'output_file': output_file,
            'confidence_file': conf_file,
            'checkpoint_file': checkpoint_file,
            'timestamp': datetime.now().isoformat(),
            'task_id': f"{model}_{prompt_type}_run{run_id}{task_id_suffix}"
        }
        
        # Save metrics
        with open(metrics_file, 'w') as f:
            json.dump(run_result, f, indent=2)
        
        print(f"✓ {model} ({prompt_type}) Run {run_id}{temp_desc} completed successfully")
        if 'overall' in metrics:
            overall = metrics['overall']
            print(f"  Accuracy: {overall['accuracy']:.3f}, F1: {overall['f1']:.3f}, AUC: {overall['auc']:.3f}")
        
        return run_result
        
    except Exception as e:
        print(f"✗ {model} ({prompt_type}) Run {run_id}{temp_desc} failed: {e}")
        raise e

def is_evaluation_complete(output_file, conf_file, metrics_file):
    """Check if evaluation is already complete."""
    
    # Check if all required files exist
    if not all(os.path.exists(f) for f in [output_file, conf_file, metrics_file]):
        return False
    
    # Check if prediction file has expected content
    try:
        with open(output_file, 'r') as f:
            predictions = json.load(f)
        
        with open(conf_file, 'r') as f:
            confidence = json.load(f)
            
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Should have 180 PMIDs and valid metrics
        return (len(predictions) >= 180 and 
                len(confidence) >= 180 and 
                'metrics' in metrics and 
                'overall' in metrics['metrics'])
    except:
        return False

def main():
    """Main function for single evaluation."""
    
    parser = argparse.ArgumentParser(description='Run single model evaluation')
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--prompt_type', default='detailed', choices=['basic', 'detailed', 'examples'])
    parser.add_argument('--run_id', type=int, default=1, help='Run ID')
    parser.add_argument('--temperature', type=float, help='Temperature for model (overrides model default)')
    parser.add_argument('--api_key', help='API key (or set CHEN_OPENAI_API_KEY)')
    parser.add_argument('--no_resume', action='store_true', help='Do not resume from checkpoint')
    parser.add_argument('--max_concurrency', type=int, default=1, help='Max concurrent OpenAI calls (JSONL only)')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('CHEN_OPENAI_API_KEY')
    if not api_key:
        print("Error: Please provide API key via --api_key or CHEN_OPENAI_API_KEY environment variable")
        return 1
    
    try:
        result = run_single_evaluation(
            api_key=api_key,
            model=args.model,
            prompt_type=args.prompt_type,
            run_id=args.run_id,
            temperature=args.temperature,
            resume_checkpoint=not args.no_resume,
            max_concurrency=args.max_concurrency,
        )
        
        print(f"\nSingle evaluation completed:")
        print(f"  Model: {result['model']}")
        print(f"  Prompt: {result['prompt_type']}")
        print(f"  Run ID: {result['run_id']}")
        print(f"  Temperature: {result.get('temperature', 'default')}")
        print(f"  Predictions: {result['n_predictions']}")
        
        if 'overall' in result['metrics']:
            overall = result['metrics']['overall']
            print(f"  Overall Accuracy: {overall['accuracy']:.3f}")
            print(f"  Overall F1: {overall['f1']:.3f}")
        
        return 0
        
    except Exception as e:
        print(f"Single evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())