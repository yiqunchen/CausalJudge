#!/usr/bin/env python3
"""
Script to run causal evaluation with checkpoint support.
This allows you to interrupt and resume processing at any time.
"""

import os
import sys
from causal_evaluation import CausalEvaluationSystem

def run_with_checkpoints():
    """Run evaluation with checkpoint support."""
    
    # Initialize the system
    api_key = os.getenv('CHEN_OPENAI_API_KEY')
    if not api_key:
        print("Please set CHEN_OPENAI_API_KEY environment variable")
        return
    
    # Configuration
    model = "gpt-4o-mini"  # Change this to your desired model
    input_file = "PMID_all_text.jsonl"
    ground_truth_file = "PMID/GoldenStandard180.csv"
    max_cases = 180
    
    # Output files
    output_file = f"predictions_{model}.json"
    checkpoint_file = f"checkpoint_{model}.pkl"
    
    print(f"Starting evaluation with model: {model}")
    print(f"Input file: {input_file}")
    print(f"Ground truth: {ground_truth_file}")
    print(f"Max cases: {max_cases}")
    print(f"Checkpoint file: {checkpoint_file}")
    print(f"Checkpoint interval: Every 5 articles")
    print("\nYou can interrupt this process at any time with Ctrl+C")
    print("The system will automatically resume from the last checkpoint when restarted.")
    print("-" * 80)
    
    try:
        # Initialize system
        system = CausalEvaluationSystem(api_key=api_key, model=model)
        
        # Process articles with checkpoint support
        results = system.process_articles(
            input_file=input_file,
            max_cases=max_cases,
            prompt_type="basic",
            output_file=output_file,
            skip_existing=True,
            checkpoint_file=checkpoint_file,
            checkpoint_interval=5
        )
        
        print(f"\n✓ Processing completed successfully!")
        print(f"  Total PMIDs processed: {len(results['predictions'])}")
        
        # Evaluate performance
        print(f"\nEvaluating performance...")
        confidence_file = output_file.replace('.json', '_confidence.json')
        metrics = system.evaluate_performance(
            ground_truth_file=ground_truth_file,
            predictions_file=output_file,
            confidence_file=confidence_file
        )
        
        print(f"\n✓ Evaluation completed!")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Process interrupted by user.")
        print(f"✓ Progress has been saved to checkpoint: {checkpoint_file}")
        print(f"  You can restart this script to resume from where you left off.")
        return
        
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        print(f"✓ Progress has been saved to checkpoint: {checkpoint_file}")
        print(f"  You can restart this script to resume from where you left off.")
        return

if __name__ == "__main__":
    run_with_checkpoints() 