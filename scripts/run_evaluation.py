#!/usr/bin/env python3
"""
CausalJudge Main Evaluation Script

Provides both single and multi-model evaluation options.
Use this as the main entry point.
"""

import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='CausalJudge Evaluation System')
    parser.add_argument('--mode', choices=['single', 'multi'], default='multi',
                       help='Evaluation mode: single model or multi-model multi-run (default: multi)')
    
    # Single mode arguments
    parser.add_argument('--model', default='gpt-4o-mini', help='Model for single evaluation')
    parser.add_argument('--prompt_type', default='detailed', choices=['basic', 'detailed', 'examples'])
    parser.add_argument('--run_id', type=int, default=1, help='Run ID for single evaluation')
    
    args, unknown_args = parser.parse_known_args()
    
    if args.mode == 'single':
        print(f"🎯 Running single evaluation: {args.model} ({args.prompt_type}) Run {args.run_id}")
        cmd = [sys.executable, 'run_single_evaluation.py', 
               '--model', args.model, 
               '--prompt_type', args.prompt_type,
               '--run_id', str(args.run_id)] + unknown_args
        result = subprocess.run(cmd)
        return result.returncode
        
    else:  # multi mode
        print("🚀 Running multi-model multi-run evaluation...")
        print("This will run 5 models × 2 prompts × 5 runs = 50 evaluations")
        print("=" * 60)
        
        # Call the multi-evaluation script
        result = subprocess.run([sys.executable, 'run_multi_evaluation.py'] + unknown_args)
        return result.returncode

if __name__ == "__main__":
    sys.exit(main())