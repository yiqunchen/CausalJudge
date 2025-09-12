#!/usr/bin/env python3
"""
Standalone script to recompute causal evaluation metrics.
This script uses the original rows from 180FinalResult_Jun17.xlsx as ground truth
and doesn't require the CausalEvaluationSystem class or OpenAI dependencies.
"""

import pandas as pd
import json
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, average_precision_score
from datetime import datetime

class StandaloneMetricsComputer:
    def __init__(self):
        """Initialize the standalone metrics computer."""
        
        # The binary fields we evaluate (matching causal_evaluation.py)
        self.BINARY_FIELDS = [
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
    
    def load_ground_truth_from_excel(self, excel_file: str) -> Dict[str, Dict[str, Any]]:
        """
        Load ground truth data from Excel file, using only rows with Type="original".
        
        Args:
            excel_file: Path to the Excel file (180FinalResult_Jun17.xlsx)
            
        Returns:
            Dictionary mapping PMID to ground truth data
        """
        print(f"Loading ground truth from {excel_file}...")
        
        # Read the Excel file
        df = pd.read_excel(excel_file)
        
        # Filter to only original rows
        original_df = df[df['Type'] == 'original'].copy()
        print(f"Found {len(original_df)} original rows out of {len(df)} total rows")
        
        # Convert to dictionary format
        ground_truth = {}
        for _, row in original_df.iterrows():
            pmid = str(row.get("PMID", ""))
            if pmid and pmid != "nan":
                ground_truth[pmid] = {}
                for field in self.BINARY_FIELDS:
                    ground_truth[pmid][field] = row.get(field, "")
        
        print(f"Loaded ground truth for {len(ground_truth)} PMIDs")
        return ground_truth
    
    def load_ground_truth_from_json(self, json_file: str) -> Dict[str, Dict[str, Any]]:
        """
        Load ground truth data from clean JSON file (extracted from GoldenStandard180.csv).
        
        Args:
            json_file: Path to the clean ground truth JSON file
            
        Returns:
            Dictionary mapping PMID to ground truth data
        """
        print(f"Loading clean ground truth from {json_file}...")
        
        if not Path(json_file).exists():
            raise FileNotFoundError(f"Ground truth file not found: {json_file}. Run 'python extract_ground_truth.py' first.")
        
        with open(json_file, 'r') as f:
            ground_truth = json.load(f)
        
        print(f"Clean ground truth loaded for {len(ground_truth)} PMIDs")
        print(f"All binary fields are already in clean 0/1 format")
        
        return ground_truth
    
    def load_model_predictions(self, predictions_file: str) -> Dict[str, Dict[str, Any]]:
        """
        Load model predictions from JSON file.
        
        Args:
            predictions_file: Path to predictions JSON file
            
        Returns:
            Dictionary mapping PMID to prediction data
        """
        print(f"Loading predictions from {predictions_file}...")
        
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)
        
        print(f"Loaded predictions for {len(predictions)} PMIDs")
        return predictions
    
    def load_confidence_scores(self, confidence_file: str) -> Dict[str, Dict[str, float]]:
        """
        Load confidence scores from JSON file.
        
        Args:
            confidence_file: Path to confidence scores JSON file
            
        Returns:
            Dictionary mapping PMID to confidence scores
        """
        print(f"Loading confidence scores from {confidence_file}...")
        
        if os.path.exists(confidence_file):
            with open(confidence_file, 'r') as f:
                confidence_scores = json.load(f)
            print(f"Loaded confidence scores for {len(confidence_scores)} PMIDs")
        else:
            print(f"Confidence file {confidence_file} not found, using default scores")
            confidence_scores = {}
        
        return confidence_scores
    
    def calculate_metrics(self, ground_truth: Dict[str, Dict[str, Any]], 
                         predictions: Dict[str, Dict[str, Any]], 
                         confidence_scores: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            ground_truth: Ground truth data
            predictions: Model predictions
            confidence_scores: Confidence scores for predictions
            
        Returns:
            Dictionary with all evaluation metrics
        """
        print("Calculating metrics...")
        results = {}
        
        for field in self.BINARY_FIELDS:
            gt_values = []
            pred_values = []
            conf_values = []
            
            for pmid in ground_truth:
                if pmid in predictions:
                    # Ground truth
                    gt_val = ground_truth[pmid].get(field, "")
                    gt_binary = 1 if str(gt_val).lower() in ['1', 'yes', 'true'] else 0
                    
                    # Prediction - check if binary label is contained in the output
                    pred_val = predictions[pmid].get(field, "")
                    pred_str = str(pred_val).lower()
                    
                    # Check if "1" is contained in the prediction (handles "0/1", "1/", "1", "yes", etc.)
                    if '1' in pred_str or 'yes' in pred_str or 'true' in pred_str:
                        pred_binary = 1
                    # Check if "0" is contained in the prediction (handles "0/1", "/0", "0", "no", etc.)
                    elif '0' in pred_str or 'no' in pred_str or 'false' in pred_str:
                        pred_binary = 0
                    else:
                        # If no clear binary indicator, default to 0
                        pred_binary = 0
                    
                    # Confidence
                    conf_val = confidence_scores.get(pmid, {}).get(field, 0.5)
                    
                    gt_values.append(gt_binary)
                    pred_values.append(pred_binary)
                    conf_values.append(conf_val)
            
            if len(gt_values) > 0:
                # Debug information
                n_positive_gt = sum(gt_values)
                n_positive_pred = sum(pred_values)
                print(f"    {field}: GT positives={n_positive_gt}, Pred positives={n_positive_pred}")
                
                # Basic metrics
                accuracy = accuracy_score(gt_values, pred_values)
                precision, recall, f1, _ = precision_recall_fscore_support(gt_values, pred_values, average='binary', zero_division=0)
                
                # AUC and PR-AUC using confidence scores as probabilities
                try:
                    # Check if we have both positive and negative classes
                    if len(set(gt_values)) > 1:
                        auc = roc_auc_score(gt_values, conf_values)
                    else:
                        auc = 0.5  # Default for single class
                except:
                    auc = 0.5
                    
                try:
                    # Check if we have positive class for PR-AUC
                    if sum(gt_values) > 0:
                        pr_auc = average_precision_score(gt_values, conf_values)
                    else:
                        pr_auc = 0.0  # No positive class
                except:
                    pr_auc = 0.0
                
                results[field] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc,
                    'pr_auc': pr_auc,
                    'n_samples': len(gt_values)
                }
        
        # Overall metrics
        all_gt = []
        all_pred = []
        all_conf = []
        
        for field in self.BINARY_FIELDS:
            if field in results:
                for pmid in ground_truth:
                    if pmid in predictions:
                        gt_val = ground_truth[pmid].get(field, "")
                        gt_binary = 1 if str(gt_val).lower() in ['1', 'yes', 'true'] else 0
                        
                        # Prediction - check if binary label is contained in the output
                        pred_val = predictions[pmid].get(field, "")
                        pred_str = str(pred_val).lower()
                        
                        # Check if "1" is contained in the prediction (handles "0/1", "1/", "1", "yes", etc.)
                        if '1' in pred_str or 'yes' in pred_str or 'true' in pred_str:
                            pred_binary = 1
                        # Check if "0" is contained in the prediction (handles "0/1", "/0", "0", "no", etc.)
                        elif '0' in pred_str or 'no' in pred_str or 'false' in pred_str:
                            pred_binary = 0
                        else:
                            # If no clear binary indicator, default to 0
                            pred_binary = 0
                        
                        conf_val = confidence_scores.get(pmid, {}).get(field, 0.5)
                        
                        all_gt.append(gt_binary)
                        all_pred.append(pred_binary)
                        all_conf.append(conf_val)
        
        if len(all_gt) > 0:
            # Check for edge cases in overall metrics
            unique_classes = len(set(all_gt))
            positive_samples = sum(all_gt)
            
            results['overall'] = {
                'accuracy': accuracy_score(all_gt, all_pred),
                'precision': precision_recall_fscore_support(all_gt, all_pred, average='macro', zero_division=0)[0],
                'recall': precision_recall_fscore_support(all_gt, all_pred, average='macro', zero_division=0)[1],
                'f1': precision_recall_fscore_support(all_gt, all_pred, average='macro', zero_division=0)[2],
                'auc': roc_auc_score(all_gt, all_conf) if unique_classes > 1 else 0.5,
                'pr_auc': average_precision_score(all_gt, all_conf) if positive_samples > 0 else 0.0,
                'n_samples': len(all_gt),
                'n_positive': positive_samples,
                'n_classes': unique_classes
            }
        
        return results
    
    def evaluate_performance(self, ground_truth_file: str, predictions_file: str, 
                           confidence_file: str = None) -> Dict[str, Any]:
        """
        Evaluate model performance against ground truth.
        
        Args:
            ground_truth_file: Path to ground truth Excel file
            predictions_file: Path to predictions JSON file
            confidence_file: Path to confidence scores JSON file (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Ensure clean ground truth exists; generate if missing
        if not os.path.exists(ground_truth_file):
            print(f"Clean ground truth file not found: {ground_truth_file}")
            print("Running ground truth extraction...")
            import subprocess
            # Call extractor in src with standardized output in data/processed
            result = subprocess.run([sys.executable, 'src/extract_ground_truth.py'], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Ground truth extraction failed: {result.stderr}")
                raise FileNotFoundError("Unable to create clean ground truth file")
            else:
                print("Ground truth extraction completed successfully")

        # Load ground truth (JSON preferred)
        if ground_truth_file.endswith('.json'):
            ground_truth = self.load_ground_truth_from_json(ground_truth_file)
        else:
            ground_truth = self.load_ground_truth_from_excel(ground_truth_file)
        predictions = self.load_model_predictions(predictions_file)
        
        if confidence_file and os.path.exists(confidence_file):
            confidence_scores = self.load_confidence_scores(confidence_file)
        else:
            confidence_scores = {pmid: {field: 0.5 for field in self.BINARY_FIELDS} 
                               for pmid in predictions}
        
        metrics = self.calculate_metrics(ground_truth, predictions, confidence_scores)
        
        # Print detailed metrics by column
        print(f"\n{'='*80}")
        print("DETAILED METRICS BY COLUMN")
        print(f"{'='*80}")
        
        print(f"{'Field':<40} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10} {'PR-AUC':<10} {'GT+':<5} {'Pred+':<6}")
        print("-" * 100)
        
        for field in self.BINARY_FIELDS:
            if field in metrics:
                m = metrics[field]
                gt_positives = sum(1 for pmid in ground_truth if pmid in predictions and 
                                 str(ground_truth[pmid].get(field, "")).lower() in ['1', 'yes', 'true'])
                pred_positives = sum(1 for pmid in predictions if pmid in ground_truth and 
                                   str(predictions[pmid].get(field, "")).lower() in ['1', 'yes', 'true'])
                
                print(f"{field:<40} {m['accuracy']:<10.3f} {m['precision']:<10.3f} {m['recall']:<10.3f} "
                      f"{m['f1']:<10.3f} {m['auc']:<10.3f} {m['pr_auc']:<10.3f} {gt_positives:<5} {pred_positives:<6}")
        
        # Print overall summary
        print(f"\n{'='*80}")
        print("OVERALL SUMMARY")
        print(f"{'='*80}")
        print(f"  Ground truth PMIDs: {len(ground_truth)}")
        print(f"  Prediction PMIDs: {len(predictions)}")
        print(f"  Overlapping PMIDs: {len(set(ground_truth.keys()) & set(predictions.keys()))}")
        
        if 'overall' in metrics:
            overall = metrics['overall']
            print(f"  Overall samples: {overall['n_samples']}")
            print(f"  Positive samples: {overall['n_positive']}")
            print(f"  Unique classes: {overall['n_classes']}")
            print(f"  Overall Accuracy: {overall['accuracy']:.3f}")
            print(f"  Overall F1: {overall['f1']:.3f}")
            print(f"  Overall AUC: {overall['auc']:.3f}")
            print(f"  Overall PR-AUC: {overall['pr_auc']:.3f}")
        
        return metrics
    
    def save_metrics(self, metrics: Dict[str, Any], output_file: str):
        """
        Save metrics to JSON file.
        
        Args:
            metrics: Metrics dictionary
            output_file: Output file path
        """
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {output_file}")

def main():
    """Main function to recompute metrics for all prediction files."""
    
    # Initialize the metrics computer
    computer = StandaloneMetricsComputer()
    
    # Ground truth file (Excel with original rows)
    # Use clean ground truth file (auto-generated if needed)
    ground_truth_file = "data/processed/ground_truth_clean.json"
    
    # Find all prediction files
    # Look for prediction files in both current directory and results directory
    prediction_files = glob.glob("predictions_*.json") + glob.glob("results/predictions_*.json")
    # Only use underscore naming, exclude confidence files and gpt_3.5_turbo
    prediction_files = [f for f in prediction_files if 
                       "confidence" not in f and 
                       "_" in f and 
                       "-" not in f and
                       not any(x in f for x in ["gpt_3.5", "gpt-3.5"])]
    print(f"Found {len(prediction_files)} prediction files:")
    for file in prediction_files:
        print(f"  - {file}")
    
    if not prediction_files:
        print("No prediction files found!")
        return
    
    # Process each prediction file
    for pred_file in prediction_files:
        print(f"\n{'='*80}")
        print(f"PROCESSING: {pred_file}")
        print(f"{'='*80}")
        
        # Determine confidence file name
        conf_file = pred_file.replace('.json', '_confidence.json')
        
        # Extract model name from filename
        model_name = pred_file.replace('predictions_', '').replace('.json', '')
        
        # Normalize ALL model names to use underscores consistently
        model_name = model_name.replace("o3-2025-04-16", "o3_2025_04_16")
        model_name = model_name.replace("gpt-4o", "gpt_4o")
        model_name = model_name.replace("gpt-4.1", "gpt_4.1")
        
        # Evaluate performance
        metrics = computer.evaluate_performance(
            ground_truth_file=ground_truth_file,
            predictions_file=pred_file,
            confidence_file=conf_file
        )
        
        # Save metrics
        metrics_file = f"results/metrics_{model_name}.json"
        os.makedirs('results', exist_ok=True)
        computer.save_metrics(metrics, metrics_file)
        
        print(f"✓ Completed evaluation for {model_name}")
    
    print(f"\n{'='*80}")
    print("ALL EVALUATIONS COMPLETED")
    print(f"{'='*80}")
    print("Generated metrics files:")
    for pred_file in prediction_files:
        model_name = pred_file.replace('predictions_', '').replace('.json', '')
        metrics_file = f"results/metrics_{model_name}.json"
        print(f"  - {metrics_file}")

if __name__ == "__main__":
    main() 
