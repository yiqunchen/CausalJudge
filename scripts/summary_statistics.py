#!/usr/bin/env python3
"""
Standalone script to generate summary statistics and tables from recomputed metrics.
This script doesn't require matplotlib or other plotting dependencies.
"""

import json
import glob
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any

class SummaryStatisticsGenerator:
    def __init__(self):
        """Initialize the summary statistics generator."""
        
        # The binary fields we evaluate
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
    
    def load_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all metrics files and organize them by model.
        
        Returns:
            Dictionary mapping model names to their metrics
        """
        metrics_files = glob.glob("metrics_*.json")
        all_metrics = {}
        
        for file_path in metrics_files:
            # Extract model name from filename
            model_name = file_path.replace('metrics_', '').replace('.json', '')
            
            with open(file_path, 'r') as f:
                metrics = json.load(f)
            
            all_metrics[model_name] = metrics
        
        return all_metrics
    
    def create_overall_summary_table(self, all_metrics: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a summary table with overall performance metrics for all models.
        
        Args:
            all_metrics: Dictionary of all model metrics
            
        Returns:
            DataFrame with overall summary
        """
        summary_data = []
        
        for model_name, metrics in all_metrics.items():
            if 'overall' in metrics:
                overall = metrics['overall']
                summary_data.append({
                    'Model': model_name,
                    'Accuracy': overall.get('accuracy', 0.0),
                    'Precision': overall.get('precision', 0.0),
                    'Recall': overall.get('recall', 0.0),
                    'F1': overall.get('f1', 0.0),
                    'AUC': overall.get('auc', 0.0),
                    'PR-AUC': overall.get('pr_auc', 0.0),
                    'N_Samples': overall.get('n_samples', 0),
                    'N_Positive': overall.get('n_positive', 0)
                })
        
        df = pd.DataFrame(summary_data)
        return df.sort_values('F1', ascending=False)
    
    def create_field_summary_table(self, all_metrics: Dict[str, Dict[str, Any]], 
                                 metric_name: str = 'f1') -> pd.DataFrame:
        """
        Create a summary table for a specific metric across all fields and models.
        
        Args:
            all_metrics: Dictionary of all model metrics
            metric_name: Metric to summarize ('accuracy', 'precision', 'recall', 'f1', 'auc', 'pr_auc')
            
        Returns:
            DataFrame with field-specific summary
        """
        summary_data = []
        
        for model_name, metrics in all_metrics.items():
            for field in self.BINARY_FIELDS:
                if field in metrics:
                    value = metrics[field].get(metric_name, 0.0)
                    summary_data.append({
                        'Model': model_name,
                        'Field': field,
                        metric_name.title(): value
                    })
        
        df = pd.DataFrame(summary_data)
        
        # Pivot to create a table with models as columns and fields as rows
        pivot_df = df.pivot(index='Field', columns='Model', values=metric_name.title())
        
        return pivot_df
    
    def create_best_performing_models_table(self, all_metrics: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a table showing the best performing model for each field.
        
        Args:
            all_metrics: Dictionary of all model metrics
            
        Returns:
            DataFrame with best models per field
        """
        best_models = {}
        
        for field in self.BINARY_FIELDS:
            field_results = {}
            
            for model_name, metrics in all_metrics.items():
                if field in metrics:
                    field_results[model_name] = {
                        'accuracy': metrics[field].get('accuracy', 0.0),
                        'precision': metrics[field].get('precision', 0.0),
                        'recall': metrics[field].get('recall', 0.0),
                        'f1': metrics[field].get('f1', 0.0),
                        'auc': metrics[field].get('auc', 0.0),
                        'pr_auc': metrics[field].get('pr_auc', 0.0)
                    }
            
            if field_results:
                # Find best model by F1 score
                best_model = max(field_results.keys(), key=lambda x: field_results[x]['f1'])
                best_models[field] = {
                    'Best_Model': best_model,
                    'Best_F1': field_results[best_model]['f1'],
                    'Best_Accuracy': field_results[best_model]['accuracy'],
                    'Best_Precision': field_results[best_model]['precision'],
                    'Best_Recall': field_results[best_model]['recall'],
                    'Best_AUC': field_results[best_model]['auc'],
                    'Best_PR_AUC': field_results[best_model]['pr_auc']
                }
        
        return pd.DataFrame.from_dict(best_models, orient='index')
    
    def create_model_comparison_table(self, all_metrics: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a detailed comparison table for all models.
        
        Args:
            all_metrics: Dictionary of all model metrics
            
        Returns:
            DataFrame with detailed model comparison
        """
        comparison_data = []
        
        for model_name, metrics in all_metrics.items():
            if 'overall' in metrics:
                overall = metrics['overall']
                
                # Calculate field-specific averages
                field_accuracies = []
                field_precisions = []
                field_recalls = []
                field_f1s = []
                field_aucs = []
                field_pr_aucs = []
                
                for field in self.BINARY_FIELDS:
                    if field in metrics:
                        field_accuracies.append(metrics[field].get('accuracy', 0.0))
                        field_precisions.append(metrics[field].get('precision', 0.0))
                        field_recalls.append(metrics[field].get('recall', 0.0))
                        field_f1s.append(metrics[field].get('f1', 0.0))
                        field_aucs.append(metrics[field].get('auc', 0.0))
                        field_pr_aucs.append(metrics[field].get('pr_auc', 0.0))
                
                comparison_data.append({
                    'Model': model_name,
                    'Overall_Accuracy': overall.get('accuracy', 0.0),
                    'Overall_Precision': overall.get('precision', 0.0),
                    'Overall_Recall': overall.get('recall', 0.0),
                    'Overall_F1': overall.get('f1', 0.0),
                    'Overall_AUC': overall.get('auc', 0.0),
                    'Overall_PR_AUC': overall.get('pr_auc', 0.0),
                    'Avg_Field_Accuracy': sum(field_accuracies) / len(field_accuracies) if field_accuracies else 0.0,
                    'Avg_Field_Precision': sum(field_precisions) / len(field_precisions) if field_precisions else 0.0,
                    'Avg_Field_Recall': sum(field_recalls) / len(field_recalls) if field_recalls else 0.0,
                    'Avg_Field_F1': sum(field_f1s) / len(field_f1s) if field_f1s else 0.0,
                    'Avg_Field_AUC': sum(field_aucs) / len(field_aucs) if field_aucs else 0.0,
                    'Avg_Field_PR_AUC': sum(field_pr_aucs) / len(field_pr_aucs) if field_pr_aucs else 0.0,
                    'N_Samples': overall.get('n_samples', 0),
                    'N_Positive': overall.get('n_positive', 0)
                })
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('Overall_F1', ascending=False)
    
    def save_tables_to_csv(self, all_metrics: Dict[str, Dict[str, Any]], output_prefix: str = None):
        """
        Save all summary tables to CSV files.
        
        Args:
            all_metrics: Dictionary of all model metrics
            output_prefix: Prefix for output files
        """
        if output_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_prefix = f"summary_statistics_{timestamp}"
        
        # Overall summary table
        overall_df = self.create_overall_summary_table(all_metrics)
        overall_df.to_csv(f"{output_prefix}_overall_summary.csv", index=False)
        print(f"Overall summary saved to: {output_prefix}_overall_summary.csv")
        
        # Field-specific tables for each metric
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'pr_auc']:
            field_df = self.create_field_summary_table(all_metrics, metric)
            field_df.to_csv(f"{output_prefix}_{metric}_by_field.csv")
            print(f"{metric.title()} by field saved to: {output_prefix}_{metric}_by_field.csv")
        
        # Best performing models table
        best_models_df = self.create_best_performing_models_table(all_metrics)
        best_models_df.to_csv(f"{output_prefix}_best_models_per_field.csv")
        print(f"Best models per field saved to: {output_prefix}_best_models_per_field.csv")
        
        # Detailed model comparison table
        comparison_df = self.create_model_comparison_table(all_metrics)
        comparison_df.to_csv(f"{output_prefix}_detailed_model_comparison.csv", index=False)
        print(f"Detailed model comparison saved to: {output_prefix}_detailed_model_comparison.csv")
    
    def print_summary_report(self, all_metrics: Dict[str, Dict[str, Any]]):
        """
        Print a comprehensive summary report to console.
        
        Args:
            all_metrics: Dictionary of all model metrics
        """
        print("=" * 80)
        print("CAUSAL EVALUATION SUMMARY REPORT")
        print("=" * 80)
        print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Number of models evaluated: {len(all_metrics)}")
        print()
        
        # Overall performance summary
        print("OVERALL PERFORMANCE SUMMARY (sorted by F1 score):")
        print("-" * 80)
        overall_df = self.create_overall_summary_table(all_metrics)
        print(overall_df.to_string(index=False, float_format='%.3f'))
        print()
        
        # Best performing model for each field
        print("BEST PERFORMING MODEL FOR EACH FIELD (by F1 score):")
        print("-" * 80)
        best_models_df = self.create_best_performing_models_table(all_metrics)
        print(best_models_df.to_string(float_format='%.3f'))
        print()
        
        # Top 3 models by overall F1
        print("TOP 3 MODELS BY OVERALL F1 SCORE:")
        print("-" * 80)
        top_3 = overall_df.head(3)
        for idx, row in top_3.iterrows():
            print(f"{row['Model']}: F1={row['F1']:.3f}, Accuracy={row['Accuracy']:.3f}, "
                  f"Precision={row['Precision']:.3f}, Recall={row['Recall']:.3f}")
        print()
        
        # Field-specific insights
        print("FIELD-SPECIFIC INSIGHTS:")
        print("-" * 80)
        for field in self.BINARY_FIELDS:
            field_results = {}
            for model_name, metrics in all_metrics.items():
                if field in metrics:
                    field_results[model_name] = metrics[field].get('f1', 0.0)
            
            if field_results:
                best_model = max(field_results.keys(), key=lambda x: field_results[x])
                worst_model = min(field_results.keys(), key=lambda x: field_results[x])
                avg_f1 = sum(field_results.values()) / len(field_results)
                
                print(f"{field}:")
                print(f"  Best: {best_model} (F1={field_results[best_model]:.3f})")
                print(f"  Worst: {worst_model} (F1={field_results[worst_model]:.3f})")
                print(f"  Average F1: {avg_f1:.3f}")
                print()

def main():
    """Main function to generate summary statistics."""
    
    # Initialize the generator
    generator = SummaryStatisticsGenerator()
    
    # Load all metrics
    print("Loading metrics files...")
    all_metrics = generator.load_all_metrics()
    
    if not all_metrics:
        print("No metrics files found!")
        return
    
    print(f"Loaded metrics for {len(all_metrics)} models:")
    for model_name in sorted(all_metrics.keys()):
        print(f"  - {model_name}")
    print()
    
    # Generate and save tables
    print("Generating summary tables...")
    generator.save_tables_to_csv(all_metrics)
    print()
    
    # Print summary report
    generator.print_summary_report(all_metrics)
    
    print("=" * 80)
    print("SUMMARY STATISTICS GENERATION COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main() 