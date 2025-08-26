#!/usr/bin/env python3
"""
Script to extract and validate ground truth data from GoldenStandard180.csv.
This script processes the CSV file to extract only the "original" Type rows 
and ensures we have exactly one entry per PMID for all 14 binary fields.
"""

import pandas as pd
import json
from typing import Dict, Any, List
from pathlib import Path

class GroundTruthExtractor:
    def __init__(self):
        """Initialize the ground truth extractor."""
        
        # The 14 binary fields we evaluate
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
    
    def load_and_validate_ground_truth(self, excel_file: str) -> Dict[str, Dict[str, Any]]:
        """
        Load and validate ground truth data from Excel file.
        
        Args:
            excel_file: Path to the Excel file (180FinalResult_Jun17.xlsx)
            
        Returns:
            Dictionary mapping PMID to ground truth data
        """
        print(f"Loading ground truth from {excel_file}...")
        
        if not Path(excel_file).exists():
            raise FileNotFoundError(f"Ground truth file not found: {excel_file}")
        
        # Read the Excel file
        df = pd.read_excel(excel_file)
        print(f"Loaded CSV with {len(df)} total rows and {len(df.columns)} columns")
        
        # Show column names
        print(f"Columns: {list(df.columns)}")
        
        # Check for Type column
        if 'Type' not in df.columns:
            raise ValueError("CSV file must have a 'Type' column")
        
        # Filter to only original rows
        original_df = df[df['Type'] == 'original'].copy()
        print(f"Found {len(original_df)} rows with Type='original' out of {len(df)} total rows")
        
        if len(original_df) == 0:
            raise ValueError("No rows found with Type='original'")
        
        # Check for PMID column
        if 'PMID' not in original_df.columns:
            raise ValueError("CSV file must have a 'PMID' column")
        
        # Check for duplicate PMIDs
        pmids = original_df['PMID'].astype(str)
        duplicate_pmids = pmids[pmids.duplicated()].unique()
        if len(duplicate_pmids) > 0:
            print(f"WARNING: Found {len(duplicate_pmids)} duplicate PMIDs: {list(duplicate_pmids)[:5]}...")
            print("Using first occurrence of each PMID")
            original_df = original_df.drop_duplicates(subset=['PMID'], keep='first')
        
        print(f"Final dataset: {len(original_df)} unique PMIDs")
        
        # Validate that all binary fields are present
        missing_fields = []
        for field in self.BINARY_FIELDS:
            if field not in original_df.columns:
                missing_fields.append(field)
        
        if missing_fields:
            raise ValueError(f"Missing required binary fields: {missing_fields}")
        
        print(f"✓ All {len(self.BINARY_FIELDS)} binary fields found")
        
        # Convert to dictionary format
        ground_truth = {}
        for _, row in original_df.iterrows():
            pmid = str(row['PMID'])
            ground_truth[pmid] = row.to_dict()
        
        print(f"✓ Ground truth extracted for {len(ground_truth)} PMIDs")
        
        return ground_truth
    
    def analyze_field_values(self, ground_truth: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the values in each binary field to understand the data format.
        
        Args:
            ground_truth: Dictionary mapping PMID to ground truth data
            
        Returns:
            Analysis results
        """
        print(f"\nAnalyzing field values...")
        analysis = {}
        
        for field in self.BINARY_FIELDS:
            values = []
            for pmid, data in ground_truth.items():
                if field in data:
                    values.append(data[field])
            
            unique_values = list(set([str(v) for v in values]))
            unique_values.sort()
            
            analysis[field] = {
                'unique_values': unique_values,
                'total_count': len(values),
                'value_counts': {}
            }
            
            # Count occurrences of each value
            for val in values:
                str_val = str(val)
                analysis[field]['value_counts'][str_val] = analysis[field]['value_counts'].get(str_val, 0) + 1
            
            print(f"  {field}:")
            print(f"    Unique values: {unique_values}")
            print(f"    Value counts: {analysis[field]['value_counts']}")
        
        return analysis
    
    def convert_to_binary(self, ground_truth: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Convert ground truth values to clean binary format (0/1).
        Handles various formats like "1/0", "__/0", "0/__", etc.
        
        Args:
            ground_truth: Raw ground truth data
            
        Returns:
            Clean binary ground truth data
        """
        print(f"\nConverting to binary format...")
        
        clean_ground_truth = {}
        conversion_stats = {}
        
        for pmid, data in ground_truth.items():
            clean_data = {}
            
            # Copy non-binary fields as-is
            for key, value in data.items():
                if key not in self.BINARY_FIELDS:
                    clean_data[key] = value
            
            # Process binary fields
            for field in self.BINARY_FIELDS:
                if field in data:
                    original_value = str(data[field])
                    clean_value = self._convert_single_value_to_binary(original_value)
                    clean_data[field] = clean_value
                    
                    # Track conversion statistics
                    if field not in conversion_stats:
                        conversion_stats[field] = {}
                    if original_value not in conversion_stats[field]:
                        conversion_stats[field][original_value] = {'count': 0, 'converted_to': clean_value}
                    conversion_stats[field][original_value]['count'] += 1
                else:
                    clean_data[field] = 0  # Default to 0 if missing
            
            clean_ground_truth[pmid] = clean_data
        
        # Print conversion statistics
        print("Conversion statistics:")
        for field in self.BINARY_FIELDS:
            if field in conversion_stats:
                print(f"  {field}:")
                for orig_val, stats in conversion_stats[field].items():
                    print(f"    '{orig_val}' -> {stats['converted_to']} ({stats['count']} times)")
        
        print(f"✓ Converted {len(clean_ground_truth)} PMIDs to binary format")
        
        return clean_ground_truth
    
    def _convert_single_value_to_binary(self, value: str) -> int:
        """
        Convert a single value to binary (0 or 1).
        
        Conversion rules:
        - "1", "1/0", "1/__" -> 1
        - "0", "0/1", "__/0", "0/__" -> 0
        - "__", "nan", empty -> 0
        - Other values -> 0 (with warning)
        """
        value = str(value).strip().lower()
        
        # Handle NaN and empty values
        if value in ['nan', 'none', '', '__']:
            return 0
        
        # Handle clear binary values
        if value == '1':
            return 1
        if value == '0':
            return 0
        
        # Handle compound values like "1/0", "0/1", etc.
        if '/' in value:
            parts = value.split('/')
            if len(parts) == 2:
                left, right = parts[0].strip(), parts[1].strip()
                # Priority to left side, but check for clear indicators
                if left == '1' or (left != '0' and left != '__' and right in ['0', '__']):
                    return 1
                elif left == '0' or left == '__':
                    return 0
        
        # Default to 0 for unclear values
        return 0
    
    def save_clean_ground_truth(self, ground_truth: Dict[str, Dict[str, Any]], output_file: str):
        """
        Save the clean ground truth data to a JSON file.
        
        Args:
            ground_truth: Clean ground truth data
            output_file: Output JSON file path
        """
        print(f"\nSaving clean ground truth to {output_file}...")
        
        with open(output_file, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        print(f"✓ Clean ground truth saved with {len(ground_truth)} PMIDs")
    
    def validate_against_pmids(self, ground_truth: Dict[str, Dict[str, Any]], 
                              jsonl_file: str = "PMID_all_text.jsonl") -> Dict[str, Any]:
        """
        Validate ground truth PMIDs against the input JSONL file.
        
        Args:
            ground_truth: Ground truth data
            jsonl_file: Path to JSONL file with articles
            
        Returns:
            Validation results
        """
        print(f"\nValidating PMIDs against {jsonl_file}...")
        
        if not Path(jsonl_file).exists():
            print(f"WARNING: JSONL file not found: {jsonl_file}")
            return {'status': 'skipped', 'reason': 'file_not_found'}
        
        # Load PMIDs from JSONL
        jsonl_pmids = set()
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    import json
                    article = json.loads(line.strip())
                    if 'pmid' in article:
                        jsonl_pmids.add(str(article['pmid']))
        
        gt_pmids = set(ground_truth.keys())
        
        overlap = gt_pmids & jsonl_pmids
        gt_only = gt_pmids - jsonl_pmids
        jsonl_only = jsonl_pmids - gt_pmids
        
        validation_results = {
            'status': 'completed',
            'ground_truth_pmids': len(gt_pmids),
            'jsonl_pmids': len(jsonl_pmids),
            'overlap': len(overlap),
            'gt_only': len(gt_only),
            'jsonl_only': len(jsonl_only),
            'gt_only_list': list(gt_only)[:10] if gt_only else [],
            'jsonl_only_list': list(jsonl_only)[:10] if jsonl_only else []
        }
        
        print(f"  Ground truth PMIDs: {len(gt_pmids)}")
        print(f"  JSONL PMIDs: {len(jsonl_pmids)}")
        print(f"  Overlap: {len(overlap)}")
        print(f"  GT only: {len(gt_only)}")
        print(f"  JSONL only: {len(jsonl_only)}")
        
        if gt_only:
            print(f"  GT PMIDs not in JSONL (first 10): {list(gt_only)[:10]}")
        if jsonl_only:
            print(f"  JSONL PMIDs not in GT (first 10): {list(jsonl_only)[:10]}")
        
        if len(overlap) == len(gt_pmids):
            print("✓ All ground truth PMIDs found in JSONL file")
        else:
            print(f"⚠️  {len(gt_only)} ground truth PMIDs missing from JSONL")
        
        return validation_results

def main():
    """Main function to extract and validate ground truth."""
    
    print("Ground Truth Extractor for CausalJudge")
    print("=" * 50)
    
    extractor = GroundTruthExtractor()
    
    # Input and output files
    excel_file = "PMID/180FinalResult_Jun17.xlsx"
    clean_gt_file = "ground_truth_clean.json"
    analysis_file = "ground_truth_analysis.json"
    
    try:
        # Step 1: Load and validate raw ground truth
        print("Step 1: Loading and validating raw ground truth...")
        ground_truth = extractor.load_and_validate_ground_truth(excel_file)
        
        # Step 2: Analyze field values
        print("\nStep 2: Analyzing field values...")
        analysis = extractor.analyze_field_values(ground_truth)
        
        # Step 3: Convert to clean binary format
        print("\nStep 3: Converting to clean binary format...")
        clean_ground_truth = extractor.convert_to_binary(ground_truth)
        
        # Step 4: Save clean ground truth
        print("\nStep 4: Saving clean ground truth...")
        extractor.save_clean_ground_truth(clean_ground_truth, clean_gt_file)
        
        # Step 5: Validate against JSONL file
        print("\nStep 5: Validating against JSONL file...")
        validation_results = extractor.validate_against_pmids(clean_ground_truth)
        
        # Save analysis results
        with open(analysis_file, 'w') as f:
            json.dump({
                'field_analysis': analysis,
                'validation_results': validation_results,
                'summary': {
                    'total_pmids': len(clean_ground_truth),
                    'binary_fields': len(extractor.BINARY_FIELDS),
                    'clean_gt_file': clean_gt_file
                }
            }, f, indent=2)
        
        print(f"\n{'='*50}")
        print("GROUND TRUTH EXTRACTION COMPLETE")
        print(f"{'='*50}")
        print(f"✓ Extracted {len(clean_ground_truth)} PMIDs")
        print(f"✓ Clean ground truth: {clean_gt_file}")
        print(f"✓ Analysis results: {analysis_file}")
        print(f"✓ All {len(extractor.BINARY_FIELDS)} binary fields processed")
        
        if validation_results['status'] == 'completed':
            print(f"✓ PMID validation: {validation_results['overlap']}/{validation_results['ground_truth_pmids']} overlap with JSONL")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during extraction: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)