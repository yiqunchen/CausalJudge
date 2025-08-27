import csv
import sys
import json
import openai
import os
import asyncio
import pandas as pd
import numpy as np
from openai import OpenAI, AsyncOpenAI
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
import argparse
from typing import Dict, List, Tuple, Any
import re
import pickle
from datetime import datetime

csv.field_size_limit(sys.maxsize)

class CausalEvaluationSystem:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialize the causal evaluation system.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for evaluation (default: gpt-4o-mini)
        """
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.model = model
        
        # The columns we expect in our final sheet (matching causalcode.py)
        self.OUTPUT_COLUMNS = [
            "PMID", 
            "Title of the Study",
            "Mediation Method Used",
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
            "Control for Other Post-Exposure Variables",
        ]
        
        # Binary fields that should be converted to 0/1
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
            "Control for Other Post-Exposure Variables",
        ]

    def truncate_text_at_references(self, text: str) -> str:
        """
        Truncate text at "## References" section.
        
        Args:
            text: Full article text
            
        Returns:
            Truncated text
        """
        if "## References" in text:
            return text.split("## References")[0].strip()
        return text

    def load_jsonl_data(self, jsonl_file: str) -> List[Dict[str, Any]]:
        """
        Load data from JSONL file.
        
        Args:
            jsonl_file: Path to JSONL file
            
        Returns:
            List of article data dictionaries
        """
        articles = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    article = json.loads(line.strip())
                    # Truncate text at references
                    if 'text' in article:
                        article['text'] = self.truncate_text_at_references(article['text'])
                    articles.append(article)
        return articles

    def create_enhanced_prompt(self, pmid: str, text: str, prompt_type: str = "basic") -> str:
        """
        Create enhanced prompts with different levels of detail and examples.
        
        Args:
            pmid: PMID identifier
            text: Article text
            prompt_type: Type of prompt ("basic", "detailed", "examples")
            
        Returns:
            Formatted prompt string
        """
        binary_fields = [
            "Randomized Exposure", "Causal Mediation", "Examined Mediator-Outcome Linearity",
            "Examined Exposure-Mediator Interaction", "Covariates in Exposure-Mediator Model",
            "Covariates in Exposure-Outcome Model", "Covariates in Mediator-Outcome Model",
            "Control for Baseline Mediator", "Control for Baseline Outcome",
            "Temporal Ordering Exposure Before Mediator", "Temporal Ordering Mediator Before Outcome",
            "Discussed Mediator Assumptions", "Sensitivity Analysis to Assumption",
            "Control for Other Post-Exposure Variables"
        ]
        
        # Create JSON structure specification
        json_structure = {
            "PMID": pmid,
            "Title of the Study": "[extracted title]",
            "Mediation Method Used": "[method name or 'Not mentioned']"
        }
        
        # Add binary fields and their confidence scores
        for field in binary_fields:
            json_structure[field] = "[0 or 1]"
            json_structure[f"{field}_confidence"] = "[0.0 to 1.0]"
        
        base_prompt = f"""
You are a causal inference expert specialized in mediation analysis. Extract information from this article:

PMID: {pmid}
TEXT: {text}

Extract the following information and provide it in JSON format:

For each binary field (0/1), also provide a confidence score (0.0 to 1.0) indicating how certain you are about your classification.

Return a JSON object with this exact structure:
{json.dumps(json_structure, indent=2)}

IMPORTANT: 
- Binary fields should be exactly 0 or 1
- Confidence scores should be decimal values between 0.0 and 1.0
- Higher confidence (0.8-1.0) when evidence is clear and explicit
- Medium confidence (0.5-0.7) when evidence is moderate or requires inference
- Lower confidence (0.1-0.4) when evidence is weak or ambiguous

Updated field naming guidance (for clarity only; keep the JSON keys exactly as specified above):
- Temporal Ordering Exposure Before Mediator ≈ "Temporal ordering of exposure and mediator"
- Temporal Ordering Mediator Before Outcome ≈ "Temporal ordering of mediator and outcome"
- When both of the above are 1, this implies "Temporal ordering of exposure, mediator, and outcome"
- Covariates in Exposure-Mediator/Exposure-Outcome/Mediator-Outcome Model together capture whether the study "Adjusted for covariates"
- Control for Baseline Mediator ≈ "Adjusted for covariates that include baseline measure of mediator"
- Control for Baseline Outcome ≈ "Adjusted for covariates that include baseline measure of outcome"
- When both controls are 1, this implies "Adjusted for covariates that include baseline measures of mediator and outcome"
- Discussed Mediator Assumptions ≈ "Discussed mediation assumptions at all"
- Examined Exposure-Mediator Interaction ≈ "Examined assumption of no interaction of exposure and mediator" (reported as 1 if they explicitly test/assess interaction or the assumption)
- Sensitivity Analysis to Assumption ≈ "Sensitivity analysis performed to mediation assumptions"
"""

        if prompt_type == "detailed":
            detailed_prompt = base_prompt + """

DETAILS AND EXAMPLES FOR EACH FIELD (names aligned to clearer phrasing; keep JSON keys as specified):

1. Randomized Exposure (1=Yes): Look for "randomized", "random assignment", "RCT", "experimental design".
2. Causal Mediation (1=Yes): Mentions "causal mediation", "natural direct/indirect effects", counterfactual framework.
3. Examined Mediator-Outcome Linearity (1=Yes): Tests of non-linearity (polynomial/spline/curvilinear terms).
4. Examined Exposure-Mediator Interaction (1=Yes): Explicitly tests interaction or states the no-interaction assumption and examines it.
5–7. Adjusted for covariates (1=Yes): Indicated by covariate adjustment in any of: exposure→mediator, exposure→outcome, mediator→outcome models.
8–9. Adjusted for covariates including baselines: Control for Baseline Mediator/Outcome indicate inclusion of respective baseline measures.
10. Temporal ordering of exposure and mediator (1=Yes): Evidence that exposure precedes mediator.
11. Temporal ordering of mediator and outcome (1=Yes): Evidence that mediator precedes outcome.
    If both 10 and 11 are 1, this implies temporal ordering of exposure, mediator, and outcome.
12. Discussed mediation assumptions at all (1=Yes): Any discussion of key assumptions (e.g., no unmeasured confounding).
13. Sensitivity analysis performed to mediation assumptions (1=Yes): Any sensitivity/robustness analyses relevant to mediation.
14. Control for Other Post-Exposure Variables (1=Yes): Adjustment for post-treatment covariates.

CONFIDENCE SCORING:
- 0.9–1.0: Explicit statements (e.g., "we randomized participants"; "we conducted sensitivity analysis").
- 0.7–0.8: Strong inferential evidence.
- 0.5–0.6: Moderate evidence.
- 0.3–0.4: Weak evidence.
- 0.1–0.2: Very uncertain.
"""
            return detailed_prompt
            
        elif prompt_type == "examples":
            examples_prompt = base_prompt + """

EXAMPLES OF KEY PHRASES TO LOOK FOR:

Randomized Exposure (1 = Yes):
- "Participants were randomly assigned to treatment or control"
- "RCT design with randomization"
- "Experimental manipulation of exposure"
- "Random assignment to intervention groups"

Causal Mediation (1 = Yes):
- "Causal mediation analysis"
- "Natural direct and indirect effects"
- "Counterfactual framework"
- "Potential outcomes approach"
- "Causal inference methods"

Examined Mediator-Outcome Linearity (1 = Yes):
- "Tested for non-linear relationships"
- "Polynomial terms included"
- "Quadratic effects examined"
- "Curvilinear relationship tested"
- "Spline analysis"

Examined Exposure-Mediator Interaction (1 = Yes):
- "Moderation analysis"
- "Interaction effects tested"
- "Conditional effects examined"
- "Effect modification assessed"

Covariates in Models (1 = Yes):
- "Adjusted for age, sex, education"
- "Controlled for confounding variables"
- "Covariates included in regression"
- "Demographic variables controlled"

Baseline Control (1 = Yes):
- "Baseline mediator controlled"
- "Pre-treatment values adjusted"
- "Initial levels accounted for"
- "Baseline outcome included"

Temporal Ordering (1 = Yes):
- "Exposure measured at time 1"
- "Mediator measured at time 2"
- "Outcome measured at time 3"
- "Longitudinal design with proper timing"

Assumptions Discussion (1 = Yes):
- "Mediation assumptions discussed"
- "No unmeasured confounding assumption"
- "Sequential ignorability"
- "Limitations of mediation analysis"

Sensitivity Analysis (1 = Yes):
- "Sensitivity analysis conducted"
- "Robustness checks performed"
- "Alternative specifications tested"
- "Bias analysis conducted"

Post-Exposure Variables (1 = Yes):
- "Controlled for variables occurring after exposure"
- "Intermediate variables adjusted"
- "Post-treatment confounders included"

CONFIDENCE SCORING GUIDANCE:
- 0.9-1.0: Explicit, clear evidence (exact phrases found)
- 0.7-0.8: Strong inferential evidence (methods clearly imply the practice)
- 0.5-0.6: Moderate evidence (some indication but not definitive)
- 0.3-0.4: Weak evidence (minimal or ambiguous mention)
- 0.1-0.2: Very uncertain (conflicting or unclear information)
"""
            return examples_prompt
            
        else:  # basic
            return base_prompt

    def extract_structured_response(self, response_text: str) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Extract structured information from model response and add confidence scores.
        
        Args:
            response_text: Raw response from the model
            
        Returns:
            Tuple of (response_dict, confidence_scores)
        """
        response_dict = {col: "" for col in self.OUTPUT_COLUMNS}
        confidence_scores = {}
        
        lines = response_text.split("\n")
        
        for line in lines:
            line = line.strip()
            if not line or ":" not in line:
                continue
            
            if line.startswith("- "):
                line = line[2:].strip()

            parts = line.split(":", 1)
            if len(parts) < 2:
                continue
            
            key_raw = parts[0].strip()
            val_raw = parts[1].strip()
            
            # Remove brackets from values
            val_raw = val_raw.strip("[]")
            
            if key_raw in response_dict:
                response_dict[key_raw] = val_raw
                
                # For binary fields, try to extract confidence score
                if key_raw in self.BINARY_FIELDS:
                    # Look for confidence score in parentheses or after comma
                    confidence_match = re.search(r'\((\d+\.?\d*)\)|confidence:\s*(\d+\.?\d*)', val_raw.lower())
                    if confidence_match:
                        confidence = float(confidence_match.group(1) or confidence_match.group(2))
                        confidence_scores[key_raw] = confidence
                    else:
                        # Default confidence based on response clarity
                        if val_raw in ['0', '1']:
                            confidence_scores[key_raw] = 0.9
                        elif val_raw in ['yes', 'no']:
                            confidence_scores[key_raw] = 0.8
                        else:
                            confidence_scores[key_raw] = 0.5

        return response_dict, confidence_scores

    def get_model_prediction_with_confidence(self, pmid: str, text: str, prompt_type: str = "basic", temperature: float = None) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Get model prediction with confidence scores.
        
        Args:
            pmid: PMID identifier
            text: Article text
            prompt_type: Type of prompt to use
            
        Returns:
            Tuple of (response_dict, confidence_scores)
        """
        prompt = self.create_enhanced_prompt(pmid, text, prompt_type)
        
        try:
            # Set temperature based on parameter or model type
            if temperature is None:
                if "o1" in self.model.lower() or "o3" in self.model.lower():
                    temperature = 1.0
                elif "gpt-5" in self.model.lower():
                    temperature = 1.0
                else:
                    temperature = 0.0
            
            # Configure request parameters for different model types
            request_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a causal inference expert specialized in mediation analysis. Provide structured responses with confidence scores in JSON format."},
                    {"role": "user", "content": prompt + "\n\nPlease respond in JSON format with the extracted information."}
                ],
                "temperature": temperature,
                "response_format": {"type": "json_object"}
            }
            
            # GPT-5 specific parameters removed for clean re-run
            
            completion = self.client.chat.completions.create(**request_params)
            
            response_text = completion.choices[0].message.content.strip()
            
            # Try to parse as JSON first
            try:
                response_json = json.loads(response_text)
                response_dict = {col: response_json.get(col, "") for col in self.OUTPUT_COLUMNS}
                confidence_scores = {col: response_json.get(f"{col}_confidence", 0.5) for col in self.BINARY_FIELDS}
                return response_dict, confidence_scores
            except json.JSONDecodeError:
                # Fall back to text parsing
                return self.extract_structured_response(response_text)
                
        except Exception as e:
            print(f"Error processing PMID {pmid}: {e}")
            return {col: "" for col in self.OUTPUT_COLUMNS}, {col: 0.0 for col in self.BINARY_FIELDS}

    async def async_get_model_prediction_with_confidence(self, pmid: str, text: str, prompt_type: str = "basic", temperature: float = None) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Async version: Get model prediction with confidence scores using AsyncOpenAI client.
        """
        prompt = self.create_enhanced_prompt(pmid, text, prompt_type)

        try:
            # Set temperature based on parameter or model type
            if temperature is None:
                if "o1" in self.model.lower() or "o3" in self.model.lower():
                    temperature = 1.0
                elif "gpt-5" in self.model.lower():
                    temperature = 1.0
                else:
                    temperature = 0.0

            request_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a causal inference expert specialized in mediation analysis. Provide structured responses with confidence scores in JSON format."},
                    {"role": "user", "content": prompt + "\n\nPlease respond in JSON format with the extracted information."}
                ],
                "temperature": temperature,
                "response_format": {"type": "json_object"}
            }

            completion = await self.async_client.chat.completions.create(**request_params)
            response_text = completion.choices[0].message.content.strip()

            try:
                response_json = json.loads(response_text)
                response_dict = {col: response_json.get(col, "") for col in self.OUTPUT_COLUMNS}
                confidence_scores = {col: response_json.get(f"{col}_confidence", 0.5) for col in self.BINARY_FIELDS}
                return response_dict, confidence_scores
            except json.JSONDecodeError:
                return self.extract_structured_response(response_text)

        except Exception as e:
            print(f"Error processing PMID {pmid} (async): {e}")
            return {col: "" for col in self.OUTPUT_COLUMNS}, {col: 0.0 for col in self.BINARY_FIELDS}

    def load_ground_truth(self, ground_truth_file: str) -> Dict[str, Dict[str, Any]]:
        """
        Load ground truth data from JSON file.
        
        Args:
            ground_truth_file: Path to ground truth JSON file
            
        Returns:
            Dictionary mapping PMID to ground truth data
        """
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
                    
        return ground_truth

    def load_model_predictions(self, predictions_file: str) -> Dict[str, Dict[str, Any]]:
        """
        Load model predictions from JSON file.
        
        Args:
            predictions_file: Path to predictions JSON file
            
        Returns:
            Dictionary mapping PMID to prediction data
        """
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)
        return predictions

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
        results = {}
        
        for field in self.BINARY_FIELDS:
            gt_values = []
            pred_values = []
            conf_values = []
            
            for pmid in ground_truth:
                if pmid in predictions:
                    # Ground truth
                    gt_val = ground_truth[pmid].get(field, "")
                    # Handle various ground truth formats
                    if isinstance(gt_val, (int, float)):
                        gt_binary = int(gt_val) if gt_val > 0 else 0
                    else:
                        gt_str = str(gt_val).lower().strip()
                        gt_binary = 1 if gt_str in ['1', 'yes', 'true', '1.0'] else 0
                    
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
                    
                    gt_values.append(int(gt_binary))
                    pred_values.append(int(pred_binary))
                    conf_values.append(float(conf_val))
            
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
                        # Handle various ground truth formats
                        if isinstance(gt_val, (int, float)):
                            gt_binary = int(gt_val) if gt_val > 0 else 0
                        else:
                            gt_str = str(gt_val).lower().strip()
                            gt_binary = 1 if gt_str in ['1', 'yes', 'true', '1.0'] else 0
                        
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
                        
                        all_gt.append(int(gt_binary))
                        all_pred.append(int(pred_binary))
                        all_conf.append(float(conf_val))
        
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

    def process_articles_jsonl(self, input_file: str, max_cases: int = 200, 
                              prompt_type: str = "basic", output_file: str = None,
                              skip_existing: bool = True, checkpoint_file: str = None, 
                              checkpoint_interval: int = 10, temperature: float = None,
                              max_concurrency: int = 1) -> Dict[str, Any]:
        """
        Process articles from JSONL file and generate predictions with confidence scores.
        Can skip processing if results already exist and supports checkpointing.
        
        Args:
            input_file: Path to input JSONL file
            max_cases: Maximum number of cases to process
            prompt_type: Type of prompt to use
            output_file: Output file path (optional)
            skip_existing: Whether to skip processing if results already exist
            checkpoint_file: Checkpoint file path (optional)
            checkpoint_interval: Save checkpoint every N articles
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        # Check for existing results first
        if skip_existing and output_file and self.check_existing_results(output_file, expected=max_cases):
            predictions, confidence_scores = self.load_existing_results(output_file)
            return {
                'predictions': predictions,
                'confidence_scores': confidence_scores
            }
        
        # If concurrency requested, use async processing path
        if max_concurrency and max_concurrency > 1:
            return asyncio.run(self.process_articles_jsonl_async(
                input_file=input_file,
                max_cases=max_cases,
                prompt_type=prompt_type,
                output_file=output_file,
                skip_existing=skip_existing,
                checkpoint_file=checkpoint_file,
                checkpoint_interval=checkpoint_interval,
                temperature=temperature,
                max_concurrency=max_concurrency,
            ))

        predictions = {}
        confidence_scores = {}
        processed_pmids = []
        
        # Load checkpoint if exists
        if checkpoint_file:
            predictions, confidence_scores, processed_pmids = self.load_checkpoint(checkpoint_file)
        
        # Load articles from JSONL
        articles = self.load_jsonl_data(input_file)
        print(f"Loaded {len(articles)} articles from {input_file}")
        
        # Filter out already processed articles
        remaining_articles = [article for article in articles if article.get("pmid") not in processed_pmids]
        print(f"Remaining articles to process: {len(remaining_articles)}")
        
        # Track the number of already processed articles to report accurate progress
        start_count = len(processed_pmids)

        for idx, article in enumerate(remaining_articles):
            pmid = article.get("pmid", "")
            text = article.get("text", "")

            current_number = start_count + idx + 1
            total_to_process = min(len(articles), max_cases)
            if current_number > total_to_process:
                break
            print(f"Processing article {current_number}/{total_to_process}: PMID {pmid}")

            try:
                response_dict, conf_scores = self.get_model_prediction_with_confidence(
                    pmid, text, prompt_type, temperature
                )
                
                predictions[pmid] = response_dict
                confidence_scores[pmid] = conf_scores
                processed_pmids.append(pmid)
                
                print(f"✓ Successfully completed PMID {pmid}")
                
                # Save checkpoint periodically
                if checkpoint_file and (idx + 1) % checkpoint_interval == 0:
                    self.save_checkpoint(predictions, confidence_scores, processed_pmids, checkpoint_file)
                    
            except Exception as e:
                print(f"✗ Error processing PMID {pmid}: {e}")
                # Still save checkpoint even on error
                if checkpoint_file:
                    self.save_checkpoint(predictions, confidence_scores, processed_pmids, checkpoint_file)
                continue

        # Save final checkpoint
        if checkpoint_file:
            self.save_checkpoint(predictions, confidence_scores, processed_pmids, checkpoint_file)

        # Save results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(predictions, f, indent=2)
            
            # Save confidence scores separately
            conf_file = output_file.replace('.json', '_confidence.json')
            with open(conf_file, 'w') as f:
                json.dump(confidence_scores, f, indent=2)

        return {
            'predictions': predictions,
            'confidence_scores': confidence_scores
        }

    async def process_articles_jsonl_async(self, input_file: str, max_cases: int = 200,
                                           prompt_type: str = "basic", output_file: str = None,
                                           skip_existing: bool = True, checkpoint_file: str = None,
                                           checkpoint_interval: int = 10, temperature: float = None,
                                           max_concurrency: int = 4) -> Dict[str, Any]:
        """
        Async processing path with bounded concurrency for JSONL inputs.
        """
        # Check for existing results first
        if skip_existing and output_file and self.check_existing_results(output_file, expected=max_cases):
            predictions, confidence_scores = self.load_existing_results(output_file)
            return {
                'predictions': predictions,
                'confidence_scores': confidence_scores
            }

        predictions: Dict[str, Any] = {}
        confidence_scores: Dict[str, Any] = {}
        processed_pmids: List[str] = []

        # Load checkpoint if exists
        if checkpoint_file:
            predictions, confidence_scores, processed_pmids = self.load_checkpoint(checkpoint_file)

        # Load and filter articles
        articles = self.load_jsonl_data(input_file)
        print(f"Loaded {len(articles)} articles from {input_file}")

        remaining_articles = [a for a in articles if a.get("pmid") not in processed_pmids]
        start_count = len(processed_pmids)
        total_to_process = min(len(articles), max_cases)
        remaining_quota = max(total_to_process - start_count, 0)
        remaining_articles = remaining_articles[:remaining_quota]
        print(f"Remaining articles to process: {len(remaining_articles)} (quota {remaining_quota})")

        semaphore = asyncio.Semaphore(max(1, max_concurrency))
        lock = asyncio.Lock()
        progress_counter = start_count

        async def handle_article(article):
            nonlocal progress_counter
            pmid = article.get("pmid", "")
            text = article.get("text", "")
            async with semaphore:
                try:
                    response_dict, conf_scores = await self.async_get_model_prediction_with_confidence(
                        pmid, text, prompt_type, temperature
                    )
                    async with lock:
                        predictions[pmid] = response_dict
                        confidence_scores[pmid] = conf_scores
                        processed_pmids.append(pmid)
                        progress_counter += 1
                        print(f"✓ Completed PMID {pmid} ({progress_counter}/{total_to_process})")
                        # Periodic checkpoint
                        if checkpoint_file and progress_counter % checkpoint_interval == 0:
                            await asyncio.to_thread(
                                self.save_checkpoint,
                                predictions,
                                confidence_scores,
                                processed_pmids,
                                checkpoint_file,
                            )
                except Exception as e:
                    print(f"✗ Error processing PMID {pmid} (async): {e}")
                    async with lock:
                        # Save checkpoint on error as well
                        if checkpoint_file:
                            await asyncio.to_thread(
                                self.save_checkpoint,
                                predictions,
                                confidence_scores,
                                processed_pmids,
                                checkpoint_file,
                            )

        await asyncio.gather(*(handle_article(a) for a in remaining_articles))

        # Final checkpoint
        if checkpoint_file:
            await asyncio.to_thread(self.save_checkpoint, predictions, confidence_scores, processed_pmids, checkpoint_file)

        # Save results
        if output_file:
            def _write_json(path: str, data: Dict[str, Any]):
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2)
            await asyncio.to_thread(_write_json, output_file, predictions)
            conf_file = output_file.replace('.json', '_confidence.json')
            await asyncio.to_thread(_write_json, conf_file, confidence_scores)

        return {
            'predictions': predictions,
            'confidence_scores': confidence_scores,
        }

    def process_articles(self, input_file: str, max_cases: int = 200, 
                        prompt_type: str = "basic", output_file: str = None,
                        skip_existing: bool = True, checkpoint_file: str = None,
                        checkpoint_interval: int = 10, temperature: float = None,
                        max_concurrency: int = 1) -> Dict[str, Any]:
        """
        Process articles and generate predictions with confidence scores.
        Automatically detects if input is JSONL or CSV.
        Can skip processing if results already exist and supports checkpointing.
        
        Args:
            input_file: Path to input file (JSONL or CSV)
            max_cases: Maximum number of cases to process
            prompt_type: Type of prompt to use
            output_file: Output file path (optional)
            skip_existing: Whether to skip processing if results already exist
            checkpoint_file: Checkpoint file path (optional)
            checkpoint_interval: Save checkpoint every N articles
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        if input_file.endswith('.jsonl'):
            return self.process_articles_jsonl(
                input_file,
                max_cases,
                prompt_type,
                output_file,
                skip_existing,
                checkpoint_file,
                checkpoint_interval,
                temperature,
                max_concurrency,
            )
        else:
            # Original CSV processing with checkpoint support
            predictions = {}
            confidence_scores = {}
            processed_pmids = []
            
            # Load checkpoint if exists
            if checkpoint_file:
                predictions, confidence_scores, processed_pmids = self.load_checkpoint(checkpoint_file)
            
            # Check for existing results first
            if skip_existing and output_file and self.check_existing_results(output_file, expected=max_cases):
                predictions, confidence_scores = self.load_existing_results(output_file)
                return {
                    'predictions': predictions,
                    'confidence_scores': confidence_scores
                }
            
            with open(input_file, mode='r', encoding='utf-8', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                
                # Filter out already processed articles
                remaining_rows = [row for row in rows if row.get("pmid") not in processed_pmids]
                print(f"Remaining articles to process: {len(remaining_rows)}")
                
                # Track the number of already processed rows to report accurate progress
                start_count = len(processed_pmids)

                for idx, row in enumerate(remaining_rows):
                    if len(processed_pmids) + idx >= max_cases:
                        break

                    current_number = start_count + idx + 1
                    total_to_process = min(len(rows), max_cases)
                    print(f"Processing row {current_number}/{total_to_process}: PMID {row['pmid']}")

                    try:
                        response_dict, conf_scores = self.get_model_prediction_with_confidence(
                            row['pmid'], row['text'], prompt_type, temperature
                        )
                        
                        predictions[row['pmid']] = response_dict
                        confidence_scores[row['pmid']] = conf_scores
                        processed_pmids.append(row['pmid'])
                        
                        print(f"✓ Successfully completed PMID {row['pmid']}")
                        
                        # Save checkpoint periodically
                        if checkpoint_file and (idx + 1) % checkpoint_interval == 0:
                            self.save_checkpoint(predictions, confidence_scores, processed_pmids, checkpoint_file)
                            
                    except Exception as e:
                        print(f"✗ Error processing PMID {row['pmid']}: {e}")
                        # Still save checkpoint even on error
                        if checkpoint_file:
                            self.save_checkpoint(predictions, confidence_scores, processed_pmids, checkpoint_file)
                        continue

            # Save final checkpoint
            if checkpoint_file:
                self.save_checkpoint(predictions, confidence_scores, processed_pmids, checkpoint_file)

            # Save results
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(predictions, f, indent=2)
                
                # Save confidence scores separately
                conf_file = output_file.replace('.json', '_confidence.json')
                with open(conf_file, 'w') as f:
                    json.dump(confidence_scores, f, indent=2)

            return {
                'predictions': predictions,
                'confidence_scores': confidence_scores
            }

    def evaluate_performance(self, ground_truth_file: str, predictions_file: str, 
                           confidence_file: str = None) -> Dict[str, Any]:
        """
        Evaluate model performance against ground truth.
        
        Args:
            ground_truth_file: Path to ground truth CSV file
            predictions_file: Path to predictions JSON file
            confidence_file: Path to confidence scores JSON file (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        ground_truth = self.load_ground_truth(ground_truth_file)
        predictions = self.load_model_predictions(predictions_file)
        
        if confidence_file and os.path.exists(confidence_file):
            with open(confidence_file, 'r') as f:
                confidence_scores = json.load(f)
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

    def check_existing_results(self, output_file: str, expected: int = None) -> bool:
        """
        Check if results already exist and are complete.
        
        Args:
            output_file: Path to predictions JSON file
            expected: Expected number of PMIDs (optional)
            
        Returns:
            True if results exist and are complete, False otherwise
        """
        if not os.path.exists(output_file):
            return False
        
        try:
            with open(output_file, 'r') as f:
                predictions = json.load(f)
            
            # Check if we have predictions
            if len(predictions) == 0:
                return False
            
            # If expected count is provided, check if we have enough results
            if expected is not None and len(predictions) < expected:
                print(f"✓ Found existing results: {len(predictions)} PMIDs, but expected {expected}")
                return False
            
            print(f"✓ Found existing results: {len(predictions)} PMIDs already processed")
            return True
            
        except Exception as e:
            print(f"Error checking existing results: {e}")
            return False

    def load_existing_results(self, output_file: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load existing results from files.
        
        Args:
            output_file: Path to predictions JSON file
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        predictions = {}
        confidence_scores = {}
        
        try:
            with open(output_file, 'r') as f:
                predictions = json.load(f)
            
            # Try to load confidence scores
            conf_file = output_file.replace('.json', '_confidence.json')
            if os.path.exists(conf_file):
                with open(conf_file, 'r') as f:
                    confidence_scores = json.load(f)
            else:
                # Create default confidence scores if file doesn't exist
                confidence_scores = {pmid: {field: 0.5 for field in self.BINARY_FIELDS} 
                                   for pmid in predictions}
            
            print(f"✓ Loaded existing results: {len(predictions)} PMIDs")
            return predictions, confidence_scores
            
        except Exception as e:
            print(f"Error loading existing results: {e}")
            return {}, {}

    def save_checkpoint(self, predictions: Dict[str, Any], confidence_scores: Dict[str, Any], 
                       processed_pmids: List[str], checkpoint_file: str):
        """
        Save progress checkpoint.
        
        Args:
            predictions: Current predictions
            confidence_scores: Current confidence scores
            processed_pmids: List of processed PMIDs
            checkpoint_file: Path to checkpoint file
        """
        checkpoint_data = {
            'predictions': predictions,
            'confidence_scores': confidence_scores,
            'processed_pmids': processed_pmids,
            'timestamp': datetime.now().isoformat(),
            'model': self.model
        }
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        print(f"✓ Checkpoint saved: {len(processed_pmids)} PMIDs processed")

    def load_checkpoint(self, checkpoint_file: str) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
        """
        Load progress checkpoint.
        
        Args:
            checkpoint_file: Path to checkpoint file
            
        Returns:
            Tuple of (predictions, confidence_scores, processed_pmids)
        """
        if not os.path.exists(checkpoint_file):
            return {}, {}, []
        
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        predictions = checkpoint_data.get('predictions', {})
        confidence_scores = checkpoint_data.get('confidence_scores', {})
        processed_pmids = checkpoint_data.get('processed_pmids', [])
        timestamp = checkpoint_data.get('timestamp', 'unknown')
        model = checkpoint_data.get('model', 'unknown')
        
        print(f"✓ Checkpoint loaded: {len(processed_pmids)} PMIDs already processed")
        print(f"  Model: {model}")
        print(f"  Timestamp: {timestamp}")
        
        return predictions, confidence_scores, processed_pmids

def main():
    parser = argparse.ArgumentParser(description='Causal Inference Evaluation System')
    parser.add_argument('--api_key', required=True, help='OpenAI API key')
    parser.add_argument('--model', default='gpt-4o-mini', help='Model to use')
    parser.add_argument('--input_file', help='Input file (JSONL or CSV) with articles')
    parser.add_argument('--ground_truth_file', help='Ground truth CSV file')
    parser.add_argument('--predictions_file', help='Predictions JSON file')
    parser.add_argument('--confidence_file', help='Confidence scores JSON file')
    parser.add_argument('--output_file', help='Output file for predictions')
    parser.add_argument('--max_cases', type=int, default=200, help='Maximum cases to process')
    parser.add_argument('--prompt_type', default='basic', choices=['basic', 'detailed', 'examples'], 
                       help='Type of prompt to use')
    parser.add_argument('--mode', required=True, choices=['process', 'evaluate'], 
                       help='Mode: process articles or evaluate performance')
    
    args = parser.parse_args()
    
    # Initialize system
    evaluator = CausalEvaluationSystem(args.api_key, args.model)
    
    if args.mode == 'process':
        if not args.input_file:
            print("Error: input_file required for process mode")
            return
            
        results = evaluator.process_articles(
            args.input_file, 
            args.max_cases, 
            args.prompt_type, 
            args.output_file
        )
        
        print(f"Processing complete. Results saved to {args.output_file}")
        
    elif args.mode == 'evaluate':
        if not args.ground_truth_file or not args.predictions_file:
            print("Error: ground_truth_file and predictions_file required for evaluate mode")
            return
            
        metrics = evaluator.evaluate_performance(
            args.ground_truth_file, 
            args.predictions_file, 
            args.confidence_file
        )
        
        print("\n=== EVALUATION RESULTS ===")
        print(f"Model: {args.model}")
        print(f"Prompt Type: {args.prompt_type}")
        
        if 'overall' in metrics:
            print(f"\nOverall Performance:")
            print(f"  Accuracy: {metrics['overall']['accuracy']:.3f}")
            print(f"  Precision: {metrics['overall']['precision']:.3f}")
            print(f"  Recall: {metrics['overall']['recall']:.3f}")
            print(f"  F1: {metrics['overall']['f1']:.3f}")
            print(f"  AUC: {metrics['overall']['auc']:.3f}")
            print(f"  PR-AUC: {metrics['overall']['pr_auc']:.3f}")
            print(f"  N Samples: {metrics['overall']['n_samples']}")
        
        print(f"\nField-specific Performance:")
        for field in evaluator.BINARY_FIELDS:
            if field in metrics:
                print(f"\n{field}:")
                print(f"  Accuracy: {metrics[field]['accuracy']:.3f}")
                print(f"  Precision: {metrics[field]['precision']:.3f}")
                print(f"  Recall: {metrics[field]['recall']:.3f}")
                print(f"  F1: {metrics[field]['f1']:.3f}")
                print(f"  AUC: {metrics[field]['auc']:.3f}")
                print(f"  PR-AUC: {metrics[field]['pr_auc']:.3f}")
                print(f"  N Samples: {metrics[field]['n_samples']}")

if __name__ == "__main__":
    main() 