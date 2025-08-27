#!/usr/bin/env python3
"""
Script to synthesize reasoning patterns and generate comprehensive findings.
"""

import json
import openai
from openai import OpenAI
import os
from typing import Dict, List
from collections import defaultdict, Counter

class ReasoningPatternSynthesizer:
    def __init__(self, api_key: str):
        """Initialize with OpenAI API key."""
        self.client = OpenAI(api_key=api_key)
    
    def load_reasoning_analysis(self) -> Dict:
        """Load the reasoning analysis results."""
        with open('/Users/yiquntchen/Desktop/chen-lab/CausalJudge/reasoning_analysis_results.json', 'r') as f:
            return json.load(f)
    
    def analyze_error_patterns(self, results: Dict) -> Dict:
        """Analyze error patterns across models and fields."""
        error_patterns = {
            'by_model': {},
            'by_field': {},
            'error_types': defaultdict(list),
            'cross_model_patterns': defaultdict(list)
        }
        
        # Analyze patterns by model
        for model, model_data in results.items():
            error_types = []
            field_errors = []
            
            for mistake in model_data['all_mistakes']:
                if 'reasoning_analysis' in mistake:
                    analysis = mistake['reasoning_analysis']
                    
                    # Extract error type
                    if 'ERROR TYPE:' in analysis:
                        error_type_section = analysis.split('ERROR TYPE:')[1].split('4.')[0]
                        error_types.append(error_type_section.strip())
                    
                    field_errors.append({
                        'field': mistake['field'],
                        'pmid': mistake['pmid'],
                        'error_description': analysis
                    })
            
            error_patterns['by_model'][model] = {
                'error_types': error_types,
                'field_errors': field_errors,
                'total_errors': len(model_data['all_mistakes'])
            }
        
        # Analyze patterns by field
        field_mistakes = defaultdict(list)
        for model, model_data in results.items():
            for field, mistakes in model_data['mistakes_by_field'].items():
                for mistake in mistakes:
                    field_mistakes[field].append({
                        'model': model,
                        'mistake': mistake
                    })
        
        error_patterns['by_field'] = dict(field_mistakes)
        
        return error_patterns
    
    def create_synthesis_prompt(self, patterns: Dict, results: Dict) -> str:
        """Create prompt for synthesizing reasoning patterns."""
        
        # Extract key information
        models = list(results.keys())
        total_mistakes = sum(len(model_data['all_mistakes']) for model_data in results.values())
        
        # Count error types across models
        error_type_counts = defaultdict(int)
        for model_data in results.values():
            for mistake in model_data['all_mistakes']:
                if 'reasoning_analysis' in mistake:
                    analysis = mistake['reasoning_analysis']
                    if 'OVERINTERPRETATION' in analysis:
                        error_type_counts['OVERINTERPRETATION'] += 1
                    elif 'UNDERINTERPRETATION' in analysis:
                        error_type_counts['UNDERINTERPRETATION'] += 1
                    elif 'TECHNICAL_MISUNDERSTANDING' in analysis:
                        error_type_counts['TECHNICAL_MISUNDERSTANDING'] += 1
                    elif 'KEYWORD_BIAS' in analysis:
                        error_type_counts['KEYWORD_BIAS'] += 1
                    elif 'AMBIGUITY' in analysis:
                        error_type_counts['AMBIGUITY'] += 1
        
        # Field performance summary
        field_performance_summary = {}
        for model, model_data in results.items():
            if 'field_performance' in model_data:
                for field, perf in model_data['field_performance'].items():
                    if field not in field_performance_summary:
                        field_performance_summary[field] = {}
                    field_performance_summary[field][model] = perf.get('accuracy', 0)
        
        prompt = f"""
You are analyzing reasoning patterns and mistakes made by large language models (GPT-5, GPT-4o, GPT-4o-mini) when extracting causal methodology information from scientific papers.

OVERVIEW:
- Total models analyzed: {len(models)}
- Models: {', '.join(models)}
- Total mistakes analyzed: {total_mistakes}
- Error type distribution: {dict(error_type_counts)}

KEY FINDINGS FROM ANALYSIS:

{json.dumps(patterns, indent=2)[:3000]}...

PERFORMANCE DATA:
{json.dumps(field_performance_summary, indent=2)[:2000]}...

SAMPLE REASONING ANALYSES:
{json.dumps({model: model_data['all_mistakes'][:2] for model, model_data in results.items()}, indent=2)[:4000]}...

Based on this comprehensive analysis, please provide:

1. **MODEL-SPECIFIC PATTERNS**: For each model (GPT-5, GPT-4o, GPT-4o-mini), identify the most common types of mistakes and their underlying reasoning patterns.

2. **FIELD-SPECIFIC CHALLENGES**: Identify which methodological fields are most challenging across models and why certain concepts lead to systematic errors.

3. **ERROR TYPE ANALYSIS**: Analyze the prevalence and implications of different error types:
   - OVERINTERPRETATION: Models inferring presence from weak evidence
   - UNDERINTERPRETATION: Models missing clear evidence
   - TECHNICAL_MISUNDERSTANDING: Models misunderstanding methodological concepts
   - KEYWORD_BIAS: Models over-relying on specific terms
   - AMBIGUITY: Text genuinely unclear

4. **SYSTEMATIC TRENDS**: Identify cross-cutting patterns that explain why models make certain mistakes consistently.

5. **IMPLICATIONS FOR METHODOLOGY EXTRACTION**: What do these patterns tell us about the challenges of automated scientific text analysis and how to improve model performance?

Please structure your response as a comprehensive analysis suitable for inclusion in a research paper's discussion section.
"""
        return prompt
    
    def synthesize_patterns(self, patterns: Dict, results: Dict) -> str:
        """Use LLM to synthesize reasoning patterns."""
        prompt = self.create_synthesis_prompt(patterns, results)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error in synthesis: {e}"

def main():
    """Main synthesis function."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    synthesizer = ReasoningPatternSynthesizer(api_key)
    
    # Load reasoning analysis results
    print("Loading reasoning analysis results...")
    results = synthesizer.load_reasoning_analysis()
    
    # Analyze error patterns
    print("Analyzing error patterns...")
    patterns = synthesizer.analyze_error_patterns(results)
    
    # Synthesize patterns using LLM
    print("Synthesizing patterns with LLM...")
    synthesis = synthesizer.synthesize_patterns(patterns, results)
    
    # Save results
    output = {
        'error_patterns': patterns,
        'synthesis': synthesis,
        'summary_stats': {
            'total_models': len(results),
            'total_mistakes_analyzed': sum(len(model_data['all_mistakes']) for model_data in results.values()),
            'models': list(results.keys())
        }
    }
    
    output_file = '/Users/yiquntchen/Desktop/chen-lab/CausalJudge/reasoning_pattern_synthesis.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    # Also save just the synthesis as a text file
    synthesis_file = '/Users/yiquntchen/Desktop/chen-lab/CausalJudge/qualitative_findings.md'
    with open(synthesis_file, 'w') as f:
        f.write("# Qualitative Analysis of Model Reasoning Patterns\n\n")
        f.write(synthesis)
    
    print(f"Pattern synthesis complete. Results saved to:")
    print(f"- {output_file}")
    print(f"- {synthesis_file}")
    
    # Print key findings
    print("\n" + "="*50)
    print("KEY FINDINGS SUMMARY:")
    print("="*50)
    print(synthesis[:1000] + "..." if len(synthesis) > 1000 else synthesis)

if __name__ == "__main__":
    main()