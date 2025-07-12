#!/usr/bin/env python3
"""
Full Evaluation Script for DebateRAG

This script runs comprehensive evaluation of the DebateRAG system:
1. Generate counter-narratives for a benchmark dataset
2. Evaluate using LLM judge and toxicity metrics
3. Compare against gold standard (if available)
4. Generate reports and export metrics

Usage:
    python scripts/run_full_evaluation.py --dataset data/raw/MultitargetCONAN.csv \
                                         --output_dir results \
                                         --sample_size 100 \
                                         --together_api_key your_key \
                                         --perspective_api_key your_key
"""

import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.evaluation_metrics import HateCounterEvaluator


def check_requirements():
    """Check if all required components are available."""
    issues = []
    
    # Check API keys
    if not os.getenv("TOGETHER_API_KEY"):
        issues.append("TOGETHER_API_KEY environment variable not set")
    
    if not os.getenv("PERSPECTIVE_API_KEY"):
        issues.append("PERSPECTIVE_API_KEY environment variable not set (toxicity evaluation will be skipped)")
    
    # Check Memgraph connection
    try:
        from gqlalchemy import Memgraph
        memgraph = Memgraph("127.0.0.1", 7687)
        list(memgraph.execute_and_fetch("RETURN 1 as test"))
        print("✓ Memgraph connection successful")
    except Exception as e:
        issues.append(f"Memgraph connection failed: {e}")
    
    # Check if knowledge base exists
    try:
        from gqlalchemy import Memgraph
        memgraph = Memgraph("127.0.0.1", 7687)
        result = list(memgraph.execute_and_fetch("MATCH (hp:HateParagraph) RETURN count(hp) as count"))
        if result and result[0]['count'] > 0:
            print(f"✓ Knowledge base found with {result[0]['count']} hate paragraphs")
        else:
            issues.append("Knowledge base appears to be empty. Run build_knowledge_base.py first.")
    except Exception as e:
        issues.append(f"Could not verify knowledge base: {e}")
    
    return issues


def load_and_validate_dataset(dataset_path, sample_size=None):
    """Load and validate the evaluation dataset."""
    print(f"\n=== Loading Dataset: {dataset_path} ===")
    
    if not os.path.exists(dataset_path):
        print(f"✗ Dataset file not found: {dataset_path}")
        return None
    
    try:
        import pandas as pd
        df = pd.read_csv(dataset_path)
        
        print(f"✓ Dataset loaded: {len(df)} examples")
        
        # Check required columns
        required_cols = ['HATE_SPEECH'] if 'HATE_SPEECH' in df.columns else ['hate_speech']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"✗ Missing required columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        # Apply sampling if requested
        if sample_size and sample_size < len(df):
            df = df.sample(sample_size, random_state=42)
            print(f"✓ Sampled {len(df)} examples for evaluation")
        
        # Check for gold standard counter-narratives
        has_gold = 'COUNTER_NARRATIVE' in df.columns or 'counter_narrative' in df.columns
        if has_gold:
            print("✓ Gold standard counter-narratives found")
        else:
            print("⚠ No gold standard counter-narratives found")
        
        return df
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None


def run_evaluation(dataset, output_dir, search_method, together_api_key, perspective_api_key, 
                  llm_model, memgraph_host, memgraph_port):
    """Run the full evaluation pipeline."""
    print(f"\n=== Running Evaluation ===")
    print(f"Search method: {search_method}")
    print(f"LLM model: {llm_model}")
    
    try:
        # Initialize evaluator
        evaluator = HateCounterEvaluator(
            perspective_api_key=perspective_api_key,
            together_api_key=together_api_key,
            llm_model_name=llm_model,
            memgraph_host=memgraph_host,
            memgraph_port=memgraph_port
        )
        
        # Save dataset temporarily for the evaluator
        temp_dataset_path = os.path.join(output_dir, "temp_evaluation_dataset.csv")
        dataset.to_csv(temp_dataset_path, index=False)
        
        # Run evaluation
        results = evaluator.run_full_evaluation(
            dataset_path=temp_dataset_path,
            output_path=os.path.join(output_dir, "evaluation_results.json"),
            sample_size=len(dataset),
            search_method=search_method
        )
        
        # Clean up temp file
        os.remove(temp_dataset_path)
        
        print("✓ Evaluation completed successfully")
        return results
        
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_reports(results_path, output_dir, evaluator):
    """Generate human-readable reports and CSV exports."""
    print(f"\n=== Generating Reports ===")
    
    try:
        # Generate markdown report
        report_path = os.path.join(output_dir, "evaluation_report.md")
        evaluator.generate_comparison_report(results_path, report_path)
        print(f"✓ Markdown report: {report_path}")
        
        # Export metrics to CSV
        csv_path = os.path.join(output_dir, "evaluation_metrics.csv")
        evaluator.export_metrics_csv(results_path, csv_path)
        print(f"✓ CSV metrics: {csv_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Report generation failed: {e}")
        return False


def print_summary(results_path):
    """Print a summary of the evaluation results."""
    print(f"\n=== Evaluation Summary ===")
    
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        config = results.get('config', {})
        metrics = results.get('metrics', {})
        
        print(f"Dataset: {config.get('dataset', 'Unknown')}")
        print(f"Sample size: {config.get('sample_size', 'Unknown')}")
        print(f"Search method: {config.get('search_method', 'Unknown')}")
        
        # LLM evaluation metrics
        if 'llm_evaluation' in metrics:
            llm_metrics = metrics['llm_evaluation']
            print(f"\nLLM Evaluation Results:")
            print(f"  Overall Score: {llm_metrics.get('avg_overall_score', 0):.2f}/10")
            print(f"  Effectiveness: {llm_metrics.get('avg_effectiveness', 0):.2f}/10")
            print(f"  Relevance: {llm_metrics.get('avg_relevance', 0):.2f}/10")
            print(f"  Evidence Quality: {llm_metrics.get('avg_evidence_quality', 0):.2f}/10")
            print(f"  Persuasiveness: {llm_metrics.get('avg_persuasiveness', 0):.2f}/10")
            print(f"  Directness: {llm_metrics.get('avg_directness', 0):.2f}/10")
        
        # Toxicity metrics
        if 'toxicity' in metrics:
            tox_metrics = metrics['toxicity']
            if 'avg_counter_scores' in tox_metrics:
                avg_toxicity = tox_metrics['avg_counter_scores'].get('toxicity', 0)
                print(f"\nToxicity Results:")
                print(f"  Avg Counter-Narrative Toxicity: {avg_toxicity:.4f}")
                
                if 'reduction_metrics' in tox_metrics:
                    reduction = tox_metrics['reduction_metrics']['avg_reduction_percent'].get('toxicity', 0)
                    print(f"  Avg Toxicity Reduction: {reduction:.1f}%")
        
        # Gold standard comparison
        if 'gold_standard' in metrics:
            print(f"\n✓ Comparison with gold standard available in detailed results")
        
    except Exception as e:
        print(f"Could not print summary: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run Full DebateRAG Evaluation")
    
    # Required arguments
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to evaluation dataset CSV file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save evaluation results")
    
    # Optional arguments
    parser.add_argument("--sample_size", type=int,
                       help="Number of examples to evaluate (default: all)")
    parser.add_argument("--search_method", type=str, default="hybrid",
                       choices=["syntax", "semantic", "hybrid"],
                       help="Retrieval method to use (default: hybrid)")
    parser.add_argument("--llm_model", type=str,
                       default="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                       help="LLM model for evaluation")
    
    # API keys
    parser.add_argument("--together_api_key", type=str,
                       help="Together AI API key")
    parser.add_argument("--perspective_api_key", type=str,
                       help="Google Perspective API key")
    
    # Database connection
    parser.add_argument("--memgraph_host", type=str, default="127.0.0.1",
                       help="Memgraph host (default: 127.0.0.1)")
    parser.add_argument("--memgraph_port", type=int, default=7687,
                       help="Memgraph port (default: 7687)")
    
    # Control flags
    parser.add_argument("--skip_reports", action="store_true",
                       help="Skip report generation")
    parser.add_argument("--force", action="store_true",
                       help="Force evaluation even if issues are detected")
    
    args = parser.parse_args()
    
    print("DebateRAG Full Evaluation")
    print("=" * 50)
    
    # Set API keys if provided
    if args.together_api_key:
        os.environ["TOGETHER_API_KEY"] = args.together_api_key
    if args.perspective_api_key:
        os.environ["PERSPECTIVE_API_KEY"] = args.perspective_api_key
    
    # Check requirements
    issues = check_requirements()
    if issues:
        print("⚠ Issues detected:")
        for issue in issues:
            print(f"  - {issue}")
        
        if not args.force:
            print("\nUse --force to proceed anyway, or fix the issues above.")
            sys.exit(1)
        else:
            print("\nProceeding with --force flag...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and validate dataset
    dataset = load_and_validate_dataset(args.dataset, args.sample_size)
    if dataset is None:
        print("\nFailed to load dataset.")
        sys.exit(1)
    
    # Run evaluation
    results = run_evaluation(
        dataset=dataset,
        output_dir=args.output_dir,
        search_method=args.search_method,
        together_api_key=os.getenv("TOGETHER_API_KEY"),
        perspective_api_key=os.getenv("PERSPECTIVE_API_KEY"),
        llm_model=args.llm_model,
        memgraph_host=args.memgraph_host,
        memgraph_port=args.memgraph_port
    )
    
    if not results:
        print("\nEvaluation failed.")
        sys.exit(1)
    
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    
    # Generate reports (optional)
    if not args.skip_reports:
        try:
            from evaluation.evaluation_metrics import HateCounterEvaluator
            evaluator = HateCounterEvaluator(
                perspective_api_key=os.getenv("PERSPECTIVE_API_KEY"),
                together_api_key=os.getenv("TOGETHER_API_KEY"),
                llm_model_name=args.llm_model,
                memgraph_host=args.memgraph_host,
                memgraph_port=args.memgraph_port
            )
            
            if not generate_reports(results_path, args.output_dir, evaluator):
                print("Report generation failed, but evaluation results are still available.")
        except Exception as e:
            print(f"Report generation failed: {e}")
    
    # Print summary
    print_summary(results_path)
    
    # Final summary
    print("\n" + "=" * 50)
    print("✓ Evaluation Complete!")
    print(f"\nResults saved in: {args.output_dir}")
    print(f"  - evaluation_results.json (detailed results)")
    if not args.skip_reports:
        print(f"  - evaluation_report.md (human-readable report)")
        print(f"  - evaluation_metrics.csv (spreadsheet-friendly metrics)")
    
    # Add timestamp to results
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        results['evaluation_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_script_version': '1.0.0',
            'command_line_args': vars(args)
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
    except Exception as e:
        print(f"Could not add metadata: {e}")
    
    print(f"\nEvaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()