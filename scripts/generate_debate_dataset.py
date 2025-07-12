import argparse
import os
import sys
import json
from pathlib import Path

# Add src to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_generation.debate_gen import main as generate_debates_main
#from data_generation.short_hate_sentence_gen import save_generated_examples_to_file
from data_generation.llm_eval import run_pair_evaluation_pipeline

def check_api_key():
    """Check if Together AI API key is available."""
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("✗ TOGETHER_API_KEY environment variable not set")
        print("Please set your Together AI API key:")
        print("export TOGETHER_API_KEY=your_api_key_here")
        return False
    print("✓ Together AI API key found")
    return True

def generate_debates(topics_csv, output_dir, api_key, max_turns=4, num_debates=None):
    """Generate structured debates from topics and positions."""
    print("\n=== Generating Structured Debates ===")
    
    if not os.path.exists(topics_csv):
        print(f"✗ Topics file not found: {topics_csv}")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    debate_output_file = os.path.join(output_dir, "raw_debates.json")
    
    # Prepare arguments for debate generation
    debate_args = [
        "--output_dir", output_dir,
        "--data_path", topics_csv,
        "--max_turns", str(max_turns),
        "--model_prefix", "debate",
        "--data_suffix", "generated"
    ]
    
    if api_key:
        debate_args.extend(["--together_api_key", api_key])
    
    try:
        # Temporarily modify sys.argv to pass arguments to the main function
        original_argv = sys.argv.copy()
        sys.argv = ["generate_debates"] + debate_args
        
        # Import and run debate generation
        from data_generation.debate_gen import main as debate_main
        debate_main()
        
        # Restore original argv
        sys.argv = original_argv
        
        # Check if output file was created
        generated_file = os.path.join(output_dir, "debate_generated_results.json")
        if os.path.exists(generated_file):
            print(f"✓ Debates generated successfully: {generated_file}")
            return generated_file
        else:
            print("✗ Debate generation completed but output file not found")
            return None
            
    except Exception as e:
        print(f"✗ Debate generation failed: {e}")
        sys.argv = original_argv  # Restore on error
        return None
    
# def generate_synthetic_hate(topics_list, output_dir, examples_per_topic=15):
#     """Generate synthetic hate speech examples."""
#     print("\n=== Generating Synthetic Hate Speech ===")
    
#     if not topics_list:
#         print("⚠ No topics provided, skipping synthetic hate generation")
#         return None
    
#     synthetic_output_file = os.path.join(output_dir, "synthetic_hate_speech.jsonl")
    
#     try:
#         # Generate synthetic examples
#         results = save_generated_examples_to_file(
#             topic_list=topics_list,
#             examples_per_topic=examples_per_topic,
#             output_file=synthetic_output_file
#         )
        
#         print(f"✓ Generated synthetic hate speech: {synthetic_output_file}")
#         return synthetic_output_file
        
#     except Exception as e:
#         print(f"✗ Synthetic hate generation failed: {e}")
#         return None
    
def evaluate_debates(debate_file, output_dir, api_key):
    """Evaluate generated debates using LLM judge."""
    print("\n=== Evaluating Debates with LLM Judge ===")
    
    if not debate_file or not os.path.exists(debate_file):
        print(f"✗ Debate file not found: {debate_file}")
        return None
    
    evaluation_output_file = os.path.join(output_dir, "evaluated_debates.json")
    
    try:
        # Run evaluation pipeline
        results = run_pair_evaluation_pipeline(
            input_path=debate_file,
            output_path=evaluation_output_file,
            config={
                'quality_threshold': 6.0,
                'evaluation_model': 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
                'api_temperature': 0.3,
                'batch_size': 10
            }
        )
        
        print(f"✓ Debates evaluated successfully: {evaluation_output_file}")
        return evaluation_output_file
        
    except Exception as e:
        print(f"✗ Debate evaluation failed: {e}")
        return None
    
def process_to_csv(evaluated_debates_file, output_dir):
    """Process evaluated debates into CSV format for knowledge base."""
    print("\n=== Processing Debates to CSV Format ===")
    
    if not evaluated_debates_file or not os.path.exists(evaluated_debates_file):
        print(f"✗ Evaluated debates file not found: {evaluated_debates_file}")
        return False
    
    try:
        # Temporarily set the json_path variable that csv_build.py expects
        import sys
        original_modules = sys.modules.copy()
        
        # Import and configure the csv builder
        from knowledge_base.data_processor import process_debates
        
        # Set the expected global variable
        sys.modules['knowledge_base.data_processor'].json_path = evaluated_debates_file
        
        # Process the debates
        result = process_debates(evaluated_debates_file)
        
        print(f"✓ Processed debates to CSV format:")
        print(f"  - Hate paragraphs: {result.get('hate_paragraphs', 0)}")
        print(f"  - Hate sentences: {result.get('hate_sentences', 0)}")
        print(f"  - Counter paragraphs: {result.get('counter_paragraphs', 0)}")
        
        return True
        
    except Exception as e:
        print(f"✗ CSV processing failed: {e}")
        return False


def extract_topics_from_csv(topics_csv):
    """Extract unique topics from the topics CSV file."""
    try:
        import pandas as pd
        df = pd.read_csv(topics_csv)
        if 'topic' in df.columns:
            topics = df['topic'].unique().tolist()
            print(f"✓ Extracted {len(topics)} unique topics")
            return topics
        else:
            print("⚠ No 'topic' column found in CSV")
            return []
    except Exception as e:
        print(f"⚠ Could not extract topics: {e}")
        return []
    

def main():
    parser = argparse.ArgumentParser(description="Generate DebateRAG Dataset")
    
    # Required arguments
    parser.add_argument("--topics", type=str, required=True,
                       help="Path to CSV file with topics and positions")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save generated data")
    
    # Optional arguments
    parser.add_argument("--api_key", type=str,
                       help="Together AI API key (or set TOGETHER_API_KEY env var)")
    parser.add_argument("--max_turns", type=int, default=8,
                       help="Maximum turns per debate (default: 8)")
    parser.add_argument("--synthetic_examples", type=int, default=15,
                       help="Synthetic examples per topic (default: 15)")
    parser.add_argument("--skip_synthetic", action="store_true",
                       help="Skip synthetic hate speech generation")
    parser.add_argument("--skip_evaluation", action="store_true",
                       help="Skip LLM evaluation of debates")
    parser.add_argument("--skip_csv", action="store_true",
                       help="Skip CSV processing")
    
    args = parser.parse_args()
    
    print("DebateRAG Dataset Generator")
    print("=" * 50)
    
    # Set API key if provided
    if args.api_key:
        os.environ["TOGETHER_API_KEY"] = args.api_key
    
    # Check API key
    if not check_api_key():
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Generate structured debates
    debate_file = generate_debates(
        topics_csv=args.topics,
        output_dir=args.output_dir,
        api_key=os.getenv("TOGETHER_API_KEY"),
        max_turns=args.max_turns
    )
    
    if not debate_file:
        print("\nFailed to generate debates.")
        sys.exit(1)
    
    # Step 2: Generate synthetic hate speech (optional)
    # synthetic_file = None
    # if not args.skip_synthetic:
    #     topics_list = extract_topics_from_csv(args.topics)
    #     if topics_list:
    #         synthetic_file = generate_synthetic_hate(
    #             topics_list=topics_list,
    #             output_dir=args.output_dir,
    #             examples_per_topic=args.synthetic_examples
    #         )
    
    # Step 3: Evaluate debates with LLM judge (optional)
    evaluated_file = debate_file  # Default to original file
    if not args.skip_evaluation:
        eval_result = evaluate_debates(
            debate_file=debate_file,
            output_dir=args.output_dir,
            api_key=os.getenv("TOGETHER_API_KEY")
        )
        if eval_result:
            evaluated_file = eval_result
    
    # Step 4: Process to CSV format (optional)
    if not args.skip_csv:
        if not process_to_csv(evaluated_file, args.output_dir):
            print("\nFailed to process debates to CSV format.")
            # Don't exit here, the JSON files are still useful
    
    # Summary
    print("\n" + "=" * 50)
    print("✓ Dataset Generation Complete!")
    print(f"\nGenerated files in {args.output_dir}:")
    print(f"  - Raw debates: {os.path.basename(debate_file) if debate_file else 'None'}")
    print(f"  - Evaluated debates: {os.path.basename(evaluated_file) if evaluated_file else 'None'}")
    # print(f"  - Synthetic hate: {os.path.basename(synthetic_file) if synthetic_file else 'None'}")
    
    if not args.skip_csv:
        print(f"  - CSV files: output1/ directory")
    
    print("\nNext steps:")
    print("1. Review the generated data")
    print("2. Run build_knowledge_base.py to create the knowledge base")
    print("3. Test the system with your generated data")


if __name__ == "__main__":
    main()