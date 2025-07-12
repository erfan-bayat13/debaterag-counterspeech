import argparse
import os
import sys
from pathlib import Path
import subprocess
import json

# Add src to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from knowledge_base.kb_builder import import_all_data, sample_query, basic_query
from knowledge_base.kb_build_level2 import extend_knowledge_base_with_generated_content
from retrieval.vector_utils import fix_existing_embeddings

def check_memgraph_connection():
    """Check if Memgraph is running and accessible."""
    try:
        from gqlalchemy import Memgraph
        memgraph = Memgraph("127.0.0.1", 7687)
        # Try a simple query
        list(memgraph.execute_and_fetch("RETURN 1 as test"))
        print("✓ Memgraph connection successful")
        return True
    except Exception as e:
        print(f"✗ Memgraph connection failed: {e}")
        print("Please ensure Memgraph is running on localhost:7687")
        return False

def copy_files_to_docker(hate_paragraphs_path, hate_sentences_path, counter_paragraphs_path):
    """Copy CSV files to the Memgraph Docker container."""
    print("\n=== Copying Files to Docker Container ===")
    
    try:
        # Get running containers
        result = subprocess.run(
            ["docker", "ps", "--format", "json"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse container info
        containers = []
        for line in result.stdout.strip().split('\n'):
            if line:
                containers.append(json.loads(line))
        
        # Find Memgraph container
        memgraph_container = None
        for container in containers:
            if 'memgraph' in container.get('Image', '').lower():
                memgraph_container = container
                break
        
        if not memgraph_container:
            print("✗ No running Memgraph container found")
            print("Make sure Memgraph is running in Docker")
            return False
        
        container_id = memgraph_container['ID']
        print(f"✓ Found Memgraph container: {container_id}")
        
        # Copy files to container
        files_to_copy = [
            (hate_paragraphs_path, "nodes_hate_paragraphs.csv"),
            (hate_sentences_path, "nodes_hate_sentences.csv"), 
            (counter_paragraphs_path, "nodes_counter_paragraphs.csv")
        ]
        
        for local_path, container_filename in files_to_copy:
            if not os.path.exists(local_path):
                print(f"✗ File not found: {local_path}")
                return False
            
            try:
                subprocess.run([
                    "docker", "cp", 
                    local_path, 
                    f"{container_id}:/{container_filename}"
                ], check=True, capture_output=True)
                
                print(f"✓ Copied {os.path.basename(local_path)} to container")
                
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to copy {local_path}: {e}")
                return False
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Docker command failed: {e}")
        return False
    except json.JSONDecodeError:
        print("✗ Failed to parse docker ps output")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def build_initial_kb(hate_paragraphs_path, hate_sentences_path, counter_paragraphs_path):
    print("\n=== Building Initial Knowledge Base ===")

    for path in [hate_paragraphs_path, hate_sentences_path, counter_paragraphs_path]:
        if not os.path.exists(path):
            print(f"✗ File not found: {path}")
            return False
    
    try:
        import_all_data(
            "nodes_hate_paragraphs.csv",
            "nodes_hate_sentences.csv", 
            "nodes_counter_paragraphs.csv"
        )
        print("✓ Initial knowledge base built successfully")
        return True
    except Exception as e:
        print(f"✗ Error building initial knowledge base: {e}")
        return False
    
def extend_with_synthetic_data(synthetic_data_path):
    """Extend knowledge base with synthetic hate speech data."""
    print("\n=== Extending Knowledge Base with Synthetic Data ===")
    
    if not os.path.exists(synthetic_data_path):
        print(f"⚠ Synthetic data file not found: {synthetic_data_path}")
        print("Skipping synthetic data extension...")
        return True
    
    try:
        from gqlalchemy import Memgraph
        memgraph = Memgraph("127.0.0.1", 7687)
        
        stats = extend_knowledge_base_with_generated_content(
            jsonl_file_path=synthetic_data_path,
            memgraph_connection=memgraph
        )
        
        print(f"✓ Extended KB with {stats.get('examples_imported', 0)} synthetic examples")
        return True
    except Exception as e:
        print(f"✗ Failed to extend KB with synthetic data: {e}")
        return False
    
def fix_vector_embeddings():
    """Fix and rebuild vector embeddings for semantic search."""
    print("\n=== Fixing Vector Embeddings ===")
    
    try:
        success = fix_existing_embeddings()
        if success:
            print("✓ Vector embeddings fixed successfully")
            return True
        else:
            print("⚠ Vector embeddings fix encountered issues")
            return False
    except Exception as e:
        print(f"✗ Failed to fix vector embeddings: {e}")
        return False
    
def verify_kb():
    """Verify the knowledge base was built correctly."""
    print("\n=== Verifying Knowledge Base ===")
    
    try:
        # Run basic verification queries
        print("Running basic verification...")
        basic_query()
        
        print("\nRunning sample schema query...")
        sample_query()
        
        print("✓ Knowledge base verification completed")
        return True
    except Exception as e:
        print(f"✗ Knowledge base verification failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Build DebateRAG Knowledge Base")
    
    # Required arguments
    parser.add_argument("--hate_paragraphs", type=str, required=True,
                       help="Path to hate paragraphs CSV file")
    parser.add_argument("--hate_sentences", type=str, required=True,
                       help="Path to hate sentences CSV file")
    parser.add_argument("--counter_paragraphs", type=str, required=True,
                       help="Path to counter paragraphs CSV file")
    
    # Optional arguments
    parser.add_argument("--synthetic_data", type=str,
                       help="Path to synthetic hate speech JSONL file")
    parser.add_argument("--skip_vector_fix", action="store_true",
                       help="Skip vector embedding fix step")
    parser.add_argument("--skip_verification", action="store_true",
                       help="Skip knowledge base verification")
    
    args = parser.parse_args()
    
    print("DebateRAG Knowledge Base Builder")
    print("=" * 50)
    
    # Step 1: Check Memgraph connection
    if not check_memgraph_connection():
        print("\nPlease start Memgraph and try again.")
        sys.exit(1)

    # Step 1.5: Copy files to Docker container
    if not copy_files_to_docker(args.hate_paragraphs, args.hate_sentences, args.counter_paragraphs):
        print("\nFailed to copy files to Docker container.")
        sys.exit(1)
    
    # Step 2: Build initial knowledge base
    if not build_initial_kb(args.hate_paragraphs, args.hate_sentences, args.counter_paragraphs):
        print("\nFailed to build initial knowledge base.")
        sys.exit(1)
    
    # Step 3: Extend with synthetic data (optional)
    if args.synthetic_data:
        if not extend_with_synthetic_data(args.synthetic_data):
            print("\nFailed to extend with synthetic data.")
            # Don't exit here, continue without synthetic data
    
    # Step 4: Fix vector embeddings (optional)
    if not args.skip_vector_fix:
        if not fix_vector_embeddings():
            print("\nVector embedding fix failed, but continuing...")
            # Don't exit here, semantic search might still work
    
    # Step 5: Verify knowledge base (optional)
    if not args.skip_verification:
        if not verify_kb():
            print("\nKnowledge base verification failed.")
            sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✓ Knowledge Base Build Complete!")
    print("\nYour DebateRAG knowledge base is ready for use.")
    print("You can now run retrieval and generation scripts.")


if __name__ == "__main__":
    main()