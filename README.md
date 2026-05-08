# DebateRAG-Counterspeech

A Retrieval-Augmented Generation system for generating effective counter-narratives to hate speech through structured debate generation and knowledge base construction.

## Overview

DebateRAG combines structured debate simulation with knowledge base construction to create an intelligent system that can assess potentially harmful queries and generate appropriate counter-speech responses. The system uses a graph database to store hate speech patterns and their corresponding counter-narratives, enabling sophisticated retrieval and response generation.

## Key Features

- **Structured Debate Generation**: Creates multi-turn debates between opposing viewpoints on sensitive topics
- **Knowledge Base Construction**: Builds a graph database of hate speech patterns and counter-narratives
- **Intelligent Query Assessment**: Evaluates incoming queries for potential hate content using similarity and harmfulness scoring
- **Adaptive Response Generation**: Provides different levels of mitigation based on assessed threat level
- **Comprehensive Evaluation**: Includes metrics for effectiveness, relevance, evidence quality, and toxicity reduction

## System Architecture

### Core Components

1. **Data Generation** (`src/data_generation/`)
   - `debate_gen.py`: Generates structured debates using LLM players
   - `llm_eval.py`: Evaluates generated debates for quality
   - `short_hate_sentence_gen.py`: Creates synthetic hate speech examples

2. **Knowledge Base** (`src/knowledge_base/`)
   - `kb_builder.py`: Constructs graph database with nodes and relationships
   - `data_processor.py`: Processes debate data into structured format
   - `kb_build_level2.py`: Advanced knowledge base construction

3. **Retrieval System** (`src/retrieval/`)
   - `retrieve.py`: Implements syntax, semantic, and hybrid search
   - `vector_utils.py`: Vector embedding utilities

4. **Response Generation** (`src/generation/`)
   - `assessment_system.py`: Core system for query assessment and response generation

5. **Evaluation** (`src/evaluation/`)
   - `evaluation_metrics.py`: Comprehensive evaluation metrics and comparison tools

## Installation

1. **Prerequisites**
   - Python 3.8+
   - Memgraph database
   - Together AI API key
   - Google Perspective API key (optional, for toxicity evaluation)

2. **Install Dependencies**
   ```bash
   pip install together gqlalchemy pandas numpy tqdm
   ```

3. **Database Setup**
   - Install and start Memgraph on `127.0.0.1:7687`
   - Or use Docker: `docker run -p 7687:7687 memgraph/memgraph`

4. **API Configuration**
   ```bash
   export TOGETHER_API_KEY="your_together_api_key"
   export PERSPECTIVE_API_KEY="your_perspective_api_key"  # Optional
   ```

## Quick Start

### 1. Generate Debate Dataset

Create structured debates from a topics CSV file:

```bash
python scripts/generate_debate_dataset.py \
    --topics data/raw/hs.csv \
    --output_dir data/generated \
    --max_turns 8
```

### 2. Build Knowledge Base

Process the generated debates into a graph database:

```bash
python scripts/build_knowledge_base.py \
    --debate_file data/generated/debate_generated_results.json \
    --output_dir data/processed
```

### 3. Run Assessment System

Use the system to assess and respond to queries:

```python
from src.generation.assessment_system import HateAssessmentSystem
from src.retrieval.retrieve import RAGRetriever

# Initialize components
retriever = RAGRetriever()
system = HateAssessmentSystem(retriever, api_key="your_api_key")

# Process a query
result = system.process_query(
    user_query="Your query here",
    search_method="hybrid",
    num_results=5
)

print(f"Response: {result['response']}")
print(f"Hate Score: {result['hate_score']}")
```

### 4. Replicating the datasets
To replicate the datasets used in the paper follow these steps:

1. Download the MultitargetCONAN dataset from the official source:
   - [MultitargetCONAN Dataset](https://github.com/marcoguerini/CONAN)

2. Download the SSTF dataset from the official source:
   - [SSTF Dataset](https://huggingface.co/datasets/SetFit/sst5)

Place the downloaded files (e.g., `multitargetconan.csv`, `sstf.csv`) in the `data/raw/` directory.


To match the distribution and size used in our experiments, we use a stratified subsample of 996 examples.  
Assuming you have loaded the dataset into a pandas DataFrame called `df` and imported `resample` from `sklearn.utils`:

```python
from sklearn.utils import resample

# Stratified sampling maintaining TARGET distribution
subsample = df.groupby("TARGET", group_keys=False).apply(
    lambda x: resample(
        x, 
        replace=False, 
        n_samples=int(996 * len(x) / len(df)), 
        random_state=42
    )
)
```

### 5. Run Full Evaluation

Evaluate the system against a benchmark dataset:

```bash
python scripts/run_full_evaluation.py \
    --dataset data/evaluation/benchmark.csv \
    --output_dir results \
    --sample_size 100
```

## Data Format

### Input Topics CSV
```csv
topic,position
Immigration,Immigrants are dangerous criminals who should be deported
Religion,Religious minorities pose a threat to our society
```

### Knowledge Base Structure

The system creates a graph database with:
- **Topics**: General subject areas
- **HateParagraphs**: Hate speech content with quality metrics
- **CounterParagraphs**: Counter-narrative responses
- **HateContent**: Individual hate speech sentences
- **Relationships**: TALKS_ABOUT, COUNTERED_WITH, CONTAINS

## Assessment Scoring

The system uses a hybrid scoring approach:

- **Similarity Score**: Semantic similarity to known hate patterns
- **Harmfulness Score**: Assessed potential for harm (1-10 scale)
- **Final Score**: Weighted combination with calibrated normalization

**Mitigation Levels:**
- **None** (0-4): Standard response
- **Mild** (4-7): Educational tone with gentle correction
- **Strong** (7+): Direct counter-narrative with evidence

## Evaluation Metrics

- **LLM Evaluation**: Overall quality, effectiveness, relevance, evidence quality, persuasiveness, directness
- **Toxicity Metrics**: Using Google Perspective API
- **Gold Standard Comparison**: Against human-annotated counter-narratives

## Configuration

### Search Methods
- `syntax`: Keyword-based search
- `semantic`: Vector similarity search
- `hybrid`: Combined approach

### Model Options
- `together`: Use Together AI models for generation
- `memgraph`: Use Memgraph for knowledge base queries

## Scripts Reference

### Core Scripts
- `scripts/generate_debate_dataset.py`: Generate training data
- `scripts/build_knowledge_base.py`: Create graph database
- `scripts/run_full_evaluation.py`: Comprehensive evaluation

### Options
```bash
# Generate with custom parameters
python scripts/generate_debate_dataset.py \
    --topics data/topics.csv \
    --output_dir output \
    --max_turns 8 \
    --skip_evaluation

# Evaluate with specific settings
python scripts/run_full_evaluation.py \
    --dataset benchmark.csv \
    --output_dir results \
    --search_method hybrid \
    --sample_size 50
```

## Output Files

- `debate_generated_results.json`: Raw debate data
- `evaluated_debates.json`: Quality-scored debates
- `evaluation_results.json`: Comprehensive evaluation metrics
- `evaluation_report.md`: Human-readable analysis
- `evaluation_metrics.csv`: Spreadsheet-compatible results

## Research Applications

This system is designed for academic research in:
- Counter-speech generation
- Hate speech detection and mitigation
- Conversational AI safety
- Content moderation systems
- Social media platform safety

## Contributing

1. Ensure all generated content is for research purposes only
2. Follow ethical guidelines for hate speech research
3. Test thoroughly with the evaluation framework
4. Document any new features or modifications

## License


## Citation



## Support
