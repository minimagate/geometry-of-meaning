# Geometry of Meaning

A research project investigating the geometric structure of semantic space through systematic perturbation, compression, and transformation of multilingual texts.

> **Current state**: Early-stage. The repository architecture and first experiment (Translation Embedding Baseline) are scaffolded. One experimental run has been executed.

## Research Idea

When we embed text into high-dimensional vector spaces, meaning acquires geometric structure. This project asks: **what is the shape of that structure?**

By applying controlled transformations to texts — compression, negation, paraphrase, logical alteration — and measuring how their embeddings move, we aim to uncover the geometry of meaning itself. What happens to semantics under progressive information deletion? How do different languages trace different paths through the same meaning space? Do semantic representations undergo phase transitions at critical compression thresholds?

## Four Research Areas

The project is organized around four core research directions, each with its own experiments:

| Area | Question |
|------|----------|
| **Semantic Preservation** | How much can you compress a text before meaning collapses? |
| **Semantic Transformation** | How do meaning vectors shift under deliberate alteration? |
| **Local Topology** | What is the structure of meaning in a small semantic neighborhood? |
| **Logical Boundaries** | How do embeddings respond to logical operations like negation? |

Each area is a directory under `experiments/`, with mirrored locations in `data/variants/` and `runs/`.

## Cross-Language and Cross-Model Dimensions

Language and embedding model are **experimental variables**, not directory categories. A single experiment runs across multiple languages and models, producing results that can be compared directly:

```
text_id | language | model | compression_level | cosine_similarity
--------|----------|-------|-------------------|-------------------
p_and_p | en       | e5    | 1.00              | 0.98
p_and_p | it       | e5    | 1.00              | 0.98
p_and_p | zh       | e5    | 1.00              | 0.97
...
```

This design means you can answer questions like "Which language preserves meaning best under compression?" without duplicating datasets or analyses.

## Repository Architecture

The repository follows a mirrored hierarchy: every experiment has three corresponding locations representing three stages of the research pipeline.

```
experiments/<area>/<experiment>/    ← protocol, config, generate, analyze
data/variants/<area>/<experiment>/   ← generated textual variants
runs/<area>/<experiment>/            ← numerical results (immutable)
```

For example, the Translation Embedding Baseline experiment lives at:

```
experiments/semantic_preservation/translation_embedding_baseline/
data/variants/semantic_preservation/translation_embedding_baseline/
runs/semantic_preservation/translation_embedding_baseline/
```

The stable textual corpus lives separately under `data/texts/`.

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd geometry-of-meaning

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package in development mode
pip install -e ".[dev]"
```

Requirements: Python 3.11+.

## Basic Usage

### Running an existing experiment

```bash
# Generate variants for the translation embedding baseline experiment
python experiments/semantic_preservation/translation_embedding_baseline/generate.py

# Run the numerical analysis
python experiments/semantic_preservation/translation_embedding_baseline/analyze.py
```

Each execution creates an immutable, timestamped run under `runs/`.

### Adding a new experiment

1. Create the three mirrored directories:
   ```bash
   mkdir -p experiments/<area>/<experiment>/prompts
   mkdir -p data/variants/<area>/<experiment>
   mkdir -p runs/<area>/<experiment>
   ```

2. Add the required files:
   - `README.md` — scientific protocol
   - `config.yaml` — experimental variables
   - `dataset.jsonl` — text selection
   - `prompts/` — generation and evaluation prompts
   - `generate.py` — variant generation script
   - `analyze.py` — numerical analysis script

3. Reference texts from `data/texts/` using their `text_id`. Do not copy them.

4. Place reusable logic in `src/geometry_of_meaning/`, not in experiment scripts.

5. Never overwrite existing runs — every change produces a new run.

## Complete Research Flow

```
data/texts/              ← stable canonical corpus
      ↓
experiments/             ← protocol, config, prompts, scripts
      ↓
data/variants/           ← generated textual variants
      ↓
src/geometry_of_meaning/ ← shared embedding & metric code
      ↓
runs/                    ← immutable numerical results
```

## License

MIT — see [LICENSE](LICENSE)
