# Progressive Semantic Compression

## Research question

How does semantic similarity change as information is progressively removed from equivalent texts in different languages?

## Motivation

Text compression is not a uniform process. As we progressively remove information from a text, the semantic embedding may not drift linearly. Some compression steps may cause negligible semantic displacement, while others may trigger sudden collapses. Understanding this geometry — and how it differs across languages — reveals how meaning is encoded in vector space.

## Hypothesis

1. **Cross-language convergence**: Under extreme compression, semantically equivalent texts in different languages converge toward the same point in embedding space, as language-specific surface features are stripped away.

2. **Non-linear collapse**: Semantic similarity to the canonical text decreases non-linearly with compression. There exists a critical compression threshold beyond which meaning collapses rapidly (a semantic phase transition).

3. **Language-dependent resilience**: Some languages preserve meaning longer under compression due to structural properties (e.g., information density, morphological richness).

## Independent variables

- **Compression level**: 1.00, 0.75, 0.50, 0.25, 0.10 (fraction of original length retained)
- **Language**: en, it, zh, ja, da
- **Text category**: novel, drama, scientific, philosophy
- **Embedding model**: multilingual-e5-large (initially; bge-m3 to be added)

## Dependent variables

- **Cosine similarity** to the canonical (uncompressed) text in the source language
- **Step-to-step displacement**: Euclidean distance between consecutive compression levels
- **Cumulative trajectory length**: total distance traveled through embedding space across all compression steps
- **Estimated semantic collapse point**: the compression level where cosine similarity drops below a threshold (e.g., 0.8)

## Controls

- Shared source meaning (all translations derive from the same original)
- Shared compression protocol (identical prompt across languages)
- Shared embedding configuration
- Equivalent compression budgets (same fractional levels for all texts)
- Fixed random seed for any stochastic generation

## Dataset

Texts are selected from the canonical corpus (`data/texts/originals/`) and their translations (`data/texts/translations/`). The dataset is defined in `dataset.jsonl`.

Initially, four texts across different categories:
- `pride_and_prejudice_opening` (novel, en)
- `hamlet_to_be_or_not_to_be` (drama, en)
- `einstein_relativity_extract` (scientific, de→en)
- `kafka_metamorphosis_opening` (novel/philosophical, de→en)

Each text is available in all five languages.

## Compression method

An LLM is instructed to compress the text to a target length while preserving core meaning. The prompt specifies:
- Target fraction (e.g., "reduce to 50% of original length")
- Preservation requirement ("retain all essential information")
- Format constraint ("output only the compressed text")

The exact prompt is version-controlled in `prompts/compress.md`.

## Metrics

All metrics are computed in the shared embedding space defined by the current embedding model.

| Metric | Definition | Range |
|--------|-----------|-------|
| Cosine similarity | cos(embed(canonical), embed(compressed)) | [-1, 1] |
| Step displacement | ‖embed(level_i) - embed(level_{i-1})‖₂ | [0, ∞) |
| Cumulative trajectory length | Σ step displacements from level 1.0 to current | [0, ∞) |

## Procedure

1. Load canonical texts and all translations from `data/texts/`
2. For each (text, language, compression level) combination, generate a compressed variant using the LLM
3. Validate each variant: non-empty, unique variant_id, correct parent reference
4. Write validated variants to `data/variants/semantic_preservation/progressive_compression/`
5. Embed all original, translated, and compressed texts
6. Compute cosine similarities, step displacements, and trajectory lengths
7. Save results as a timestamped run under `runs/semantic_preservation/progressive_compression/`

## Expected outputs

- `metrics.parquet`: full table of all measurements
- `summary.json`: aggregate statistics per language, model, and compression level
- `manifest.yaml`: run metadata with config snapshot and git commit

## Limitations

- LLM-based compression is stochastic; repeated compressions of the same text may differ
- Embedding models have their own biases and may not represent semantics uniformly across languages
- The four-text corpus is small; findings may not generalize
- Compression fraction is a rough proxy for information content; some sentences carry more information per word than others
- The "semantic collapse point" threshold (0.8) is arbitrary and should be varied in analysis
