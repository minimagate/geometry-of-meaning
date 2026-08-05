# Cross-Lingual Compression

## Research question

How well does each language preserve meaning when text is progressively compressed? Do some languages retain semantic fidelity better than others at the same compression ratio?

## Motivation

When text is shortened — via summarization, keyword extraction, or aggressive editing — the meaning necessarily degrades. But the rate of semantic decay may depend on the language itself. Languages differ in syntactic flexibility, morphological density, and information-theoretic properties. A more synthetic language might pack more meaning per word, while a more analytic language might require more words to express the same proposition.

This experiment systematically measures meaning preservation under compression across five languages (en, it, zh, ja, da), using 12 canonical texts that have been translated into all five languages. By compressing each translation at identical ratios (100% → 50% → 25% → 12.5%) and measuring embedding displacement from the original, we can compare how each language handles information loss.

## Hypothesis

1. **H1**: All languages show monotonically decreasing semantic similarity as compression increases.
2. **H2**: Analytic languages (en, zh) lose meaning faster per compression step than synthetic/fusional languages (it, ja) because they rely more on word count to encode grammatical relations.
3. **H3**: The phase transition point — where meaning collapses disproportionately — varies by language.

## Independent variables

- **Language**: en, it, zh, ja, da
- **Compression level**: 1.00 (verbatim), 0.50 (half), 0.25 (summary), 0.125 (keywords)
- **Text**: 12 canonical texts across 7 categories
- **Original language**: en (8 texts), de (2), fr (1), grc (1)
- **Embedding model**: multilingual-e5-large

## Dependent variables

- **Cosine similarity** between uncompressed (1.00) and compressed variant, per text per language
- **Euclidean distance** between uncompressed and compressed variant
- **Step displacement**: embedding displacement between adjacent compression levels
- **Cumulative trajectory length**: total embedding path length from 1.00 to 0.125
- **Phase transition score**: ratio of max step displacement to mean step displacement

## Controls

- Identical compression ratios applied to all languages and texts
- Shared embedding model for all measurements
- Fixed random seed (42)
- Word counts measured programmatically, not eyeballed
- All translations derive from the same source meaning
- LLM compression uses the same prompt template across all languages

## Dataset

All 12 canonical texts from `data/texts/v0.1.0/originals/` and their translations in all 5 languages from `data/texts/v0.1.0/translations/`:

| text_id | original_language | category |
|---|---|---|
| pride_and_prejudice_opening | en | novel |
| hamlet_to_be_or_not_to_be | en | drama |
| darwin_natural_selection | en | scientific |
| declaration_human_equality | en | political |
| descartes_methodic_doubt | fr | philosophy |
| douglass_learning_to_read | en | autobiography |
| einstein_special_relativity | de | scientific |
| invictus | en | poetry |
| marcus_aurelius_control_and_judgment | grc | philosophy |
| metamorphosis_opening | de | novel |
| odyssey_opening_invocation | grc | poetry |
| wollstonecraft_womens_education | en | philosophy |

## Procedure

1. Load all 12 canonical texts and their translations in en, it, zh, ja, da
2. For each language, generate 4 compression variants per text using an LLM:
   - 1.00: verbatim copy (no LLM needed)
   - 0.50: reduce to ~50% word count (remove adverbs/adjectives first)
   - 0.25: very short summary (~25% word count)
   - 0.125: telegraphic keywords (~12.5% word count)
3. Count words programmatically for all source texts and variants
4. Validate variants with `validate.py` to ensure structural integrity
5. Embed all 240 variants (12 texts × 5 languages × 4 levels) using the configured model
6. For each text and language, measure the embedding trajectory across compression levels
7. Compute per-language statistics: mean cosine decay, displacement curves, phase transitions
8. Save results as a timestamped immutable run

## Metrics

| Metric | Definition | Range |
|---|---|---|
| Cosine similarity | cos(emb_100, emb_level) per text, language, level | [-1, 1] |
| Euclidean distance | ‖emb_100 - emb_level‖₂ | [0, ∞) |
| Step displacement | ‖emb_level_{i} - emb_level_{i+1}‖₂ | [0, ∞) |
| Cumulative trajectory length | sum of step displacements from 1.00 to 0.125 | [0, ∞) |
| Phase transition score | max step displacement / mean step displacement | [1, ∞) |

## Expected outputs

- `manifest.yaml`: run metadata with config snapshot and git commit
- `config.snapshot.yaml`: frozen configuration
- `dataset.snapshot.jsonl`: frozen dataset selection
- `embeddings.parquet`: embedding vectors for all 240 variants
- `metrics.parquet`: flat table with per-text per-language per-level metrics
- `summary.json`: aggregate per-language statistics and trajectory analyses
- `logs.txt`: execution log

## Limitations

- Only one embedding model (multilingual-e5-large); results may not generalize
- 12 texts is a small sample for per-category conclusions
- LLM compression is stochastic and depends on the generator model used
- The "verbatim" anchor may itself have cross-lingual embedding displacement (see `translation_embedding_baseline`)
- Compression heuristics (remove adverbs/adjectives first) assume Indo-European grammar; may not apply equally to Chinese or Japanese
- Word count is a proxy for information content that varies in accuracy across languages (especially Chinese vs. Italian)