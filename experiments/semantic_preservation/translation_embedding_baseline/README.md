# Translation Embedding Baseline

## Research question

How much does translation alone shift the semantic embedding of a text? Before we can study compression, negation, or any other transformation, we need to measure the baseline effect of expressing the same meaning in different languages.

## Motivation

Every downstream experiment in this project compares embeddings across languages. Without a baseline measurement of translation-induced displacement, we cannot distinguish between:
- A signal caused by our experimental manipulation (e.g., compression)
- The background "noise" of expressing the same content in different languages

This experiment establishes that baseline. It embeds all 12 canonical texts in all 5 languages with zero textual manipulation, then measures pairwise distances.

## Hypothesis

...

## Independent variables

- **Language**: en, it, zh, ja, da
- **Text**: 12 canonical texts across 7 categories (novel, drama, poetry, scientific, philosophy, political, autobiography)
- **Original language**: en (8 texts), de (2), fr (1), grc (1)
- **Embedding model**: multilingual-e5-large

## Dependent variables

- **Cosine similarity** between English version and each translation, per text
- **Euclidean distance** between English version and each translation, per text
- **Pairwise similarity matrix**: 5×5 cosine similarity across all language pairs, per text
- **Cross-language centroid distances**: mean embedding per language (across all texts), then pairwise distances between language centroids
- **Intra-text vs inter-text similarity**: similarity distributions for same-text cross-language vs different-text same-language comparisons

## Controls

- Identical embedding model and configuration for all texts
- No textual manipulation — all texts are canonical translations
- Fixed random seed for reproducibility
- All translations derive from the same source meaning

## Dataset

All 12 canonical texts from `data/texts/originals/` and their translations in all 5 languages from `data/texts/translations/`:

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
2. Save translations as "variants" at compression_level=1.0 (no manipulation)
3. Validate variants with `validate.py` to ensure structural integrity
4. Embed all 60 texts (12 × 5 languages) using the configured embedding model
5. For each text, compute the 5×5 pairwise cosine similarity matrix across languages
6. For each text, measure cosine similarity and Euclidean distance from English to each translation
7. Compute aggregate statistics: per-language similarity distributions, centroid distances
8. Save results as a timestamped immutable run

## Metrics

| Metric | Definition | Range |
|---|---|---|
| Cosine similarity | cos(emb_en, emb_lang) per text per language | [-1, 1] |
| Euclidean distance | ‖emb_en - emb_lang‖₂ per text per language | [0, ∞) |
| Pairwise similarity | 5×5 matrix of all language-pair cosine similarities per text | [-1, 1] |
| Centroid distance | 1 - cos(mean_en, mean_lang) across all texts | [0, 2] |
| Intra/inter-text similarity ratio | mean(same-text, cross-lang sim) / mean(diff-text, same-lang sim) | [0, ∞) |

## Expected outputs

- `embeddings.parquet`: all 60 embedding vectors with text_id, language, and metadata
- `metrics.parquet`: flat table with per-text per-language similarity/distance measurements
- `pairwise_matrices.json`: 5×5 similarity matrices for each text
- `summary.json`: aggregate statistics and centroids
- `manifest.yaml`: run metadata with config snapshot and git commit

## Limitations

- Only one embedding model (multilingual-e5-large); results may not generalize to other models
- 12 texts is a small sample; per-category conclusions are tentative
- Translation quality varies — some translations are model-generated with human review, others are machine-only
- Embedding models have known cross-lingual biases (e.g., high-resource languages may embed more faithfully)
- The English "reference" is sometimes the original (for en-original texts) and sometimes a translation (for non-en originals), introducing asymmetry
