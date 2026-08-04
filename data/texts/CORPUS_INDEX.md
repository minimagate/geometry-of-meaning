# Corpus Index — Geometry of Meaning

Complete canonical corpus: **10 texts** across **5 comparison languages** (en, it, zh, ja, da).

Last updated: 2025-07-28

---

## All Entries

| # | text_id | Title | Author | Category | Orig. Lang | ~Words | Source Langs | Validation |
|---|---------|-------|--------|----------|------------|--------|--------------|------------|
| 1 | `pride_and_prejudice_opening` | Pride and Prejudice — Opening | Jane Austen | novel | en | 153 | en, it, zh, ja, da | canonical |
| 2 | `hamlet_to_be_or_not_to_be` | Hamlet — "To be, or not to be" | William Shakespeare | drama | en | 260 | en, it, zh, ja, da | canonical |
| 3 | `einstein_special_relativity` | Electrodynamics of Moving Bodies | Albert Einstein | scientific | de | 175 | en, it, zh, ja, da | canonical |
| 4 | `metamorphosis_opening` | The Metamorphosis — Opening | Franz Kafka | novel | de | 182 | en, it, zh, ja, da | canonical |
| 5 | `darwin_natural_selection` | On the Origin of Species — Natural Selection | Charles Darwin | scientific | en | 175 | en, it, zh, ja, da | unreviewed |
| 6 | `wollstonecraft_womens_education` | A Vindication of the Rights of Woman | Mary Wollstonecraft | philosophy | en | 193 | en, it, zh, ja, da | unreviewed |
| 7 | `descartes_methodic_doubt` | Discourse on the Method — Methodic Doubt | René Descartes | philosophy | fr | 275 | en, it, zh, ja, da | unreviewed |
| 8 | `douglass_learning_to_read` | Narrative — Learning to Read | Frederick Douglass | autobiography | en | 249 | en, it, zh, ja, da | unreviewed |
| 9 | `marcus_aurelius_control_and_judgment` | Meditations — Judgment and Cooperation | Marcus Aurelius | philosophy | grc | 87 | en, it, zh, ja, da | unreviewed |
| 10 | `declaration_human_equality` | Declaration of Independence — Equality | Jefferson et al. | political | en | 207 | en, it, zh, ja, da | unreviewed |

## Category Distribution

| Category | Count | Texts |
|----------|-------|-------|
| novel | 2 | pride_and_prejudice_opening, metamorphosis_opening |
| drama | 1 | hamlet_to_be_or_not_to_be |
| scientific | 2 | einstein_special_relativity, darwin_natural_selection |
| philosophy | 3 | wollstonecraft_womens_education, descartes_methodic_doubt, marcus_aurelius_control_and_judgment |
| autobiography | 1 | douglass_learning_to_read |
| political | 1 | declaration_human_equality |

## Language Distribution (Originals)

| Original Language | Count | Texts |
|-------------------|-------|-------|
| English (en) | 6 | pride_and_prejudice_opening, hamlet_to_be_or_not_to_be, darwin_natural_selection, wollstonecraft_womens_education, douglass_learning_to_read, declaration_human_equality |
| German (de) | 2 | einstein_special_relativity, metamorphosis_opening |
| French (fr) | 1 | descartes_methodic_doubt |
| Ancient Greek (grc) | 1 | marcus_aurelius_control_and_judgment |

## Translation Coverage

All 10 texts are available in all 5 comparison languages (en, it, zh, ja, da). Coverage: 50/50 files.

For texts originally in one of the 5 comparison languages, the canonical version in the source language is identical to the original. For texts in non-comparison languages (de, fr, grc), all 5 language directories contain model translations.

## Known Issues

- **marcus_aurelius_control_and_judgment**: Greek text (87 words) is shorter than the 150-word target. The passage is a complete meditation (Book II, section 1) that cannot be extended without crossing into a new topic. Greek text should be verified against the Loeb edition.
- **descartes_methodic_doubt**: Uses modernized French orthography rather than 1637 original spelling.
- All 6 new texts and their 30 translations are marked `validation_status: unreviewed` pending human review.
- **Structure**: As of 2025-07-28, originals use a folder-per-text layout (`<text_id>/source.txt` + `<text_id>/metadata.json`).