# Corpus Index — Geometry of Meaning

Complete canonical corpus: **12 texts** across **5 comparison languages** (en, it, zh, ja, da).

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
| 11 | `odyssey_opening_invocation` | The Odyssey — Opening Invocation | Homer | poetry | grc | 179 | en, it, zh, ja, da | unreviewed |
| 12 | `invictus` | Invictus | William Ernest Henley | poetry | en | 108 | en, it, zh, ja, da | unreviewed |

## Category Distribution

| Category | Count | Texts |
|----------|-------|-------|
| novel | 2 | pride_and_prejudice_opening, metamorphosis_opening |
| drama | 1 | hamlet_to_be_or_not_to_be |
| scientific | 2 | einstein_special_relativity, darwin_natural_selection |
| philosophy | 3 | wollstonecraft_womens_education, descartes_methodic_doubt, marcus_aurelius_control_and_judgment |
| autobiography | 1 | douglass_learning_to_read |
| political | 1 | declaration_human_equality |
| poetry | 2 | odyssey_opening_invocation, invictus |

## Language Distribution (Originals)

| Original Language | Count | Texts |
|-------------------|-------|-------|
| English (en) | 7 | pride_and_prejudice_opening, hamlet_to_be_or_not_to_be, darwin_natural_selection, wollstonecraft_womens_education, douglass_learning_to_read, declaration_human_equality, invictus |
| German (de) | 2 | einstein_special_relativity, metamorphosis_opening |
| French (fr) | 1 | descartes_methodic_doubt |
| Ancient Greek (grc) | 2 | marcus_aurelius_control_and_judgment, odyssey_opening_invocation |

## Translation Coverage

All 12 texts are available in all 5 comparison languages (en, it, zh, ja, da). Coverage: 60/60 files.

For texts originally in one of the 5 comparison languages, the canonical version in the source language is identical to the original. For texts in non-comparison languages (de, fr, grc), all 5 language directories contain model translations.

## Known Issues

- **marcus_aurelius_control_and_judgment**: Greek text (87 words) is shorter than the 150-word target. The passage is a complete meditation (Book II, section 1) that cannot be extended without crossing into a new topic. Greek text should be verified against the Loeb edition.
- **invictus**: The poem (108 words) is below the typical corpus length target of 150–300 words. It is kept complete because it forms an indivisible semantic unit and its syntactic/semantic density is exceptionally high.
- **descartes_methodic_doubt**: Uses modernized French orthography rather than 1637 original spelling.
- **odyssey_opening_invocation**: Greek text (Book 1, lines 1–32) ends mid-scene at Zeus beginning to speak. Verification against the printed Oxford Classical Text (Allen, 1917) is recommended. The end boundary was chosen because line 32 is a natural transition — Zeus has just recalled Aegisthus and is about to deliver his speech.
- All texts and translations added 2025-07-28 are marked `validation_status: unreviewed` pending human review.
This is **corpus version v0.1.0**, stored under `data/texts/v0.1.0/`. When the corpus changes (texts added, removed, or corrected), the version is bumped and a new directory is created under `data/texts/`. Within this version, originals use a folder-per-text layout (`<text_id>/source.txt` + `<text_id>/metadata.json`).
