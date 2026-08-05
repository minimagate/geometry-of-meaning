# Compression Generation Prompt

You are a text compression engine. Your task is to produce compressed variants
of a source text at four distinct compression levels. The language of the output
must match the language of the input. Do not translate.

## Input

- **Language**: {language}
- **Source text**:
```
{source_text}
```

## Output Format

You must output exactly four variants, each wrapped in XML-like tags as shown
below. Include nothing else in your response — no explanations, no commentary,
just the four tagged blocks.

```xml
<variant level="1.00">
[Verbatim copy of the source text — no changes whatsoever]
</variant>

<variant level="0.50">
[Approximately 50% of the source text word count.
Remove adverbs and adjectives first, then trim less essential clauses.
The core meaning and argument structure must survive.]
</variant>

<variant level="0.25">
[Very short summary — approximately 25% of the source text word count.
Preserve only the core proposition. Collapse sub-clauses.]
</variant>

<variant level="0.125">
[Telegraphic keywords — approximately 12.5% of the source text word count.
Reduce to key concepts, named entities, and essential predicates only.
Final output resembles a sequence of keywords.]
</variant>
```

## Compression Guidelines

### Level 1.00 (100% — verbatim)
Copy the source text exactly, including all whitespace, punctuation,
line breaks, and formatting. No changes of any kind.

### Level 0.50 (50% — half length)
- Target: approximately half the word count of the source.
- Priority of removal: (1) adverbs, (2) adjectives, (3) parentheticals
  and asides, (4) redundant clauses.
- The core proposition, main argument, key named entities, and
  structural backbone must survive intact.
- The output must still read as coherent text in {language}, not
  as notes or bullet points.

### Level 0.25 (25% — short summary)
- Target: approximately one quarter of the source word count.
- Collapse sub-clauses into their parent clause.
- Remove all elaboration beyond the core claim or narrative beat.
- Output should be a very short, coherent summary.

### Level 0.125 (12.5% — telegraphic keywords)
- Target: approximately one eighth of the source word count.
- Reduce to key concepts, named entities, and essential predicates.
- Output may consist of keywords, short phrases, or fragments.
- It does not need to read as prose, but the core meaning must
  be recoverable from the keywords.

## Important

- Do NOT translate the text. The output language must be {language}.
- Count your output words and ensure they are close to the target ratio
  of the source word count.
- For the 1.00 level, copy the text verbatim with zero changes.