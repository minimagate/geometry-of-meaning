# Semantic Preservation Evaluation Prompt

You are evaluating how well a compressed text preserves the meaning of the original, across a series of semantic dimensions.

## Task

Given an original text and its compressed version, rate the preservation of meaning on each dimension.

## Original Text

{original_text}

## Compressed Text

{compressed_text}

## Evaluation Dimensions

Rate each dimension from 1 (completely lost) to 5 (perfectly preserved):

1. **Factual content**: Are the key facts, claims, and assertions preserved?
2. **Logical structure**: Are causal, conditional, and logical relationships intact?
3. **Entities and references**: Are the main entities (people, objects, concepts) still present and correctly referenced?
4. **Temporal structure**: Is the sequence of events or temporal relationships preserved?
5. **Tone and stance**: Is the author's attitude, tone, or evaluative stance preserved?
6. **Core message**: Does the compressed text communicate the same central message?

## Response Format

Respond in JSON format:
```json
{
  "factual_content": 1-5,
  "logical_structure": 1-5,
  "entities_and_references": 1-5,
  "temporal_structure": 1-5,
  "tone_and_stance": 1-5,
  "core_message": 1-5,
  "overall_preservation": 0.0-1.0,
  "comments": "brief overall assessment"
}
```
