# Validation Prompt

You are evaluating whether a compressed text faithfully preserves the meaning of the original.

## Task

Given an original text and its compressed version, answer the following:

1. Is the compressed text a valid compression of the original? (yes/no)
2. What fraction of the original's essential meaning is preserved? (0.0 to 1.0)
3. Is any new information introduced that was not in the original? (yes/no)
4. Are there any factual errors in the compressed version? (yes/no)
5. Brief explanation of any issues found.

## Original Text

{original_text}

## Compressed Text

{compressed_text}

## Evaluation

Respond in JSON format:
```json
{
  "valid_compression": true/false,
  "meaning_preservation": 0.0-1.0,
  "new_information": true/false,
  "factual_errors": true/false,
  "issues": "brief explanation or null"
}
```
