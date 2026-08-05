"""
Word counting utilities for the Geometry of Meaning project.

Provides language-aware word counting:
  - Latin-script languages (en, it, da, de, fr): split on whitespace
  - CJK languages (zh, ja): character-based counting with Latin token detection
"""

import re
from typing import Any

# Language codes that use character-based counting (no spaces between words)
_CJK_LANGUAGES = {"zh", "ja"}

# Pattern to match sequences of Latin alphanumeric characters (for CJK text)
_LATIN_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def count_words(text: str, language: str) -> int:
    """
    Count words in a text, using language-appropriate heuristics.

    For Latin-script languages, splits on whitespace and counts tokens.
    For CJK languages (zh, ja), counts each CJK character as a word unit
    and each contiguous Latin/alphanumeric sequence as a word.

    Args:
        text: The input text to count.
        language: ISO 639-1 language code.

    Returns:
        Number of words in the text.

    Raises:
        ValueError: If the language is not recognized.
    """
    text = text.strip()
    if not text:
        return 0

    if language in _CJK_LANGUAGES:
        return _count_cjk_words(text)

    return _count_latin_words(text)


def _count_latin_words(text: str) -> int:
    """Count words by whitespace splitting."""
    return len(text.split())


def _count_cjk_words(text: str) -> int:
    """
    Count words in CJK text.
    Each CJK character counts as one word unit.
    Contiguous Latin/alphanumeric sequences count as one word each.
    Punctuation and whitespace are not counted.
    """
    count = 0

    # Count Latin word sequences
    count += len(_LATIN_WORD_RE.findall(text))

    # Count CJK characters (Unicode ranges)
    for ch in text:
        code = ord(ch)
        if (
            0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
            or 0x3400 <= code <= 0x4DBF  # CJK Unified Ideographs Extension A
            or 0x3000 <= code <= 0x303F  # CJK Symbols and Punctuation (as words)
            or 0xF900 <= code <= 0xFAFF  # CJK Compatibility Ideographs
            or 0x3040 <= code <= 0x309F  # Hiragana
            or 0x30A0 <= code <= 0x30FF  # Katakana
            or 0xFF00 <= code <= 0xFFEF  # Halfwidth/Fullwidth (includes halfwidth katakana)
        ):
            count += 1

    return count


def count_words_for_texts(
    texts: list[str],
    language: str,
) -> list[int]:
    """
    Count words for a list of texts.

    Args:
        texts: List of text strings.
        language: ISO 639-1 language code.

    Returns:
        List of word counts matching the input order.
    """
    return [count_words(t, language) for t in texts]


def word_count_summary(texts: list[str], language: str) -> dict[str, Any]:
    """
    Compute summary statistics for a list of texts.

    Args:
        texts: List of text strings.
        language: ISO 639-1 language code.

    Returns:
        Dict with min, max, mean, total word counts.
    """
    counts = count_words_for_texts(texts, language)
    if not counts:
        return {"min": 0, "max": 0, "mean": 0.0, "total": 0}

    return {
        "min": min(counts),
        "max": max(counts),
        "mean": sum(counts) / len(counts),
        "total": sum(counts),
    }
