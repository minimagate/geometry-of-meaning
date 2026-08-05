"""
Cross-Lingual Compression — Variant Generation Utilities

Provides helper functions for generating compressed text variants:
  - Loading source texts for a given language
  - Counting words
  - Saving variant records to JSONL
  - Preparing verbatim (100%) variants without LLM

The actual LLM-based compression is performed by subagents. This module
provides the shared utilities they need.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

from geometry_of_meaning.data import (
    VariantRecord,
    load_experiment_dataset,
    load_original_text,
    load_translation,
    save_variants,
)
from geometry_of_meaning.utils import (
    generate_variant_id,
    load_config,
    resolve_paths,
    setup_logging,
)
from geometry_of_meaning.word_count import count_words

logger = logging.getLogger(__name__)


def load_source_texts_for_language(
    experiment_dir: Path,
    texts_dir: Path,
    language: str,
) -> list[dict[str, Any]]:
    """
    Load all source texts for a given language.

    For texts whose original_language matches the requested language,
    loads the canonical original. Otherwise loads the translation.

    Args:
        experiment_dir: Path to the experiment directory.
        texts_dir: Path to data/texts/.
        language: ISO 639-1 language code.

    Returns:
        List of dicts with keys: text_id, language, source_language, text, category,
        original_language, word_count.
    """
    dataset = load_experiment_dataset(experiment_dir / "dataset.jsonl")
    enabled_ids = [e["text_id"] for e in dataset if e.get("enabled", True)]

    originals_dir = texts_dir / "originals"
    translations_dir = texts_dir / "translations"

    results: list[dict[str, Any]] = []
    for text_id in enabled_ids:
        original = load_original_text(originals_dir, text_id)

        if original.original_language == language:
            source_text = original.text
            source_language = language
        else:
            translation = load_translation(translations_dir, text_id, language)
            source_text = translation.text
            source_language = translation.source_language

        wc = count_words(source_text, language)
        results.append({
            "text_id": text_id,
            "language": language,
            "source_language": source_language,
            "category": original.category,
            "original_language": original.original_language,
            "text": source_text,
            "word_count": wc,
        })

    return results


def build_verbatim_variant(
    text_id: str,
    language: str,
    source_language: str,
    text: str,
    word_count: int,
) -> VariantRecord:
    """
    Build a verbatim (100%) variant record from a source text.

    No LLM needed — this is a direct copy.
    """
    compression_level = 1.0
    return VariantRecord(
        variant_id=generate_variant_id(text_id, language, compression_level),
        text_id=text_id,
        language=language,
        source_language=source_language,
        compression_level=compression_level,
        text=text,
        metadata={
            "generation_method": "verbatim_copy",
            "source_word_count": word_count,
            "variant_word_count": word_count,
        },
    )


def build_compressed_variant(
    text_id: str,
    language: str,
    source_language: str,
    compression_level: float,
    compressed_text: str,
    source_word_count: int,
    generator_model: str,
) -> VariantRecord:
    """
    Build a compressed variant record from LLM-generated text.

    Args:
        text_id: The parent text identifier.
        language: ISO 639-1 language code.
        source_language: Source language of the text.
        compression_level: Compression level (0.0 to 1.0).
        compressed_text: The LLM-generated compressed text.
        source_word_count: Word count of the source text.
        generator_model: Identifier of the LLM that generated this variant.

    Returns:
        A VariantRecord with metadata including word counts.
    """
    variant_wc = count_words(compressed_text, language)
    return VariantRecord(
        variant_id=generate_variant_id(text_id, language, compression_level),
        text_id=text_id,
        language=language,
        source_language=source_language,
        compression_level=compression_level,
        text=compressed_text.strip(),
        metadata={
            "generation_method": "llm_compression",
            "generator_model": generator_model,
            "source_word_count": source_word_count,
            "variant_word_count": variant_wc,
        },
    )


def save_language_variants(
    variants_dir: Path,
    language: str,
    records: list[VariantRecord],
) -> None:
    """
    Save variant records for a single language to the variants directory.
    """
    save_variants(variants_dir, {language: records})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare source texts and verbatim variants for cross-lingual compression."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to the experiment directory.",
    )
    parser.add_argument(
        "--texts-dir",
        type=Path,
        default=None,
        help="Path to data/texts/.",
    )
    parser.add_argument(
        "--variants-dir",
        type=Path,
        default=None,
        help="Path to the mirrored variants directory.",
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Generate variants for a single language only.",
    )
    parser.add_argument(
        "--verbatim-only",
        action="store_true",
        help="Only generate verbatim (100%) variants, skip LLM compression.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load texts and print summary without writing files.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    repo_root = resolve_paths(args.experiment_dir)
    texts_dir = args.texts_dir or repo_root / "data" / "texts"

    experiment_relative = args.experiment_dir.resolve().relative_to(repo_root)
    mirror_path = Path(*experiment_relative.parts[1:])
    variants_dir = args.variants_dir or repo_root / "data" / "variants" / mirror_path

    config = load_config(args.experiment_dir / "config.yaml")
    languages = [args.language] if args.language else config["languages"]

    all_verbatim: list[VariantRecord] = []

    for language in languages:
        texts = load_source_texts_for_language(args.experiment_dir, texts_dir, language)
        logger.info(
            f"Loaded {len(texts)} source texts for language '{language}'"
        )

        for t in texts:
            logger.info(
                f"  {t['text_id']}: {t['word_count']} words "
                f"(source_language={t['source_language']})"
            )

        verbatim = [
            build_verbatim_variant(
                t["text_id"], t["language"], t["source_language"],
                t["text"], t["word_count"],
            )
            for t in texts
        ]
        all_verbatim.extend(verbatim)

        if args.verbatim_only and not args.dry_run:
            save_language_variants(variants_dir, language, verbatim)
            logger.info(f"Saved {len(verbatim)} verbatim variants for '{language}'")

    if args.dry_run:
        logger.info(
            f"Dry run complete. Would generate {len(all_verbatim)} verbatim "
            f"variants across {len(languages)} language(s)."
        )
    elif not args.verbatim_only:
        logger.info(
            "Verbatim variants prepared. LLM compression must be performed "
            "by subagents for the remaining compression levels."
        )


if __name__ == "__main__":
    main()
