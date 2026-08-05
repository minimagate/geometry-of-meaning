"""
Cross-Lingual Compression — Variant Validation

Validates that subagent-generated variants under data/variants/ conform to
the experiment's schema, naming conventions, and dataset references.

Validates:
  - All (text_id, language, compression_level) combinations are covered
  - variant_ids follow the naming convention
  - text_ids reference existing canonical texts
  - language codes are in the experiment config
  - texts are non-empty
  - word counts are present in metadata
"""

import argparse
import logging
import sys
from pathlib import Path

from geometry_of_meaning.data import (
    load_experiment_dataset,
    load_original_text,
    load_variants,
)
from geometry_of_meaning.utils import generate_variant_id, load_config, resolve_paths, setup_logging
from geometry_of_meaning.word_count import count_words

logger = logging.getLogger(__name__)


def validate_variants(
    experiment_dir: Path,
    texts_dir: Path,
    variants_dir: Path,
) -> tuple[list[str], list[str]]:
    """
    Validate all variant files under variants_dir.

    Returns:
        A tuple of (errors, warnings).
    """
    config = load_config(experiment_dir / "config.yaml")
    dataset = load_experiment_dataset(experiment_dir / "dataset.jsonl")

    errors: list[str] = []
    warnings: list[str] = []

    enabled_ids = {e["text_id"] for e in dataset if e.get("enabled", True)}
    configured_languages = config["languages"]
    compression_levels = config["compression"]["levels"]

    if not variants_dir.exists():
        errors.append(f"Variants directory does not exist: {variants_dir}")
        return errors, warnings

    existing_variants = load_variants(variants_dir)

    for language in configured_languages:
        if language not in existing_variants:
            errors.append(f"Missing language directory: '{language}' under {variants_dir}")

    variant_ids_seen: set[str] = set()
    lang_coverage: dict[str, dict[str, set[float]]] = {}

    for language, records in existing_variants.items():
        if language not in configured_languages:
            warnings.append(
                f"Language '{language}' found in variants but not in experiment config"
            )
            continue

        lang_coverage.setdefault(language, {})
        for record in records:
            variant_ids_seen.add(record.variant_id)

            lang_coverage[language].setdefault(record.text_id, set())
            lang_coverage[language][record.text_id].add(record.compression_level)

            expected_id = generate_variant_id(
                record.text_id, record.language, record.compression_level,
            )
            if record.variant_id != expected_id:
                errors.append(
                    f"Variant ID mismatch: got '{record.variant_id}', "
                    f"expected '{expected_id}'"
                )

            if record.language not in configured_languages:
                errors.append(
                    f"Variant '{record.variant_id}' uses unconfigured language: "
                    f"'{record.language}'"
                )

            if record.text_id not in enabled_ids:
                warnings.append(
                    f"Variant '{record.variant_id}' references text_id not in "
                    f"dataset: '{record.text_id}'"
                )

            try:
                load_original_text(texts_dir / "originals", record.text_id)
            except FileNotFoundError:
                errors.append(
                    f"Variant '{record.variant_id}' references non-existent "
                    f"text_id: '{record.text_id}'"
                )
            except Exception as e:
                errors.append(
                    f"Variant '{record.variant_id}' text_id lookup failed: {e}"
                )

            if not record.text.strip():
                errors.append(f"Variant '{record.variant_id}' has empty text")

            expected_wc = count_words(record.text, record.language)
            metadata_wc = record.metadata.get("variant_word_count")
            if metadata_wc is not None and metadata_wc != expected_wc:
                errors.append(
                    f"Variant '{record.variant_id}': metadata word_count "
                    f"{metadata_wc} != computed word_count {expected_wc}"
                )

    for text_id in enabled_ids:
        for language in configured_languages:
            for level in compression_levels:
                if (
                    language not in lang_coverage
                    or text_id not in lang_coverage[language]
                    or level not in lang_coverage[language][text_id]
                ):
                    errors.append(
                        f"Missing variant: text_id='{text_id}', "
                        f"language='{language}', compression_level={level}"
                    )

    total_variants = sum(len(v) for v in existing_variants.values())
    duplicate_count = total_variants - len(variant_ids_seen)
    if duplicate_count > 0:
        errors.append(f"Found {duplicate_count} duplicate variant_id(s)")

    expected_total = len(enabled_ids) * len(configured_languages) * len(compression_levels)
    logger.info(
        f"Validated {total_variants}/{expected_total} variants across "
        f"{len(existing_variants)} languages at {len(compression_levels)} "
        f"compression levels"
    )

    return errors, warnings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate subagent-generated variants for cross-lingual compression."
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
        help="Path to data/texts/. Defaults to repo root / data/texts.",
    )
    parser.add_argument(
        "--variants-dir",
        type=Path,
        default=None,
        help="Path to the variants directory. Defaults to mirrored data/variants/ path.",
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

    errors, warnings = validate_variants(args.experiment_dir, texts_dir, variants_dir)

    if warnings:
        for w in warnings:
            logger.warning(w)

    if errors:
        logger.error(f"Validation FAILED — {len(errors)} error(s):")
        for e in errors:
            logger.error(f"  {e}")
        sys.exit(1)

    logger.info("Validation PASSED — all variants conform to the experiment schema.")


if __name__ == "__main__":
    main()
