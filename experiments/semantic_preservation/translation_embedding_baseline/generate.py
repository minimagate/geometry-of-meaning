"""
Translation Embedding Baseline — Variant Generation

Copies canonical translations as "variants" at compression_level=1.0.
No LLM generation — this is a zero-manipulation baseline experiment.

Pipeline:
  1. Read experiment configuration (config.yaml)
  2. Load dataset selection (dataset.jsonl)
  3. Load all translations for the configured languages
  4. Save each translation as a VariantRecord at compression_level=1.0
  5. Write variants to data/variants/semantic_preservation/translation_embedding_baseline/
"""

import argparse
import logging
from pathlib import Path

from geometry_of_meaning.data import (
    load_experiment_dataset,
    load_original_text,
    load_translation,
    save_variants,
    VariantRecord,
)
from geometry_of_meaning.utils import (
    load_config,
    resolve_paths,
    setup_logging,
)

logger = logging.getLogger(__name__)


def generate_variants(
    experiment_dir: Path,
    texts_dir: Path,
    variants_dir: Path,
) -> None:
    """
    Load translations and save them as variants at compression_level=1.0.

    No textual manipulation is performed. The translations themselves are the
    variants, labeled at full compression (1.0).

    Args:
        experiment_dir: Path to the experiment directory.
        texts_dir: Path to data/texts/.
        variants_dir: Path to write generated variants.
    """
    config = load_config(experiment_dir / "config.yaml")
    dataset = load_experiment_dataset(experiment_dir / "dataset.jsonl")

    languages = config["languages"]
    all_variants: dict[str, list[VariantRecord]] = {}

    for entry in dataset:
        if not entry.get("enabled", True):
            continue

        text_id = entry["text_id"]
        logger.info(f"Processing text: {text_id}")

        # Load the original to get the source language
        original = load_original_text(texts_dir / "originals", text_id)
        source_language = original.original_language

        for language in languages:
            variant_id = f"{text_id}_{language}_100"

            if language == source_language:
                # Use the original text directly
                source_text = original.text
            else:
                translation = load_translation(texts_dir / "translations", text_id, language)
                source_text = translation.text

            variant = VariantRecord(
                variant_id=variant_id,
                text_id=text_id,
                language=language,
                source_language=source_language,
                compression_level=1.0,
                text=source_text,
                metadata={
                    "category": entry.get("category", "unknown"),
                    "original_language": source_language,
                },
            )

            all_variants.setdefault(language, []).append(variant)
            logger.info(f"  Created variant: {variant_id} ({language})")

    # Write variants to disk, organized by language
    save_variants(variants_dir, all_variants)
    logger.info(f"Generated {sum(len(v) for v in all_variants.values())} variants in {variants_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate translation-as-variants for the translation embedding baseline."
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
        help="Path to write variants. Defaults to mirrored data/variants/ path.",
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
    variants_dir = (
        args.variants_dir
        or repo_root / "data" / "variants" / "semantic_preservation" / "translation_embedding_baseline"
    )

    variants_dir.mkdir(parents=True, exist_ok=True)

    generate_variants(
        experiment_dir=args.experiment_dir,
        texts_dir=texts_dir,
        variants_dir=variants_dir,
    )


if __name__ == "__main__":
    main()