"""
Progressive Semantic Compression — Variant Generation

Generates compressed textual variants for the progressive compression experiment.

Pipeline:
  1. Read experiment configuration (config.yaml)
  2. Load dataset selection (dataset.jsonl)
  3. Load canonical texts and all requested translations
  4. For each (text, language, compression_level) combination:
     a. Format the compression prompt
     b. Call the LLM to generate the compressed variant
     c. Validate the generated record
  5. Write validated variants to data/variants/semantic_preservation/progressive_compression/
"""

import argparse
import json
import logging
from pathlib import Path

from geometry_of_meaning.data import (
    load_experiment_dataset,
    load_original_text,
    load_translation,
    save_variants,
    validate_variant_record,
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
    dry_run: bool = False,
) -> None:
    """
    Generate compressed variants for all combinations of text, language, and compression level.

    Args:
        experiment_dir: Path to the experiment directory (contains config.yaml, dataset.jsonl, prompts/)
        texts_dir: Path to data/texts/ (contains originals/ and translations/)
        variants_dir: Path to write generated variants
        dry_run: If True, log what would be generated without calling the LLM
    """
    config = load_config(experiment_dir / "config.yaml")
    dataset = load_experiment_dataset(experiment_dir / "dataset.jsonl")

    languages = config["languages"]
    compression_levels = config["compression"]["levels"]
    prompt_path = experiment_dir / config["compression"]["prompt_file"]

    with open(prompt_path, encoding="utf-8") as f:
        prompt_template = f.read()

    all_variants: dict[str, list[VariantRecord]] = {}

    for entry in dataset:
        if not entry.get("enabled", True):
            continue

        text_id = entry["text_id"]
        logger.info(f"Processing text: {text_id}")

        # Load original text
        original = load_original_text(texts_dir / "originals", text_id)
        source_language = original.original_language

        for language in languages:
            # Get the text to compress: original if same language, otherwise translation
            if language == source_language:
                source_text = original.text
            else:
                translation = load_translation(texts_dir / "translations", text_id, language)
                source_text = translation.text

            language_variants: list[VariantRecord] = []

            for level in compression_levels:
                variant_id = f"{text_id}_{language}_{int(level * 100):03d}"

                logger.info(f"  Generating: {variant_id}")

                if dry_run:
                    # In dry-run mode, store placeholder text
                    compressed_text = f"[DRY RUN] Compressed {text_id} ({language}) to {level:.0%}"
                else:
                    # Format the prompt and call the LLM
                    prompt = prompt_template.format(
                        target_fraction=f"{level:.0%}",
                        source_language=source_language,
                        target_language=language,
                        text=source_text,
                    )
                    compressed_text = _call_llm(prompt)

                variant = VariantRecord(
                    variant_id=variant_id,
                    text_id=text_id,
                    language=language,
                    source_language=source_language,
                    compression_level=level,
                    text=compressed_text,
                )

                validate_variant_record(variant)
                language_variants.append(variant)

            all_variants[language] = language_variants

    # Write variants to disk, organized by language
    save_variants(variants_dir, all_variants)
    logger.info(f"Generated variants written to {variants_dir}")


def _call_llm(prompt: str) -> str:
    """
    Call the LLM with the given prompt and return the generated text.

    This is a placeholder. Replace with actual API call (OpenAI, Anthropic, etc.)
    or integrate with a local model runner.

    Args:
        prompt: The formatted prompt to send to the LLM.

    Returns:
        The generated compressed text.
    """
    # TODO: Integrate with actual LLM API
    # Example with OpenAI:
    # import openai
    # response = openai.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[{"role": "user", "content": prompt}],
    #     temperature=0.3,
    # )
    # return response.choices[0].message.content

    raise NotImplementedError(
        "LLM integration not yet implemented. "
        "Use --dry-run to test the pipeline without calling an LLM."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate compressed variants for the progressive compression experiment."
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
        "--dry-run",
        action="store_true",
        help="Run without calling the LLM (uses placeholder text).",
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
    variants_dir = args.variants_dir or repo_root / "data" / "variants" / "semantic_preservation" / "progressive_compression"

    variants_dir.mkdir(parents=True, exist_ok=True)

    generate_variants(
        experiment_dir=args.experiment_dir,
        texts_dir=texts_dir,
        variants_dir=variants_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
