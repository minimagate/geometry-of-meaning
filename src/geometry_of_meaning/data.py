"""
Data management for the Geometry of Meaning project.

Handles loading, validation, and storage of:
  - Canonical original texts (data/texts/originals/)
  - Canonical translations (data/texts/translations/)
  - Experiment datasets (experiments/.../dataset.jsonl)
  - Generated variants (data/variants/.../)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Valid ISO 639-1 language codes used in this project
VALID_LANGUAGES = {"en", "it", "zh", "ja", "da"}
VALID_ORIGINAL_LANGUAGES = {"en", "it", "zh", "ja", "da", "de", "fr", "grc"}
VALID_SOURCE_LANGUAGES = {"en", "it", "zh", "ja", "da", "de", "fr", "grc"}


@dataclass
class OriginalText:
    """A canonical source text in its original language."""

    text_id: str
    title: str
    author: str
    category: str
    original_language: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text_id:
            raise ValueError("text_id must not be empty")
        if self.original_language not in VALID_ORIGINAL_LANGUAGES:
            raise ValueError(
                f"Invalid language '{self.original_language}'. "
                f"Must be one of: {sorted(VALID_ORIGINAL_LANGUAGES)}"
            )
        if not self.text:
            raise ValueError("text must not be empty")


@dataclass
class Translation:
    """A canonical translation of an original text."""

    text_id: str
    language: str
    source_language: str
    text: str
    translation_method: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text_id:
            raise ValueError("text_id must not be empty")
        if self.language not in VALID_LANGUAGES:
            raise ValueError(
                f"Invalid language '{self.language}'. "
                f"Must be one of: {sorted(VALID_LANGUAGES)}"
            )
        if self.source_language not in VALID_SOURCE_LANGUAGES:
            raise ValueError(
                f"Invalid source_language '{self.source_language}'. "
                f"Must be one of: {sorted(VALID_SOURCE_LANGUAGES)}"
            )
        if not self.text:
            raise ValueError("text must not be empty")


@dataclass
class VariantRecord:
    """A generated textual variant produced by an experiment."""

    variant_id: str
    text_id: str
    language: str
    source_language: str
    compression_level: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.variant_id:
            raise ValueError("variant_id must not be empty")
        if not self.text_id:
            raise ValueError("text_id must not be empty")
        if self.language not in VALID_LANGUAGES:
            raise ValueError(
                f"Invalid language '{self.language}'. "
                f"Must be one of: {sorted(VALID_LANGUAGES)}"
            )
        if not (0.0 <= self.compression_level <= 1.0):
            raise ValueError(
                f"compression_level must be between 0.0 and 1.0, got {self.compression_level}"
            )
        if not self.text:
            raise ValueError("text must not be empty")


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------


def load_original_text(originals_dir: Path, text_id: str) -> OriginalText:
    """
    Load a canonical original text by its text_id.

    Args:
        originals_dir: Path to data/texts/originals/
        text_id: The text identifier (matches folder name, e.g., 'pride_and_prejudice_opening')

    Returns:
        An OriginalText dataclass instance.

    Raises:
        FileNotFoundError: If the text folder or its source.txt / metadata.json does not exist.
        ValueError: If the file contains invalid data.
    """
    text_dir = originals_dir / text_id
    source_path = text_dir / "source.txt"
    metadata_path = text_dir / "metadata.json"

    if not source_path.exists():
        raise FileNotFoundError(f"Original text file not found: {source_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Original metadata file not found: {metadata_path}")

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    with open(source_path, encoding="utf-8") as f:
        text = f.read()

    return OriginalText(
        text_id=metadata["text_id"],
        title=metadata.get("title", ""),
        author=metadata.get("author", "unknown"),
        category=metadata.get("category", "unknown"),
        original_language=metadata["original_language"],
        text=text,
        metadata={k: v for k, v in metadata.items() if k not in OriginalText.__dataclass_fields__},
    )


def load_translation(translations_dir: Path, text_id: str, language: str) -> Translation:
    """
    Load a canonical translation by text_id and language.

    Args:
        translations_dir: Path to data/texts/translations/
        text_id: The text identifier.
        language: The ISO 639-1 language code of the translation.

    Returns:
        A Translation dataclass instance.

    Raises:
        FileNotFoundError: If the translation file does not exist.
    """
    filepath = translations_dir / language / f"{text_id}.json"
    if not filepath.exists():
        raise FileNotFoundError(f"Translation not found: {filepath}")

    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    return Translation(
        text_id=data["text_id"],
        language=data["language"],
        source_language=data.get("source_language", "unknown"),
        text=data["text"],
        translation_method=data.get("translation_method", "unknown"),
        metadata={k: v for k, v in data.items() if k not in Translation.__dataclass_fields__},
    )


def load_experiment_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    """
    Load an experiment's dataset.jsonl file.

    Args:
        dataset_path: Path to the dataset.jsonl file.

    Returns:
        A list of dataset entry dictionaries.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    entries: list[dict[str, Any]] = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_variants(variants_dir: Path) -> dict[str, list[VariantRecord]]:
    """
    Load all variant records from a variants directory.

    Variants are organized by language subdirectories, each containing JSONL files.

    Args:
        variants_dir: Path to the experiment's variants directory.

    Returns:
        A dict mapping language code to list of VariantRecord instances.
    """
    all_variants: dict[str, list[VariantRecord]] = {}

    if not variants_dir.exists():
        logger.warning(f"Variants directory does not exist: {variants_dir}")
        return all_variants

    for lang_dir in sorted(variants_dir.iterdir()):
        if not lang_dir.is_dir():
            continue

        language = lang_dir.name
        if language not in VALID_LANGUAGES:
            logger.warning(f"Skipping unrecognized language directory: {lang_dir}")
            continue

        lang_variants: list[VariantRecord] = []
        for jsonl_file in sorted(lang_dir.glob("*.jsonl")):
            with open(jsonl_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    variant = VariantRecord(
                        variant_id=data["variant_id"],
                        text_id=data["text_id"],
                        language=data["language"],
                        source_language=data.get("source_language", "unknown"),
                        compression_level=data["compression_level"],
                        text=data["text"],
                        metadata={k: v for k, v in data.items()
                                  if k not in VariantRecord.__dataclass_fields__},
                    )
                    lang_variants.append(variant)

        if lang_variants:
            all_variants[language] = lang_variants

    return all_variants


# ---------------------------------------------------------------------------
# Saving functions
# ---------------------------------------------------------------------------


def save_variants(
    variants_dir: Path,
    variants: dict[str, list[VariantRecord]],
) -> None:
    """
    Save variant records to disk, organized by language.

    Each language gets a subdirectory, and each text_id gets a JSONL file.

    Args:
        variants_dir: Path to the experiment's variants directory.
        variants: Dict mapping language to list of VariantRecord instances.
    """
    variants_dir.mkdir(parents=True, exist_ok=True)

    for language, records in variants.items():
        lang_dir = variants_dir / language
        lang_dir.mkdir(parents=True, exist_ok=True)

        # Group records by text_id
        by_text: dict[str, list[VariantRecord]] = {}
        for record in records:
            by_text.setdefault(record.text_id, []).append(record)

        for text_id, text_records in by_text.items():
            filepath = lang_dir / f"{text_id}.jsonl"
            with open(filepath, "w", encoding="utf-8") as f:
                for record in text_records:
                    entry = {
                        "variant_id": record.variant_id,
                        "text_id": record.text_id,
                        "language": record.language,
                        "source_language": record.source_language,
                        "compression_level": record.compression_level,
                        "text": record.text,
                        **record.metadata,
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Saved {sum(len(v) for v in variants.values())} variants to {variants_dir}")


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------


def validate_text_record(data: dict[str, Any]) -> list[str]:
    """
    Validate a raw original text record dictionary.

    Returns a list of validation error messages (empty if valid).
    """
    errors: list[str] = []

    if "text_id" not in data or not data["text_id"]:
        errors.append("Missing required field: text_id")
    if "original_language" not in data:
        errors.append("Missing required field: original_language")
    elif data["original_language"] not in VALID_ORIGINAL_LANGUAGES:
        errors.append(f"Invalid original_language: {data['original_language']}")
    if "text" not in data or not data["text"]:
        errors.append("Missing required field: text")

    return errors


def validate_variant_record(variant: VariantRecord) -> None:
    """
    Validate a VariantRecord, raising ValueError on invalid data.

    This is in addition to the __post_init__ checks on the dataclass itself.
    """
    errors: list[str] = []

    if not variant.variant_id:
        errors.append("variant_id must not be empty")
    if not variant.text_id:
        errors.append("text_id must not be empty")
    if variant.language not in VALID_LANGUAGES:
        errors.append(f"Invalid language: {variant.language}")
    if not (0.0 <= variant.compression_level <= 1.0):
        errors.append(f"Invalid compression_level: {variant.compression_level}")
    if not variant.text.strip():
        errors.append("text must not be empty or whitespace-only")

    if errors:
        raise ValueError(f"Invalid variant {variant.variant_id}: {'; '.join(errors)}")
