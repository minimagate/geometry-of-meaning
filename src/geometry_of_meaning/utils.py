"""
Shared utilities for the Geometry of Meaning project.

Small helper functions used across experiments and library code:
  - Configuration loading
  - Path resolution
  - ID generation
  - Random seed control
  - Logging setup
"""

import hashlib
import logging
import os
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to the config.yaml file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Configuration file is empty: {config_path}")

    return config


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_paths(experiment_dir: Path) -> Path:
    """
    Resolve the repository root from an experiment directory.

    Assumes the standard layout: <repo>/experiments/<area>/<experiment>/

    Args:
        experiment_dir: Path to any experiment directory.

    Returns:
        Path to the repository root.
    """
    # Walk up from experiment_dir to find repo root
    # (repo root has pyproject.toml or README.md)
    current = experiment_dir.resolve()
    for _ in range(10):  # Safety limit
        if (current / "pyproject.toml").exists():
            return current
        if (current / "README.md").exists() and (current / ".git").exists():
            return current
        if current.parent == current:
            break
        current = current.parent

    # Fallback: assume standard structure (three levels up from experiment dir)
    raise FileNotFoundError(
        f"Could not find repository root from {experiment_dir}. "
        f"Ensure pyproject.toml exists at the repository root."
    )


# ---------------------------------------------------------------------------
# ID and hash generation
# ---------------------------------------------------------------------------


def generate_variant_id(text_id: str, language: str, compression_level: float) -> str:
    """
    Generate a deterministic variant ID.

    Args:
        text_id: The parent text identifier.
        language: ISO 639-1 language code.
        compression_level: Compression level (0.0 to 1.0).

    Returns:
        Variant ID string, e.g., 'pride_and_prejudice_opening_it_075'.
    """
    level_int = int(compression_level * 100)
    return f"{text_id}_{language}_{level_int:03d}"


def deterministic_hash(text: str, length: int = 8) -> str:
    """
    Compute a deterministic SHA256 hash of a string.

    Args:
        text: Input string.
        length: Number of hex characters to return.

    Returns:
        Hex digest truncated to the specified length.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def hash_file(filepath: Path) -> str:
    """
    Compute SHA256 hash of a file's contents.

    Args:
        filepath: Path to the file.

    Returns:
        Full SHA256 hex digest.
    """
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


def timestamp_now() -> str:
    """
    Get current UTC timestamp in ISO 8601 format, suitable for run directory names.

    Returns:
        Timestamp string, e.g., '2026-08-04T135700'.
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H%M%S")


def iso_timestamp() -> str:
    """
    Get current UTC timestamp in full ISO 8601 format.

    Returns:
        Timestamp string, e.g., '2026-08-04T13:57:00+00:00'.
    """
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Random seed control
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """
    Set random seed for Python, NumPy, and (if available) PyTorch.

    Args:
        seed: The random seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(
    verbose: bool = False,
    log_file: Path | None = None,
) -> None:
    """
    Configure logging for experiment scripts.

    Args:
        verbose: If True, set log level to DEBUG. Otherwise INFO.
        log_file: Optional path to a log file for persistent logs.
    """
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------


def progress_bar(iterable, desc: str = "", total: int | None = None):
    """
    Thin wrapper around tqdm for progress bars.

    Args:
        iterable: The iterable to track.
        desc: Description prefix.
        total: Total number of items (optional, auto-detected if possible).

    Returns:
        A tqdm-wrapped iterable, or the original iterable if tqdm is unavailable.
    """
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc, total=total)
    except ImportError:
        return iterable
