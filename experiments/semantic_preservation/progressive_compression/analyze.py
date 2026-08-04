"""
Progressive Semantic Compression — Numerical Analysis

Embeds compressed variants, computes semantic metrics, and saves an immutable run.

Pipeline:
  1. Load canonical texts and compressed variants
  2. Embed all texts using the configured embedding model(s)
  3. Compute cosine similarities, step displacements, and trajectory lengths
  4. Compare across languages and embedding models
  5. Save results as a new immutable timestamped run
"""

import argparse
import hashlib
import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from geometry_of_meaning.data import (
    load_experiment_dataset,
    load_original_text,
    load_translation,
    load_variants,
)
from geometry_of_meaning.embeddings import embed_texts, get_model_info
from geometry_of_meaning.metrics import (
    cosine_similarity,
    cumulative_trajectory_length,
    step_displacements,
)
from geometry_of_meaning.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def run_analysis(
    experiment_dir: Path,
    texts_dir: Path,
    variants_dir: Path,
    runs_dir: Path,
) -> Path:
    """
    Run the full numerical analysis and save results as an immutable run.

    Args:
        experiment_dir: Path to experiment directory.
        texts_dir: Path to data/texts/.
        variants_dir: Path to generated variants.
        runs_dir: Path to runs output directory.

    Returns:
        Path to the created run directory.
    """
    config = load_config(experiment_dir / "config.yaml")
    dataset = load_experiment_dataset(experiment_dir / "dataset.jsonl")

    # Create timestamped run directory
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    logger.info(f"Starting run: {run_id}")
    logger.info(f"Run directory: {run_dir}")

    # Get git commit
    git_commit = _get_git_commit()

    # Get prompt hashes
    prompt_hashes = _hash_prompts(experiment_dir / "prompts")

    # Load canonical texts
    canonical_texts: dict[str, str] = {}
    for entry in dataset:
        if not entry.get("enabled", True):
            continue
        text_id = entry["text_id"]
        original = load_original_text(texts_dir / "originals", text_id)
        canonical_texts[text_id] = original.text

    # Load variants
    all_variants = load_variants(variants_dir)

    # Prepare results table
    records: list[dict] = []

    for model_id in config["embedding_models"]:
        logger.info(f"Embedding with model: {model_id}")
        model_info = get_model_info(model_id)

        # Collect all texts to embed: canonicals + translations + variants
        texts_to_embed: list[tuple[str, str, str, str, float | None]] = []
        # (text_id, language, source_language, text, compression_level)

        for text_id, text in canonical_texts.items():
            original = load_original_text(texts_dir / "originals", text_id)
            texts_to_embed.append((text_id, original.original_language, original.original_language, text, 1.0))

        for text_id in canonical_texts:
            for language in config["languages"]:
                original = load_original_text(texts_dir / "originals", text_id)
                if language != original.original_language:
                    translation = load_translation(texts_dir / "translations", text_id, language)
                    texts_to_embed.append((text_id, language, original.original_language, translation.text, 1.0))

        for language, variants in all_variants.items():
            for variant in variants:
                texts_to_embed.append((
                    variant.text_id, variant.language, variant.source_language,
                    variant.text, variant.compression_level,
                ))

        # Batch embed
        texts_only = [t[3] for t in texts_to_embed]
        embeddings = embed_texts(texts_only, model_id=model_id)

        # Build lookup: (text_id, language, compression_level) -> embedding
        embedding_lookup: dict[tuple[str, str, float], np.ndarray] = {}
        for i, (text_id, language, source_language, _, level) in enumerate(texts_to_embed):
            key = (text_id, language, level)
            embedding_lookup[key] = embeddings[i]

        # Compute metrics for each variant
        for language in config["languages"]:
            original = load_original_text(texts_dir / "originals", list(canonical_texts.keys())[0])
            canonical_lang = original.original_language

            for entry in dataset:
                if not entry.get("enabled", True):
                    continue
                text_id = entry["text_id"]

                # Get canonical embedding in the source language
                canonical_key = (text_id, canonical_lang, 1.0)
                if canonical_key not in embedding_lookup:
                    continue
                canonical_emb = embedding_lookup[canonical_key]

                # Get variant embeddings at each compression level
                variant_embs: list[tuple[float, np.ndarray]] = []
                for level in sorted(config["compression"]["levels"]):
                    variant_key = (text_id, language, level)
                    if variant_key not in embedding_lookup:
                        continue
                    variant_emb = embedding_lookup[variant_key]
                    variant_embs.append((level, variant_emb))

                if not variant_embs:
                    continue

                # Sort by compression level descending (1.0 → 0.1)
                variant_embs.sort(key=lambda x: x[0], reverse=True)

                # Compute cosine similarities
                for level, emb in variant_embs:
                    sim = cosine_similarity(canonical_emb, emb)
                    records.append({
                        "text_id": text_id,
                        "language": language,
                        "model": model_id,
                        "model_version": model_info.get("version", "unknown"),
                        "compression_level": level,
                        "cosine_similarity": sim,
                        "category": entry.get("category", "unknown"),
                    })

                # Compute step displacements and cumulative trajectory
                embs_only = [e for _, e in variant_embs]
                levels_only = [l for l, _ in variant_embs]
                displacements = step_displacements(embs_only)
                cum_trajectory = cumulative_trajectory_length(embs_only)

                for i, level in enumerate(levels_only):
                    # Update existing records with step displacement
                    for r in records:
                        if r["text_id"] == text_id and r["language"] == language and r["model"] == model_id and r["compression_level"] == level:
                            r["step_displacement"] = displacements[i] if i < len(displacements) else 0.0
                            r["cumulative_trajectory_length"] = cum_trajectory[i] if i < len(cum_trajectory) else 0.0
                            break

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(records)

    # Compute summary statistics
    summary = _compute_summary(metrics_df, config)

    # --- Save run artifacts ---

    # Write manifest
    manifest = {
        "run_id": run_id,
        "experiment": config["experiment"]["id"],
        "research_area": config["experiment"]["research_area"],
        "languages": config["languages"],
        "embedding_models": config["embedding_models"],
        "git_commit": git_commit,
        "random_seed": config["random_seed"],
        "prompt_hashes": prompt_hashes,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(run_dir / "manifest.yaml", "w", encoding="utf-8") as f:
        import yaml
        yaml.dump(manifest, f, allow_unicode=True, default_flow_style=False)

    # Write config snapshot
    with open(run_dir / "config.snapshot.yaml", "w", encoding="utf-8") as f:
        import yaml
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    # Write dataset snapshot
    with open(run_dir / "dataset.snapshot.jsonl", "w", encoding="utf-8") as f:
        for entry in dataset:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    # Write metrics
    metrics_df.to_parquet(run_dir / "metrics.parquet", index=False)

    # Write summary
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Write placeholder for embeddings (actual embeddings may be too large; consider external storage)
    pd.DataFrame({"note": ["Embedding storage not yet implemented. See metrics.parquet for results."]}).to_parquet(
        run_dir / "embeddings.parquet", index=False
    )

    logger.info(f"Run complete. Results saved to {run_dir}")
    return run_dir


def _compute_summary(metrics_df: pd.DataFrame, config: dict) -> dict:
    """Compute aggregate summary statistics from the metrics DataFrame."""
    summary: dict = {
        "description": "Aggregate summary of progressive compression experiment",
        "num_texts": metrics_df["text_id"].nunique(),
        "num_languages": metrics_df["language"].nunique(),
        "num_models": metrics_df["model"].nunique(),
        "num_variants": len(metrics_df),
    }

    # Summary by compression level
    by_level = metrics_df.groupby("compression_level")["cosine_similarity"].agg(["mean", "std"]).reset_index()
    summary["by_compression_level"] = by_level.to_dict(orient="records")

    # Summary by language
    by_language = metrics_df.groupby("language")["cosine_similarity"].agg(["mean", "std"]).reset_index()
    summary["by_language"] = by_language.to_dict(orient="records")

    # Summary by language + compression level
    by_lang_level = (
        metrics_df.groupby(["language", "compression_level"])["cosine_similarity"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary["by_language_and_compression_level"] = by_lang_level.to_dict(orient="records")

    # Estimated collapse points (first level where mean cosine similarity < 0.8)
    collapse_points: dict[str, float | None] = {}
    for language in metrics_df["language"].unique():
        lang_df = metrics_df[metrics_df["language"] == language].copy()
        by_lev = lang_df.groupby("compression_level")["cosine_similarity"].mean().sort_index(ascending=False)
        collapse = None
        for level, sim in by_lev.items():
            if sim < 0.8:
                collapse = level
                break
        collapse_points[language] = collapse
    summary["estimated_collapse_points"] = collapse_points

    return summary


def _get_git_commit() -> str:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def _hash_prompts(prompts_dir: Path) -> dict[str, str]:
    """Compute SHA256 hashes of all prompt files."""
    hashes: dict[str, str] = {}
    if not prompts_dir.exists():
        return hashes
    for prompt_file in sorted(prompts_dir.glob("*.md")):
        content = prompt_file.read_text(encoding="utf-8")
        hashes[prompt_file.name] = hashlib.sha256(content.encode()).hexdigest()
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run numerical analysis for the progressive compression experiment."
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
        help="Path to generated variants. Defaults to mirrored data/variants/ path.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Path to runs output. Defaults to mirrored runs/ path.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    from geometry_of_meaning.utils import resolve_paths

    repo_root = resolve_paths(args.experiment_dir)
    texts_dir = args.texts_dir or repo_root / "data" / "texts"
    variants_dir = args.variants_dir or repo_root / "data" / "variants" / "semantic_preservation" / "progressive_compression"
    runs_dir = args.runs_dir or repo_root / "runs" / "semantic_preservation" / "progressive_compression"

    run_analysis(
        experiment_dir=args.experiment_dir,
        texts_dir=texts_dir,
        variants_dir=variants_dir,
        runs_dir=runs_dir,
    )


if __name__ == "__main__":
    main()
