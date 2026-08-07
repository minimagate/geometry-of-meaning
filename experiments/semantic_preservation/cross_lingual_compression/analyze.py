"""
Cross-Lingual Compression — Numerical Analysis

Embeds the already-generated compression variants (12 texts x 5 languages x 4
compression levels), computes semantic-decay metrics along each per-language
compression trajectory, and saves an immutable run.

Pipeline:
  1. Load all generated variants under data/variants/ via load_variants
  2. Embed every variant using the configured embedding model(s)
  3. For each (text_id, language) anchor on the 1.00 (verbatim) variant and:
       - cosine similarity / euclidean distance of each level vs the anchor
       - step displacement between adjacent compression levels
       - cumulative trajectory length from 1.00 down to 0.125
       - phase-transition score (max step displacement / mean step displacement)
  4. Save results as a new immutable timestamped run
"""

import argparse
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from geometry_of_meaning.data import (
    CORPUS_VERSION,
    load_experiment_dataset,
    load_variants,
)
from geometry_of_meaning.embeddings import embed_texts, get_model_info
from geometry_of_meaning.metrics import (
    cosine_similarity,
    cumulative_trajectory_length,
    euclidean_distance,
    phase_transition_score,
    step_displacements,
)
from geometry_of_meaning.utils import (
    hash_file,
    load_config,
    resolve_paths,
    set_seed,
    setup_logging,
    timestamp_now,
)

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
        texts_dir: Path to data/texts/ (unused by compression, kept for parity).
        variants_dir: Path to the mirrored variants directory.
        runs_dir: Path to runs output directory.

    Returns:
        Path to the created run directory.
    """
    del texts_dir  # compression variants are self-contained; texts_dir unused
    config = load_config(experiment_dir / "config.yaml")
    set_seed(config.get("random_seed", 42))
    dataset = load_experiment_dataset(experiment_dir / "dataset.jsonl")

    run_id = timestamp_now()
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    file_handler = logging.FileHandler(run_dir / "logs.txt", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    logger.info(f"Starting run: {run_id}")
    git_commit = _get_git_commit()

    reference_language = config.get("reference_language", "en")
    compression_levels = sorted(config["compression"]["levels"], reverse=True)
    enabled_ids = {e["text_id"] for e in dataset if e.get("enabled", True)}
    category_by_id = {e["text_id"]: e.get("category", "unknown") for e in dataset}

    # ── 1. Collect variants ──────────────────────────────────────────────

    variants_by_lang = load_variants(variants_dir)
    records: list[dict[str, Any]] = []
    texts_to_embed: list[str] = []

    for language in config["languages"]:
        for record in variants_by_lang.get(language, []):
            if record.text_id not in enabled_ids:
                continue
            entry = {
                "text_id": record.text_id,
                "language": record.language,
                "source_language": record.source_language,
                "category": category_by_id.get(record.text_id, "unknown"),
                "compression_level": record.compression_level,
                "variant_id": record.variant_id,
            }
            records.append(entry)
            texts_to_embed.append(record.text)

    logger.info(f"Collected {len(texts_to_embed)} variants to embed")

    # ── 2. Embed all variants ────────────────────────────────────────────

    all_embeddings: dict[str, np.ndarray] = {}

    for model_id in config["embedding_models"]:
        logger.info(f"Embedding with model: {model_id}")
        embeddings_array = embed_texts(texts_to_embed, model_id=model_id)
        all_embeddings[model_id] = embeddings_array

    # ── 3. Compute metrics per compression trajectory ───────────────────

    metrics_records: list[dict[str, Any]] = []
    trajectory_summaries: dict[str, dict[str, Any]] = {}
    anchor_cosine: dict[str, dict[str, float]] = {}

    for model_id in config["embedding_models"]:
        model_info = get_model_info(model_id)
        emb = all_embeddings[model_id]

        emb_lookup: dict[tuple[str, str, float], np.ndarray] = {}
        for i, rec in enumerate(records):
            emb_lookup[(rec["text_id"], rec["language"], rec["compression_level"])] = emb[i]

        for text_id in sorted(enabled_ids):
            for language in config["languages"]:
                anchor_key = (text_id, language, 1.0)
                if anchor_key not in emb_lookup:
                    continue
                anchor = emb_lookup[anchor_key]

                ordered = [
                    (level, emb_lookup[(text_id, language, level)])
                    for level in compression_levels
                    if (text_id, language, level) in emb_lookup
                ]
                ordered.sort(key=lambda item: item[0], reverse=True)
                if not ordered:
                    continue

                level_vecs = [vec for _, vec in ordered]
                steps = step_displacements(level_vecs)
                cumul = cumulative_trajectory_length(level_vecs)
                phase = phase_transition_score(level_vecs)

                for idx, (level, vec) in enumerate(ordered):
                    cos_anchor = cosine_similarity(anchor, vec)
                    euc_anchor = euclidean_distance(anchor, vec)
                    metrics_records.append({
                        "text_id": text_id,
                        "language": language,
                        "source_language": next(
                            r["source_language"]
                            for r in records
                            if (r["text_id"], r["language"], r["compression_level"])
                            == (text_id, language, level)
                        ),
                        "category": category_by_id.get(text_id, "unknown"),
                        "model": model_id,
                        "model_version": model_info.get("version", "unknown"),
                        "compression_level": level,
                        "cosine_similarity_to_anchor": cos_anchor,
                        "euclidean_distance_to_anchor": euc_anchor,
                        "step_displacement": steps[idx],
                        "cumulative_trajectory_length": cumul[idx],
                        "phase_transition_score": phase,
                    })
                    if level == 1.0:
                        anchor_cosine.setdefault(text_id, {})[language] = cos_anchor

                key = f"{text_id}|{language}"
                ordered_levels = [item[0] for item in ordered]
                trajectory_summaries[key] = {
                    "text_id": text_id,
                    "language": language,
                    "model": model_id,
                    "levels": ordered_levels,
                    "step_displacements": steps,
                    "cumulative_trajectory_length": cumul,
                    "phase_transition_score": phase,
                }

    # ── 4. Build DataFrames ──────────────────────────────────────────────

    metrics_df = pd.DataFrame(metrics_records)

    emb_records: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        for model_id in config["embedding_models"]:
            emb_records.append({
                "text_id": rec["text_id"],
                "language": rec["language"],
                "source_language": rec["source_language"],
                "category": rec["category"],
                "model": model_id,
                "compression_level": rec["compression_level"],
                "variant_id": rec["variant_id"],
                "embedding": all_embeddings[model_id][i].tolist(),
            })
    embeddings_df = pd.DataFrame(emb_records)

    # ── 5. Compute summary ───────────────────────────────────────────────

    summary = _compute_summary(metrics_df, trajectory_summaries, config)

    # ── 6. Save run artifacts ────────────────────────────────────────────

    prompt_hashes = {
        prompt_path.name: hash_file(prompt_path)
        for prompt_path in sorted((experiment_dir / "prompts").glob("*.md"))
    }

    manifest = {
        "run_id": run_id,
        "experiment": config["experiment"]["id"],
        "research_area": config["experiment"]["research_area"],
        "languages": config["languages"],
        "reference_language": reference_language,
        "embedding_models": config["embedding_models"],
        "compression_levels": compression_levels,
        "git_commit": git_commit,
        "random_seed": config["random_seed"],
        "prompt_hashes": prompt_hashes,
        "num_variants": len(texts_to_embed),
        "num_enabled_texts": len(enabled_ids),
        "created_at": datetime.now(UTC).isoformat(),
    }

    with open(run_dir / "manifest.yaml", "w", encoding="utf-8") as f:
        yaml.dump(manifest, f, allow_unicode=True, default_flow_style=False)

    with open(run_dir / "config.snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    with open(run_dir / "dataset.snapshot.jsonl", "w", encoding="utf-8") as f:
        for entry in dataset:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    with open(run_dir / "variants.manifest.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")

    metrics_df.to_parquet(run_dir / "metrics.parquet", index=False)
    embeddings_df.to_parquet(run_dir / "embeddings.parquet", index=False)

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(run_dir / "trajectories.json", "w", encoding="utf-8") as f:
        json.dump(trajectory_summaries, f, indent=2, ensure_ascii=False)

    logger.info(f"Run complete. Results saved to {run_dir}")
    return run_dir


def _compute_summary(
    metrics_df: pd.DataFrame,
    trajectory_summaries: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, object]:
    """Compute aggregate summary statistics."""
    summary: dict[str, object] = {
        "description": (
            "Cross-lingual compression — semantic decay under progressive compression"
        ),
        "num_variants": len(metrics_df),
        "num_texts": metrics_df["text_id"].nunique() if not metrics_df.empty else 0,
        "num_languages": metrics_df["language"].nunique() if not metrics_df.empty else 0,
        "num_models": metrics_df["model"].nunique() if not metrics_df.empty else 0,
    }

    if metrics_df.empty:
        return summary

    # Per-language mean cosine decay at each compression level
    by_lang_level = (
        metrics_df.groupby(["language", "compression_level"])["cosine_similarity_to_anchor"]
        .mean()
        .round(6)
        .reset_index()
    )
    decay: dict[str, dict[str, float]] = {}
    for lang in config["languages"]:
        decay[lang] = {}
    for _, row in by_lang_level.iterrows():
        decay[row["language"]][str(row["compression_level"])] = row["cosine_similarity_to_anchor"]
    summary["mean_cosine_similarity_by_language_level"] = decay

    # Overall decay curve (averaged across languages and texts)
    overall_decay = (
        metrics_df.groupby("compression_level")["cosine_similarity_to_anchor"]
        .mean()
        .round(6)
        .to_dict()
    )
    summary["mean_cosine_similarity_by_level"] = {
        str(k): v for k, v in sorted(overall_decay.items(), reverse=True)
    }

    # Per-language mean phase-transition score (sharpness of collapse)
    phase = (
        metrics_df.groupby("language")["phase_transition_score"]
        .mean()
        .round(6)
        .reset_index()
    )
    summary["mean_phase_transition_score_by_language"] = phase.to_dict(orient="records")

    # Per-language mean cumulative trajectory length
    traj = (
        metrics_df.groupby("language")["cumulative_trajectory_length"]
        .mean()
        .round(6)
        .reset_index()
    )
    summary["mean_cumulative_trajectory_length_by_language"] = traj.to_dict(orient="records")

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run numerical analysis for cross-lingual compression."
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
        help="Path to variants. Defaults to mirrored data/variants/ path.",
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

    repo_root = resolve_paths(args.experiment_dir)
    texts_dir = args.texts_dir or repo_root / "data" / "texts"

    experiment_relative = args.experiment_dir.resolve().relative_to(repo_root)
    mirror_path = Path(*experiment_relative.parts[1:])
    variants_dir = (
        args.variants_dir
        or repo_root / "data" / "variants" / mirror_path / CORPUS_VERSION
    )
    runs_dir = args.runs_dir or repo_root / "runs" / mirror_path

    run_analysis(
        experiment_dir=args.experiment_dir,
        texts_dir=texts_dir,
        variants_dir=variants_dir,
        runs_dir=runs_dir,
    )


if __name__ == "__main__":
    main()
