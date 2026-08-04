"""
Translation Embedding Baseline — Numerical Analysis

Embeds canonical texts and their translations, computes semantic metrics,
and saves an immutable run. This is the zero-manipulation baseline:
all texts are at compression_level=1.0 with no LLM modification.

Pipeline:
  1. Load originals and translations for all configured languages
  2. Embed all texts using the configured embedding model(s)
  3. Compute per-text pairwise similarity matrices across languages
  4. Compute English→translation cosine similarities and Euclidean distances
  5. Compute cross-language centroid distances
  6. Save results as a new immutable timestamped run
"""

import argparse
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from geometry_of_meaning.data import (
    load_experiment_dataset,
    load_original_text,
    load_translation,
)
from geometry_of_meaning.embeddings import embed_texts, get_model_info
from geometry_of_meaning.metrics import (
    centroid_distance,
    cosine_similarity,
    euclidean_distance,
)
from geometry_of_meaning.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def run_analysis(
    experiment_dir: Path,
    texts_dir: Path,
    runs_dir: Path,
) -> Path:
    """
    Run the full numerical analysis and save results as an immutable run.

    Args:
        experiment_dir: Path to experiment directory.
        texts_dir: Path to data/texts/.
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

    git_commit = _get_git_commit()
    reference_language = config.get("reference_language", "en")

    # ── 1. Collect all texts to embed ────────────────────────────────────

    # {text_id, language, source_language, category, is_original, is_reference}
    records: list[dict] = []
    texts_to_embed: list[str] = []

    enabled_ids = {e["text_id"] for e in dataset if e.get("enabled", True)}

    for entry in dataset:
        if not entry.get("enabled", True):
            continue

        text_id = entry["text_id"]
        category = entry.get("category", "unknown")

        original = load_original_text(texts_dir / "originals", text_id)
        source_language = original.original_language

        # Add the original text (in its original language)
        texts_to_embed.append(original.text)
        records.append({
            "text_id": text_id,
            "language": source_language,
            "source_language": source_language,
            "category": category,
            "is_original": True,
            "is_reference": (source_language == reference_language),
        })

        # Load translations for each configured language (skip source — already added)
        for language in config["languages"]:
            if language == source_language:
                continue

            translation = load_translation(texts_dir / "translations", text_id, language)
            texts_to_embed.append(translation.text)
            records.append({
                "text_id": text_id,
                "language": language,
                "source_language": source_language,
                "category": category,
                "is_original": False,
                "is_reference": (language == reference_language),
            })

    logger.info(f"Collected {len(texts_to_embed)} texts to embed")

    # ── 2. Embed all texts ───────────────────────────────────────────────

    all_embeddings: dict[str, dict[str, np.ndarray]] = {}

    for model_id in config["embedding_models"]:
        logger.info(f"Embedding with model: {model_id}")
        model_info = get_model_info(model_id)

        embeddings_array = embed_texts(texts_to_embed, model_id=model_id)

        # Build lookup: (text_id, language) -> embedding
        emb_lookup: dict[str, np.ndarray] = {}
        for i, rec in enumerate(records):
            key = f"{rec['text_id']}|{rec['language']}"
            if key not in emb_lookup:
                emb_lookup[key] = embeddings_array[i]

        all_embeddings[model_id] = emb_lookup

    # ── 3. Compute metrics ───────────────────────────────────────────────

    metrics_records: list[dict] = []
    pairwise_matrices: dict[str, dict] = {}
    language_centroid_distances: dict[str, dict[str, float]] = {}

    for model_id in config["embedding_models"]:
        model_info = get_model_info(model_id)
        emb_lookup = all_embeddings[model_id]

        # Accumulate embeddings per language for centroid computation
        lang_embs: dict[str, list[np.ndarray]] = {
            lang: [] for lang in config["languages"]
        }

        for text_id in sorted(enabled_ids):
            # Collect embeddings for each configured language
            text_embs: dict[str, np.ndarray] = {}
            for language in config["languages"]:
                key = f"{text_id}|{language}"
                if key in emb_lookup:
                    text_embs[language] = emb_lookup[key]
                    lang_embs[language].append(emb_lookup[key])

            if len(text_embs) < 2:
                continue

            # Compute pairwise cosine similarity matrix (5×5 for 5 languages)
            languages_present = sorted(text_embs.keys())
            n_langs = len(languages_present)
            pairwise_sim = np.zeros((n_langs, n_langs))

            for i, lang_i in enumerate(languages_present):
                for j, lang_j in enumerate(languages_present):
                    pairwise_sim[i, j] = cosine_similarity(
                        text_embs[lang_i], text_embs[lang_j]
                    )

            pairwise_matrices.setdefault(text_id, {})[model_id] = {
                "languages": languages_present,
                "similarity_matrix": pairwise_sim.tolist(),
            }

            # Compute English→translation metrics
            if reference_language in text_embs:
                ref_emb = text_embs[reference_language]
                for language in languages_present:
                    if language == reference_language:
                        continue

                    lang_emb = text_embs[language]
                    cos_sim = cosine_similarity(ref_emb, lang_emb)
                    euc_dist = euclidean_distance(ref_emb, lang_emb)

                    metrics_records.append({
                        "text_id": text_id,
                        "language": language,
                        "reference_language": reference_language,
                        "model": model_id,
                        "model_version": model_info.get("version", "unknown"),
                        "cosine_similarity_to_en": cos_sim,
                        "euclidean_distance_to_en": euc_dist,
                        "category": next(
                            (e.get("category") for e in dataset if e["text_id"] == text_id),
                            "unknown",
                        ),
                    })

        # Compute cross-language centroids
        lang_centroids: dict[str, np.ndarray] = {}
        for lang, embs in lang_embs.items():
            if embs:
                lang_centroids[lang] = np.mean(np.stack(embs), axis=0)

        # Pairwise centroid distances
        for lang_i in sorted(lang_centroids.keys()):
            for lang_j in sorted(lang_centroids.keys()):
                if lang_i < lang_j:
                    dist = centroid_distance(
                        lang_centroids[lang_i][np.newaxis, :],
                        lang_centroids[lang_j][np.newaxis, :],
                    )
                    language_centroid_distances.setdefault(model_id, {})[
                        f"{lang_i}_{lang_j}"
                    ] = dist

    # ── 4. Build DataFrames ──────────────────────────────────────────────

    metrics_df = pd.DataFrame(metrics_records)

    # Build embeddings DataFrame with actual vectors
    emb_records: list[dict] = []
    for i, rec in enumerate(records):
        for model_id in config["embedding_models"]:
            key = f"{rec['text_id']}|{rec['language']}"
            if key in all_embeddings[model_id]:
                emb_records.append({
                    "text_id": rec["text_id"],
                    "language": rec["language"],
                    "source_language": rec["source_language"],
                    "category": rec["category"],
                    "is_original": rec["is_original"],
                    "model": model_id,
                    "embedding": all_embeddings[model_id][key].tolist(),
                })

    embeddings_df = pd.DataFrame(emb_records)

    # ── 5. Compute summary ───────────────────────────────────────────────

    summary = _compute_summary(metrics_df, pairwise_matrices, language_centroid_distances, config)

    # ── 6. Save run artifacts ────────────────────────────────────────────

    manifest = {
        "run_id": run_id,
        "experiment": config["experiment"]["id"],
        "research_area": config["experiment"]["research_area"],
        "languages": config["languages"],
        "reference_language": reference_language,
        "embedding_models": config["embedding_models"],
        "git_commit": git_commit,
        "random_seed": config["random_seed"],
        "num_texts": len(texts_to_embed),
        "num_enabled_texts": len(enabled_ids),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(run_dir / "manifest.yaml", "w", encoding="utf-8") as f:
        yaml.dump(manifest, f, allow_unicode=True, default_flow_style=False)

    with open(run_dir / "config.snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    with open(run_dir / "dataset.snapshot.jsonl", "w", encoding="utf-8") as f:
        for entry in dataset:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    metrics_df.to_parquet(run_dir / "metrics.parquet", index=False)
    embeddings_df.to_parquet(run_dir / "embeddings.parquet", index=False)

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(run_dir / "pairwise_matrices.json", "w", encoding="utf-8") as f:
        json.dump(pairwise_matrices, f, indent=2, ensure_ascii=False)

    logger.info(f"Run complete. Results saved to {run_dir}")
    return run_dir


def _compute_summary(
    metrics_df: pd.DataFrame,
    pairwise_matrices: dict,
    language_centroid_distances: dict,
    config: dict,
) -> dict:
    """Compute aggregate summary statistics."""

    summary: dict = {
        "description": "Translation embedding baseline — zero-manipulation semantic distances",
        "num_texts": metrics_df["text_id"].nunique() if not metrics_df.empty else 0,
        "num_languages": metrics_df["language"].nunique() if not metrics_df.empty else 0,
        "num_models": metrics_df["model"].nunique() if not metrics_df.empty else 0,
        "num_measurements": len(metrics_df),
    }

    if metrics_df.empty:
        return summary

    # Per-language cosine similarity to English
    by_language = (
        metrics_df.groupby("language")["cosine_similarity_to_en"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary["cosine_similarity_to_en_by_language"] = by_language.to_dict(orient="records")

    # Per-language Euclidean distance to English
    by_lang_dist = (
        metrics_df.groupby("language")["euclidean_distance_to_en"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary["euclidean_distance_to_en_by_language"] = by_lang_dist.to_dict(orient="records")

    # Per-text average cosine similarity (across all non-English languages)
    by_text = (
        metrics_df.groupby("text_id")["cosine_similarity_to_en"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=True)
        .reset_index()
    )
    summary["cosine_similarity_to_en_by_text"] = by_text.to_dict(orient="records")

    # Per-category
    if "category" in metrics_df.columns:
        by_category = (
            metrics_df.groupby("category")["cosine_similarity_to_en"]
            .agg(["mean", "std", "count"])
            .sort_values("mean", ascending=True)
            .reset_index()
        )
        summary["cosine_similarity_to_en_by_category"] = by_category.to_dict(orient="records")

    # Overall statistics
    summary["overall"] = {
        "mean_cosine_similarity_to_en": float(metrics_df["cosine_similarity_to_en"].mean()),
        "std_cosine_similarity_to_en": float(metrics_df["cosine_similarity_to_en"].std()),
        "min_cosine_similarity_to_en": float(metrics_df["cosine_similarity_to_en"].min()),
        "max_cosine_similarity_to_en": float(metrics_df["cosine_similarity_to_en"].max()),
        "mean_euclidean_distance_to_en": float(metrics_df["euclidean_distance_to_en"].mean()),
    }

    # Centroid distances
    summary["cross_language_centroid_distances"] = language_centroid_distances

    # Per-text self-consistency (average cross-language similarity for same text)
    text_self_consistency: dict[str, float] = {}
    for text_id, models in pairwise_matrices.items():
        for model_id, matrix_data in models.items():
            sim_matrix = np.array(matrix_data["similarity_matrix"])
            languages = matrix_data["languages"]
            # Off-diagonal mean (same text, different languages)
            mask = ~np.eye(len(languages), dtype=bool)
            if mask.sum() > 0:
                text_self_consistency[text_id] = float(sim_matrix[mask].mean())
    summary["text_self_consistency"] = dict(
        sorted(text_self_consistency.items(), key=lambda x: x[1])
    )

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
        description="Run numerical analysis for the translation embedding baseline."
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
    runs_dir = (
        args.runs_dir
        or repo_root / "runs" / "semantic_preservation" / "translation_embedding_baseline"
    )

    run_analysis(
        experiment_dir=args.experiment_dir,
        texts_dir=texts_dir,
        runs_dir=runs_dir,
    )


if __name__ == "__main__":
    main()