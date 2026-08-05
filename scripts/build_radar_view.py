"""Build a self-contained radar-chart HTML page from experiment embedding vectors.

Usage:
    python scripts/build_radar_view.py [--run-dir <path>] [--output <path>] [--n-components <int>]

The script reads embeddings.parquet from the latest (or specified) run, applies PCA
to reduce 1024-d vectors to --n-components dimensions, and generates a standalone
radar_view.html with Chart.js radar charts.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "runs/semantic_preservation/translation_embedding_baseline/2026-08-04T210618"
)
DEFAULT_OUTPUT = REPO_ROOT / "radar_view.html"
COMPARISON_LANGUAGES = ("en", "it", "zh", "ja", "da")

LANGUAGE_COLORS = {
    "en": {"border": "#3366CC", "background": "rgba(51, 102, 204, 0.15)"},
    "it": {"border": "#33CC66", "background": "rgba(51, 204, 102, 0.15)"},
    "zh": {"border": "#CC3333", "background": "rgba(204, 51, 51, 0.15)"},
    "ja": {"border": "#CC9933", "background": "rgba(204, 153, 51, 0.15)"},
    "da": {"border": "#9933CC", "background": "rgba(153, 51, 204, 0.15)"},
}

LANGUAGE_LABELS = {
    "en": "English",
    "it": "Italian",
    "zh": "Chinese",
    "ja": "Japanese",
    "da": "Danish",
}


def load_titles() -> dict[str, str]:
    titles = {}
    originals_dir = REPO_ROOT / "data/texts/v0.1.0/originals"
    for meta_path in sorted(originals_dir.glob("*/metadata.json")):
        with open(meta_path) as f:
            md = json.load(f)
        titles[md["text_id"]] = md.get("title", md["text_id"])
    return titles


def load_embeddings(run_dir: Path) -> pd.DataFrame:
    parquet_path = run_dir / "embeddings.parquet"
    if not parquet_path.exists():
        print(f"Error: embeddings.parquet not found at {parquet_path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_parquet(parquet_path)
    return df


def build_chart_data(
    df: pd.DataFrame, n_components: int
) -> tuple[dict, list[str], list[float]]:
    all_vectors = np.stack(df["embedding"].values)
    scaled = StandardScaler().fit_transform(all_vectors)
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(scaled)

    df = df.copy()
    df["pca_vector"] = list(reduced)

    texts_sorted = sorted(df["text_id"].unique())
    titles = load_titles()

    charts = []
    for text_id in texts_sorted:
        rows = df[df["text_id"] == text_id]
        datasets = []
        for lang in COMPARISON_LANGUAGES:
            lang_rows = rows[rows["language"] == lang]
            if lang_rows.empty:
                continue
            row = lang_rows.iloc[0]
            data = row["pca_vector"].tolist()
            color = LANGUAGE_COLORS[lang]
            datasets.append(
                {
                    "label": f"{LANGUAGE_LABELS[lang]} ({lang})",
                    "data": data,
                    "borderColor": color["border"],
                    "backgroundColor": color["background"],
                    "borderWidth": 2,
                    "pointRadius": 4,
                    "pointBackgroundColor": color["border"],
                }
            )
        charts.append(
            {
                "text_id": text_id,
                "title": titles.get(text_id, text_id),
                "category": rows.iloc[0]["category"],
                "datasets": datasets,
            }
        )

    labels = [f"PC{i + 1}" for i in range(n_components)]
    variance = pca.explained_variance_ratio_.tolist()
    return charts, labels, variance


def render_html(charts: list[dict], labels: list[str], variance: list[float]) -> str:
    charts_json = json.dumps(charts)
    labels_json = json.dumps(labels)
    variance_json = json.dumps(variance)
    lang_colors_json = json.dumps(LANGUAGE_COLORS)

    cumulative = sum(variance) * 100

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Embedding Radar Charts — Translation Embedding Baseline</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0f1117;
    color: #e1e4e8;
    padding: 24px;
    min-height: 100vh;
  }}
  header {{
    text-align: center;
    margin-bottom: 32px;
  }}
  header h1 {{
    font-size: 1.6rem;
    font-weight: 600;
    margin-bottom: 4px;
  }}
  header p {{
    font-size: 0.85rem;
    color: #8b949e;
  }}
  .legend {{
    display: flex;
    justify-content: center;
    gap: 24px;
    margin-bottom: 32px;
    flex-wrap: wrap;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.8rem;
  }}
  .legend-dot {{
    width: 12px;
    height: 12px;
    border-radius: 50%;
    border: 2px solid;
  }}
  .charts-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
    gap: 20px;
  }}
  .chart-card {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px;
  }}
  .chart-card h3 {{
    font-size: 0.82rem;
    font-weight: 600;
    text-align: center;
    margin-bottom: 2px;
    color: #c9d1d9;
  }}
  .chart-card .category {{
    font-size: 0.7rem;
    text-align: center;
    color: #6e7681;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .chart-card canvas {{
    max-height: 380px;
  }}
</style>
</head>
<body>

<header>
  <h1>Embedding Radar Charts</h1>
  <p>
    PCA on {len(charts) * len(COMPARISON_LANGUAGES)} embeddings across {len(charts)} texts × {len(COMPARISON_LANGUAGES)} languages —
    {cumulative:.1f}% variance explained by {len(labels)} components
  </p>
</header>

<div class="legend">
{''.join(f'<div class="legend-item"><span class="legend-dot" style="background:{c["background"]};border-color:{c["border"]}"></span>{LANGUAGE_LABELS[l]} ({l})</div>' for l, c in LANGUAGE_COLORS.items())}
</div>

<div class="charts-grid" id="charts-grid"></div>

<script>
const charts = {charts_json};
const labels = {labels_json};
const variance = {variance_json};

const grid = document.getElementById('charts-grid');

charts.forEach(function(ch, idx) {{
  const card = document.createElement('div');
  card.className = 'chart-card';
  card.innerHTML = '<h3>' + ch.title + '</h3><div class="category">' + ch.category + '</div><canvas id="chart-' + idx + '"></canvas>';
  grid.appendChild(card);

  new Chart(document.getElementById('chart-' + idx), {{
    type: 'radar',
    data: {{
      labels: labels,
      datasets: ch.datasets
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: true,
      scales: {{
        r: {{
          beginAtZero: false,
          grid: {{ color: 'rgba(255,255,255,0.08)' }},
          angleLines: {{ color: 'rgba(255,255,255,0.08)' }},
          pointLabels: {{ color: '#8b949e', font: {{ size: 10 }} }},
          ticks: {{ display: false }}
        }}
      }},
      plugins: {{
        legend: {{ display: true, position: 'bottom', labels: {{ color: '#8b949e', font: {{ size: 10 }}, padding: 8, usePointStyle: true, pointStyle: 'circle' }} }}
      }}
    }}
  }});
}});
</script>

</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build radar chart HTML from embeddings")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-components", type=int, default=10)
    args = parser.parse_args()

    df = load_embeddings(args.run_dir)
    print(f"Loaded {len(df)} embeddings from {args.run_dir / 'embeddings.parquet'}")

    charts, labels, variance = build_chart_data(df, args.n_components)
    cumulative = sum(variance) * 100
    print(
        f"PCA: {args.n_components} components explain {cumulative:.1f}% of variance"
    )
    for i, v in enumerate(variance):
        print(f"  PC{i + 1}: {v * 100:.1f}%")

    html = render_html(charts, labels, variance)
    args.output.write_text(html)
    print(f"Written {args.output}")


if __name__ == "__main__":
    main()
