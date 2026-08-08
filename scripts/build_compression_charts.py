"""Run an interactive Dash app for the cross-lingual compression experiment.

Usage:
    python scripts/build_compression_charts.py [--run-dir <path>]

Instead of radar charts, this visualizer presents the compression data through a
variety of complementary chart kinds:

  * Cosine-decay line charts — cosine_similarity_to_anchor against compression
    level (100% -> 12.5%), one line per language, one chart per text.
  * Trajectory line charts — cumulative trajectory length and step displacement
    along the compression path, one line per language, one chart per text.
  * PCA projections — the compression embeddings projected into a shared 2D PCA
    plane, colored by language or compression level, with faint lines connecting
    the successive compression steps of each language so drift can be traced.

The app reads metrics.parquet and embeddings.parquet from the compression run.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from sklearn.decomposition import PCA

from geometry_of_meaning.data import get_originals_dir

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_DIR = (
    REPO_ROOT / "runs/semantic_preservation/cross_lingual_compression/2026-08-07T164304"
)
COLORS_PATH = REPO_ROOT / ".agents/skills/data-visualization/assets/colors.json"
ASSETS_DIR = REPO_ROOT / "scripts/assets"
PAPER_THEME_PATH = ASSETS_DIR / "radar_theme.json"
ORIGINALS_DIR = get_originals_dir(REPO_ROOT)
COMPARISON_LANGUAGES = ("en", "it", "zh", "ja", "da")
LEVELS_DESC = (1.00, 0.50, 0.25, 0.125)
LEVEL_LABELS = {1.00: "100%", 0.50: "50%", 0.25: "25%", 0.125: "12.5%"}
THEME_KEYS = (
    "paper",
    "surface",
    "ink",
    "muted",
    "quiet",
    "rule",
    "rule_strong",
    "soft",
    "grid",
    "accent",
)
LANGUAGE_STYLE_KEYS = ("label", "border", "background")


@dataclass(frozen=True)
class CompressionChartsData:
    """The metric and embedding tables plus their metadata."""

    metrics: pd.DataFrame
    embeddings: pd.DataFrame
    pca_x: np.ndarray
    pca_y: np.ndarray
    pca_explained_variance_ratio: np.ndarray

    @property
    def embedding_count(self) -> int:
        return len(self.embeddings)

    @property
    def metric_count(self) -> int:
        return len(self.metrics)


def load_language_styles() -> dict[str, dict[str, str]]:
    """Load canonical language labels and colors from the visualization assets."""
    with COLORS_PATH.open(encoding="utf-8") as colors_file:
        raw_config: object = json.load(colors_file)
    if not isinstance(raw_config, dict) or not isinstance(raw_config.get("languages"), dict):
        raise ValueError(f"invalid language color configuration: {COLORS_PATH}")

    configured = raw_config["languages"]
    styles: dict[str, dict[str, str]] = {}
    for language in COMPARISON_LANGUAGES:
        style = configured.get(language)
        if not isinstance(style, dict):
            raise ValueError(f"missing style for language {language!r}: {COLORS_PATH}")
        values = {key: style.get(key) for key in LANGUAGE_STYLE_KEYS}
        if not all(isinstance(value, str) for value in values.values()):
            raise ValueError(f"invalid style for language {language!r}: {COLORS_PATH}")
        styles[language] = {
            "label": str(values["label"]),
            "border": str(values["border"]),
            "background": str(values["background"]),
        }
    return styles


def load_chart_theme() -> dict[str, str]:
    """Load the paper-theme tokens used by Plotly and CSS."""
    with PAPER_THEME_PATH.open(encoding="utf-8") as theme_file:
        raw_theme: object = json.load(theme_file)
    if not isinstance(raw_theme, dict):
        raise ValueError(f"invalid chart theme: {PAPER_THEME_PATH}")

    theme: dict[str, str] = {}
    for key in THEME_KEYS:
        value = raw_theme.get(key)
        if not isinstance(value, str):
            raise ValueError(f"missing or invalid theme token {key!r}: {PAPER_THEME_PATH}")
        theme[key] = value
    return theme


def load_titles() -> dict[str, str]:
    """Load human-readable titles for every canonical text."""
    titles: dict[str, str] = {}
    for metadata_path in sorted(ORIGINALS_DIR.glob("*/metadata.json")):
        with metadata_path.open(encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        titles[metadata["text_id"]] = metadata.get("title", metadata["text_id"])
    return titles


def load_metrics(run_dir: Path) -> pd.DataFrame:
    """Load the immutable metrics table for a run."""
    parquet_path = run_dir / "metrics.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(f"metrics.parquet not found at {parquet_path}")

    frame = pd.read_parquet(parquet_path)
    required_columns = {
        "text_id",
        "language",
        "category",
        "compression_level",
        "cosine_similarity_to_anchor",
        "euclidean_distance_to_anchor",
        "step_displacement",
        "cumulative_trajectory_length",
        "phase_transition_score",
    }
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"metrics.parquet is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"metrics.parquet contains no rows: {parquet_path}")
    frame = frame.copy()
    frame["compression_level"] = frame["compression_level"].astype(float)
    return frame


def load_embeddings(run_dir: Path) -> pd.DataFrame:
    """Load the immutable embedding table for a run."""
    parquet_path = run_dir / "embeddings.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(f"embeddings.parquet not found at {parquet_path}")

    frame = pd.read_parquet(parquet_path)
    required_columns = {"text_id", "language", "category", "compression_level", "embedding"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"embeddings.parquet is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"embeddings.parquet contains no rows: {parquet_path}")
    return frame


def build_compression_charts_data(
    metrics: pd.DataFrame,
    embeddings: pd.DataFrame,
) -> CompressionChartsData:
    """Compute the shared PCA plane over all compression embeddings."""
    comparison_mask = embeddings["language"].isin(COMPARISON_LANGUAGES).to_numpy()
    comparison = cast(pd.DataFrame, embeddings.loc[comparison_mask])
    if comparison.empty:
        raise ValueError("no comparison-language embeddings were found")

    vectors = [np.asarray(vector, dtype=float) for vector in comparison["embedding"]]
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError(f"embedding vectors have inconsistent dimensions: {sorted(dimensions)}")
    matrix = np.stack(vectors)
    if not np.isfinite(matrix).all():
        raise ValueError("embedding vectors contain non-finite values")

    pca_dimension = min(len(comparison) - 1, dimensions.pop())
    if pca_dimension < 2:
        raise ValueError("at least 3 embeddings are required for a 2D PCA view")
    pca = PCA(n_components=2)
    pca_matrix = pca.fit_transform(matrix)

    ordered = comparison.copy()
    ordered["pca_x"] = pca_matrix[:, 0]
    ordered["pca_y"] = pca_matrix[:, 1]

    return CompressionChartsData(
        metrics=metrics,
        embeddings=ordered,
        pca_x=pca_matrix[:, 0],
        pca_y=pca_matrix[:, 1],
        pca_explained_variance_ratio=pca.explained_variance_ratio_,
    )


def _text_metric_rows(
    data: CompressionChartsData,
    text_id: str,
    languages: list[str],
) -> pd.DataFrame:
    """Return the metric rows for a text across the selected languages."""
    frame = data.metrics
    return cast(
        pd.DataFrame,
        frame.loc[
            (frame["text_id"] == text_id) & (frame["language"].isin(languages))
        ].copy(),
    )


def build_cosine_figure(
    rows: pd.DataFrame,
    languages: list[str],
    language_styles: dict[str, dict[str, str]],
    chart_theme: dict[str, str],
) -> go.Figure:
    """Build a line chart of cosine similarity to anchor vs compression level."""
    figure = go.Figure()
    for language in languages:
        language_rows = cast(
            pd.DataFrame,
            rows.loc[rows["language"] == language].sort_values("compression_level"),
        )
        if language_rows.empty:
            continue
        style = language_styles[language]
        figure.add_trace(
            go.Scatter(
                x=language_rows["compression_level"],
                y=language_rows["cosine_similarity_to_anchor"],
                name=f"{style['label']} ({language})",
                mode="lines+markers",
                line={"color": style["border"], "width": 1.8},
                marker={"color": style["border"], "size": 5},
                text=[
                    f"{LEVEL_LABELS[level]}"
                    for level in language_rows["compression_level"]
                ],
                customdata=language_rows["compression_level"],
                hovertemplate=(
                    "%{fullData.name}<br>level %{customdata}"
                    "<br>cosine to anchor: %{y:.4f}<extra></extra>"
                ),
            )
        )
    figure.update_layout(_shared_layout(chart_theme))
    figure.update_xaxes(
        title="compression level",
        tickmode="array",
        tickvals=list(LEVELS_DESC),
        ticktext=[LEVEL_LABELS[level] for level in LEVELS_DESC],
        range=[0.06, 1.09],
        color=chart_theme["quiet"],
        gridcolor=chart_theme["grid"],
    )
    figure.update_yaxes(
        title="cosine similarity to anchor",
        range=[0.75, 1.02],
        color=chart_theme["quiet"],
        gridcolor=chart_theme["grid"],
        tickformat=".3f",
    )
    return figure


def build_trajectory_figure(
    rows: pd.DataFrame,
    languages: list[str],
    language_styles: dict[str, dict[str, str]],
    chart_theme: dict[str, str],
    metric: str,
    y_label: str,
) -> go.Figure:
    """Build a line chart of a trajectory metric vs compression level."""
    figure = go.Figure()
    for language in languages:
        language_rows = cast(
            pd.DataFrame,
            rows.loc[rows["language"] == language].sort_values("compression_level"),
        )
        if language_rows.empty:
            continue
        style = language_styles[language]
        figure.add_trace(
            go.Scatter(
                x=language_rows["compression_level"],
                y=language_rows[metric],
                name=f"{style['label']} ({language})",
                mode="lines+markers",
                line={"color": style["border"], "width": 1.8},
                marker={"color": style["border"], "size": 5},
                customdata=language_rows["compression_level"],
                hovertemplate=(
                    "%{fullData.name}<br>level %{customdata} "
                    f"<br>{y_label}: %{{y:.4f}}<extra></extra>"
                ),
            )
        )
    figure.update_layout(_shared_layout(chart_theme))
    figure.update_xaxes(
        title="compression level",
        tickmode="array",
        tickvals=list(LEVELS_DESC),
        ticktext=[LEVEL_LABELS[level] for level in LEVELS_DESC],
        range=[0.06, 1.09],
        color=chart_theme["quiet"],
        gridcolor=chart_theme["grid"],
    )
    figure.update_yaxes(
        title=y_label,
        color=chart_theme["quiet"],
        gridcolor=chart_theme["grid"],
        tickformat=".3f",
    )
    return figure


def _shared_layout(chart_theme: dict[str, str]) -> dict[str, Any]:
    """Return the shared dark-on-light layout options."""
    return {
        "template": "plotly_white",
        "paper_bgcolor": chart_theme["surface"],
        "plot_bgcolor": chart_theme["surface"],
        "font": {
            "color": chart_theme["ink"],
            "family": "Poppins, Avenir Next, Helvetica Neue, Arial, sans-serif",
            "size": 9,
        },
        "legend": {
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.22,
            "font": {"color": chart_theme["muted"], "size": 8},
            "itemsizing": "constant",
            "itemwidth": 30,
        },
        "margin": {"l": 46, "r": 18, "t": 18, "b": 44},
        "autosize": True,
        "height": 330,
        "hovermode": "closest",
    }


def build_pca_figure(
    data: CompressionChartsData,
    text_ids: list[str],
    languages: list[str],
    color_by: str,
    language_styles: dict[str, dict[str, str]],
    chart_theme: dict[str, str],
) -> go.Figure:
    """Build a 2D PCA scatter of the compression embeddings."""
    frame = data.embeddings
    rows = cast(
        pd.DataFrame,
        frame.loc[frame["text_id"].isin(text_ids) & frame["language"].isin(languages)].copy(),
    )

    figure = go.Figure()
    if color_by == "language":
        for language in languages:
            language_rows = cast(pd.DataFrame, rows.loc[rows["language"] == language])
            if language_rows.empty:
                continue
            style = language_styles[language]
            figure.add_trace(
                go.Scatter(
                    x=language_rows["pca_x"],
                    y=language_rows["pca_y"],
                    name=f"{style['label']} ({language})",
                    mode="markers",
                    marker={
                        "color": style["border"],
                        "size": 6,
                        "opacity": 0.85,
                        "line": {"color": chart_theme["surface"], "width": 0.5},
                    },
                    customdata=language_rows["compression_level"],
                    hovertemplate=(
                        "%{fullData.name}<br>level %{customdata}"
                        "<br>PC1 %{x:.3f}<br>PC2 %{y:.3f}<extra></extra>"
                    ),
                )
            )
    else:
        for level in sorted(LEVELS_DESC):
            level_rows = cast(
                pd.DataFrame,
                rows.loc[rows["compression_level"] == level],
            )
            if level_rows.empty:
                continue
            figure.add_trace(
                go.Scatter(
                    x=level_rows["pca_x"],
                    y=level_rows["pca_y"],
                    name=LEVEL_LABELS[level],
                    mode="markers",
                    marker={
                        "size": 6,
                        "opacity": 0.85,
                        "line": {"color": chart_theme["surface"], "width": 0.5},
                    },
                    hovertemplate=(
                        "%{fullData.name}<br>PC1 %{x:.3f}<br>PC2 %{y:.3f}<extra></extra>"
                    ),
                )
            )

    if color_by == "language":
        for language in languages:
            language_rows = cast(
                pd.DataFrame,
                rows.loc[rows["language"] == language].sort_values("compression_level"),
            )
            if len(language_rows) < 2:
                continue
            style = language_styles[language]
            figure.add_trace(
                go.Scatter(
                    x=language_rows["pca_x"],
                    y=language_rows["pca_y"],
                    name=f"trajectory ({language})",
                    mode="lines",
                    line={"color": style["border"], "width": 1, "dash": "dot"},
                    opacity=0.5,
                    showlegend=False,
                    hovertemplate="%{fullData.name}<extra></extra>",
                )
            )

    explained = data.pca_explained_variance_ratio
    xlabel = f"PC1 ({explained[0] * 100:.1f}% variance)"
    ylabel = f"PC2 ({explained[1] * 100:.1f}% variance)"
    figure.update_layout(_shared_layout(chart_theme))
    figure.update_xaxes(title=xlabel, color=chart_theme["quiet"], gridcolor=chart_theme["grid"])
    figure.update_yaxes(title=ylabel, color=chart_theme["quiet"], gridcolor=chart_theme["grid"])
    return figure


def build_lines_grid(
    data: CompressionChartsData,
    titles: dict[str, str],
    text_ids: list[str],
    languages: list[str],
    language_styles: dict[str, dict[str, str]],
    chart_theme: dict[str, str],
    metric: str,
    y_label: str,
    card_class: str,
) -> list[Any]:
    """Build one line chart card per selected text."""
    cards: list[Any] = []
    for figure_number, text_id in enumerate(sorted(text_ids), start=1):
        rows = _text_metric_rows(data, text_id, languages)
        if rows.empty:
            continue
        if metric == "cosine_similarity_to_anchor":
            figure = build_cosine_figure(
                rows, languages, language_styles, chart_theme
            )
        else:
            figure = build_trajectory_figure(
                rows, languages, language_styles, chart_theme, metric, y_label
            )
        category = str(rows.iloc[0]["category"])
        cards.append(
            html.Section(
                [
                    html.Header(
                        [
                            html.P(f"Figure {figure_number:02d}", className="figure-number"),
                            html.H3(titles.get(text_id, text_id)),
                            html.P(category, className="chart-category"),
                        ],
                        className="figure-header",
                    ),
                    dcc.Graph(
                        figure=figure,
                        className=card_class,
                        style={"height": "330px"},
                        config={
                            "displaylogo": False,
                            "responsive": True,
                            "toImageButtonOptions": {"filename": f"{text_id}_{metric}"},
                        },
                    ),
                ],
                className="chart-card",
            )
        )
    return cards


def build_pca_card(
    data: CompressionChartsData,
    text_ids: list[str],
    languages: list[str],
    color_by: str,
    language_styles: dict[str, dict[str, str]],
    chart_theme: dict[str, str],
) -> list[Any]:
    """Build a single PCA scatter card encompassing the selected texts."""
    figure = build_pca_figure(
        data,
        text_ids,
        languages,
        color_by,
        language_styles,
        chart_theme,
    )
    explained = data.pca_explained_variance_ratio
    caption = (
        f"Shared PCA on {data.embedding_count} compression embeddings · "
        f"PC1 {explained[0] * 100:.1f}% + PC2 {explained[1] * 100:.1f}% variance · "
        f"colored by {color_by}. Dotted lines trace each language's path from "
        f"{LEVEL_LABELS[1.00]} to {LEVEL_LABELS[0.125]} compression."
    )
    return [
        html.Section(
            [
                html.Header(
                    [
                        html.P("Projection", className="figure-number"),
                        html.H3("Compression embeddings · 2D PCA"),
                        html.P(caption, className="figure-caption"),
                    ],
                    className="figure-header",
                ),
                dcc.Graph(
                    figure=figure,
                    className="pca-chart",
                    style={"height": "600px"},
                    config={
                        "displaylogo": False,
                        "responsive": True,
                        "toImageButtonOptions": {"filename": "compression_pca"},
                    },
                ),
            ],
            className="chart-card pca-card",
        )
    ]


def create_app(data: CompressionChartsData, run_dir: Path) -> Dash:
    """Create the Dash application and register its callbacks."""
    titles = load_titles()
    language_styles = load_language_styles()
    chart_theme = load_chart_theme()
    text_ids = sorted(data.metrics["text_id"].unique())
    categories = sorted(data.metrics["category"].unique())
    languages = [
        language
        for language in COMPARISON_LANGUAGES
        if language in data.metrics["language"].unique()
    ]

    theme_variables = {
        "--paper": chart_theme["paper"],
        "--surface": chart_theme["surface"],
        "--ink": chart_theme["ink"],
        "--muted": chart_theme["muted"],
        "--quiet": chart_theme["quiet"],
        "--rule": chart_theme["rule"],
        "--rule-strong": chart_theme["rule_strong"],
        "--soft": chart_theme["soft"],
        "--grid": chart_theme["grid"],
        "--accent": chart_theme["accent"],
    }

    app = Dash(
        __name__,
        title="Compression Charts",
        assets_folder=str(ASSETS_DIR),
    )

    controls = html.Div(
        [
            html.Div(
                [
                    html.Label("Categories", htmlFor="category-filter"),
                    dcc.Dropdown(
                        id="category-filter",
                        options=[
                            {"label": value.title(), "value": value} for value in categories
                        ],
                        value=categories,
                        multi=True,
                    ),
                ],
                className="filter-field compact-multi",
            ),
            html.Div(
                [
                    html.Label("Texts", htmlFor="text-filter"),
                    dcc.Dropdown(
                        id="text-filter",
                        options=[
                            {"label": titles.get(text_id, text_id), "value": text_id}
                            for text_id in text_ids
                        ],
                        value=text_ids,
                        multi=True,
                    ),
                ],
                className="filter-field compact-multi",
            ),
            html.Div(
                [
                    html.Label("Languages", htmlFor="language-filter"),
                    dcc.Dropdown(
                        id="language-filter",
                        options=[
                            {
                                "label": f"{language_styles[language]['label']} ({language})",
                                "value": language,
                            }
                            for language in languages
                        ],
                        value=languages,
                        multi=True,
                    ),
                ],
                className="filter-field compact-multi",
            ),
            html.Div(
                [
                    html.Label("PCA color", htmlFor="pca-color"),
                    dcc.Dropdown(
                        id="pca-color",
                        options=[
                            {"label": "By language", "value": "language"},
                            {"label": "By compression", "value": "compression"},
                        ],
                        value="language",
                        clearable=False,
                        searchable=False,
                    ),
                ],
                className="filter-field",
            ),
        ],
        className="filters-panel",
    )

    def format_title(s: str) -> str:
        """Convert snake_case to Title Case."""
        return s.replace("_", " ").title()

    def build_page_title(run_dir: Path) -> str:
        """Build a clean page title from the run path."""
        experiment = format_title(run_dir.parent.name)
        area = format_title(run_dir.parent.parent.name)
        return f"{experiment} - {area}"

    app.layout = html.Main(
        [
            html.Header(
                [
                    html.H1(build_page_title(run_dir)),
                    html.Dl(
                        [
                            html.Dt("experiment:"),
                            html.Dd(run_dir.parent.name),
                            html.Dt("research area:"),
                            html.Dd(run_dir.parent.parent.name),
                            html.Dt("run:"),
                            html.Dd(run_dir.name),
                            html.Dt("metric rows:"),
                            html.Dd(str(data.metric_count)),
                            html.Dt("embeddings:"),
                            html.Dd(str(data.embedding_count)),
                            html.Dt("levels:"),
                            html.Dd(" / ".join(LEVEL_LABELS[level] for level in LEVELS_DESC)),
                        ],
                        className="run-metadata",
                    ),
                ],
                className="app-header",
            ),
            dcc.Tabs(
                id="view-tabs",
                value="cosine",
                children=[
                    dcc.Tab(label="Cosine decay", value="cosine"),
                    dcc.Tab(label="Trajectory", value="trajectory"),
                    dcc.Tab(label="PCA projection", value="pca"),
                ],
            ),
            dcc.Loading(html.Div(id="charts", className="charts-grid")),
            html.Div(id="selection-summary", className="selection-summary"),
            html.Details(
                [
                    html.Summary("Controls"),
                    controls,
                ],
                className="controls-disclosure",
            ),
        ],
        className="app-shell",
        style=theme_variables,
    )

    @app.callback(
        Output("charts", "children"),
        Output("selection-summary", "children"),
        Input("view-tabs", "value"),
        Input("category-filter", "value"),
        Input("text-filter", "value"),
        Input("language-filter", "value"),
        Input("pca-color", "value"),
    )
    def update_charts(
        view: str,
        selected_categories: list[str] | None,
        selected_texts: list[str] | None,
        selected_languages: list[str] | None,
        pca_color: str,
    ) -> tuple[list[Any], str]:
        categories = selected_categories or []
        text_ids = selected_texts or []
        languages = selected_languages or []

        metric_rows = cast(
            pd.DataFrame,
            data.metrics.loc[
                data.metrics["category"].isin(categories)
                & data.metrics["text_id"].isin(text_ids)
            ],
        )
        if not text_ids:
            return [], "Select at least one text to display charts."
        if not languages:
            return [], "Select at least one language to display charts."
        if metric_rows.empty:
            return [], "No metrics match the current category and text filters."

        if view == "pca":
            cards = build_pca_card(
                data,
                text_ids,
                languages,
                pca_color,
                language_styles,
                chart_theme,
            )
            summary = (
                f"{len(text_ids)} text(s) x {len(languages)} language(s) "
                f"projected into a shared 2D PCA plane."
            )
        else:
            if metric_rows.empty:
                return [], "No metric rows match the current filters."
            if view == "trajectory":
                cards = build_lines_grid(
                    data,
                    titles,
                    text_ids,
                    languages,
                    language_styles,
                    chart_theme,
                    "cumulative_trajectory_length",
                    "cumulative trajectory length",
                    "line-chart",
                )
            else:
                cards = build_lines_grid(
                    data,
                    titles,
                    text_ids,
                    languages,
                    language_styles,
                    chart_theme,
                    "cosine_similarity_to_anchor",
                    "cosine similarity to anchor",
                    "line-chart",
                )
            summary = (
                f"{len(cards)} text(s) x {len(languages)} language(s) "
                f"lines across {len(LEVELS_DESC)} compression levels."
            )
        return cards, summary

    return app


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the Dash development server."""
    parser = argparse.ArgumentParser(description="Run interactive compression charts")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8052)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Load data, report provenance, and run the Dash application."""
    args = parse_args()
    metrics = load_metrics(args.run_dir)
    embeddings = load_embeddings(args.run_dir)
    data = build_compression_charts_data(metrics, embeddings)

    print(f"Loaded {len(metrics)} metric rows from {args.run_dir / 'metrics.parquet'}")
    print(f"Loaded {len(embeddings)} embeddings from {args.run_dir / 'embeddings.parquet'}")
    print(
        f"PCA variance explained: PC1 {data.pca_explained_variance_ratio[0] * 100:.1f}%, "
        f"PC2 {data.pca_explained_variance_ratio[1] * 100:.1f}%"
    )

    app = create_app(data, args.run_dir)
    print(f"Starting Dash app at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
