"""Run an interactive Dash app for projected embedding radar charts.

Usage:
    python scripts/build_radar_view.py [--run-dir <path>] [--initial-dimensions <int>]

The app reads embeddings.parquet from the specified run and renders one
interactive Plotly radar chart per canonical text. PCA is the default projection;
centered, L2-scaled coordinate blocks remain available for direct inspection.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from geometry_of_meaning.data import get_originals_dir

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_DIR = (
    REPO_ROOT / "runs/semantic_preservation/translation_embedding_baseline/2026-08-04T210618"
)
COLORS_PATH = REPO_ROOT / ".agents/skills/data-visualization/assets/colors.json"
ASSETS_DIR = REPO_ROOT / "scripts/assets"
PAPER_THEME_PATH = ASSETS_DIR / "radar_theme.json"
ORIGINALS_DIR = get_originals_dir(REPO_ROOT)
COMPARISON_LANGUAGES = ("en", "it", "zh", "ja", "da")
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
LANGUAGE_PALETTE = {
    "en": ("#00B4D8", "rgba(0, 180, 216, 0.07)"),
    "it": ("#D63384", "rgba(214, 51, 132, 0.07)"),
    "zh": ("#FF3333", "rgba(255, 51, 51, 0.07)"),
    "ja": ("#FF8800", "rgba(255, 136, 0, 0.07)"),
    "da": ("#6600FF", "rgba(102, 0, 255, 0.07)"),
}


@dataclass(frozen=True)
class RadarData:
    """Embedding data and the metadata needed to plot it."""

    frame: pd.DataFrame
    embedding_count: int
    embedding_dimension: int
    center_vector: np.ndarray
    pca_explained_variance_ratio: np.ndarray

    @property
    def center_norm(self) -> float:
        """Return the L2 norm of the run-wide embedding centroid."""
        return float(np.linalg.norm(self.center_vector))

    @property
    def pca_dimension(self) -> int:
        """Return the number of non-degenerate PCA components available."""
        return len(self.pca_explained_variance_ratio)


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
        border, background = LANGUAGE_PALETTE[language]
        styles[language] = {
            "label": str(values["label"]),
            "border": border,
            "background": background,
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


def load_embeddings(run_dir: Path) -> pd.DataFrame:
    """Load the immutable embedding table for a run."""
    parquet_path = run_dir / "embeddings.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(f"embeddings.parquet not found at {parquet_path}")

    frame = pd.read_parquet(parquet_path)
    required_columns = {"text_id", "language", "category", "embedding"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"embeddings.parquet is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"embeddings.parquet contains no rows: {parquet_path}")
    return frame


def build_radar_data(frame: pd.DataFrame) -> RadarData:
    """Validate embeddings and precompute the shared PCA coordinate system."""
    vectors = [np.asarray(vector, dtype=float) for vector in frame["embedding"]]
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError(f"embedding vectors have inconsistent dimensions: {sorted(dimensions)}")

    embedding_dimension = dimensions.pop()
    embedding_matrix = np.stack(vectors)
    if not np.isfinite(embedding_matrix).all():
        raise ValueError("embedding vectors contain non-finite values")

    scaled_matrix = StandardScaler().fit_transform(embedding_matrix)
    pca_dimension = min(len(frame) - 1, embedding_dimension)
    if pca_dimension < 10:
        raise ValueError("at least 11 embeddings are required for a 10-component PCA view")
    pca = PCA(n_components=pca_dimension)
    pca_matrix = pca.fit_transform(scaled_matrix)

    comparison_mask = frame["language"].isin(COMPARISON_LANGUAGES).to_numpy()
    comparison_frame = cast(pd.DataFrame, frame.loc[comparison_mask].copy())
    if comparison_frame.empty:
        raise ValueError("no comparison-language embeddings were found")
    comparison_frame["pca_vector"] = list(pca_matrix[comparison_mask])

    return RadarData(
        frame=comparison_frame,
        embedding_count=len(frame),
        embedding_dimension=embedding_dimension,
        center_vector=embedding_matrix.mean(axis=0),
        pca_explained_variance_ratio=pca.explained_variance_ratio_,
    )


def build_block_labels(source_dimensions: int, display_dimensions: int) -> tuple[str, ...]:
    """Label raw dimensions or contiguous dimension ranges used by each block."""
    if not 1 <= display_dimensions <= source_dimensions:
        raise ValueError(
            f"display_dimensions must be between 1 and {source_dimensions}, "
            f"got {display_dimensions}"
        )
    boundaries = np.linspace(0, source_dimensions, display_dimensions + 1, dtype=int)
    labels: list[str] = []
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        labels.append(f"D{start + 1}" if stop - start == 1 else f"D{start + 1}–D{stop}")
    return tuple(labels)


def project_coordinate_blocks(
    vector: np.ndarray,
    center_vector: np.ndarray,
    display_dimensions: int,
) -> np.ndarray:
    """Center an embedding and project it onto orthonormal block indicators."""
    source_vector = np.asarray(vector, dtype=float)
    center = np.asarray(center_vector, dtype=float)
    if source_vector.shape != center.shape:
        raise ValueError(
            f"embedding shape {source_vector.shape} does not match center shape {center.shape}"
        )
    if not 1 <= display_dimensions <= len(source_vector):
        raise ValueError(
            f"display_dimensions must be between 1 and {len(source_vector)}, "
            f"got {display_dimensions}"
        )

    centered_vector = source_vector - center
    boundaries = np.linspace(0, len(source_vector), display_dimensions + 1, dtype=int)
    block_sums = np.add.reduceat(centered_vector, boundaries[:-1])
    return block_sums / np.sqrt(np.diff(boundaries))


def project_row(
    row: pd.Series,
    radar_data: RadarData,
    display_dimensions: int,
    projection_method: str,
) -> np.ndarray:
    """Project one embedding with the selected shared coordinate map."""
    if projection_method == "pca":
        if display_dimensions > radar_data.pca_dimension:
            raise ValueError(
                f"PCA supports at most {radar_data.pca_dimension} components, "
                f"got {display_dimensions}"
            )
        return np.asarray(row["pca_vector"], dtype=float)[:display_dimensions]
    if projection_method == "blocks":
        return project_coordinate_blocks(
            row["embedding"],
            radar_data.center_vector,
            display_dimensions,
        )
    raise ValueError(f"unknown projection method: {projection_method!r}")


def build_projection_labels(
    radar_data: RadarData,
    display_dimensions: int,
    projection_method: str,
) -> tuple[str, ...]:
    """Build axis labels for PCA components or raw coordinate blocks."""
    if projection_method == "pca":
        if not 1 <= display_dimensions <= radar_data.pca_dimension:
            raise ValueError(
                f"display_dimensions must be between 1 and {radar_data.pca_dimension}, "
                f"got {display_dimensions}"
            )
        return tuple(f"PC{component}" for component in range(1, display_dimensions + 1))
    if projection_method == "blocks":
        return build_block_labels(radar_data.embedding_dimension, display_dimensions)
    raise ValueError(f"unknown projection method: {projection_method!r}")


def get_radial_range(projected_vectors: np.ndarray) -> tuple[float, float]:
    """Return a padded range without forcing zero to the center of every chart."""
    radial_min = float(np.min(projected_vectors))
    radial_max = float(np.max(projected_vectors))
    span = radial_max - radial_min
    padding = max(span * 0.05, np.finfo(float).eps)
    return radial_min - padding, radial_max + padding


def build_radar_figure(
    rows: pd.DataFrame,
    dimension_labels: tuple[str, ...],
    selected_languages: list[str],
    language_styles: dict[str, dict[str, str]],
    chart_theme: dict[str, str],
    radar_data: RadarData,
    projection_method: str,
    radial_range: tuple[float, float],
) -> go.Figure:
    """Build an interactive Plotly radar chart for one canonical text."""
    figure = go.Figure()
    for language in COMPARISON_LANGUAGES:
        if language not in selected_languages:
            continue
        language_rows = rows[rows["language"] == language]
        if language_rows.empty:
            continue

        vector = project_row(
            language_rows.iloc[0],
            radar_data,
            len(dimension_labels),
            projection_method,
        )
        # Close the loop for radar chart (repeat first point at end)
        vector = np.concatenate([vector, vector[:1]])
        closed_labels = dimension_labels + (dimension_labels[0],)
        style = language_styles[language]
        show_detail = len(dimension_labels) <= 64
        figure.add_trace(
            go.Scatterpolar(
                r=vector,
                theta=closed_labels,
                name=f"{style['label']} ({language})",
                mode="lines+markers" if show_detail else "lines",
                fill="toself",
                fillcolor=style["background"],
                line={"color": style["border"], "width": 1.3 if show_detail else 1},
                marker={"color": style["border"], "size": 3.5},
                hovertemplate="%{fullData.name}<br>%{theta}: %{r:.3f}<extra></extra>",
            )
        )

    figure.update_layout(
        template="plotly_white",
        paper_bgcolor=chart_theme["surface"],
        plot_bgcolor=chart_theme["surface"],
        font={
            "color": chart_theme["ink"],
            "family": "Poppins, Avenir Next, Helvetica Neue, Arial, sans-serif",
            "size": 9,
        },
        polar={
            "bgcolor": chart_theme["surface"],
            "radialaxis": {
                "showticklabels": len(dimension_labels) <= 32,
                "color": chart_theme["quiet"],
                "gridcolor": chart_theme["grid"],
                "linecolor": chart_theme["grid"],
                "range": radial_range,
                "tickformat": ".3f",
                "tickfont": {"size": 8},
            },
            "angularaxis": {
                "direction": "clockwise",
                "color": chart_theme["quiet"],
                "gridcolor": chart_theme["grid"],
                "linecolor": chart_theme["grid"],
                "showgrid": len(dimension_labels) <= 32,
                "showticklabels": len(dimension_labels) <= 32,
                "tickfont": {"size": 8},
            },
        },
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.08,
            "font": {"color": chart_theme["muted"], "size": 8},
            "itemsizing": "constant",
            "itemwidth": 30,
        },
        margin={"l": 28, "r": 28, "t": 28, "b": 44},
        autosize=True,
        height=380,
        hovermode="closest",
    )
    return figure


def build_chart_cards(
    radar_data: RadarData,
    titles: dict[str, str],
    language_styles: dict[str, dict[str, str]],
    chart_theme: dict[str, str],
    display_dimensions: int,
    projection_method: str,
    scale_mode: str,
    selected_categories: list[str] | None,
    selected_texts: list[str] | None,
    selected_languages: list[str] | None,
) -> tuple[list[Any], str]:
    """Filter embedding rows and build the Dash chart cards and selection summary."""
    categories = selected_categories or []
    text_ids = selected_texts or []
    languages = selected_languages or []
    filtered = cast(
        pd.DataFrame,
        radar_data.frame.loc[
            radar_data.frame["category"].isin(categories)
            & radar_data.frame["text_id"].isin(text_ids)
        ],
    )
    if not languages:
        return [], "Select at least one language to display charts."
    if filtered.empty:
        return [], "No texts match the current category and text filters."

    plotted_rows = cast(pd.DataFrame, filtered.loc[filtered["language"].isin(languages)])
    if plotted_rows.empty:
        return [], "No embeddings match the selected text and language filters."

    dimension_labels = build_projection_labels(
        radar_data,
        display_dimensions,
        projection_method,
    )
    projected_vectors = np.stack(
        [
            project_row(
                row,
                radar_data,
                display_dimensions,
                projection_method,
            )
            for _, row in plotted_rows.iterrows()
        ]
    )
    shared_radial_range = get_radial_range(projected_vectors)
    if scale_mode not in {"local", "shared"}:
        raise ValueError(f"unknown scale mode: {scale_mode!r}")

    cards: list[Any] = []
    for figure_number, text_id in enumerate(sorted(filtered["text_id"].unique()), start=1):
        rows = cast(pd.DataFrame, filtered.loc[filtered["text_id"] == text_id])
        visible_rows = cast(pd.DataFrame, rows.loc[rows["language"].isin(languages)])
        if scale_mode == "local":
            text_vectors = np.stack(
                [
                    project_row(
                        row,
                        radar_data,
                        display_dimensions,
                        projection_method,
                    )
                    for _, row in visible_rows.iterrows()
                ]
            )
            radial_range = get_radial_range(text_vectors)
        else:
            radial_range = shared_radial_range
        category = str(rows.iloc[0]["category"])
        figure = build_radar_figure(
            rows,
            dimension_labels,
            languages,
            language_styles,
            chart_theme,
            radar_data,
            projection_method,
            radial_range,
        )
        if projection_method == "pca":
            explained = radar_data.pca_explained_variance_ratio[:display_dimensions].sum() * 100
            caption = (
                f"Global PCA · {display_dimensions} components · "
                f"{explained:.1f}% cumulative variance · {scale_mode} radial scale."
            )
        else:
            caption = (
                f"Coordinate map f{display_dimensions}: "
                f"ℝ{radar_data.embedding_dimension} → ℝ{display_dimensions}; "
                f"centered orthogonal blocks · {scale_mode} radial scale."
            )
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
                        className="radar-chart",
                        style={"height": "380px"},
                        config={
                            "displaylogo": False,
                            "responsive": True,
                            "toImageButtonOptions": {"filename": f"{text_id}_embedding_radar"},
                        },
                    ),
                    html.P(caption, className="figure-caption"),
                ],
                className="chart-card",
            )
        )

    projection_label = "PCA" if projection_method == "pca" else "coordinate blocks"
    range_note = (
        f"shared radial range {shared_radial_range[0]:.3f} to {shared_radial_range[1]:.3f}"
        if scale_mode == "shared"
        else "independent per-text radial ranges"
    )
    summary = ""
    return cards, summary


def create_app(radar_data: RadarData, run_dir: Path, initial_dimensions: int) -> Dash:
    """Create the Dash application and register its filter callback."""
    titles = load_titles()
    language_styles = load_language_styles()
    chart_theme = load_chart_theme()
    text_ids = sorted(radar_data.frame["text_id"].unique())
    categories = sorted(radar_data.frame["category"].unique())
    languages = [
        language
        for language in COMPARISON_LANGUAGES
        if language in radar_data.frame["language"].unique()
    ]
    if not 10 <= initial_dimensions <= radar_data.pca_dimension:
        raise ValueError(
            f"initial_dimensions must be between 10 and {radar_data.pca_dimension}, "
            f"got {initial_dimensions}"
        )

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
        title="Embedding Radar Charts",
        assets_folder=str(ASSETS_DIR),
    )
    controls = html.Div(
        [
            html.Div(
                [
                    html.Label("Categories", htmlFor="category-filter"),
                    dcc.Dropdown(
                        id="category-filter",
                        options=[{"label": value.title(), "value": value} for value in categories],
                        value=categories,
                        multi=True,
                    ),
                ],
                className="filter-field compact-multi",
            ),
            html.Div(
                [
                    html.Label("Projection", htmlFor="projection-method"),
                    dcc.Dropdown(
                        id="projection-method",
                        options=[
                            {"label": "PCA", "value": "pca"},
                            {"label": "Coordinate blocks", "value": "blocks"},
                        ],
                        value="pca",
                        clearable=False,
                        searchable=False,
                    ),
                ],
                className="filter-field",
            ),
            html.Div(
                [
                    html.Label("Scale", htmlFor="scale-mode"),
                    dcc.Dropdown(
                        id="scale-mode",
                        options=[
                            {"label": "Per text", "value": "local"},
                            {"label": "Shared", "value": "shared"},
                        ],
                        value="local",
                        clearable=False,
                        searchable=False,
                    ),
                ],
                className="filter-field",
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
                    html.Label(
                        [
                            "Dimensions ",
                            html.Output(
                                str(initial_dimensions),
                                id="dimension-value",
                            ),
                        ],
                        htmlFor="dimension-slider",
                    ),
                    dcc.Slider(
                        id="dimension-slider",
                        min=10,
                        max=radar_data.pca_dimension,
                        step=1,
                        value=initial_dimensions,
                        marks={
                            value: str(value)
                            for value in (10, 20, 32, radar_data.pca_dimension)
                            if value <= radar_data.pca_dimension
                        },
                        tooltip={"placement": "bottom"},
                        updatemode="mouseup",
                    ),
                ],
                className="filter-field dimension-field",
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
        return f"{experiment} · {area}"

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
                            html.Dt("embeddings:"),
                            html.Dd(str(radar_data.embedding_count)),
                            html.Dt("dimensions:"),
                            html.Dd(str(radar_data.embedding_dimension)),
                            html.Dt("projection:"),
                            html.Dd("global_pca"),
                            html.Dt("centroid_norm:"),
                            html.Dd(f"{radar_data.center_norm:.4f}"),
                        ],
                        className="run-metadata",
                    ),
                ],
                className="app-header",
            ),
            dcc.Loading(html.Div(id="radar-grid", className="charts-grid")),
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
        Output("dimension-slider", "max"),
        Output("dimension-slider", "marks"),
        Output("dimension-slider", "value"),
        Input("projection-method", "value"),
        State("dimension-slider", "value"),
    )
    def update_dimension_control(
        projection_method: str,
        current_dimensions: int,
    ) -> tuple[int, dict[int, str], int]:
        maximum = (
            radar_data.pca_dimension
            if projection_method == "pca"
            else radar_data.embedding_dimension
        )
        candidates = (
            (10, 20, 32, radar_data.pca_dimension)
            if projection_method == "pca"
            else (10, 32, 64, 128, 256, 512, radar_data.embedding_dimension)
        )
        marks = {value: str(value) for value in candidates if value <= maximum}
        return maximum, marks, min(current_dimensions, maximum)

    @app.callback(
        Output("radar-grid", "children"),
        Output("selection-summary", "children"),
        Output("dimension-value", "children"),
        Input("category-filter", "value"),
        Input("text-filter", "value"),
        Input("language-filter", "value"),
        Input("dimension-slider", "value"),
        Input("projection-method", "value"),
        Input("scale-mode", "value"),
    )
    def update_charts(
        selected_categories: list[str] | None,
        selected_texts: list[str] | None,
        selected_languages: list[str] | None,
        display_dimensions: int,
        projection_method: str,
        scale_mode: str,
    ) -> tuple[list[Any], str, str]:
        cards, summary = build_chart_cards(
            radar_data,
            titles,
            language_styles,
            chart_theme,
            display_dimensions,
            projection_method,
            scale_mode,
            selected_categories,
            selected_texts,
            selected_languages,
        )
        return cards, summary, str(display_dimensions)

    return app


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the Dash development server."""
    parser = argparse.ArgumentParser(description="Run interactive embedding radar charts")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--initial-dimensions", type=int, default=10)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Load data, report embedding provenance, and run the Dash application."""
    args = parse_args()
    frame = load_embeddings(args.run_dir)
    radar_data = build_radar_data(frame)

    print(f"Loaded {len(frame)} embeddings from {args.run_dir / 'embeddings.parquet'}")
    print(f"Embedding dimensions: {radar_data.embedding_dimension}")
    print(f"Initial display dimensions: {args.initial_dimensions}")

    app = create_app(radar_data, args.run_dir, args.initial_dimensions)
    print(f"Starting Dash app at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
