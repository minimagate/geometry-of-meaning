from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from scripts import build_compression_view as view


def test_load_reference_embeddings_requires_existing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing-baseline"

    with pytest.raises(FileNotFoundError):
        view.load_reference_embeddings(missing)


def test_main_uses_explicit_reference_run_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    run_frame = pd.DataFrame(
        [{"text_id": "text_a", "language": "en", "category": "novel", "compression_level": 1.0, "embedding": [1.0, 0.0]}]
    )
    reference_frame = pd.DataFrame(
        [{"text_id": "text_a", "language": "en", "category": "novel", "compression_level": 1.0, "embedding": [0.0, 1.0]}]
    )

    calls: list[Path] = []
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        view,
        "parse_args",
        lambda: Namespace(
            run_dir=Path("/tmp/compression-run"),
            reference_run_dir=Path("/tmp/baseline-run"),
            initial_dimensions=10,
            host="127.0.0.1",
            port=8051,
            debug=False,
        ),
    )
    monkeypatch.setattr(view, "load_embeddings", lambda run_dir: calls.append(run_dir) or run_frame)
    monkeypatch.setattr(
        view,
        "load_reference_embeddings",
        lambda run_dir: calls.append(run_dir) or reference_frame,
    )

    def fake_build_compression_data(frame: pd.DataFrame, reference_frame: pd.DataFrame | None = None):
        captured["frame"] = frame
        captured["reference_frame"] = reference_frame
        return view.CompressionData(
            frame=frame,
            embedding_count=len(frame),
            embedding_dimension=2,
            center_vector=pd.Series([0.0, 0.0]).to_numpy(),
            pca_explained_variance_ratio=pd.Series([1.0, 0.0]).to_numpy(),
        )

    class FakeApp:
        def run(self, **kwargs):
            captured["run_kwargs"] = kwargs

    monkeypatch.setattr(view, "build_compression_data", fake_build_compression_data)
    monkeypatch.setattr(view, "create_app", lambda compression_data, run_dir, initial_dimensions: FakeApp())

    view.main()

    assert calls == [Path("/tmp/compression-run"), Path("/tmp/baseline-run")]
    assert captured["frame"] is run_frame
    assert captured["reference_frame"] is reference_frame
    assert captured["run_kwargs"] == {"host": "127.0.0.1", "port": 8051, "debug": False}
