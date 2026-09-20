"""Tests für das MVP-Suchsystem (``ai_agents.search_engine_cli.SearchEngine``).

Ausgangslage (Defekt): ``test_system_components`` gab ausschließlich
``True``/``False`` zurück (pytest verwirft Rückgabewerte → immer grün, „stilles
Grün“) und las aus den Produktionspfaden ``raw_construction_kits/`` und
``processed_database/embeddings.pkl``. ``demonstrate_search`` war reine
Statusausgabe ganz ohne Assertion.

Jetzt: echte Assertions auf einem temporären Embedding-Index in ``tmp_path``.
Zusätzlich abgedeckt: ein leerer Index liefert keine Treffer statt eines
Shape-Fehlers (vorher unbehandelt).
"""

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from ai_agents.search_engine_cli import SearchEngine

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_index(index_path: Path, paths: list, embeddings: np.ndarray) -> Path:
    """Schreibt einen Embedding-Index im Format des Suchsystems."""
    payload = [
        {"path": str(path), "embedding": embedding}
        for path, embedding in zip(paths, embeddings)
    ]
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_bytes(pickle.dumps(payload))
    return index_path


@pytest.fixture
def embeddings_index(tmp_path: Path):
    """Temporärer Index mit 6 Stems à 512 Dimensionen."""
    rng = np.random.default_rng(1234)
    stems_dir = tmp_path / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for index in range(6):
        stem = stems_dir / f"stem_{index:02d}.wav"
        stem.write_bytes(b"RIFF....WAVEfmt ")
        paths.append(stem)

    embeddings = rng.normal(size=(len(paths), 512)).astype(np.float32)
    index_path = _write_index(tmp_path / "embeddings.pkl", paths, embeddings)

    return {"index_path": index_path, "paths": [str(p) for p in paths], "dir": tmp_path}


def test_engine_loads_every_embedding(embeddings_index):
    """Der Index wird vollständig geladen — Anzahl und Form sind prüfbar."""
    engine = SearchEngine(str(embeddings_index["index_path"]))

    assert engine.file_paths == embeddings_index["paths"]
    assert engine.embedding_tensors.shape == (len(embeddings_index["paths"]), 512)
    assert engine.embedding_tensors.dtype == torch.float32


def test_search_respects_top_k_and_returns_scores(embeddings_index):
    """``search`` liefert höchstens ``top_k`` Treffer mit absteigender Ähnlichkeit."""
    engine = SearchEngine(str(embeddings_index["index_path"]))

    results = engine.search("kick drum", top_k=3)

    assert len(results) == 3
    for entry in results:
        assert isinstance(entry, tuple) and len(entry) == 2
        path, score = entry
        assert path in embeddings_index["paths"]
        assert isinstance(score, float)

    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True), f"Treffer nicht sortiert: {scores}"


def test_search_with_top_k_beyond_index_returns_all(embeddings_index):
    """Ein zu großes ``top_k`` ist kein Fehler, sondern liefert alle Treffer."""
    engine = SearchEngine(str(embeddings_index["index_path"]))

    results = engine.search("dark industrial atmosphere", top_k=99)

    assert len(results) == len(embeddings_index["paths"])
    assert {path for path, _ in results} == set(embeddings_index["paths"])


def test_search_does_not_write_anything(tmp_path: Path, embeddings_index):
    """Die Suche ist rein lesend — kein neuer Dateiname neben dem Index."""
    engine = SearchEngine(str(embeddings_index["index_path"]))
    before = sorted(p.name for p in embeddings_index["dir"].rglob("*"))

    engine.search("punchy attack", top_k=2)
    engine.search("hypnotic arpeggio", top_k=1)

    after = sorted(p.name for p in embeddings_index["dir"].rglob("*"))
    assert after == before


def test_empty_index_returns_no_results(tmp_path: Path):
    """Ein leerer Index führt zu ``[]`` — nicht zu einem RuntimeError."""
    index_path = _write_index(tmp_path / "empty.pkl", [], np.zeros((0, 512)))

    engine = SearchEngine(str(index_path))

    assert engine.file_paths == []
    assert engine.search("anything", top_k=3) == []


def test_missing_index_raises_instead_of_silently_succeeding(tmp_path: Path):
    """Ein fehlender Index ist ein Fehler — kein stilles Grün."""
    with pytest.raises(FileNotFoundError):
        SearchEngine(str(tmp_path / "does-not-exist.pkl"))


def test_production_index_is_not_used(embeddings_index):
    """Der Test benutzt den temporären Index, nicht ``processed_database/``."""
    engine = SearchEngine(str(embeddings_index["index_path"]))

    assert all(
        REPO_ROOT not in Path(path).parents for path in engine.file_paths
    ), engine.file_paths
    assert str(embeddings_index["index_path"]).startswith(str(embeddings_index["dir"]))
