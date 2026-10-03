from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.ensure_curated_index as index_startup
from scripts.ensure_curated_index import ensure_curated_index


class _Store:
    def __init__(self, count: int) -> None:
        self.records = count

    def count(self) -> int:
        return self.records


def test_completed_index_is_reused_with_matching_identity(tmp_path: Path) -> None:
    builds = 0
    store = _Store(3)
    marker = tmp_path / "index.json"

    def build() -> int:
        nonlocal builds
        builds += 1
        return 0

    def store_factory(_path: Path, _collection: str) -> _Store:
        return store

    assert ensure_curated_index(
        build=build, store_factory=store_factory, marker_path=marker
    ) is True
    assert ensure_curated_index(
        build=build, store_factory=store_factory, marker_path=marker
    ) is False
    assert builds == 1


def test_incomplete_index_is_built_and_marked(tmp_path: Path) -> None:
    store = _Store(0)
    opened = 0

    def store_factory(_path: Path, _collection: str) -> _Store:
        nonlocal opened
        opened += 1
        return store

    def build() -> int:
        store.records = 5
        return 0

    assert ensure_curated_index(
        build=build, store_factory=store_factory, marker_path=tmp_path / "index.json"
    ) is True
    assert opened == 2


def test_failed_or_empty_index_build_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Could not build"):
        ensure_curated_index(
            build=lambda: 1,
            store_factory=lambda *_: _Store(0),
            marker_path=tmp_path / "failed-index.json",
        )

    with pytest.raises(RuntimeError, match="still empty"):
        ensure_curated_index(
            build=lambda: 0,
            store_factory=lambda *_: _Store(0),
            marker_path=tmp_path / "empty-index.json",
        )


def test_incompatible_existing_index_fails_loudly(tmp_path: Path) -> None:
    marker = tmp_path / "index.json"
    store_factory = lambda *_: _Store(3)
    ensure_curated_index(
        build=lambda: 0, store_factory=store_factory, marker_path=marker
    )
    marker.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="different corpus or indexing configuration"):
        ensure_curated_index(
            build=lambda: 0, store_factory=store_factory, marker_path=marker
        )


def test_corpus_checksum_mismatch_prevents_startup(tmp_path: Path, monkeypatch) -> None:
    manifest = json.loads(index_startup.CORPUS_MANIFEST.read_text(encoding="utf-8"))
    manifest["documents"][0]["sha256"] = "invalid-checksum"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(index_startup, "CORPUS_MANIFEST", manifest_path)

    with pytest.raises(RuntimeError, match="Corpus checksum mismatch"):
        ensure_curated_index(store_factory=lambda *_: _Store(5))
