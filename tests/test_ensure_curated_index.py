from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ensure_curated_index import ensure_curated_index


class _Store:
    def __init__(self, count: int) -> None:
        self.records = count

    def count(self) -> int:
        return self.records


def test_existing_index_is_reused() -> None:
    builds = 0

    def build() -> int:
        nonlocal builds
        builds += 1
        return 0

    assert ensure_curated_index(build=build, store_factory=lambda *_: _Store(3)) is False
    assert builds == 0


def test_empty_index_is_built_and_verified() -> None:
    store = _Store(0)
    opened = 0

    def store_factory(_path: Path, _collection: str) -> _Store:
        nonlocal opened
        opened += 1
        return store

    def build() -> int:
        store.records = 5
        return 0

    assert ensure_curated_index(build=build, store_factory=store_factory) is True
    assert opened == 2


def test_failed_or_empty_index_build_fails_loudly() -> None:
    with pytest.raises(RuntimeError, match="Could not build"):
        ensure_curated_index(build=lambda: 1, store_factory=lambda *_: _Store(0))

    with pytest.raises(RuntimeError, match="still empty"):
        ensure_curated_index(build=lambda: 0, store_factory=lambda *_: _Store(0))
