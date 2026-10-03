"""Readiness checks use configured paths and a fake vector-store connection."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_phy.api import make_readiness_check
from rag_phy.api.service import create_app
from rag_phy.knowledge import ChromaVectorStore


def test_readiness_loads_configuration_and_probes_the_configured_collection() -> None:
    observed: list[tuple[Path, str]] = []

    class Store:
        def check_ready(self) -> None:
            pass

    def factory(path: Path, collection: str) -> Store:
        observed.append((path, collection))
        return Store()

    check = make_readiness_check(vector_store_factory=factory)

    assert check()
    assert observed == [(Path("data/chroma"), "rag_phy_documents")]


def test_readiness_configuration_failure_returns_not_ready(tmp_path: Path) -> None:
    invalid_config = tmp_path / "app.yaml"
    invalid_config.write_text("logging: [invalid", encoding="utf-8")
    check = make_readiness_check(app_config_path=invalid_config)

    with TestClient(create_app(readiness_check=check)) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_chroma_readiness_operation_raises_typed_error_when_store_is_down(
    tmp_path: Path,
) -> None:
    class Collection:
        def count(self) -> int:
            raise OSError("unavailable")

    class Client:
        def get_or_create_collection(self, **kwargs) -> Collection:
            return Collection()

    store = ChromaVectorStore(tmp_path, "test", client=Client())

    with pytest.raises(RuntimeError, match="not reachable"):
        store.check_ready()


def test_chroma_readiness_rejects_an_empty_collection(tmp_path: Path) -> None:
    class Collection:
        def count(self) -> int:
            return 0

    class Client:
        def get_or_create_collection(self, **kwargs) -> Collection:
            return Collection()

    store = ChromaVectorStore(tmp_path, "test", client=Client())

    with pytest.raises(RuntimeError, match="collection is empty"):
        store.check_ready()
