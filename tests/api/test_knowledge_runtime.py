"""Configured API retrieval adapter tests with a fake embedder and vector store."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from rag_phy.api import ConfiguredKnowledgeSearch
from rag_phy.api.service import create_app
from rag_phy.config import EmbeddingConfig
from rag_phy.knowledge import RetrievalResult


class _FakeEmbedder:
    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.25, 0.75]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.25, 0.75] for _ in texts]


class _FakeVectorStore:
    def __init__(self) -> None:
        self.query_vectors: list[list[float]] = []
        self.limits: list[int] = []

    def upsert(self, chunks, embeddings) -> None:
        raise AssertionError("Search-only adapter must not ingest documents")

    def search(self, query_embedding, limit: int) -> list[RetrievalResult]:
        self.query_vectors.append(list(query_embedding))
        self.limits.append(limit)
        return [
            RetrievalResult(
                chunk_id="chunk-1",
                text="Source-backed passage",
                distance=0.12,
                source_refs=("report.pdf#page=7",),
            )
        ]


def test_configured_knowledge_search_is_lazy_and_preserves_sources() -> None:
    embedder_instances: list[_FakeEmbedder] = []
    stores: list[_FakeVectorStore] = []

    def make_embedder(config: EmbeddingConfig) -> _FakeEmbedder:
        embedder = _FakeEmbedder(config)
        embedder_instances.append(embedder)
        return embedder

    def make_store(path: Path, collection_name: str) -> _FakeVectorStore:
        assert path == Path("data/chroma")
        assert collection_name == "rag_phy_documents"
        store = _FakeVectorStore()
        stores.append(store)
        return store

    search = ConfiguredKnowledgeSearch(
        embedder_factory=make_embedder,
        vector_store_factory=make_store,
    )
    assert not embedder_instances
    assert not stores

    with TestClient(create_app(knowledge_search=search)) as client:
        response = client.get("/knowledge/search", params={"q": "compressor pressure ratio"})

    assert response.status_code == 200
    results = response.json()

    assert len(embedder_instances) == 1
    assert embedder_instances[0].queries == ["compressor pressure ratio"]
    assert stores[0].query_vectors == [[0.25, 0.75]]
    assert stores[0].limits == [5]
    assert results[0] == {
        "text": "Source-backed passage",
        "source_refs": ["report.pdf#page=7"],
        "distance": 0.12,
    }

    search("another query")
    assert len(embedder_instances) == 1
    assert len(stores) == 1
