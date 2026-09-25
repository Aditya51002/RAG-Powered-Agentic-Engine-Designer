from __future__ import annotations

from pathlib import Path
from typing import Sequence

import chromadb
import pymupdf
import pytest

from rag_phy.config import ModelsConfig, load_models_config
from rag_phy.ingestion import DocumentLoader, IngestionPipeline, SentenceTransformerEmbedder
from rag_phy.knowledge import ChromaVectorStore


class FixtureEmbeddingModel:
    """Test-only embedding double mapping fixture topics to orthogonal vectors."""

    def get_sentence_embedding_dimension(self) -> int:
        """Report the BGE-small dimensionality expected by the application config."""
        return 384

    @property
    def max_seq_length(self) -> int:
        """Return the configured test model token limit."""
        return 512

    @property
    def tokenizer(self) -> FixtureTokenizer:
        """Return a lightweight tokenizer used only to test input-bound enforcement."""
        return FixtureTokenizer()

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int,
        show_progress_bar: bool,
        normalize_embeddings: bool,
    ) -> list[list[float]]:
        """Return fixture-only vectors to exercise dimensionality and index contracts."""
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * 384
            topic = text.casefold()
            vector[0 if "compressor" in topic else 1] = 1.0
            vectors.append(vector)
        return vectors


class FixtureTokenizer:
    """Test-only token counter for fixture text."""

    def encode(self, text: str, add_special_tokens: bool) -> list[str]:
        """Split fixture input on whitespace into deterministic token placeholders."""
        return text.split()


@pytest.fixture
def models_config() -> ModelsConfig:
    """Load the typed model and vector configuration from the project YAML file."""
    return load_models_config(Path(__file__).parents[2] / "config" / "models.yaml")


def test_fixture_ingestion_deduplicates_and_retrieves_source_backed_chunk(
    models_config: ModelsConfig,
) -> None:
    """Index fixture docs in Chroma, check BGE shape, deduplication, and retrieval."""
    fixture_root = Path(__file__).parents[1] / "fixtures" / "documents"
    source_paths = [
        fixture_root / "compressor_notes.txt",
        fixture_root / "compressor_notes_duplicate.txt",
        fixture_root / "turbine_notes.txt",
    ]
    embedder = SentenceTransformerEmbedder(models_config.embedding, model=FixtureEmbeddingModel())
    vector_store = ChromaVectorStore(
        persist_directory=models_config.vector_store.persist_directory,
        collection_name="test_ingestion",
        client=chromadb.EphemeralClient(),
    )
    pipeline = IngestionPipeline(models_config, DocumentLoader(), embedder, vector_store)

    report = pipeline.ingest(source_paths)
    results = pipeline.retrieve("compressor pressure ratio", limit=1)

    assert report.source_documents_loaded == 3
    assert report.unique_chunks_indexed == 2
    assert report.duplicate_chunks_removed == 1
    assert results[0].text.startswith("SYNTHETIC TEST FIXTURE")
    assert len(results[0].source_refs) == 2
    assert len(embedder.embed_query("dimensionality check")) == 384


def test_document_loader_rejects_empty_and_unsupported_files(tmp_path: Path) -> None:
    """Make parser gaps visible rather than silently indexing empty content."""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("  ", encoding="utf-8")
    unsupported_file = tmp_path / "source.docx"
    unsupported_file.write_text("not parsed", encoding="utf-8")

    with pytest.raises(ValueError, match="no extractable text"):
        DocumentLoader().load(empty_file)
    with pytest.raises(ValueError, match="Unsupported document format"):
        DocumentLoader().load(unsupported_file)


def test_pdf_loader_preserves_page_level_source_references(tmp_path: Path) -> None:
    """Extract each text PDF page as a separately citable source unit."""
    pdf_path = tmp_path / "two_pages.pdf"
    with pymupdf.open() as pdf:
        first_page = pdf.new_page()
        first_page.insert_text((72, 72), "Synthetic page one text")
        second_page = pdf.new_page()
        second_page.insert_text((72, 72), "Synthetic page two text")
        pdf.save(pdf_path)

    documents = DocumentLoader().load(pdf_path)

    assert len(documents) == 2
    assert documents[0].source_ref.endswith("#page=1")
    assert "page two" in documents[1].text
