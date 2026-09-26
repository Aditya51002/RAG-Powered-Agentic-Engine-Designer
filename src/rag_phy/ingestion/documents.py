"""Document loading for text, Markdown, and text-based PDF sources."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentParseError(ValueError):
    """Raised when a source cannot be parsed or contains no extractable text."""


@dataclass(frozen=True)
class SourceDocument:
    """A parsed text unit with a stable source reference."""

    source_ref: str
    title: str
    text: str


class DocumentLoader:
    """Load supported source files into page-level text documents."""

    def load(self, path: str | Path) -> list[SourceDocument]:
        """Parse a UTF-8 text/Markdown file or PDF into non-empty source documents.

        Args:
            path: Source file path.

        Returns:
            Parsed text units; PDFs return one unit per page.

        Raises:
            FileNotFoundError: If the source file is missing.
            DocumentParseError: For unsupported formats, parse errors, or empty text.
        """
        source_path = Path(path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Document source does not exist: {source_path}")
        suffix = source_path.suffix.casefold()
        stable_path = source_path.as_posix() if not source_path.is_absolute() else str(source_path)
        try:
            if suffix in {".txt", ".md", ".markdown"}:
                text = source_path.read_text(encoding="utf-8")
                documents = [
                    SourceDocument(
                        source_ref=stable_path,
                        title=source_path.name,
                        text=text,
                    )
                ]
            elif suffix == ".pdf":
                documents = self._load_pdf(source_path, stable_path)
            else:
                raise DocumentParseError(f"Unsupported document format: {suffix or '(no suffix)'}")
        except (FileNotFoundError, DocumentParseError):
            raise
        except Exception as exc:
            logger.exception("Document parsing failed", extra={"source": str(source_path)})
            raise DocumentParseError(f"Could not parse document {source_path}") from exc

        non_empty = [document for document in documents if document.text.strip()]
        if not non_empty:
            raise DocumentParseError(f"Document contains no extractable text: {source_path}")
        return non_empty

    @staticmethod
    def _load_pdf(path: Path, source_ref: str | None = None) -> list[SourceDocument]:
        """Extract page-level text from a PDF using PyMuPDF."""
        try:
            import pymupdf
        except ImportError as exc:
            raise DocumentParseError(
                "PDF parsing requires the optional PyMuPDF dependency"
            ) from exc

        with pymupdf.open(path) as pdf:
            return [
                SourceDocument(
                    source_ref=f"{source_ref or path}#page={page_number}",
                    title=path.name,
                    text=page.get_text("text", sort=True),
                )
                for page_number, page in enumerate(pdf, start=1)
            ]


__all__ = ["DocumentLoader", "DocumentParseError", "SourceDocument"]
