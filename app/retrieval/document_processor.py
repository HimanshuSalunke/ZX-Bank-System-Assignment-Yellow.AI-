"""
ZX Bank AI — Document Processor.

Loads markdown files from ``data/documents/``, splits them into
structured chunks preserving heading hierarchy, and attaches rich
metadata to each chunk for downstream retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_text_splitters import MarkdownHeaderTextSplitter

from app.config import DOCUMENTS_DIR, get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__, component="document_processor")

# ── Headers to split on (preserves section hierarchy) ─────────────────────
_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


@dataclass
class DocumentChunk:
    """A single chunk of a document with metadata."""

    content: str
    doc_title: str
    doc_filename: str
    section_heading: str = ""
    subsection_heading: str = ""
    doc_type: str = "general"
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """Unique identifier for this chunk."""
        return f"{self.doc_filename}::{self.chunk_index}"

    @property
    def display_source(self) -> str:
        """Human-readable source reference."""
        parts = [self.doc_title]
        if self.section_heading:
            parts.append(self.section_heading)
        if self.subsection_heading:
            parts.append(self.subsection_heading)
        return " > ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "doc_title": self.doc_title,
            "doc_filename": self.doc_filename,
            "section_heading": self.section_heading,
            "subsection_heading": self.subsection_heading,
            "doc_type": self.doc_type,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }


def _extract_doc_title(content: str, filename: str) -> str:
    """Extract the top-level heading (h1) from markdown content."""
    match = re.search(r"^#\s+(.+?)(?:\s*$)", content, re.MULTILINE)
    if match:
        return match.group(1).strip().replace("ZX Bank — ", "").replace("ZX Bank — ", "")
    # Fallback: derive from filename
    name = filename.replace(".md", "").lstrip("0123456789_")
    return name.replace("_", " ").title()


def _classify_doc_type(filename: str) -> str:
    """Classify document type from filename for metadata enrichment."""
    mapping = {
        "about": "company_info",
        "savings": "account",
        "current": "account",
        "fixed_deposit": "deposit",
        "home_loan": "loan",
        "personal_loan": "loan",
        "education_loan": "loan",
        "business": "loan",
        "vehicle": "loan",
        "credit_card": "card",
        "debit_card": "card",
        "digital": "digital_banking",
        "mobile": "digital_banking",
        "security": "security",
        "fraud": "security",
        "fees": "charges",
        "charges": "charges",
        "wealth": "investment",
        "forex": "forex",
        "nri": "nri",
        "customer": "support",
        "support": "support",
        "grievance": "support",
        "atm": "infrastructure",
        "branch": "infrastructure",
        "government": "government_scheme",
        "scheme": "government_scheme",
        "account_management": "procedures",
        "terms": "regulatory",
        "policy": "regulatory",
    }
    lower = filename.lower()
    for key, doc_type in mapping.items():
        if key in lower:
            return doc_type
    return "general"


def load_and_split_documents(
    documents_dir: Path | None = None,
) -> list[DocumentChunk]:
    """
    Load all markdown files and split into structured chunks.

    Returns a flat list of ``DocumentChunk`` objects ready for indexing.
    """
    docs_dir = documents_dir or DOCUMENTS_DIR

    if not docs_dir.exists():
        logger.warning("documents_dir_missing", path=str(docs_dir))
        return []

    md_files = sorted(docs_dir.glob("*.md"))
    if not md_files:
        logger.warning("no_markdown_files", path=str(docs_dir))
        return []

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )

    all_chunks: list[DocumentChunk] = []

    for md_file in md_files:
        raw_text = md_file.read_text(encoding="utf-8")
        doc_title = _extract_doc_title(raw_text, md_file.name)
        doc_type = _classify_doc_type(md_file.name)

        splits = splitter.split_text(raw_text)

        for idx, split in enumerate(splits):
            section = split.metadata.get("h2", "")
            subsection = split.metadata.get("h3", "")
            content = split.page_content.strip()

            if not content or len(content) < 20:
                continue

            chunk = DocumentChunk(
                content=content,
                doc_title=doc_title,
                doc_filename=md_file.name,
                section_heading=section,
                subsection_heading=subsection,
                doc_type=doc_type,
                chunk_index=idx,
                metadata={
                    "h1": split.metadata.get("h1", doc_title),
                    "h2": section,
                    "h3": subsection,
                    "char_count": len(content),
                    "doc_type": doc_type,
                },
            )
            all_chunks.append(chunk)

    logger.info(
        "documents_loaded",
        total_files=len(md_files),
        total_chunks=len(all_chunks),
        avg_chunk_size=round(
            sum(len(c.content) for c in all_chunks) / max(len(all_chunks), 1)
        ),
    )
    return all_chunks
