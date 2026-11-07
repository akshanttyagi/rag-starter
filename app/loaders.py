"""Turn PDF, DOCX and plain-text files into text, page by page."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@dataclass
class Page:
    text: str
    number: int | None  # 1-based page number for PDFs, None for formats without pages


@dataclass
class LoadedDocument:
    title: str
    pages: list[Page]


class UnsupportedFileType(ValueError):
    pass


def load(path: Path) -> LoadedDocument:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _load_pdf(path)
    if ext == ".docx":
        return _load_docx(path)
    if ext in (".txt", ".md"):
        text = path.read_text(encoding="utf-8", errors="replace")
        return LoadedDocument(title=_title_from_text(text, path), pages=[Page(text, None)])
    raise UnsupportedFileType(
        f"{path.name}: unsupported file type (use {', '.join(sorted(SUPPORTED_EXTENSIONS))})"
    )


def _title_from_name(path: Path) -> str:
    return path.stem.replace("-", " ").replace("_", " ").strip() or path.name


_SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to"}


def _title_from_text(text: str, path: Path) -> str:
    """Use a short first line as the title ("# Heading" or "TITLE IN CAPS"), else the file name."""
    first = next((ln.strip().lstrip("#").strip() for ln in text.splitlines() if ln.strip()), "")
    if not first or len(first) > 80:
        return _title_from_name(path)
    if first.isupper():
        words = first.lower().split()
        first = " ".join(
            w if (i and w in _SMALL_WORDS) else w.capitalize() for i, w in enumerate(words)
        )
    return first


def _load_pdf(path: Path) -> LoadedDocument:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    raw = [page.extract_text() or "" for page in reader.pages]
    cleaned = [_fix_pdf_text(t) for t in strip_repeated_lines(raw)]
    meta_title = (reader.metadata.title if reader.metadata else None) or ""
    title = meta_title.strip() or _title_from_name(path)
    pages = [Page(text, i) for i, text in enumerate(cleaned, start=1) if text.strip()]
    return LoadedDocument(title=title, pages=pages)


def _fix_pdf_text(text: str) -> str:
    # Re-join words hyphenated across line breaks: "manage-\nment" -> "management".
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def strip_repeated_lines(
    pages: list[str], edge_lines: int = 4, min_share: float = 0.5
) -> list[str]:
    """Remove running headers, footers and page numbers.

    A line counts as boilerplate when it sits in the first or last few lines of a
    page and the same line (digits ignored) shows up on at least ``min_share`` of
    all pages. Body text is left alone.
    """
    if len(pages) < 4:
        return pages

    def norm(line: str) -> str:
        return re.sub(r"\d+", "#", " ".join(line.split()).lower())

    def edges(lines: list[str]) -> list[str]:
        return lines[:edge_lines] + lines[-edge_lines:]

    counts: Counter[str] = Counter()
    for text in pages:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        counts.update({norm(ln) for ln in edges(lines)})
    threshold = max(3, int(min_share * len(pages)))
    repeated = {line for line, n in counts.items() if n >= threshold}

    out = []
    for text in pages:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        tail_start = max(edge_lines, len(lines) - edge_lines)
        head, middle, tail = lines[:edge_lines], lines[edge_lines:tail_start], lines[tail_start:]
        keep = [ln for ln in head if norm(ln) not in repeated]
        keep += middle
        keep += [ln for ln in tail if norm(ln) not in repeated]
        out.append("\n".join(keep))
    return out


def _load_docx(path: Path) -> LoadedDocument:
    import docx

    document = docx.Document(str(path))
    blocks: list[str] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            blocks.append(text)
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                blocks.append(" | ".join(cells))
    title = (document.core_properties.title or "").strip() or _title_from_name(path)
    return LoadedDocument(title=title, pages=[Page("\n\n".join(blocks), None)])
