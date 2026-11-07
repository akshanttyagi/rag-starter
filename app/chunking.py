"""Split text into overlapping, sentence-aligned chunks.

Chunks are packed from whole sentences up to ``max_words`` words, and each new
chunk repeats the last sentences of the previous one (up to ``overlap_words``)
so that a fact sitting on a boundary is still found in one piece. Paragraph
breaks inside a chunk are kept. Word counts stand in for tokens: 200 English
words is roughly 260 tokens, well inside the 512-token limit of the default
embedding and reranker models.
"""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+(?=[\"'(\[]?[A-Z0-9])")


def split_sentences(paragraph: str) -> list[str]:
    text = " ".join(paragraph.split())
    if not text:
        return []
    return [s for s in _SENTENCE_END.split(text) if s]


def _split_long(sentence: str, max_words: int) -> list[str]:
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def _units(text: str, max_words: int) -> list[tuple[str, bool]]:
    """Sentences as (text, starts_paragraph) pairs."""
    units: list[tuple[str, bool]] = []
    for paragraph in re.split(r"\n\s*\n", text):
        first = True
        for sentence in split_sentences(paragraph):
            for piece in _split_long(sentence, max_words):
                units.append((piece, first))
                first = False
    return units


def _words(unit: tuple[str, bool]) -> int:
    return len(unit[0].split())


def _join(units: list[tuple[str, bool]]) -> str:
    out = ""
    for i, (text, new_paragraph) in enumerate(units):
        if i == 0:
            out = text
        else:
            out += ("\n\n" if new_paragraph else " ") + text
    return out


def chunk_text(text: str, max_words: int = 200, overlap_words: int = 40) -> list[str]:
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be between 0 and max_words - 1")

    units = _units(text, max_words)
    if not units:
        return []

    chunks: list[list[tuple[str, bool]]] = []
    fresh: list[int] = []  # how many units in each chunk are new (not overlap)
    current: list[tuple[str, bool]] = []
    current_words = 0
    current_fresh = 0

    for unit in units:
        n = _words(unit)
        if current and current_words + n > max_words:
            chunks.append(current)
            fresh.append(current_fresh)
            tail: list[tuple[str, bool]] = []
            tail_words = 0
            for previous in reversed(current):
                if tail_words + _words(previous) > overlap_words:
                    break
                tail.insert(0, previous)
                tail_words += _words(previous)
            if tail_words + n > max_words:
                tail, tail_words = [], 0
            current, current_words, current_fresh = tail, tail_words, 0
        current.append(unit)
        current_words += n
        current_fresh += 1
    chunks.append(current)
    fresh.append(current_fresh)

    # A short leftover reads badly on its own; fold its new sentences into the previous chunk.
    if len(chunks) > 1:
        last_new = chunks[-1][len(chunks[-1]) - fresh[-1] :]
        if sum(_words(u) for u in last_new) < max_words // 4:
            chunks[-2] = chunks[-2] + last_new
            chunks.pop()

    return [_join(c) for c in chunks]
