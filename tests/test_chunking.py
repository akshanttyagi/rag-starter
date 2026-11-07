import itertools

import pytest

from app.chunking import chunk_text, split_sentences


def words(n: int, start: int = 0) -> str:
    """A sentence of n distinct words, e.g. "W0 w1 w2."."""
    return " ".join(f"w{i}" for i in range(start, start + n)).capitalize() + "."


def test_empty_text_gives_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_short_text_is_one_chunk():
    assert chunk_text("One sentence. Another one.") == ["One sentence. Another one."]


def test_split_sentences_keeps_punctuation():
    assert split_sentences("First line\nwraps here. Second! Third? Fourth") == [
        "First line wraps here.",
        "Second!",
        "Third?",
        "Fourth",
    ]


def test_chunks_respect_max_words():
    text = " ".join(words(30, i * 30) for i in range(20))  # 600 words, 20 sentences
    chunks = chunk_text(text, max_words=100, overlap_words=0)
    assert len(chunks) == 7  # 3 sentences (90 words) per chunk, 2 in the last
    assert all(len(c.split()) <= 100 for c in chunks)


def test_overlap_repeats_trailing_sentences():
    sentences = [words(30, i * 30) for i in range(10)]
    chunks = chunk_text(" ".join(sentences), max_words=100, overlap_words=40)
    assert len(chunks) > 1
    for previous, current in itertools.pairwise(chunks):
        last_sentence = previous.split(". ")[-1].rstrip(".")
        assert current.startswith(last_sentence)


def test_long_sentence_is_split_by_words():
    chunks = chunk_text(words(250), max_words=100, overlap_words=0)
    assert [len(c.split()) for c in chunks] == [100, 100, 50]


def test_short_tail_is_merged_into_previous_chunk():
    text = words(95) + " " + words(5, 1000)
    chunks = chunk_text(text, max_words=100, overlap_words=0)
    assert len(chunks) == 1
    assert chunks[0].endswith("w1004.")


def test_paragraph_breaks_are_kept():
    chunks = chunk_text("First paragraph.\n\nSecond paragraph.", max_words=50)
    assert chunks == ["First paragraph.\n\nSecond paragraph."]


@pytest.mark.parametrize("max_words,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_sizes_raise(max_words, overlap):
    with pytest.raises(ValueError):
        chunk_text("text", max_words=max_words, overlap_words=overlap)
