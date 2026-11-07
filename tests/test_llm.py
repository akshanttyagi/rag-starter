from app.llm import NOT_FOUND, build_messages, cited_numbers


def test_cited_numbers_in_order_and_deduplicated():
    answer = "Thirty years [2]. Also nine years a citizen [2][1]. See [1, 3]."
    assert cited_numbers(answer, n_sources=3) == [2, 1, 3]


def test_out_of_range_citations_are_ignored():
    assert cited_numbers("Claim [0] and [7].", n_sources=4) == []


def test_no_citations():
    assert cited_numbers("I could not find this in the documents.", n_sources=4) == []


def test_build_messages_numbers_sources():
    messages = build_messages("Q?", [("a.txt", "alpha"), ("b.pdf, page 2", "beta")])
    assert messages[0]["role"] == "system"
    assert [m["role"] for m in messages[1:]] == ["user", "assistant"] * 2 + ["user"]
    user = messages[-1]["content"]
    assert "[1] (a.txt)\nalpha" in user
    assert "[2] (b.pdf, page 2)\nbeta" in user
    assert user.endswith("Question: Q?")


def test_examples_show_both_citing_and_refusing():
    answers = [m["content"] for m in build_messages("Q?", []) if m["role"] == "assistant"]
    assert cited_numbers(answers[0], n_sources=2) == [1]
    assert answers[1] == NOT_FOUND
